from torch import nn
from torch.distributions import Normal
from typing import Optional, Union, Literal, List
from .networks import MLP
from ...rl.reinforce import *


class PolicyNetwork(nn.Module):
    def __init__(self, config):
        super(PolicyNetwork, self).__init__()

        net_config = [{'in': config.feature_dim, 'out': 32, 'drop_out': 0, 'activation': 'ReLU'},
                      {'in': 32, 'out': 8, 'drop_out': 0, 'activation': 'ReLU'},
                      {'in': 8, 'out': config.action_dim, 'drop_out': 0, 'activation': 'None'}]

        self.__mu_net = MLP(net_config)
        self.__sigma_net = MLP(net_config)

        self.__max_sigma = config.max_sigma
        self.__min_sigma = config.min_sigma

    def forward(self, x_in, require_entropy=False, require_musigma=False):
        mu = self.__mu_net(x_in)
        mu = (torch.tanh(mu) + 1.) / 2.
        sigma = self.__sigma_net(x_in)
        sigma = (torch.tanh(sigma) + 1.) / 2.
        sigma = torch.clamp(sigma, min=self.__min_sigma, max=self.__max_sigma)

        policy = Normal(mu, sigma)
        action = policy.sample()

        filter = torch.abs(action - 0.5) >= 0.5
        action = torch.where(filter, (action + 3 * sigma.detach() - mu.detach()) * (1. / 6 * sigma.detach()), action)
        log_prob = policy.log_prob(action)

        if require_entropy:
            entropy = policy.entropy()

            out = (action, log_prob, entropy)
        else:
            if require_musigma:
                out = (action, log_prob, mu, sigma)
            else:
                out = (action, log_prob)

        return out


class RLPSO(REINFORCE_Agent):
    def __init__(self, config):

        # add specified config
        self.config = config
        self.config.feature_dim = 2 * config.dim
        self.config.action_dim = 1
        self.config.action_shape = (1,)
        self.config.max_sigma = 0.7
        self.config.min_sigma = 0.01
        # origin RLPSO doesnt have gamma : set a default value
        self.config.gamma = self.config.min_sigma
        self.config.lr_model = 1e-5

        model = PolicyNetwork(config)

        # optimizer
        self.config.optimizer = 'Adam'
        # origin RLPSO doesn't have clip
        self.config.max_grad_norm = math.inf

        super().__init__(self.config, {'model': model}, [self.config.lr_model])

    def __str__(self):
        return "RLPSO"

    def train_episode(self,
                      envs,
                      seeds: Optional[Union[int, List[int], np.ndarray]],
                      para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc'] = 'dummy',
                      asynchronous: Literal[None, 'idle', 'restart', 'continue'] = None,
                      num_cpus: Optional[Union[int, None]] = 1,
                      num_gpus: int = 0,
                      tb_logger=None,
                      required_info={}):
        if self.device != 'cpu':
            num_gpus = max(num_gpus, 1)
        env = ParallelEnv(envs, para_mode, asynchronous, num_cpus, num_gpus)
        env.seed(seeds)
        # input action_dim should be : bs, ps
        # action in (0,1) the ratio to learn from pbest & gbest
        state = env.reset()
        try:
            state = torch.Tensor(state).to(self.device)
        except:
            pass

        _R = torch.zeros(len(env)).to(self.device)
        _loss = []
        _reward = []
        # sample trajectory
        while not env.all_done():
            action, log_prob = self.model(state)
            action = action.reshape(len(env))
            action = action.cpu().numpy()
            log_prob = log_prob.reshape(len(env))

            next_state, reward, is_done, _ = env.step(action)
            reward = torch.Tensor(reward).to(self.device)
            _R += reward
            _reward.append(reward)
            state = torch.Tensor(next_state).to(self.device)
            policy_gradient = -log_prob * reward
            loss = policy_gradient.mean()
            self.optimizer.zero_grad()
            loss.mean().backward()
            grad_norms = clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)
            _loss.append(loss.item())
            self.optimizer.step()
            self.learning_time += 1
            if self.learning_time >= (
                    self.config.save_interval * self.cur_checkpoint) and self.config.end_mode == "step":
                save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
                self.cur_checkpoint += 1

            if not self.config.no_tb and self.learning_time % int(self.config.log_step) == 0:
                self.log_to_tb_train(tb_logger, self.learning_time,
                                     grad_norms,
                                     loss,
                                     _R, _reward,
                                     log_prob)

        is_train_ended = self.learning_time >= self.config.max_learning_step
        return_info = {'return': _R.detach().cpu().numpy(), 'loss': _loss, 'learn_steps': self.learning_time, }
        env_cost = env.get_env_attr('cost')
        return_info['normalizer'] = env_cost[0]
        return_info['gbest'] = env_cost[-1]
        for key in required_info.keys():
            return_info[key] = env.get_env_attr(required_info[key])
        env.close()

        return is_train_ended, return_info

    def rollout_batch_episode(self,
                              envs,
                              seeds=None,
                              para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc'] = 'dummy',
                              asynchronous: Literal[None, 'idle', 'restart', 'continue'] = None,
                              num_cpus: Optional[Union[int, None]] = 1,
                              num_gpus: int = 0,
                              required_info={}):
        if self.device != 'cpu':
            num_gpus = max(num_gpus, 1)
        env = ParallelEnv(envs, para_mode, asynchronous, num_cpus, num_gpus)

        env.seed(seeds)
        state = env.reset()
        try:
            state = torch.Tensor(state).to(self.device)
        except:
            pass

        R = torch.zeros(len(env))
        # sample trajectory
        while not env.all_done():
            with torch.no_grad():
                action, _ = self.model(state)
            action = action.reshape(len(env))
            action = action.cpu().numpy()
            # state transient
            state, rewards, is_end, info = env.step(action)
            # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))
            R += torch.Tensor(rewards).squeeze()
            # store info
            try:
                state = torch.Tensor(state).to(self.device)
            except:
                pass
        _Rs = R.detach().numpy().tolist()
        env_cost = env.get_env_attr('cost')
        env_fes = env.get_env_attr('fes')
        results = {'cost': env_cost, 'fes': env_fes, 'return': _Rs}
        for key in required_info.keys():
            results[key] = env.get_env_attr(required_info[key])
        return results