from typing import Tuple
# from src.agent.basic_agent import Basic_Agent
import torch
import math, copy
from typing import Any, Callable, List, Optional, Tuple, Union, Literal

from torch import nn
import torch
from torch.distributions import Normal
import torch.nn.functional as F
from agent.utils import *
from agent.basic_agent import Basic_Agent


from VectorEnv.great_para_env import ParallelEnv


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for idx, group in enumerate(param_groups)
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


class VDN_Agent(Basic_Agent):
    def __init__(self, config, n_agent, available_action: list):
        super().__init__(config)
        self.config = config

        # define parameters
        self.gamma = self.config.gamma
        self.n_act = self.config.n_act
        self.epsilon = self.config.epsilon
        self.max_grad_norm = self.config.max_grad_norm
        self.memory_size = self.config.memory_size
        self.batch_size = self.config.batch_size
        self.warm_up_size = self.config.warm_up_size
        self.chunk_size = self.config.chunk_size
        self.update_iter = self.config.update_iter
        self.device = self.config.device
        self.n_agent = n_agent
        self.available_action = available_action

        # figure out the actor network
        # self.model = None
        assert hasattr(self, 'model')

        # figure out the optimizer
        assert hasattr(torch.optim, self.config.optimizer)
        self.optimizer = eval('torch.optim.' + self.config.optimizer)(
            [{'params': self.model.parameters(), 'lr': self.config.lr_model}])
        # figure out the lr schedule
        assert hasattr(torch.optim.lr_scheduler, self.config.lr_scheduler)
        self.lr_scheduler = eval('torch.optim.lr_scheduler.' + self.config.lr_scheduler)(self.optimizer,
                                                                                         self.config.lr_decay,
                                                                                         last_epoch=-1, )

        assert hasattr(torch.nn, self.config.criterion)
        self.criterion = eval('torch.nn.' + self.config.criterion)()

        self.replay_buffer = MultiAgent_ReplayBuffer(self.memory_size)

        # move to device
        self.model.to(self.device)

        # init learning time
        self.learning_time = 0
        self.cur_checkpoint = 0

        # save init agent
        save_class(self.config.agent_save_dir, 'checkpoint' + str(self.cur_checkpoint), self)
        self.cur_checkpoint += 1

    def update_setting(self, config):
        self.config.max_learning_step = config.max_learning_step
        self.config.agent_save_dir = config.agent_save_dir
        self.learning_time = 0
        save_class(self.config.agent_save_dir, 'checkpoint0', self)
        self.config.save_interval = config.save_interval
        self.cur_checkpoint = 1

    def get_action(self, state, epsilon_greedy=False):
        state = torch.Tensor(state).to(self.device)
        with torch.no_grad():
            Q_list = self.model(state)
        if epsilon_greedy and np.random.rand() < self.epsilon:
            action = np.zeros((len(state), self.n_agent), dtype=int)
            for i in range(self.n_agent):
                action[:, i] = np.random.randint(low=0, high=self.available_action[i], size=len(state))
        else:
            action = torch.argmax(Q_list, -1).detach().cpu().numpy().astype(int)
        return action,Q_list

    def train_episode(self,
                      envs,
                      para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc'] = 'dummy',
                      asynchronous: Literal[None, 'idle', 'restart', 'continue'] = None,
                      num_cpus: Optional[Union[int, None]] = 1,
                      num_gpus: int = 0,
                      required_info={}):
        if required_info is None:
            required_info = {'best_igd': 'best_value',
                             }
        if self.device != 'cpu':
            num_gpus = max(num_gpus, 1)
        env = ParallelEnv(envs, para_mode, asynchronous, num_cpus, num_gpus)

        # params for training
        gamma = self.gamma

        state = env.reset()
        try:
            state = torch.FloatTensor(state)
        except:
            pass

        _R = torch.zeros(len(env))
        _Q = []
        _loss = []
        # sample trajectory
        while not env.all_done():
            action,Q_list = self.get_action(state=state, epsilon_greedy=True)
            _Q.append(Q_list)

            # state transient
            next_state, reward, is_end, info = env.step(action)
            _R += reward[:, 0]
            # store info
            for s, a, r, ns, d in zip(state.numpy(), action, reward, next_state, is_end):
                self.replay_buffer.append((s, a, r, ns, d))
            try:
                state = torch.FloatTensor(next_state).to(self.device)
            except:
                state = copy.deepcopy(next_state)
            # begin update
            self.target_model = copy.deepcopy(self.model)
            if len(self.replay_buffer) >= self.warm_up_size:
                for _ in range(self.update_iter):
                    batch_obs, batch_action, batch_reward, batch_next_obs, batch_done \
                        = self.replay_buffer.sample_chunk(self.batch_size,self.chunk_size)
                    loss = 0
                    for step_i in range(self.chunk_size):
                        q_out = self.model(batch_obs[:,step_i,:,:].to(self.device))
                        q_a = q_out.gather(2, batch_action[:, step_i, :].unsqueeze(-1).long()).squeeze(-1)
                        sum_q = q_a.sum(dim=1, keepdims=True)
                        max_q_prime = self.target_model(batch_next_obs[:, step_i, :, :])
                        max_q_prime = max_q_prime.max(dim=2)[0].squeeze(-1)
                        target_q = batch_reward[:, step_i, :].sum(dim=1, keepdims=True)
                        target_q += self.gamma * max_q_prime.sum(dim=1, keepdims=True) * (1 - batch_done[:, step_i])

                        loss += self.criterion(sum_q, target_q.detach())
                    loss = loss / (self.batch_size * self.chunk_size)
                    _loss.append(loss.item())
                    self.optimizer.zero_grad()
                    loss.backward()
                    grad_norms = clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)
                    self.optimizer.step()
                    self.learning_time += 1
                    if self.config.target_update_interval is not None and self.learning_time % self.config.target_update_interval == 0:
                        self.target_model.load_state_dict(self.model.state_dict())


                if self.learning_time >= (self.config.save_interval * self.cur_checkpoint):
                    save_class(self.config.agent_save_dir, 'checkpoint' + str(self.cur_checkpoint), self)
                    self.cur_checkpoint += 1

                if self.learning_time >= self.config.max_learning_step:
                    return_info = {'return': _R, 'learn_steps': self.learning_time, 'Q_values':torch.tensor(Q_list), 'loss':_loss}
                    for key in required_info.keys():
                        return_info[key] = env.get_env_attr(required_info[key])
                    env.close()
                    return self.learning_time >= self.config.max_learning_step, return_info

        is_train_ended = self.learning_time >= self.config.max_learning_step
        return_info = {'return': _R, 'learn_steps': self.learning_time, 'q_values':torch.tensor(Q_list), 'loss':_loss}
        for key in required_info.keys():
            return_info[key] = env.get_env_attr(required_info[key])
        env.close()

        return is_train_ended, return_info

    def rollout_episode(self,
                        env,
                        seed=None,
                        required_info={}):
        with torch.no_grad():
            if seed is not None:
                env.seed(seed)
            is_done = False
            state = env.reset()
            R = 0
            while not is_done:
                try:
                    state = torch.FloatTensor(state).to(self.device)
                except:
                    pass
                action = self.get_action([state])[0]
                action = action.cpu().numpy().squeeze()
                state, reward, is_done = env.step(action)
                R += reward[0]
            results = {'return': R}
            for key in required_info.keys():
                results[key] = getattr(env, required_info[key])
            return results

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
        if seeds is not None:
            env.seed(seeds)
        state = env.reset()
        try:
            state = torch.FloatTensor(state).to(self.device)
        except:
            pass

        R = torch.zeros(len(env))
        # sample trajectory
        while not env.all_done():
            with torch.no_grad():
                action = self.get_action(state)

            # state transient
            state, rewards, is_end, info = env.step(action)
            # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))
            R += torch.FloatTensor(rewards[:,0]).squeeze()
            # store info
            try:
                state = torch.FloatTensor(state).to(self.device)
            except:
                pass
        results = {'return': R}
        for key in required_info.keys():
            results[key] = env.get_env_attr(required_info[key])
        return results
