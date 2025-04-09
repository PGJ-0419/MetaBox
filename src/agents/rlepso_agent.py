from torch import nn
from torch.distributions import Normal

from agents.networks import MLP
from basic_agent.PPO_Agent import *


class Actor(nn.Module):
    def __init__(self,
                 config,
                 ):
        super(Actor, self).__init__()
        net_config = [{'in': config.feature_dim, 'out': 64, 'drop_out': 0, 'activation': 'ReLU'},
                      {'in': 64, 'out': 32, 'drop_out': 0, 'activation': 'ReLU'},
                      {'in': 32, 'out': config.action_dim, 'drop_out': 0, 'activation': 'None'}]
        self.__mu_net = MLP(net_config)
        self.__sigma_net = MLP(net_config)
        self.__max_sigma = config.max_sigma
        self.__min_sigma = config.min_sigma

    def forward(self, x_in, fixed_action=None, require_entropy=False):  # x-in: bs*gs*9

        mu = (torch.tanh(self.__mu_net(x_in)) + 1.) / 2.
        sigma = (torch.tanh(self.__sigma_net(x_in)) + 1.) / 2. * (self.__max_sigma - self.__min_sigma) + self.__min_sigma

        policy = Normal(mu, sigma)

        if fixed_action is not None:
            action = fixed_action
        else:
            action = torch.clamp(policy.sample(), min=0, max=1)
        log_prob = policy.log_prob(action)

        log_prob = torch.sum(log_prob, dim = 1)

        if require_entropy:
            entropy = policy.entropy()  # for logging only bs,ps,2

            out = (action,
                   log_prob,
                   entropy)
        else:
            out = (action,
                   log_prob,
                   )
        return out


class Critic(nn.Module):
    def __init__(self,
                 config
                 ):
        super(Critic, self).__init__()
        self.__value_head = MLP([{'in': config.feature_dim, 'out': 16, 'drop_out': 0, 'activation': 'ReLU'},
                                 {'in': 16, 'out': 8, 'drop_out': 0, 'activation': 'ReLU'},
                                 {'in': 8, 'out': 1, 'drop_out': 0, 'activation': 'None'}])

    def forward(self, h_features):
        baseline_value = self.__value_head(h_features)
        # baseline_value = baseline_value[0]
        # print(type(baseline_value.detach()))
        return baseline_value.detach().squeeze(), baseline_value.squeeze()


class RLEPSO_Agent(PPO_Agent):
    def __init__(self, config):

        self.config = config
        # add specified config
        self.config.feature_dim=1
        self.config.action_dim=35
        self.config.action_shape=(35,)
        self.config.n_step=10
        self.config.K_epochs=3
        self.config.eps_clip=0.1
        self.config.gamma=0.999
        self.config.max_sigma=0.7
        self.config.min_sigma=0.01
        self.config.lr=1e-5
        self.config.optimizer = 'Adam'
        self.config.max_grad_norm = math.inf
        # config.lr_decay=0.99


        # figure out the actor
        actor = Actor(config)

        # figure out the critic
        critic = Critic(config)

        # figure out the optimizer
        # self.__optimizer_actor = torch.optim.Adam(
        #     [{'params': self.__actor.parameters(), 'lr': config.lr}])
        # self.__optimizer_critic = torch.optim.Adam(
        #     [{'params': self.__critic.parameters(), 'lr': config.lr}])

        # init learning time
        # self.__learning_time=0
        #
        # self.__cur_checkpoint=0

        # save init agent
        # if self.__cur_checkpoint==0:
        #     save_class(self.__config.agent_save_dir,'checkpoint'+str(self.__cur_checkpoint),self)
        #     self.__cur_checkpoint+=1

        super().__init__(self.config, {'actor': actor, 'critic': critic}, self.config.lr)

    def __str__(self):
        return "RLEPSO"
    # def update_setting(self, config):
    #     self.__config.max_learning_step = config.max_learning_step
    #     self.__config.agent_save_dir = config.agent_save_dir
    #     self.__learning_time = 0
    #     save_class(self.__config.agent_save_dir, 'checkpoint0', self)
    #     self.__config.save_interval = config.save_interval
    #     self.__cur_checkpoint = 1

    def train_episode(self,
                      envs,
                      seeds: Optional[Union[int, List[int], np.ndarray]],
                      para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc'] = 'dummy',
                      # todo: asynchronous: Literal[None, 'idle', 'restart', 'continue'] = None,
                      # num_cpus: Optional[Union[int, None]] = 1,
                      # num_gpus: int = 0,
                      compute_resource = {},
                      tb_logger = None,
                      required_info = {}):
        num_cpus = None
        num_gpus = 0
        if 'num_cpus' in compute_resource.keys():
            num_cpus = compute_resource['num_cpus']
        if 'num_gpus' in compute_resource.keys():
            num_gpus = compute_resource['num_gpus']
        env = ParallelEnv(envs, para_mode, num_cpus=num_cpus, num_gpus=num_gpus)
        env.seed(seeds)
        memory = Memory()

        # params for training
        gamma = self.gamma
        n_step = self.n_step

        K_epochs = self.K_epochs
        eps_clip = self.eps_clip

        state = env.reset()
        try:
            state = torch.FloatTensor(state).to(self.device)
        except:
            pass

        t = 0
        # initial_cost = obj
        _R = torch.zeros(len(env))
        _loss = []
        # sample trajectory
        while not env.all_done():
            t_s = t
            total_cost = 0
            bl_val_detached = []
            bl_val = []
            entropy = []

            # accumulate transition
            while t - t_s < n_step and not env.all_done():

                memory.states.append(state.clone())
                action, log_lh, entro_p = self.actor(state,require_entropy=True)

                # action = action.reshape(self.config.action_shape)

                memory.actions.append(action.clone() if isinstance(action, torch.Tensor) else copy.deepcopy(action))


                memory.logprobs.append(log_lh)
                entropy.append(entro_p.detach().cpu())

                baseline_val_detached, baseline_val = self.critic(state)
                bl_val_detached.append(baseline_val_detached)
                bl_val.append(baseline_val)

                # state transient
                state, rewards, is_end, info = env.step(action.detach().cpu().numpy())
                memory.rewards.append(torch.FloatTensor(rewards).to(self.device))
                # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))
                _R += rewards
                # store info

                # next
                t = t + 1

                try:
                    state = torch.FloatTensor(state).to(self.device)
                except:
                    pass

            # store info
            t_time = t - t_s
            total_cost = total_cost / t_time

            # begin update
            old_actions = torch.stack(memory.actions)
            try:
                old_states = torch.stack(memory.states).detach()  # .view(t_time, bs, ps, dim_f)
            except:
                pass
            # old_actions = all_actions.view(t_time, bs, ps, -1)
            old_logprobs = torch.stack(memory.logprobs).detach().view(-1)

            # Optimize PPO policy for K mini-epochs:
            old_value = None
            for _k in range(K_epochs):
                if _k == 0:
                    logprobs = memory.logprobs

                else:
                    # Evaluating old actions and values :
                    logprobs = []
                    bl_val_detached = []
                    bl_val = []
                    entropy = []

                    for tt in range(t_time):
                        # get new action_prob
                        _, log_p, entro_p = self.actor(state, fixed_action = old_actions[tt], require_entropy=True)

                        logprobs.append(log_p)
                        entropy.append(entro_p.detach().cpu())

                        baseline_val_detached, baseline_val = self.critic(state)

                        bl_val_detached.append(baseline_val_detached)
                        bl_val.append(baseline_val)

                logprobs = torch.stack(logprobs).view(-1)
                entropy = torch.stack(entropy).view(-1)
                bl_val_detached = torch.stack(bl_val_detached).view(-1)
                bl_val = torch.stack(bl_val).view(-1)

                # get traget value for critic
                Reward = []
                reward_reversed = memory.rewards[::-1]
                # get next value
                R = self.critic(state)[0]
                critic_output = R.clone()
                # print(R.shape)
                for r in range(len(reward_reversed)):
                    # print(reward_reversed[r].shape)
                    R = R * gamma + reward_reversed[r]
                    # print(R)
                    Reward.append(R)
                # clip the target:
                Reward = torch.stack(Reward[::-1], 0)

                Reward = Reward.view(-1)

                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs.detach())
                # print(Reward.shape,bl_val_detached.shape)
                # Finding Surrogate Loss:
                advantages = Reward - bl_val_detached

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
                reinforce_loss = -torch.min(surr1, surr2).mean()

                # define baseline loss
                if old_value is None:
                    baseline_loss = ((bl_val - Reward) ** 2).mean()
                    old_value = bl_val.detach()
                else:
                    vpredclipped = old_value + torch.clamp(bl_val - old_value, - eps_clip, eps_clip)
                    v_max = torch.max(((bl_val - Reward) ** 2), ((vpredclipped - Reward) ** 2))
                    baseline_loss = v_max.mean()

                # check K-L divergence (for logging only)
                approx_kl_divergence = (.5 * (old_logprobs.detach() - logprobs) ** 2).mean().detach()
                approx_kl_divergence[torch.isinf(approx_kl_divergence)] = 0
                # calculate loss
                loss = baseline_loss + reinforce_loss

                # update gradient step
                self.optimizer.zero_grad()
                loss.backward()
                _loss.append(loss.item())
                # Clip gradient norm and get (clipped) gradient norms for logging
                # current_step = int(pre_step + t//n_step * K_epochs  + _k)
                grad_norms = clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)

                # perform gradient descent
                self.optimizer.step()
                self.learning_time += 1
                if self.learning_time >= (self.config.save_interval * self.cur_checkpoint):
                    save_class(self.config.agent_save_dir, 'checkpoint' + str(self.cur_checkpoint), self)
                    self.cur_checkpoint += 1

                if not self.config.no_tb:
                    self.log_to_tb_train(tb_logger, self.learning_time,
                                         grad_norms,
                                         reinforce_loss, baseline_loss,
                                         _R, Reward, memory.rewards,
                                         critic_output, logprobs, entropy, approx_kl_divergence)

                if self.learning_time >= self.config.max_learning_step:
                    memory.clear_memory()
                    _Rs = _R.detach().numpy().tolist()
                    return_info = {'return': _Rs, 'loss': np.mean(_loss),'learn_steps': self.learning_time, }
                    env_cost = env.get_env_attr('cost')
                    return_info['normalizer'] = env_cost[0]
                    return_info['gbest'] = env_cost[-1]
                    for key in required_info.keys():
                        return_info[key] = env.get_env_attr(required_info[key])
                    env.close()
                    return self.learning_time >= self.config.max_learning_step, return_info

            memory.clear_memory()

        is_train_ended = self.learning_time >= self.config.max_learning_step
        _Rs = _R.detach().numpy().tolist()
        return_info = {'return': _Rs, 'loss': np.mean(_loss),'learn_steps': self.learning_time,}
        env_cost = env.get_env_attr('cost')
        return_info['normalizer'] = env_cost[0]
        return_info['gbest'] = env_cost[-1]
        for key in required_info.keys():
            return_info[key] = env.get_env_attr(required_info[key])
        env.close()
        return is_train_ended, return_info


    def rollout_batch_episode(self,
                              envs,
                              seeds = None,
                              para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc'] = 'dummy',
                              # todo: asynchronous: Literal[None, 'idle', 'restart', 'continue'] = None,
                              # num_cpus: Optional[Union[int, None]] = 1,
                              # num_gpus: int = 0,
                              compute_resource = {},
                              required_info = {}):
        num_cpus = None
        num_gpus = 0
        if 'num_cpus' in compute_resource.keys():
            num_cpus = compute_resource['num_cpus']
        if 'num_gpus' in compute_resource.keys():
            num_gpus = compute_resource['num_gpus']
        env = ParallelEnv(envs, para_mode, num_cpus=num_cpus, num_gpus=num_gpus)

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
                action, log_lh = self.actor(state, require_entropy = False)

            # state transient
            state, rewards, is_end, info = env.step(action.detach().cpu().numpy())
            # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))
            R += torch.FloatTensor(rewards).squeeze()
            # store info
            try:
                state = torch.FloatTensor(state).to(self.device)
            except:
                pass
        _Rs = R.detach().numpy().tolist()
        env_cost = env.get_env_attr('cost')
        env_fes = env.get_env_attr('fes')
        results = {'cost': env_cost, 'fes': env_fes, 'return': _Rs}
        for key in required_info.keys():
            results[key] = env.get_env_attr(required_info[key])
        return results
    # def train_episode(self, env):
    #     config = self.__config
    #     # setup
    #     memory = Memory()
    #     # initial instances and solutions
    #     state = env.reset()
    #     state = torch.FloatTensor(state).to(self.__device)
    #
    #
    #     # params for training
    #     gamma = config.gamma
    #     n_step = config.n_step
    #     K_epochs = config.K_epochs
    #     eps_clip = config.eps_clip
    #
    #     t = 0
    #     _R = 0
    #     # initial_cost = obj
    #     is_done = False
    #     # sample trajectory
    #     while not is_done:
    #         t_s = t
    #         total_cost = 0
    #         entropy = []
    #         bl_val_detached = []
    #         bl_val = []
    #
    #         while t - t_s < n_step:
    #             # encoding the state
    #
    #             memory.states.append(state.clone())
    #
    #             # get model output
    #             action, log_lh,  entro_p = self.__actor(state,
    #                                                     require_entropy=True,
    #                                                     )
    #             action = action.reshape(config.action_shape)
    #             memory.actions.append(action.clone().detach())
    #             action = action.cpu().numpy()
    #             memory.logprobs.append(log_lh)
    #
    #             entropy.append(entro_p.detach().cpu())
    #
    #             baseline_val_detached, baseline_val = self.__critic(state)
    #             bl_val_detached.append(baseline_val_detached)
    #             bl_val.append(baseline_val)
    #
    #             # state transient
    #             next_state,rewards,is_done = env.step(action)
    #             _R += rewards
    #             memory.rewards.append(torch.FloatTensor([rewards]).to(config.device))
    #             # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))
    #
    #             # store info
    #             # total_cost = total_cost + gbest_val
    #
    #             # next
    #             t = t + 1
    #             state=next_state
    #             state=torch.FloatTensor(state).to(config.device)
    #             if is_done:
    #
    #                 break
    #
    #         # store info
    #         t_time = t - t_s
    #         total_cost = total_cost / t_time
    #
    #         # begin update        =======================
    #
    #         # bs, ps, dim_f = state.size()
    #
    #         old_actions = torch.stack(memory.actions)
    #         old_states = torch.stack(memory.states).detach()
    #
    #         old_logprobs = torch.stack(memory.logprobs).detach().view(-1)
    #
    #         # Optimize PPO policy for K mini-epochs:
    #         old_value = None
    #         for _k in range(K_epochs):
    #             if _k == 0:
    #                 logprobs = memory.logprobs
    #
    #             else:
    #                 # Evaluating old actions and values :
    #                 logprobs = []
    #                 entropy = []
    #                 bl_val_detached = []
    #                 bl_val = []
    #
    #                 for tt in range(t_time):
    #
    #                     # get new action_prob
    #                     _, log_p,  entro_p = self.__actor(old_states[tt],
    #                                                       fixed_action=old_actions[tt],
    #                                                       require_entropy=True,  # take same action
    #                                                       )
    #
    #                     logprobs.append(log_p)
    #                     entropy.append(entro_p.detach().cpu())
    #
    #                     baseline_val_detached, baseline_val = self.__critic(old_states[tt])
    #
    #                     bl_val_detached.append(baseline_val_detached)
    #                     bl_val.append(baseline_val)
    #
    #             logprobs = torch.stack(logprobs).view(-1)
    #             entropy = torch.stack(entropy).view(-1)
    #             bl_val_detached = torch.stack(bl_val_detached).view(-1)
    #             bl_val = torch.stack(bl_val).view(-1)
    #
    #             # get target value for critic
    #             Reward = []
    #             reward_reversed = memory.rewards[::-1]
    #             # get next value
    #             R = self.__critic(state)[0]
    #
    #             # R = agent.critic(state)[0]
    #             critic_output = R.clone()
    #             for r in range(len(reward_reversed)):
    #                 R = R * gamma + reward_reversed[r]
    #                 Reward.append(R)
    #             # clip the target:
    #             Reward = torch.stack(Reward[::-1], 0)
    #             Reward = Reward.view(-1)
    #
    #             # Finding the ratio (pi_theta / pi_theta__old):
    #             ratios = torch.exp(logprobs - old_logprobs.detach())
    #
    #             # Finding Surrogate Loss:
    #             advantages = Reward - bl_val_detached
    #
    #             surr1 = ratios * advantages
    #             surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
    #             reinforce_loss = -torch.min(surr1, surr2).mean()
    #
    #             # define baseline loss
    #             if old_value is None:
    #                 baseline_loss = ((bl_val - Reward) ** 2).mean()
    #                 old_value = bl_val.detach()
    #             else:
    #                 vpredclipped = old_value + torch.clamp(bl_val - old_value, - eps_clip, eps_clip)
    #                 v_max = torch.max(((bl_val - Reward) ** 2), ((vpredclipped - Reward) ** 2))
    #                 baseline_loss = v_max.mean()
    #
    #             # check K-L divergence (for logging only)
    #             approx_kl_divergence = (.5 * (old_logprobs.detach() - logprobs) ** 2).mean().detach()
    #             approx_kl_divergence[torch.isinf(approx_kl_divergence)] = 0
    #             # calculate loss
    #             loss = baseline_loss + reinforce_loss
    #
    #             # update gradient step
    #             # agent.optimizer.zero_grad()
    #             self.__optimizer_actor.zero_grad()
    #             self.__optimizer_critic.zero_grad()
    #             baseline_loss.backward()
    #             reinforce_loss.backward()
    #             # loss.backward()
    #
    #
    #             # perform gradient descent
    #             self.__optimizer_actor.step()
    #             self.__optimizer_critic.step()
    #             self.__learning_time += 1
    #
    #             if self.__learning_time >= (self.__config.save_interval * self.__cur_checkpoint):
    #                 save_class(self.__config.agent_save_dir, 'checkpoint'+str(self.__cur_checkpoint), self)
    #                 self.__cur_checkpoint += 1
    #
    #             if self.__learning_time >= config.max_learning_step:
    #                 return self.__learning_time >= config.max_learning_step, {'normalizer': env.optimizer.cost[0],
    #                                                                           'gbest': env.optimizer.cost[-1],
    #                                                                           'return': _R,
    #                                                                           'learn_steps': self.__learning_time}
    #
    #         memory.clear_memory()
    #     return self.__learning_time >= config.max_learning_step, {'normalizer': env.optimizer.cost[0],
    #                                                               'gbest': env.optimizer.cost[-1],
    #                                                               'return': _R,
    #                                                               'learn_steps': self.__learning_time}
    #
    # def rollout_episode(self, env):
    #     is_done = False
    #     state = env.reset()
    #     R = 0
    #     while not is_done:
    #         state = torch.FloatTensor(state).to(self.__config.device)
    #         action = self.__actor(state)[0].cpu().numpy()
    #         state, reward, is_done = env.step(action)
    #         R += reward
    #     return {'cost': env.optimizer.cost, 'fes': env.optimizer.fes, 'return': R}
