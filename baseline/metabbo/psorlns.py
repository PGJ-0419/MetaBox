from .networks import MLP
from ...rl.dqn import *


class PSORLNS(DQN_Agent):
    def __init__(self, config):

        self.config = config
        self.config.state_size = 1
        self.config.n_act = 5
        self.config.mlp_config = [{'in': self.config.state_size, 'out': 10, 'drop_out': 0, 'activation': 'ReLU'},
                                  {'in': 10, 'out': 10, 'drop_out': 0, 'activation': 'ReLU'},
                                  {'in': 10, 'out': config.n_act, 'drop_out': 0, 'activation': 'None'}]
        self.config.lr_model = 1e-4

        self.config.lr_decay = 1
        self.config.epsilon = 0.1
        self.config.gamma = 0.8
        self.config.memory_size = 1000  # todo
        self.config.batch_size = 256  # todo
        self.config.warm_up_size = config.batch_size

        self.config.device = config.device

        self.config.max_grad_norm = math.inf
        self.config.optimizer = 'AdamW'

        self.config.lr_scheduler = 'ExponentialLR'
        self.config.criterion = 'MSELoss'
        model = MLP(self.config.mlp_config).to(self.config.device)

        # self.__cur_checkpoint=0
        super().__init__(self.config, {'model': model}, self.config.lr_model)

    def __str__(self):
        return "PSORLNS"

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
        num_gpus = 0 if self.config.device == 'cpu' else torch.cuda.device_count()
        if 'num_cpus' in compute_resource.keys():
            num_cpus = compute_resource['num_cpus']
        if 'num_gpus' in compute_resource.keys():
            num_gpus = compute_resource['num_gpus']
        env = ParallelEnv(envs, para_mode, num_cpus = num_cpus, num_gpus = num_gpus)
        env.seed(seeds)
        self.ps = int(env.get_env_attr('ps')[0])
        # params for training
        gamma = self.gamma

        state = env.reset()
        try:
            state = torch.Tensor(state).reshape(len(env) * self.ps, self.config.state_size).to(self.device)
        except:
            pass

        _R = torch.zeros(len(env) * self.ps)
        _loss = []
        _reward = []
        # sample trajectory
        while not env.all_done():
            assert state.shape == (len(env) * self.ps, self.config.state_size)
            action = self.get_action(state = state, epsilon_greedy = True)

            # state transient
            next_state, reward, is_end, info = env.step(action.reshape(len(env), self.ps))
            reward = reward.reshape(len(env) * self.ps, )
            is_end = is_end.reshape(len(env) * self.ps, )
            _R += reward
            _reward.append(torch.Tensor(reward))
            # store info
            # convert next_state into tensor
            try:
                next_state = torch.Tensor(next_state).reshape(len(env) * self.ps, self.config.state_size).to(self.device)
            except:
                pass
            for s, a, r, ns, d in zip(state, action, reward, next_state, is_end):
                self.replay_buffer.append((s, a, r, ns, d))
            try:
                state = torch.Tensor(next_state).to(self.device)
            except:
                state = copy.deepcopy(next_state)

            # begin update
            if len(self.replay_buffer) >= self.warm_up_size:
                batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = self.replay_buffer.sample(self.batch_size)
                pred_Vs = self.model(batch_obs.to(self.device))  # [batch_size, n_act]
                action_onehot = torch.nn.functional.one_hot(batch_action.to(self.device), self.n_act)  # [batch_size, n_act]

                _avg_predict_Q = (pred_Vs * action_onehot).mean(0)  # [n_act]
                predict_Q = (pred_Vs * action_onehot).sum(1)  # [batch_size]

                target_output = self.model(batch_next_obs.to(self.device))
                _avg_target_Q = batch_reward.to(self.device)[:, None] + (1 - batch_done.to(self.device))[:, None] * gamma * target_output
                target_Q = batch_reward.to(self.device) + (1 - batch_done.to(self.device)) * gamma * target_output.max(1)[0]
                _avg_target_Q = _avg_target_Q.mean(0)  # [n_act]

                self.optimizer.zero_grad()
                loss = self.criterion(predict_Q, target_Q)
                loss.backward()
                grad_norms = clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)
                self.optimizer.step()

                _loss.append(loss.item())

                self.learning_time += 1
                if self.learning_time >= (self.config.save_interval * self.cur_checkpoint) and self.config.end_mode == "step":
                    save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
                    self.cur_checkpoint += 1

                if not self.config.no_tb and self.learning_time % int(self.config.log_step) == 0:
                    self.log_to_tb_train(tb_logger, self.learning_time,
                                         grad_norms,
                                         loss,
                                         _R, _reward,
                                         _avg_predict_Q, _avg_target_Q)

                if self.learning_time >= self.config.max_learning_step:
                    _Rs = _R.detach().numpy().tolist()
                    return_info = {'return': _Rs, 'loss': _loss, 'learn_steps': self.learning_time, }
                    env_cost = env.get_env_attr('cost')
                    return_info['gbest'] = env_cost[-1]
                    for key in required_info:
                        return_info[key] = env.get_env_attr(key)
                    env.close()
                    return self.learning_time >= self.config.max_learning_step, return_info

        is_train_ended = self.learning_time >= self.config.max_learning_step
        _Rs = _R.detach().numpy().tolist()
        return_info = {'return': _Rs, 'loss': _loss, 'learn_steps': self.learning_time}
        env_cost = env.get_env_attr('cost')
        return_info['gbest'] = env_cost[-1]
        for key in required_info:
            return_info[key] = env.get_env_attr(key)
            # print(f"{key} : {return_info[key]}")
        env.close()

        return is_train_ended, return_info

    def rollout_episode(self,
                        env,
                        seed = None,
                        required_info = {}):
        self.ps = env.get_env_attr('ps')
        with torch.no_grad():
            if seed is not None:
                env.seed(seed)
            is_done = [False] * self.ps
            state = env.reset()
            R = np.zeros(self.ps)
            while not is_done[0]:
                try:
                    state = torch.Tensor(state).unsqueeze(0).to(self.device).reshape(self.ps, self.config.state_size)
                except:
                    st = state.reshape(self.ps, self.config.state_size).to(self.device)
                action = self.get_action(state)
                action = action.squeeze()
                state, reward, is_done, _ = env.step(action.reshape(self.ps, ))
                reward = reward.reshape(self.ps, )
                is_done = is_done.reshape(self.ps, )
                R += reward
            # _Rs = np.mean(R).tolist()
            _Rs = np.mean(R)
            env_cost = env.get_env_attr('cost')
            env_fes = env.get_env_attr('fes')
            env_pr = env.get_env_attr('pr')
            env_sr = env.get_env_attr('sr')
            results = {'cost': env_cost, 'fes': env_fes, 'return': _Rs, 'pr': env_pr, 'sr': env_sr}

            if self.config.full_meta_data:
                meta_X = env.get_env_attr('meta_X')
                meta_Cost = env.get_env_attr('meta_Cost')
                meta_Pr = env.get_env_attr('meta_Pr')
                meta_Sr = env.get_env_attr('meta_Sr')
                metadata = {'X': meta_X, 'Cost': meta_Cost, 'Pr': meta_Pr, 'Sr': meta_Sr}
                results['metadata'] = metadata
            for key in required_info.keys():
                results[key] = getattr(env, required_info[key])
            return results

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
        num_gpus = 0 if self.config.device == 'cpu' else torch.cuda.device_count()
        if 'num_cpus' in compute_resource.keys():
            num_cpus = compute_resource['num_cpus']
        if 'num_gpus' in compute_resource.keys():
            num_gpus = compute_resource['num_gpus']
        env = ParallelEnv(envs, para_mode, num_cpus = num_cpus, num_gpus = num_gpus)
        env.seed(seeds)
        self.ps = env.get_env_attr('ps')
        state = env.reset()
        try:
            state = torch.Tensor(state).to(self.device).reshape(-1, self.state_size)
        except:
            pass

        R = torch.zeros(len(env) * self.ps)
        # sample trajectory
        while not env.all_done():
            with torch.no_grad():
                action = self.get_action(state)

            # state transient
            state, rewards, is_end, info = env.step(action.reshape(len(env), self.ps))
            rewards = rewards.reshape(len(env) * self.ps)
            # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))
            R += torch.Tensor(rewards).squeeze()
            # store info
            try:
                state = torch.Tensor(state).to(self.device).reshape(-1, self.state_size)
            except:
                pass
        _Rs = torch.mean(R.reshape(len(env), self.ps), dim = -1).detach().numpy().tolist()

        env_cost = env.get_env_attr('cost')
        env_fes = env.get_env_attr('fes')
        env_pr = env.get_env_attr('pr')
        env_sr = env.get_env_attr('sr')
        results = {'cost': env_cost, 'fes': env_fes, 'return': _Rs, 'pr': env_pr, 'sr': env_sr}

        if self.config.full_meta_data:
            meta_X = env.get_env_attr('meta_X')
            meta_Cost = env.get_env_attr('meta_Cost')
            meta_Pr = env.get_env_attr('meta_Pr')
            meta_Sr = env.get_env_attr('meta_Sr')
            metadata = {'X': meta_X, 'Cost': meta_Cost, 'Pr': meta_Pr, 'Sr': meta_Sr}
            results['metadata'] = metadata
        for key in required_info.keys():
            results[key] = getattr(env, required_info[key])
        return results
