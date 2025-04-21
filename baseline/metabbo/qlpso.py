from scipy.special import softmax
from typing import Optional, Union, Literal, List
from ...rl.qlearning import *
from ...rl.utils import save_class


class QLPSO(QLearning_Agent):
    def __init__(self, config):
        self.config = config
        # define hyperparameters that agent needs
        self.config.n_state = 4
        self.config.n_act = 4
        self.config.lr_model = 1
        self.config.alpha_decay = True
        self.config.gamma = 0.8

        self.config.epsilon = None

        # self.__q_table = np.zeros((config.n_states, config.n_actions))

        self.__alpha_max = self.config.lr_model
        # self.lr_model = self.config.lr_model
        self.__alpha_decay = self.config.alpha_decay
        self.__max_learning_step = self.config.max_learning_step
        # self.__global_ls = 0  # a counter of accumulated learned steps
        self.device = self.config.device
        # self.__cur_checkpoint = 0
        super().__init__(self.config)

    def __str__(self):
        return "QLPSO"

    def __get_action(self, state):  # Make action decision according to the given state
        # Get the corresponding rows from the Q-table and compute the softmax
        q_values = self.q_table[state]  # shape: (bs, n_actions)

        # Compute the action probabilities for each state
        prob = torch.softmax(q_values, dim=0)  # shape: (bs, n_actions)

        # Choose an action based on the probabilities
        action = torch.multinomial(prob, 1)  # shape: (bs, 1)

        # Return the action
        return action.squeeze().detach().cpu().numpy()  # Return the action and remove unnecessary dimensions

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
        # params for training
        gamma = self.gamma

        state = env.reset()
        state = torch.tensor(state, dtype=torch.int64)

        _R = torch.zeros(len(env))
        _loss = []
        _reward = []
        # sample trajectory
        while not env.all_done():
            action = self.__get_action(state)
            # state transient
            next_state, reward, is_end, info = env.step(action)
            _R += reward

            reward = torch.Tensor(reward).to(self.device)
            _reward.append(reward)
            TD_error = reward + gamma * torch.max(self.q_table[next_state], dim=1)[0] - self.q_table[state, action]

            _loss.append(TD_error.mean().item())
            self.q_table[state, action] += self.lr_model * TD_error

            self.learning_time += 1

            if self.learning_time >= (
                    self.config.save_interval * self.cur_checkpoint) and self.config.end_mode == "step":
                save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
                self.cur_checkpoint += 1

            if not self.config.no_tb and self.learning_time % int(self.config.log_step) == 0:
                self.log_to_tb_train(tb_logger, self.learning_time,
                                     TD_error.mean(),
                                     _R, _reward,
                                     )

            if self.learning_time >= self.config.max_learning_step:
                _Rs = _R.detach().numpy().tolist()
                return_info = {'return': _Rs, 'loss': _loss, 'learn_steps': self.learning_time, }
                env_cost = env.get_env_attr('cost')
                return_info['normalizer'] = env_cost[0]
                return_info['gbest'] = env_cost[-1]
                for key in required_info.keys():
                    return_info[key] = env.get_env_attr(required_info[key])
                env.close()
                return self.learning_time >= self.config.max_learning_step, return_info

            if self.__alpha_decay:
                self.lr_model = self.__alpha_max - (
                            self.__alpha_max - 0.1) * self.learning_time / self.__max_learning_step

            # store info
            state = torch.tensor(next_state, dtype=torch.int64)

        is_train_ended = self.learning_time >= self.config.max_learning_step
        _Rs = _R.detach().numpy().tolist()
        return_info = {'return': _Rs, 'loss': _loss, 'learn_steps': self.learning_time, }
        env_cost = env.get_env_attr('cost')
        return_info['normalizer'] = env_cost[0]
        return_info['gbest'] = env_cost[-1]
        for key in required_info.keys():
            return_info[key] = env.get_env_attr(required_info[key])
        env.close()

        return is_train_ended, return_info


