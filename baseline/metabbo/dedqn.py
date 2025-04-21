from .networks import MLP
from ...rl.dqn import *


class DEDQN(DQN_Agent):
    def __init__(self, config):
        
        self.config = config
        self.config.state_size = 4
        self.config.n_act = 3
        self.config.mlp_config = [{'in': config.state_size, 'out': 10, 'drop_out': 0, 'activation': 'ReLU'},
                             {'in': 10, 'out': 10, 'drop_out': 0, 'activation': 'ReLU'},
                             {'in': 10, 'out': config.n_act, 'drop_out': 0, 'activation': 'None'}]
        self.config.lr_model = 1e-4
        # origin DEDDQN doesn't have decay
        self.config.lr_decay = 1
        self.config.epsilon = 0.1
        self.config.gamma = 0.8
        self.config.memory_size = 100
        self.config.batch_size = 64
        self.config.warm_up_size = config.batch_size

        self.config.device = config.device
        # origin DEDDQN doesn't have clip 
        self.config.max_grad_norm = math.inf
        self.config.optimizer = 'AdamW'
        # origin code does not have lr_scheduler
        self.config.lr_scheduler = 'ExponentialLR'
        self.config.criterion = 'MSELoss'
        model = MLP(self.config.mlp_config).to(self.config.device)

        # self.__cur_checkpoint=0
        super().__init__(self.config, {'model': model}, self.config.lr_model)

    def __str__(self):
        return "DEDQN"

