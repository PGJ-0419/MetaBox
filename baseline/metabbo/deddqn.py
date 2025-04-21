from ...rl.ddqn import *
from .networks import MLP

class DEDDQN(DDQN_Agent):
    def __init__(self, config):
        self.config = config
        self.config.state_size = 99
        self.config.n_act = 4
        self.config.lr_model = 1e-4
        # origin DEDDQN doesn't have decay
        self.config.lr_decay = 1
        self.config.batch_size = 64
        self.config.epsilon = 0.1
        self.config.gamma = 0.99
        self.config.target_update_interval = 1000
        self.config.memory_size = 100000
        self.config.warm_up_size = 10000
        self.config.net_config = [{'in': config.state_size, 'out': 100, 'drop_out': 0, 'activation': 'ReLU'},
                                  {'in': 100, 'out': 100, 'drop_out': 0, 'activation': 'ReLU'},
                                  {'in': 100, 'out': 100, 'drop_out': 0, 'activation': 'ReLU'},
                                  {'in': 100, 'out': 100, 'drop_out': 0, 'activation': 'ReLU'},
                                  {'in': 100, 'out': config.n_act, 'drop_out': 0, 'activation': 'None'}]

        self.config.device = config.device
        # origin DEDDQN doesn't have clip
        self.config.max_grad_norm = math.inf

        # self.target_model is defined in DDQN_Agent

        self.config.optimizer = 'Adam'
        # origin code does not have lr_scheduler
        self.config.lr_scheduler = 'ExponentialLR'
        self.config.criterion = 'MSELoss'

        model = MLP(self.config.net_config).to(self.config.device)
        super().__init__(self.config, {'model': model}, self.config.lr_model)

    def __str__(self):
        return "DEDDQN"