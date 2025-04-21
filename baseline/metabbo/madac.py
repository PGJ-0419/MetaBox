import collections
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from ...rl.vdn import VDN_Agent


class MultiAgentQNet(nn.Module):
    def __init__(self, input_shape, agent_configs):
        """
        Args:
            input_shape (int): 输入特征维度
            agent_configs (list of dict): 每个 agent 的配置字典，包括 'name', 'n_actions', 'n_valid_actions'
        """
        super(MultiAgentQNet, self).__init__()
        self.agents = nn.ModuleList()

        for config in agent_configs:
            agent = nn.Sequential(
                nn.Linear(input_shape, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, config['n_actions'])
            )
            # 保存每个 agent 的有效动作数
            agent.n_valid_actions = config['n_valid_actions']
            self.agents.append(agent)

        self.n_agents = len(agent_configs)

    def forward(self, obs):
        q_values = []
        for i, agent in enumerate(self.agents):
            x = agent(obs[:, i, :])
            # 无效动作置为-999
            x[:, agent.n_valid_actions:] = -999
            q_values.append(x.unsqueeze(1))
        return th.cat(q_values, dim=1)


class MADAC_Agent(VDN_Agent):
    def __init__(self, config):
        
        agent_configs = [
            {'name': 'ns_agent', 'n_actions': 4, 'n_valid_actions': 4},
            {'name': 'os_agent', 'n_actions': 4, 'n_valid_actions': 4},
            {'name': 'pc_agent', 'n_actions': 4, 'n_valid_actions': 4},
            {'name': 'weight_agent', 'n_actions': 4, 'n_valid_actions': 2}
        ]
        model = MultiAgentQNet(input_shape=22, agent_configs=agent_configs)
        self.set_network({'model': model},0.001)
        config.gamma = 0.01
        config.n_act = 4
        config.epsilon = 0.1
        config.max_grad_norm = 5
        config.memory_size = 5000
        config.batch_size = 64
        config.warm_up_size = 500
        config.chunk_size = 1
        config.update_iter = 10
        config.device = 'cuda' if th.cuda.is_available() else 'cpu'
        config.n_agent = 4
        config.available_action = [4, 4, 4, 2]
        config.optimizer = 'Adam'       
        # config.lr_model = 1e-3           
        # config.lr_scheduler = 'StepLR'
        # config.lr_decay = 0.99
        config.criterion = 'MSELoss'  
        config.agent_save_dir = 'model'  
        config.max_learning_step = 10000  
        config.target_update_interval = 10  
        config.save_interval = 1000 

        super().__init__(config)
        

class Config:
    def __init__(self):
        # 设置默认值，具体数值可以根据需要进行调整
        self.gamma = 0.99               # 折扣因子
        self.n_act = 4                  # 动作空间大小
        self.epsilon = 0.5             # epsilon-greedy 策略中的 epsilon 值
        self.max_grad_norm = 10.0      # 最大梯度裁剪
        self.memory_size = 10000       # 经验回放池大小
        self.batch_size = 64           # 批量大小
        self.warm_up_size = 1000      # 预热步数
        self.device = 'cuda' if th.cuda.is_available() else 'cpu'  # 使用的设备，默认为 'cuda' 或 'cpu'
        self.n_agent = 4               # 代理数量
        self.available_action = 4      # 可用动作数量

if __name__ == '__main__':
    # 输入数据 shape: (batch_size, n_agents, input_shape)

    # 创建一个 config 实例
    config = Config()
    obs = th.randn(32, 4, 22)
    madac_agent = MADAC_Agent(config)

    print()


