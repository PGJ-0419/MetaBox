import sys
import subprocess
import numpy as np
import time
import torch
import torch.nn as nn
from evox.problems.neuroevolution.brax import BraxProblem
from evox.utils import ParamsAndVector

from ...basic_problem import Basic_Problem


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layer_num):
        super(MLP,self).__init__()
        self.networks = nn.ModuleList()
        # self.in_layer = nn.Sequential(nn.Linear(state_dim,32),nn.Tanh())
        self.networks.append(nn.Sequential(nn.Linear(state_dim,32),nn.Tanh()))
        # self.hidden_layers = []
        for _ in range(hidden_layer_num):
            self.networks.append(nn.Sequential(nn.Linear(32,32),nn.Tanh()))
        # self.out_layer = nn.Linear(32,action_dim)
        self.networks.append(nn.Linear(32,action_dim))
    def forward(self, state):
        # h = self.in_layer(state)
        for layer in self.networks:
            state = layer(state)
        return torch.tanh(state)


envs = {
    'ant': {'state_dim':27, 'action_dim':8,}, # https://github.com/google/brax/blob/main/brax/envs/ant.py
    'halfcheetah': {'state_dim':18, 'action_dim':6,}, # https://github.com/google/brax/blob/main/brax/envs/half_cheetah.py
    'hopper': {'state_dim':11, 'action_dim':3,}, # https://github.com/google/brax/blob/main/brax/envs/hopper.py
    'humanoid':{'state_dim':376, 'action_dim':17,}, # https://github.com/google/brax/blob/main/brax/envs/humanoid.py
    'humanoidstandup':{'state_dim':376, 'action_dim':17,}, # https://github.com/google/brax/blob/main/brax/envs/humanoidstandup.py
    'inverted_pendulum':{'state_dim':4, 'action_dim':1,}, # https://github.com/google/brax/blob/main/brax/envs/inverted_pendulum.py
    'inverted_double_pendulum':{'state_dim':8, 'action_dim':1,}, # https://github.com/google/brax/blob/main/brax/envs/inverted_double_pendulum.py
    'pusher':{'state_dim':23, 'action_dim':7,}, # https://github.com/google/brax/blob/main/brax/envs/pusher.py
    'reacher':{'state_dim':11, 'action_dim':2,}, # https://github.com/google/brax/blob/main/brax/envs/reacher.py
    'swimmer':{'state_dim':8, 'action_dim':2,}, # https://github.com/google/brax/blob/main/brax/envs/swimmer.py
    'walker2d':{'state_dim':17, 'action_dim':6,}, # https://github.com/google/brax/blob/main/brax/envs/ant.py
}

model_depth = [
    0,
    1,
    2,
    3,
    4,
    5
]

class NE_Problem(Basic_Problem):
    def __init__(self,env_name,model_depth, seed):
        self.env_state_dim = envs[env_name]['state_dim']
        self.env_action_dim = envs[env_name]['action_dim']
        self.nn_model = MLP(self.env_state_dim, self.env_action_dim, model_depth)
        self.dim = sum(p.numel() for p in self.nn_model.parameters())
        self.ub = 5.
        self.lb = -5.
        self.pop_size = 500
        self.adapter = ParamsAndVector(dummy_model= self.nn_model) 
        self.evaluator = BraxProblem(
            policy=self.nn_model,
            env_name=env_name,
            max_episode_length=200, #todo: 10,3,5,1 should be indicated in config.py and loaded here
            num_episodes=10,
            pop_size=self.pop_size,
            seed=seed,
            reduce_fn=torch.mean,
        )

    def func(self,x): # x is a batch of neural network parameters: bs * num_params, type: numpy.array
        # x_cuda = torch.from_numpy(x).double().to(torch.get_default_device())
        # x_cuda = torch.from_numpy(x)
        # print(1)
        torch.set_default_device("cuda")
        assert x.shape[-1] == self.dim, "solution dimension not equal to problem dimension!"
        x = torch.tensor(x, device=torch.get_default_device()).float()
        pop_size = x.shape[0]
        if x.shape[0] < self.pop_size:
            x = torch.concat([x, torch.zeros(self.pop_size - pop_size, self.dim)], 0)
        nn_population = self.adapter.batched_to_params(x)
        # for key in nn_population.keys():
        #     print(nn_population[key].shape)
        rewards = self.evaluator.evaluate(nn_population)
        torch.set_default_device("cpu")
        return rewards[:pop_size]


