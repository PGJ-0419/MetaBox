"""
This is the basic optimizer class. All traditional optimizers should inherit from this class.
Your own traditional should have the following functions:
    1. __init__(self, config) : to initialize the optimizer
    2. run_episode(self, problem) : to run the optimizer for an episode
"""
from ..problem.basic_problem import Basic_Problem
import numpy as np
import torch
import time
class Basic_Optimizer:
    def __init__(self, config):
        self.__config = config
        self.rng_seed = None

    def seed(self, seed = None):
        rng_seed = int(time.time()) if seed is None else seed

        self.rng_seed = rng_seed

        self.rng = np.random.RandomState(rng_seed)

        self.rng_cpu = torch.Generator().manual_seed(rng_seed)

        self.rng_gpu = None
        if self.__config.device != 'cpu':
            self.rng_gpu = torch.Generator(device = self.__config.device).manual_seed(rng_seed)
        # GPU: torch.rand(4, generator = rng_gpu, device = 'self.__config.device')
        # CPU: torch.rand(4, generator = rng_cpu)

    def run_episode(self, problem: Basic_Problem):
        raise NotImplementedError
