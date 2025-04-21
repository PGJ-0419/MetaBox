from typing import Any, Callable, List, Optional, Tuple, Union, Literal
from .problem.basic_problem import Basic_Problem
from .optimizer.learnable_optimizer import Learnable_Optimizer
from ..rl import Basic_Agent
import gym
import numpy as np


class PBO_Env(gym.Env):
    """
    An environment with a problem and an optimizer.
    """
    def __init__(self,
                 problem: Basic_Problem,
                 optimizer: Learnable_Optimizer,
                 ):
        super(PBO_Env, self).__init__()
        self.problem = problem
        self.optimizer = optimizer

    def reset(self):
        if isinstance(self.problem, list):
            for _ in range(len(self.problem)):
                self.problem[_].reset()
        else:
            self.problem.reset()
        reset_ = self.optimizer.init_population(self.problem)
        return reset_

    def step(self, action: Any):
        update_ = self.optimizer.update(action, self.problem)
        return update_

    def seed(self, seed):
        self.optimizer.seed(seed)

    def get_env_attr(self, 
                     key: str):
        if hasattr(self, key):
            return getattr(self, key)
        elif hasattr(self.optimizer, key):
            return getattr(self.optimizer, key)
        elif hasattr(self.problem, key):
            return getattr(self.problem, key)
        else:
            return None
        
    def set_env_attr(self, key: str, value: Any):
        if hasattr(self, key):
            return setattr(self, key, value)
        elif hasattr(self.optimizer, key):
            return setattr(self.optimizer, key, value)
        elif hasattr(self.problem, key):
            return setattr(self.problem, key, value)
        else:
            raise ModuleNotFoundError
