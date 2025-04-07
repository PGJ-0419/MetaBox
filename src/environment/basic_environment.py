from typing import Any
from problem.basic_problem import Basic_Problem
from optimizer.learnable_optimizer import Learnable_Optimizer


class PBO_Env:
    """
    An environment with a problem and an optimizer.
    """
    def __init__(self,
                 problem: Basic_Problem,
                 optimizer: Learnable_Optimizer,
                 ):
        self.problem = problem
        self.optimizer = optimizer

    def reset(self):
        if isinstance(self.problem, list):
            for _ in range(len(self.problem)):
                self.problem[_].reset()
        else:
            self.problem.reset()
        return self.optimizer.init_population(self.problem)

    def step(self, action: Any):
        return self.optimizer.update(action, self.problem)
    
    def seed(self, seed):
        self.optimizer.seed(seed)
