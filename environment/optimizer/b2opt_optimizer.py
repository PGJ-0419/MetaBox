from typing import Any

from ..optimizer.learnable_optimizer import Learnable_Optimizer
import torch
import numpy as np

# torch
class B2OPT_Optimizer(Learnable_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.NP = 100

        self.MaxFEs = config.maxFEs
        self.ems = (self.MaxFEs + self.NP - 1) // self.NP - 1

        self.fes = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

        self.ems_index = 0

    def __str__(self):
        return "B2OPT_Optimizer"

    def get_costs(self, position, problem):
        if problem.optimum is None:
            cost = problem.eval(position)
        else:
            cost = problem.eval(position) - problem.optimum

        return cost

    def __sort(self):
        _, index = torch.sort(self.c_cost)
        self.population = self.population[index]
        self.c_cost = self.c_cost[index]

    def init_population(self, problem):
        dim = problem.dim
        self.rng_torch = self.rng_cpu
        if self.config.device != "cpu":
            self.rng_torch = self.rng_gpu

        self.fes = 0
        self.population = (problem.ub - problem.lb) * torch.rand((self.NP, dim), generator = self.rng_torch, device = self.config.device, dtype = torch.float64) + problem.lb
        self.c_cost = self.get_costs(position = self.population, problem = problem)

        self.fes += self.NP

        self.ems_index = 0 # opt ob pointer

        self.gbest_val = torch.min(self.c_cost).detach().cpu().numpy()

        self.init_gbest = torch.min(self.c_cost).detach().cpu()

        self.cost = [self.gbest_val]
        self.log_index = 1

        self.__sort()

        if self.config.full_meta_data:
            self.meta_X = [self.population.detach().cpu().numpy()]
            self.meta_Cost = [self.c_cost.detach().cpu().numpy()]

        return self.get_state()
    def get_state(self):
        Y = self.c_cost
        return Y

    def update(self, action, problem):
        # 这里的action 是policy 网络
        pre_gbest = torch.min(self.c_cost.detach().cpu())


        v = action(self.population[None, :].clone().detach(), self.c_cost[None, :].clone().detach(), self.ems_index)[0] # off
        self.ems_index += 1

        new_cost = self.get_costs(position = v, problem = problem)
        self.fes += self.NP

        old_population = self.population.clone().detach()
        old_c_cost = self.c_cost.clone().detach()
        optim = new_cost.detach() < old_c_cost

        old_population[optim] = v[optim]
        old_c_cost[optim] = new_cost[optim]
        self.population = old_population
        self.c_cost = old_c_cost

        new_gbest_val = torch.min(self.c_cost).detach().cpu()

        reward = (pre_gbest - new_gbest_val) / self.init_gbest

        new_gbest_val = new_gbest_val.numpy()

        self.gbest_val = np.minimum(self.gbest_val, new_gbest_val)

        if problem.optimum is None:
            is_end = self.fes >= self.MaxFEs
        else:
            is_end = self.fes >= self.MaxFEs

        self.__sort()

        if self.config.full_meta_data:
            self.meta_X.append(self.population.detach().cpu().numpy())
            self.meta_Cost.append(self.c_cost.detach().cpu().numpy())

        next_state = self.get_state()

        if self.fes >= self.log_interval * self.log_index:
            self.log_index += 1
            self.cost.append(self.gbest_val)

        if is_end:
            if len(self.cost) >= self.config.n_logpoint + 1:
                self.cost[-1] = self.gbest_val
            else:
                while len(self.cost) < self.config.n_logpoint + 1:
                    self.cost.append(self.gbest_val)

        info = {}

        return next_state, reward, is_end, info


