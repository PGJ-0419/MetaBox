from typing import Any
import torch
from torch import nn
from ..optimizer.learnable_optimizer import Learnable_Optimizer
import torch
import numpy as np

def vector2nn(x,net):
    assert len(x) == sum([param.nelement() for param in net.parameters()]), 'dim of x and net not match!'
    params = net.parameters()
    ptr = 0
    for v in params:
        num_of_params = v.nelement()
        temp = torch.Tensor(x[ptr: ptr+num_of_params])
        v.data = temp.reshape(v.shape)
        ptr += num_of_params
    return net

class Policy(nn.Module):
    def __init__(self, pop_size, mu = 0, sigma = 1.0, DK = 16, device = None):
        super(Policy, self).__init__()
        self.pop_size = pop_size
        self.mu = mu
        self.sigma = sigma
        self.DF = 2
        self.D_sigma = 1
        self.DK = DK
        self.device = device

        # Linear layers with bias
        self.W_QP = nn.Linear(self.DF, DK, bias = True) # 32 + 16 = 48
        self.W_KC = nn.Linear(self.DF, DK, bias = True) # 32 + 16 = 48
        self.W_VC = nn.Linear(self.DF, DK, bias = True) # 32 + 16 = 48

        self.W_QS = nn.Linear(DK, DK, bias = True) # 256 + 16 = 272
        self.W_KS = nn.Linear(self.DF, DK, bias = True) # 32 + 16 = 48

        self.W_QM = nn.Linear(self.DF + self.D_sigma, DK, bias = True) # 48 + 16 = 64
        self.W_KM = nn.Linear(self.DF + self.D_sigma, DK, bias = True) # 48 + 16 = 64
        self.W_VM = nn.Linear(self.DF + self.D_sigma, DK, bias = True) # 48 + 16 = 64

        self.W_sigma = nn.Linear(DK, self.D_sigma, bias = True) # 16 + 1 = 17

        # Apply custom initialization
        self._init_weights(self.mu, self.sigma)

    def _init_weights(self, mu, sigma):
        for layer in [
            self.W_QP, self.W_KC, self.W_VC,
            self.W_QS, self.W_KS,
            self.W_QM, self.W_KM, self.W_VM,
            self.W_sigma
        ]:
            nn.init.normal_(layer.weight, mean = mu, std = sigma)
            nn.init.normal_(layer.bias, mean = mu, std = sigma)

    def trans_F(self, f):
        z_score = (f - f.mean()) / (f.std() + 1e-8)
        ranks = torch.argsort(torch.argsort(-1 * z_score))
        scaled_rank = ranks / (len(ranks) - 1) - 0.5

        return torch.stack([z_score, scaled_rank], dim=1) # [NP, 2]

    def adaptation(self, fitness, sigma):
        # 先变 torch
        fitness = torch.Tensor(fitness).to(self.device)
        sigma = torch.Tensor(sigma).to(self.device)

        F_P = self.trans_F(fitness)
        F_M = torch.cat([F_P, sigma.unsqueeze(1)], dim = 1) # [NP, 3]

        K_M = self.W_KM(F_M) # [NP, DK]
        Q_M = self.W_QM(F_M) # [NP, DK]
        V_M = self.W_VM(F_M) # [NP, DK]

        scale = Q_M.shape[-1] ** 0.5
        A_M = torch.softmax(torch.matmul(Q_M, K_M.T) / scale, dim = 1)
        A_M = torch.matmul(A_M, V_M) # [NP, DK]

        delta_sigma = torch.exp(0.5 * self.W_sigma(A_M))[:, 0] # [NP]

        return delta_sigma * sigma

    def selection(self, fitness_c, fitness_p):
        # 先变 torch
        fitness_c = torch.Tensor(fitness_c).to(self.device)
        fitness_p = torch.Tensor(fitness_p).to(self.device)

        F_C = self.trans_F(fitness_c)
        F_P = self.trans_F(fitness_p)

        K_C = self.W_KC(F_C) # [N, DK]
        Q_P = self.W_QP(F_P) # [E, DK]
        V_C = self.W_VC(F_C) # [N, DK]

        scale = Q_P.shape[-1] ** 0.5
        A_S = torch.softmax(torch.matmul(Q_P, K_C.T) / scale, dim = 1)
        A_S = torch.matmul(A_S, V_C)

        Q_S = self.W_QS(A_S) # [E, DK]
        K_S = self.W_KS(F_C) # [N, DK]

        M_S = torch.matmul(Q_S, K_S.T) / scale
        # 创建一个全是 1 的列，shape [NP, 1]
        ones_column = torch.ones(M_S.size(0), 1, device = M_S.device)

        # 拼接到 attn_scores 的最后一列
        M_S = torch.cat((M_S, ones_column), dim = 1)  # [E, NP+1]
        M_S = torch.softmax(M_S, dim = 1) # [E, NP + 1]

        idx = torch.distributions.Categorical(probs = M_S).sample() # [E]

        S = torch.nn.functional.one_hot(idx, num_classes = M_S.size(1)).float() # [E, NP + 1]
        return S



class LGA_Optimizer(Learnable_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.NP = 16

        self.MaxFEs = config.maxFEs

        self.policy = Policy(self.NP, 0, 1, 16, self.config.device)

        self.fes = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

    def __str__(self):
        return "LGA_Optimizer"

    def get_costs(self, position, problem):
        if problem.optimum is None:
            cost = problem.eval(position)
        else:
            cost = problem.eval(position) - problem.optimum

        return cost

    def get_state(self):
        return self.fitness

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / (e_x.sum(axis = 0) + 1e-8)

    def init_population(self, problem):
        self.fes = 0
        dim = problem.dim

        self.population = (problem.ub - problem.lb) * self.rng.rand(self.NP, dim) + problem.lb
        self.sigma = np.ones(self.NP) * 0.2

        self.c_cost = self.get_costs(self.population, problem)

        self.fitness = (1 - self.softmax(self.c_cost)) / (self.NP - 1)

        self.fes += self.NP

        self.gbest_val = np.min(self.c_cost)

        self.cost = [self.gbest_val]
        self.log_index = 1

        self.init_gbest = self.gbest_val

        if self.config.full_meta_data:
            self.meta_X = [self.population.copy()]
            self.meta_Cost = [self.c_cost.copy()]

        return None

    def update(self, action, problem):
        # action 是 网络

        self.policy = vector2nn(action['net'], self.policy).to(self.config.device)

        skip_step = action['skip_step']

        step = 0
        is_end = False
        init_y = None
        dim = problem.dim

        while not is_end:
            indices = self.rng.choice(np.arange(self.NP), size = self.NP, replace = True, p = self.fitness)

            population = self.population[indices]

            sigma = self.sigma[indices]
            fitness = self.fitness[indices]
            c_cost = self.c_cost[indices]

            # cal MRA

            sigma_C = self.policy.adaptation(fitness, sigma).detach().cpu().numpy()  # [NP]
            sigma_C_dim = np.tile(sigma_C[:, None], (1, dim))

            # mutate
            child_population = population + sigma_C_dim * self.rng.randn(self.NP, dim)  # [NP, dim]

            child_population = np.clip(child_population, problem.lb, problem.ub)


            child_c_cost = self.get_costs(child_population, problem)
            self.fes += self.NP

            child_fitness = (1 - self.softmax(child_c_cost)) / (self.NP - 1)  # [NP]

            S = self.policy.selection(child_fitness, fitness).detach().cpu().numpy()  # [E, NP + 1]

            self.population = S[:, :self.NP] @ child_population + np.diag(S[:, -1]) @ population

            self.sigma = S[:, :self.NP] @ sigma_C + np.diag(S[:, -1]) @ sigma

            self.fitness = S[:, :self.NP] @ child_fitness + np.diag(S[:, -1]) @ fitness

            self.fitness = self.softmax(self.fitness)

            self.c_cost = S[:, :self.NP] @ child_c_cost + np.diag(S[:, -1]) @ c_cost

            if step == 0:
                init_y = np.min(self.c_cost)

            step += 1

            self.gbest_val = np.minimum(self.gbest_val, np.min(self.c_cost))

            if problem.optimum is None:
                is_end = (self.fes >= self.MaxFEs)
            else:
                is_end = (self.fes >= self.MaxFEs or np.min(self.c_cost) <= 1e-8)

            if skip_step is not None:
                is_end = is_end or step >= skip_step

            if self.fes >= self.log_index * self.log_interval:
                self.log_index += 1
                self.cost.append(self.gbest_val)

            if self.config.full_meta_data:
                self.meta_X.append(self.population.copy())
                self.meta_Cost.append(self.c_cost.copy())

            if is_end:
                if len(self.cost) >= self.config.n_logpoint + 1:
                    self.cost[-1] = self.gbest_val
                else:
                    self.cost.append(self.gbest_val)
        info = {}
        return self.fitness, (init_y - self.gbest_val) / init_y, is_end, info























        dim = problem.dim
        # sample
        indices = self.rng.choice(np.arange(self.NP), size = self.NP, replace = True, p = self.fitness)

        population = self.population[indices]

        sigma = self.sigma[indices]
        fitness = self.fitness[indices]
        c_cost = self.c_cost[indices]

        # cal MRA

        sigma_C = action.adaptation(fitness, sigma).detach().cpu().numpy() # [NP]
        sigma_C_dim = np.tile(sigma_C[:, None], (1, dim))

        # mutate
        child_population = population + sigma_C_dim * self.rng.randn((self.NP, dim)) # [NP, dim]

        child_c_cost = self.get_costs(child_population, problem)
        self.fes += self.NP

        child_fitness = (1 - self.softmax(child_c_cost)) / (self.NP - 1) # [NP]

        S = action.selection(child_fitness, fitness).detach().cpu().numpy() # [E, NP + 1]

        self.population = S[:, :self.NP] @ child_population + np.diag(S[:, -1]) @ population

        self.sigma = S[:, :self.NP] @ sigma_C + np.diag(S[:, -1]) @ sigma

        self.fitness = S[:, :self.NP] @ child_fitness + np.diag(S[:, -1]) @ fitness

        self.c_cost = S[:, :self.NP] @ child_c_cost + np.diag(S[:, -1]) @ c_cost

        new_gbest_val = np.min(self.c_cost)

        reward = (pre_gbest_val - new_gbest_val) / self.init_gbest

        self.gbest_val = np.minimum(pre_gbest_val, new_gbest_val)

        if problem.optimum is None:
            is_end = self.fes >= self.MaxFEs
        else:
            is_end = self.fes >= self.MaxFEs

        if self.config.full_meta_data:
            self.meta_X.append(self.population.copy())
            self.meta_Cost.append(self.c_cost.copy())

        next_state = self.get_state()

        if self.fes >= self.log_interval * self.log_index:
            self.log_index += 1
            self.cost.append(self.gbest_val)

        if is_end:
            if len(self.cost) >= self.config.n_logpoint + 1:
                self.cost[-1] = self.gbest_val
            else:
                while len(self.cost) < self.__config.n_logpoint + 1:
                    self.cost.append(self.gbest_val)

        info = {}

        return next_state, reward, is_end, info


