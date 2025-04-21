from ..optimizer.learnable_optimizer import Learnable_Optimizer
import numpy as np
from typing import Union, Iterable

def cal_fdc(sample, fitness):
    best = np.argmin(fitness)
    distance = np.linalg.norm(sample-sample[best], axis=-1)
    cfd = np.mean((fitness - np.mean(fitness)) * (distance - np.mean(distance)))
    return cfd / (np.var(distance)*np.var(fitness) + 1e-6)


def cal_rie(fitness):
    epsilon_star = 0
    for i in range(1,len(fitness)):
        if (fitness[i] - fitness[i-1]) > epsilon_star:
            epsilon_star = fitness[i] - fitness[i-1]
    # cal rie
    hs = []
    for k in range(9):
        epsilon = 0
        if k < 8:
            epsilon = epsilon_star / (2 ** k)
        s = []
        for i in range(len(fitness) - 1):
            if (fitness[i+1] - fitness[i]) < -epsilon:
                s.append(-1)
            elif (fitness[i+1] - fitness[i]) > epsilon:
                s.append(1)
            else:
                s.append(0)
        freq = np.zeros(6)
        for i in range(len(fitness) - 2):
            if s[i] == -1 and s[i+1] == 0:
                freq[0] += 1
            elif s[i] == -1 and s[i+1] == 1:
                freq[1] += 1
            elif s[i] == 0 and s[i+1] == 1:
                freq[2] += 1
            elif s[i] == 0 and s[i+1] == -1:
                freq[3] += 1
            elif s[i] == 1 and s[i+1] == -1:
                freq[4] += 1
            else:
                freq[5] += 1
        freq[freq == 0] = len(fitness)
        freq /= len(fitness)
        entropy = -np.sum(freq * np.log(freq) / np.log(6))
        hs.append(entropy)
    return max(hs)


def cal_acf(fitness):
    avg_f = np.mean(fitness)
    a = np.sum((fitness - avg_f) ** 2) + 1e-6
    acf = 0
    for i in  range(len(fitness) - 1):
        acf += (fitness[i] - avg_f) * (fitness[i + 1] - avg_f)
    acf /= a
    return acf


def cal_nop(sample, fitness):
    best = np.argmin(fitness)
    distance = np.linalg.norm(sample - sample[best], axis=-1)
    data = np.stack([fitness, distance], axis=0)
    data = data.T
    data = data[np.argsort(data[:, 1]), :]
    fitness_sorted = data[:,0]
    r = 0
    for i in range(len(fitness) - 1):
        if fitness_sorted[i+1] < fitness_sorted[i]:
            r += 1
    return r / len(fitness)


def random_walk_sampling(population, dim, steps, rng):
    pmin = np.min(population, axis=0)
    pmax = np.max(population, axis=0)
    walks = []
    start_point = rng.rand(dim)
    walks.append(start_point.tolist())
    for _ in range(steps - 1):
        move = rng.rand(dim)
        start_point = (start_point + move) % 1
        walks.append(start_point.tolist())
    return pmin + (pmax - pmin) * np.array(walks)


def cal_reward(survival, pointer):
    reward = 0
    for i in range(len(survival)):
        if i == pointer:
            if survival[i] == 1:
                reward += 1
        else:
            reward += 1/survival[i]
    return reward / len(survival)


class DEDQN_Optimizer(Learnable_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        config.NP = 100
        config.F = 0.5
        config.Cr = 0.5
        config.rwsteps = config.NP
        self.__config = config

        self.__dim = config.dim
        self.__NP = config.NP
        self.__F = config.F
        self.__Cr = config.Cr
        self.__maxFEs = config.maxFEs
        self.__rwsteps = config.rwsteps
        self.__solution_pointer = 0 #indicate which solution receive the action
        self.__population = None
        self.__cost = None
        self.__gbest = None
        self.__gbest_cost = None
        self.__state = None
        self.__survival = None
        self.fes = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

    def __cal_feature(self, problem):
        samples = random_walk_sampling(self.__population, self.__dim, self.__rwsteps, self.rng)
        if problem.optimum is None:
            samples_cost = problem.eval(self.__population)
        else:
            samples_cost = problem.eval(self.__population) - problem.optimum
        # calculate fdc
        fdc = cal_fdc(samples, samples_cost)
        rie = cal_rie(samples_cost)
        acf = cal_acf(samples_cost)
        nop = cal_nop(samples, samples_cost)
        self.fes += self.__rwsteps
        return np.array([fdc, rie, acf, nop])

    def init_population(self, problem):
        self.__population = self.rng.rand(self.__NP, self.__dim) * (problem.ub - problem.lb) + problem.lb  # [lb, ub]
        self.__survival = np.ones(self.__population.shape[0])
        if problem.optimum is None:
            self.__cost = problem.eval(self.__population)
        else:
            self.__cost = problem.eval(self.__population) - problem.optimum
        self.__gbest = self.__population[self.__cost.argmin()]
        self.__gbest_cost = self.__cost.min()
        self.fes = self.__NP
        self.log_index = 1
        self.cost = [self.__gbest_cost]
        self.__state = self.__cal_feature(problem)

        if self.__config.full_meta_data:
            self.meta_X = [self.__population.copy()]
            self.meta_Cost = [self.__cost.copy()]

        return self.__state

    def update(self, action, problem):
        # mutate first
        if action == 0:
            u = rand_1_single(self.__population, self.__F, self.__solution_pointer, rng=self.rng)
        elif action == 1:
            u = cur_to_rand_1_single(self.__population, self.__F, self.__solution_pointer, rng=self.rng)
        elif action == 2:
            u = best_2_single(self.__population, self.__gbest, self.__F, self.__solution_pointer, rng=self.rng)
        else:
            raise ValueError(f'action error: {action}')
        # BC
        u = clipping(u, problem.lb, problem.ub)
        # then crossover
        u = binomial(self.__population[self.__solution_pointer], u, self.__Cr, self.rng)
        # select from u and x
        if problem.optimum is None:
            u_cost = problem.eval(u)
        else:
            u_cost = problem.eval(u) - problem.optimum
        self.fes += self.__NP
        if u_cost <= self.__cost[self.__solution_pointer]:
            self.__population[self.__solution_pointer] = u
            self.__cost[self.__solution_pointer] = u_cost
            self.__survival[self.__solution_pointer] = 1
            if u_cost < self.__gbest_cost:
                self.__gbest = u
                self.__gbest_cost = u_cost
        else:
            self.__survival[self.__solution_pointer] += 1
        self.__state = self.__cal_feature(problem)

        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.__gbest_cost)

        reward = cal_reward(self.__survival, self.__solution_pointer)

        
        if problem.optimum is None:
            is_done = self.fes >= self.__maxFEs
        else:
            is_done = self.fes >= self.__maxFEs

        if self.__config.full_meta_data:
            self.meta_X.append(self.__population.copy())
            self.meta_Cost.append(self.__cost.copy())

        if is_done:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.__gbest_cost
            else:
                while len(self.cost) < self.__config.n_logpoint + 1:
                    self.cost.append(self.__gbest_cost)
        self.__solution_pointer = (self.__solution_pointer + 1) % self.__population.shape[0]
        info = {}
        return self.__state, reward, is_done , info

def clipping(x: Union[np.ndarray, Iterable],
             lb: Union[np.ndarray, Iterable, int, float, None],
             ub: Union[np.ndarray, Iterable, int, float, None]
             ) -> np.ndarray:
    return np.clip(x, lb, ub)

def binomial(x: np.ndarray, v: np.ndarray, Cr: Union[np.ndarray, float], rng) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(1, -1)
        v = v.reshape(1, -1)
    NP, dim = x.shape
    jrand = rng.randint(dim, size=NP)
    if isinstance(Cr, np.ndarray) and Cr.ndim == 1:
        Cr = Cr.reshape(-1, 1)
    u = np.where(rng.rand(NP, dim) < Cr, v, x)
    u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
    if u.shape[0] == 1:
        u = u.squeeze(axis=0)
    return u

def generate_random_int_single(NP: int, cols: int, pointer: int, rng: np.random.RandomState = None) -> np.ndarray:
    r = rng.randint(low=0, high=NP, size=cols)
    while pointer in r:
        r = rng.randint(low=0, high=NP, size=cols)
    return r

def rand_1_single(x: np.ndarray, F: float, pointer: int, r: np.ndarray = None, rng: np.random.RandomState = None) -> np.ndarray:
    if r is None:
        r = generate_random_int_single(x.shape[0], 3, pointer,rng=rng)
    return x[r[0]] + F * (x[r[1]] - x[r[2]])

def best_2_single(x: np.ndarray, best: np.ndarray, F: float, pointer: int, r: np.ndarray = None, rng: np.random.RandomState = None) -> np.ndarray:
    if r is None:
        r = generate_random_int_single(x.shape[0], 4, pointer, rng=rng)
    return best + F * (x[r[0]] - x[r[1]] + x[r[2]] - x[r[3]])

def cur_to_rand_1_single(x: np.ndarray, F: float, pointer: int, r: np.ndarray = None, rng: np.random.RandomState = None) -> np.ndarray:
    if r is None:
        r = generate_random_int_single(x.shape[0], 3, pointer, rng=rng)
    return x[pointer] + F * (x[r[0]] - x[pointer] + x[r[1]] - x[r[2]])
