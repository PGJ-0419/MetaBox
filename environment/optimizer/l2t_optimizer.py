import numpy as np
import torch
import copy
from typing import Any, Tuple
import time

def DE_mutation(populations):
    # input: pupulations [population_cnt, dim]
    # output: mutants [population_cnt, dim]
    F = 0.5
    pop_cnt, dim = populations.shape
    mutants = copy.deepcopy(populations)
    for j in range(pop_cnt):
        r1 = np.random.randint(low=0, high=pop_cnt)
        r2 = np.random.randint(low=0, high=pop_cnt)
        r3 = np.random.randint(low=0, high=pop_cnt)
        while r1 == j:
            r1 = np.random.randint(low=0, high=pop_cnt)
        while r2 == r1 or r2 == j:
            r2 = np.random.randint(low=0, high=pop_cnt)
        while r3 == r2 or r3 == r1 or r3 == j:
            r3 = np.random.randint(low=0, high=pop_cnt)

        x1 = populations[r1]
        x2 = populations[r2]
        x3 = populations[r3]
        mutant = x1 + F * (x2 - x3)
        mutant = np.clip(mutant, a_min=0, a_max=1)
        mutants[j] = mutant

    return mutants

def DE_crossover(mutants, populations):
    CR = 0.7
    U = copy.deepcopy(mutants)
    try:
        population_cnt, dim = mutants.shape
    except ValueError as e:
        print("ValueError occurred:", e)
        print('mutant_shape',mutants.shape)

    #population_cnt, dim = mutants.shape
    for j in range(population_cnt):
        rand_pos = np.random.randint(low=0, high=dim)
        for k in range(dim):
            mutant = mutants[j]
            rand = np.random.rand()
            if rand <= CR or k == rand_pos:
                U[j][k] = mutant[k]

            if rand > CR and k != rand_pos:
                U[j][k] = populations[j][k]
    return U

def DE_rand_1(populations):
    mutants = DE_mutation(populations)
    DE_offsprings = DE_crossover(mutants, populations)
    return DE_offsprings


def mixed_DE(populations, source_pupulations, KT_index, action_2, action_3):
    population_target = populations[KT_index]
    pop_cnt, dim = source_pupulations.shape
    mutants = []
    F = 0.5
    for i in range(population_target.shape[0]):
        r1, r2, r3, r4, r5, r6 = np.random.choice(np.arange(pop_cnt),size=6, replace=False)
        X_r1 = populations[r1]
        X_r2 = source_pupulations[r2]
        X_r3 = populations[r3]
        X_r4 = populations[r4]
        X_r5 = source_pupulations[r5]
        X_r6 = source_pupulations[r6]

        mutant = (1 - action_2) * X_r1 + action_2 * X_r2 + F * (1 - action_3) * (X_r3 - X_r4) + F * action_3 * (
                    X_r5 - X_r6)

        mutants.append(mutant)

    mutants = np.array(mutants)
    U = DE_crossover(mutants, population_target)

    return U



"""
This is a basic class for learnable backbone optimizer.
Your own backbone optimizer should inherit from this class and have the following methods:
    1. __init__(self, config) : to initialize the backbone optimizer.
    2. init_population(self, problem) : to initialize the population, calculate costs using problem.eval()
       and record some information such as pbest and gbest if needed. It's expected to return a state for
       agent to make decisions.
    3. update(self, action, problem) : to update the population or one individual in population as you wish
       using the action given by agent, calculate new costs using problem.eval() and update some records
       if needed. It's expected to return a tuple of [next_state, reward, is_done] for agent to learn.
"""
class Learnable_Optimizer:
    """
    Abstract super class for learnable backbone optimizers.
    """
    def __init__(self, config):
        self.__config = config

    def init_population(self,
                        tasks:Any) -> Any:
        raise NotImplementedError

    def update(self,
               action: Any,
               tasks:Any) -> Tuple[Any]:
        raise NotImplementedError

    def seed(self, seed = None):
        rng_seed = int(time.time()) if seed is None else seed

        self.rng = np.random.default_rng(rng_seed)

        self.rng_cpu = torch.Generator().manual_seed(rng_seed)

        self.rng_gpu = None
        if self.__config.device.type == 'cuda':
            self.rng_gpu = torch.Generator(device = self.__config.device).manual_seed(rng_seed)
        # GPU: torch.rand(4, generator = rng_gpu, device = 'self.__config.device')
        # CPU: torch.rand(4, generator = rng_cpu)


class L2T_Optimizer(Learnable_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.__config = config
        self.task_cnt = config.task_cnt
        self.dim = config.dim
        self.generation = 0
        self.pop_cnt = 50
        self.total_generation = config.generation

        self.flag_improved = np.array([0 for _ in range(self.task_cnt)], dtype=np.float32)
        self.stagnation = np.array([0 for _ in range(self.task_cnt)], dtype=np.float32) 
        self.old_action_1 = np.array([0 for _ in range(self.task_cnt)], dtype=np.float32)
        self.old_action_2 = np.array([0 for _ in range(self.task_cnt)], dtype=np.float32)
        self.old_action_3 = np.array([0 for _ in range(self.task_cnt)], dtype=np.float32)
        self.N_kt = np.array([0 for _ in range(self.task_cnt)], dtype=np.float32)
        self.Q_kt = np.array([0 for _ in range(self.task_cnt)], dtype=np.float32)

        self.gbest = np.array([1e+32 for _ in range(self.task_cnt)],dtype=np.float32)
        self.task = None
        self.offsprings = np.array([[np.random.rand(self.dim) for i in range(self.pop_cnt)] for _ in range(self.task_cnt)])
        self.noKT_offsprings = np.array([[np.random.rand(self.dim) for i in range(self.pop_cnt)] for _ in range(self.task_cnt)])
        self.KT_offsprings = [None for _ in range(self.task_cnt)]
        self.KT_index = [None for _ in range(self.task_cnt)]
        self.parent_population = None
        self.reward = [0 for _ in range(self.task_cnt)]
        self.total_reward = 0

        self.fes = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

    def get_state(self):
        state_o = self.generation / self.total_generation
        states = np.array([state_o], dtype=np.float32)

        for i in range(self.task_cnt):
            states_t = []
            state_1 = self.stagnation[i] / self.total_generation
            states_t.append(state_1)
            state_2 = self.flag_improved[i]
            states_t.append(state_2)
            if self.N_kt[i] == 0:
                state_3 = 0
            else:
                state_3 = self.Q_kt[i]

            states_t.append(state_3)
            state_4 = np.mean(np.std(self.parent_population[i], axis=-1))
            states_t.append(state_4)
            state_5 = self.old_action_1[i]
            state_6 = self.old_action_2[i]
            state_7 = self.old_action_3[i]

            states_t.append(state_5)
            states_t.append(state_6)
            states_t.append(state_7)

            states_t = np.array(states_t, dtype=np.float32)
            states = np.concatenate((states, states_t),axis=-1)
        self.generation += 1

        return states

    def init_population(self, tasks):
        self.fes = 0
        self.task = tasks
        self.parent_population = np.array([[np.random.rand(self.dim) for i in range(self.pop_cnt)] for _ in range(self.task_cnt)])
        self.log_index = 1
        

        parent_fitnesses_list = []
        for i in range(self.task_cnt):
            fitnesses = self.task[i].eval(self.parent_population[i])
            self.gbest[i] = np.min(fitnesses, axis=-1)
            parent_fitnesses_list.append(fitnesses)
        
        parent_fitnesses_np = np.array(parent_fitnesses_list, dtype=np.float32)
        
        self.cost = [self.gbest]

        state = self.get_state()


        if self.__config.full_meta_data:
            self.meta_X = [self.parent_population.copy()]
            self.meta_Cost = [parent_fitnesses_np.copy()]
            
        return state

    def self_update(self):
        for i in range(self.task_cnt):
            self.noKT_offsprings[i] = DE_rand_1(self.parent_population[i])
    

    def transfer(self,actions):
        for i in range(self.task_cnt):
            action_1 = actions[0]
            action_2 = actions[1]
            action_3 = actions[2]

            rand_source_index = np.random.randint(low=0,high=self.task_cnt)
            while rand_source_index == i:
                rand_source_index = np.random.randint(low=0, high=self.task_cnt)

            source_population = self.parent_population[rand_source_index]

            self.N_kt[i] = 0.5 * action_1
            self.KT_count = int(np.ceil(self.N_kt[i] * self.pop_cnt))
            if self.KT_count == 0:
                self.KT_count = 1
            self.KT_index[i] = np.random.choice(np.arange(self.pop_cnt), size=self.KT_count, replace=False)

            self.KT_offsprings[i] = mixed_DE(self.parent_population[i], source_population, self.KT_index[i], action_2, action_3)
            self.offsprings[i] = copy.deepcopy(self.noKT_offsprings[i])
            for j in range(self.KT_count):
                self.offsprings[i][self.KT_index[i][j]] = self.KT_offsprings[i][j]

            self.old_action_1[i] = action_1
            self.old_action_2[i] = action_2
            self.old_action_3[i] = action_3

    def seletion(self):
        parent_finesses_list = []
        for i in range(self.task_cnt):
            ps = self.parent_population[i].shape[0]
            self.fes += ps
            parent_population_fitness = self.task[i].eval(self.parent_population[i])
            offsprings_population_fitness = self.task[i].eval(self.offsprings[i])

            next_population = copy.deepcopy(self.parent_population)
           
            S_update = 0
            S_KT = 0
            for j in range(self.pop_cnt):
                if offsprings_population_fitness[j] <= parent_population_fitness[j]:
                    if j not in self.KT_index[i]:
                        S_update += 1
                    else:
                        S_KT += 1

                    next_population[i][j] = self.offsprings[i][j]
                else:
                    next_population[i][j] = self.parent_population[i][j]

            self.reward[i] = (float)(S_update-S_KT) / self.pop_cnt
            self.Q_kt[i] = float(S_KT) / self.KT_count

            flag = 0
            fitnesses = self.task[i].eval(next_population[i])
            parent_finesses_list.append(fitnesses)
            best_fitness = np.min(fitnesses,axis=-1)
            if(best_fitness < self.gbest[i]):
                self.gbest[i] = best_fitness
                flag = 1

            if(flag):
                self.flag_improved[i] = 1
            else:
                self.flag_improved[i] = 0
                self.stagnation[i] += 1

            self.parent_population[i] = next_population[i]

        parent_finesses_np = np.array(parent_finesses_list, dtype=np.float32)
        if self.__config.full_meta_data:
            self.meta_X.append(self.parent_population.copy())
            self.meta_Cost.append(parent_finesses_np.copy())
        return self.get_state()

    def update(self, actions, tasks):
        self.self_update()
        self.transfer(actions)
        next_state = self.seletion()

        for _ in range(self.task_cnt):
            self.total_reward += self.reward[_]

        is_end = False
        if self.generation > self.total_generation:
            is_end = True
        
        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.gbest)

        if is_end:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.gbest
            else:
                while len(self.cost) < self.__config.n_logpoint + 1:
                    self.cost.append(self.gbest)

        info = {}
        return next_state, self.total_reward, is_end, info
