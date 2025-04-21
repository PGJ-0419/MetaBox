import numpy as np
from ..optimizer.learnable_optimizer import Learnable_Optimizer
from typing import Union, Iterable

class RLPSO_Optimizer(Learnable_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        
        config.w_decay = True
        config.c = 2.05
        config.NP = 100
        self.__config = config

        self.__dim = config.dim
        self.__w_decay = config.w_decay
        if self.__w_decay:
            self.__w = 0.9
        else:
            self.__w = 0.729
        self.__c = config.c
        
        self.__NP = config.NP

        self.__max_fes = config.maxFEs
        self.fes = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

    def __str__(self):
        return "RLPSO_Optimizer"

    # initialize PSO environment
    def init_population(self, problem):
        rand_pos = self.rng.uniform(low=problem.lb, high=problem.ub, size=(self.__NP, self.__dim))
        self.fes = 0
        self.__max_velocity=0.1*(problem.ub-problem.lb)
        rand_vel = self.rng.uniform(low=-self.__max_velocity, high=self.__max_velocity, size=(self.__NP, self.__dim))

        c_cost = self.__get_costs(problem,rand_pos) # ps

        gbest_val = np.min(c_cost)
        gbest_index = np.argmin(c_cost)
        gbest_position = rand_pos[gbest_index]
        self.__max_cost = np.max(c_cost)

        
        self.__particles={'current_position': rand_pos.copy(),  # ?ps, dim
                          'c_cost': c_cost.copy(),  # ?ps
                          'pbest_position': rand_pos.copy(),  # ps, dim
                          'pbest': c_cost.copy(),  # ?ps
                          'gbest_position': gbest_position.copy(),  # dim
                          'gbest_val': gbest_val,  # 1
                          'velocity': rand_vel.copy(),  # ps,dim
                          'gbest_index': gbest_index  # 1
                          }
        if self.__w_decay:
            self.__w = 0.9
            
        self.__cur_index = 0
        self.log_index = 1
        self.cost = [self.__particles['gbest_val']]

        if self.__config.full_meta_data:
            self.meta_X = [rand_pos.copy()]
            self.meta_Cost = [c_cost.copy()]

        return self.__get_state(self.__cur_index)

    def __get_state(self, index):
        return np.concatenate((self.__particles['gbest_position'], self.__particles['current_position'][index]), axis=-1)

    # calculate costs of solutions
    def __get_costs(self, problem, position):
        if len(position.shape) == 2:
            self.fes += position.shape[0]
        else:
            self.fes += 1
        if problem.optimum is None:
            cost = problem.eval(position)
        else:
            cost = problem.eval(position) - problem.optimum
        return cost

    def update(self, action, problem):
        
        is_done = False

        # record the gbest_val in the begining
        self.__pre_gbest = self.__particles['gbest_val']

        # linearly decreasing the coefficient of inertia w
        if self.__w_decay:
            self.__w -= 0.5 / (self.__max_fes / self.__NP)

        # generate two set of random val for pso velocity update
        rand1 = self.rng.rand()
        rand2 = np.squeeze(action)
       
        j = self.__cur_index
        v = self.__particles['velocity'][j]
        x = self.__particles['current_position'][j]
        pbest_pos = self.__particles['pbest_position'][j]
        gbest_pos = self.__particles['gbest_position']
        pre_cost = self.__particles['c_cost'][j]

        # update velocity
        new_velocity = self.__w*v+self.__c*rand1*(pbest_pos-x)+self.__c*rand2*(gbest_pos-x)

        # clip the velocity if exceeding the boarder
        new_velocity = np.clip(new_velocity, -self.__max_velocity, self.__max_velocity)
        
        # update position
        new_x = x+new_velocity

        # print("velocity.shape = ",new_velocity.shape)
        new_x = clipping(new_x, problem.lb, problem.ub)

        # update population
        self.__particles['current_position'][j] = new_x
        self.__particles['velocity'][j] = new_velocity

        # calculate the new costs
        new_cost = self.__get_costs(problem, new_x)
        self.__particles['c_cost'][j] = new_cost
        
        # update pbest
        if new_cost < self.__particles['pbest'][j]:
            self.__particles['pbest'][j] = new_cost
            self.__particles['pbest_position'][j] = new_x
        # update gbest
        if new_cost < self.__particles['gbest_val']:
            self.__particles['gbest_val'] = new_cost
            self.__particles['gbest_position'] = new_x
            self.__particles['gbest_index'] = j

        # see if the end condition is satisfied
        if problem.optimum is None:
            is_done = self.fes >= self.__max_fes
        else:
            is_done = self.fes >= self.__max_fes

        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.__particles['gbest_val'])

        reward = (pre_cost-new_cost)/(self.__max_cost-self.__particles['gbest_val'])

        self.__cur_index = (self.__cur_index+1) % self.__NP

        if self.__config.full_meta_data:
            self.meta_X.append(self.__particles['current_position'].copy())
            self.meta_Cost.append(self.__particles['c_cost'].copy())

        if is_done:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.__particles['gbest_val']
            else:
                while len(self.cost) < self.__config.n_logpoint + 1:
                    self.cost.append(self.__particles['gbest_val'])
                
        info = {}
        
        return self.__get_state(self.__cur_index), reward, is_done, info

def clipping(x: Union[np.ndarray, Iterable],
             lb: Union[np.ndarray, Iterable, int, float, None],
             ub: Union[np.ndarray, Iterable, int, float, None]
             ) -> np.ndarray:
    return np.clip(x, lb, ub)
