from ..optimizer.learnable_optimizer import Learnable_Optimizer
import torch
import numpy as np
from scipy.spatial import distance

class PSORLNS_Optimizer(Learnable_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.__config = config

        self.w = 1
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.ps = 1000
        self.eps =  0.1  # todo: 邻域个数判断阈值
        self.TT2 = 0.8 # 选择worse better的随机数阈值
        self.neighbor_num = [5,10,20,30,40]
        

        self.fes = None
        self.cost = None
        self.pr = None
        self.sr = None
        self.log_index = None
        self.log_interval = None

    def __str__(self):
        return "PSORLNS_Optimizer"

    def cal_pr_sr(self, problem):
        raw_PR = np.zeros(5)
        raw_SR = np.zeros(5)
        solu = self.particles['current_position'].copy()
        accuracy = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        total_pkn = problem.nopt
        for acc_level in range(5):
            nfp, _ = problem.how_many_goptima(solu, accuracy[acc_level])
            raw_PR[acc_level] = nfp / total_pkn
            if nfp >= total_pkn:
                raw_SR[acc_level] = 1
        return raw_PR, raw_SR

    # initialize GPSO environment
    def initialize_particles(self, problem):
        # randomly generate the position and velocity
        rand_pos = self.rng.uniform(low = problem.lb, high = problem.ub, size = (self.ps, self.dim))
        rand_vel = self.rng.uniform(low = -self.max_velocity, high = self.max_velocity, size = (self.ps, self.dim))

        # get the initial cost
        c_cost = self.get_costs(rand_pos, problem)  # ps

        # find out the gbest_val
        gbest_val = np.min(c_cost)
        gbest_index = np.argmin(c_cost)
        gbest_position = rand_pos[gbest_index]

        pop_dist = distance.cdist(rand_pos, rand_pos)

        # store all the information of the paraticles
        self.particles = {'current_position': rand_pos.copy(),  # ps, dim
                          'c_cost': c_cost.copy(),  # ps
                          'pbest_position': rand_pos.copy(),  # ps, dim
                          'pbest': c_cost.copy(),  # ps
                          'gbest_position': gbest_position.copy(),  # dim
                          'gbest_val': gbest_val,  # 1
                          'velocity': rand_vel.copy(),  # ps,dim
                          'pop_dist': pop_dist.copy()
                          }

    # the interface for environment reseting
    def init_population(self, problem):
        self.max_fes = problem.maxfes
        self.log_interval =(self.max_fes // self.__config.n_logpoint)
        self.dim = problem.dim
        self.fes = 0
        self.max_velocity = 0.1 * (problem.ub - problem.lb)
        # set the hyperparameters back to init value if needed
        self.w = 1

        self.max_dist = np.sqrt(np.sum((problem.ub - problem.lb) ** 2))

        # initialize the population
        self.initialize_particles(problem)

        self.log_index = 1
        self.cost = [self.particles['gbest_val']]
        raw_pr, raw_sr = self.cal_pr_sr(problem)
        self.pr = [raw_pr.copy()]
        self.sr = [raw_sr.copy()]

        # get state

        # get the population state
        state = self.observe()  # ps, 9


        if self.__config.full_meta_data:
            self.meta_X = [self.particles['current_position'].copy()]
            self.meta_Cost = [self.particles['c_cost'].copy()]
            raw_pr, raw_sr = self.cal_pr_sr(problem)
            self.meta_Pr = [raw_pr.copy()]
            self.meta_Sr = [raw_sr.copy()]

        return state

    # calculate costs of solutions
    def get_costs(self, position, problem):
        ps = position.shape[0]
        self.fes += ps
        if problem.optimum is None:
            cost = problem.eval(position)
        else:
            cost = problem.eval(position) - problem.optimum
        return cost

    # feature encoding
    def observe(self):
        state = np.zeros(self.ps, 1)
        pop_dist = self.particles['pop_dist'].copy()
        pop_dist[range(self.ps), range(self.ps)] = np.inf
        neighbor_matrix = np.zeros((self.ps, self.ps))
        neighbor_matrix[pop_dist < self.eps] = 1
        sum_neighbors = np.sum(neighbor_matrix, axis = -1)
        xtate = sum_neighbors / 100
        
        return state

    # direct reward function
    def cal_reward(self, current_cost, parent_cost):
        reward = np.zeros(self.ps)
        reward[current_cost < parent_cost]  = 1
        reward[current_cost > parent_cost] = -1
        
        return reward

    def update(self, action, problem):
        is_end = False

        # record the gbest_val in the begining
        parent_cost = self.particles['c_cost']


        # generate two set of random val for pso velocity update
        new_position = np.zeros(self.ps, self.dim)
        new_velocity = np.zeros(self.ps, self.dim)
        
        pop_dist = self.particles['pop_dist'].copy()
        pop_dist[range(self.ps), range(self.ps)] = np.inf
        rank_dist = np.argsort(pop_dist, axis = -1)
        for i in range(self.ps):
            neighbors = rank_dist[i][:neighbor_num[action[i]]]
            # neighbors = np.append(neighbors, i)
            neighbors = neighbors[np.argsort(self.particles['c_cost'][neighbors])[::-1]]
            k = len(neighbors)
            c_pos = self.particles['current_position'].copy()
            rand1 = self.rng.rand()
            rand2 = self.rng.rand()
            if self.rng.rand() <= self.TT2:
                worse = neighbors[0]
            else:
                worse = neighbors[self.rng.randint(0, 0.05*k)]

            if self.rng.rand() <= self.TT2:
                better = neighbors[-1]
            else:
                better = neighbors[self.rng.randint(0.95*k, k)]
            if self.rng.rand() < 0.5:
                # update velocity
                new_velocity[i] = self.w * self.particles['velocity'][i] + self.c1 * rand1 * (self.particles['pbest_position'][i] - self.particles['current_position'][i]) + \
                            self.c2 * rand2 * (self.particles['current_position'][better] - self.particles['current_position'][i])
                # clip the velocity if exceeding the boarder
                new_velocity[i] = np.clip(new_velocity[i], -self.max_velocity, self.max_velocity)
                # update position according the boarding method
                raw_position = self.particles['current_position'][i] + new_velocity[i]
                new_position[i] = np.clip(raw_position, problem.lb, problem.ub)
            else:
                raw_position = c_pos[i] + self.rng.rand() * (c_pos[i] - c_pos[worse]) + self.rng.rand() * (c_pos[better] - c_pos[i]) 
                new_position[i] = np.clip(raw_position, problem.lb, problem.ub)
                # update velocity
                new_velocity[i] =  self.w * self.particles['velocity'][i] + self.c1 * rand1 * (self.particles['pbest_position'][i] - self.particles['current_position'][i]) + \
                            self.c2 * rand2 * (self.particles['current_position'][better] - self.particles['current_position'][i])
                # clip the velocity if exceeding the boarder
                new_velocity[i] = np.clip(new_velocity[i], -self.max_velocity, self.max_velocity)
                # update position according the boarding method
                
        # calculate the new costs
        new_cost = self.get_costs(new_position, problem)

        # update particles
        filters = new_cost < self.particles['pbest']

        new_cbest_val = np.min(new_cost)
        new_cbest_index = np.argmin(new_cost)
        filters_best_val = new_cbest_val < self.particles['gbest_val']

        new_pop_dist = distance.cdist(new_position, new_position)

        new_particles = {'current_position': new_position.copy(),
                         'c_cost': new_cost.copy(),
                         'pbest_position': np.where(np.expand_dims(filters, axis = -1),
                                                    new_position.copy(),
                                                    self.particles['pbest_position']),
                         'pbest': np.where(filters,
                                           new_cost.copy(),
                                           self.particles['pbest']),
                         'velocity': new_velocity.copy(),
                         'gbest_val': new_cbest_val if filters_best_val else self.particles['gbest_val'],
                         'gbest_position': np.where(np.expand_dims(filters_best_val, axis = -1),
                                                    new_position[new_cbest_index],
                                                    self.particles['gbest_position']),
                         'pop_dist': new_pop_dist.copy()
                         }

        # update the population
        self.particles = new_particles

        if self.__config.full_meta_data:
            self.meta_X.append(self.particles['current_position'].copy())
            self.meta_Cost.append(self.particles['c_cost'].copy())
            raw_pr, raw_sr = self.cal_pr_sr(problem)
            self.meta_Pr.append(raw_pr.copy())
            self.meta_Sr.append(raw_sr.copy())


        # see if the end condition is satisfied
        is_end = np.array([self.fes >= self.max_fes] * self.ps)

        # cal the reward
        reward = self.cal_reward(self.particles['c_cost'], parent_cost)

        # get the population next_state
        next_state = self.observe()  # ps, 9

        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.particles['gbest_val'])
            raw_pr, raw_sr = self.cal_pr_sr(problem)
            self.pr.append(raw_pr.copy())
            self.sr.append(raw_sr.copy())

            
        if is_end[0]:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.particles['gbest_val']
                raw_pr, raw_sr = self.cal_pr_sr(problem)
                self.pr[-1] = raw_pr.copy()
                self.sr[-1] = raw_sr.copy()
            else:
                self.cost.append(self.particles['gbest_val'])
                raw_pr, raw_sr = self.cal_pr_sr(problem)
                self.pr.append(raw_pr.copy())
                self.sr.append(raw_sr.copy())
            


        info = {}
        return next_state, reward, is_end, info