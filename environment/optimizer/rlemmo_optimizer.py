from ..optimizer.learnable_optimizer import Learnable_Optimizer
import torch
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import DBSCAN


class RLEMMO_Optimizer(Learnable_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.__config = config

        self.ps = 100
        self.k_neighbors = 4
        self.n_action = 5
        self.FF = 0.5
        self.CR = 0.9
        self.eps = 0.2
        self.min_samples = 3
        self.reward_scale = 1000

        self.fes = None
        self.cost = None
        self.pr = None
        self.sr = None
        self.log_index = None
        self.log_interval = None

    def __str__(self):
        return "RLEMMO_Optimizer"

    # calculate costs of solutions
    def get_costs(self, position, problem):
        ps = position.shape[0]
        self.fes += ps
        if problem.optimum is None:
            cost = problem.eval(position)
        else:
            cost = problem.eval(position) - problem.optimum
        return cost

    def find_nei(self, pop_dist):
        pop_dist[range(self.ps), range(self.ps)] = np.inf
        pop_dist_arg = np.argsort(pop_dist.copy(), axis = -1)

        neighbor_matrix = np.zeros(shape = (self.ps, self.ps))
        for i in range(self.ps):
            neighbor_matrix[i][pop_dist_arg[i, :self.k_neighbors]] = 1
        return neighbor_matrix

    def act1(self, pop_choice):
        population = self.individuals['current_position'].copy()
        stacked_pos = population[pop_choice]
        sigmas = np.float_power(10, -np.random.randint(0, 9, (len(pop_choice), self.dim)))
        v = stacked_pos + np.random.normal(loc = 0, scale = sigmas)
        return v

    def act2(self, pop_choice):
        population = self.individuals['current_position'].copy()
        neighbor_matrix = self.individuals['neighbor_matrix'].copy()
        val = self.individuals['c_cost'].copy()
        stacked_pos = population[pop_choice]
        stacked_rs = np.zeros((len(pop_choice), 2), dtype = 'int')
        stacked_best = np.zeros(len(pop_choice), dtype = 'int')
        for idx in range(len(pop_choice)):
            neibor = np.argwhere(neighbor_matrix[pop_choice[idx]] > 0).squeeze(-1)
            stacked_rs[idx, :] = np.random.choice(neibor, 2, replace = False)
            niching = np.append(neibor.copy(), pop_choice[idx])
            stacked_best[idx] = niching[np.argmin(val[niching])]

        r1 = stacked_rs[:, 0]
        r2 = stacked_rs[:, 1]
        v = stacked_pos + self.FF * (population[stacked_best] - stacked_pos) + self.FF * (population[r1] - population[r2])
        return v

    def act3(self, pop_choice):
        population = self.individuals['current_position'].copy()
        neighbor_matrix = self.individuals['neighbor_matrix'].copy()
        stacked_pos = population[pop_choice]
        stacked_rs = np.zeros((len(pop_choice), 3), dtype = 'int')
        for idx in range(len(pop_choice)):
            neibor = np.argwhere(neighbor_matrix[pop_choice[idx]] > 0).squeeze(-1)
            stacked_rs[idx, :] = np.random.choice(neibor, 3, replace = False)

        r1 = stacked_rs[:, 0]
        r2 = stacked_rs[:, 1]
        r3 = stacked_rs[:, 2]
        v = population[r1] + self.FF * (population[r2] - population[r3])
        return v

    def act4(self, pop_choice):
        population = self.individuals['current_position'].copy()
        neighbor_matrix = self.individuals['neighbor_matrix'].copy()
        val = self.individuals['c_cost'].copy()
        stacked_pos = population[pop_choice]
        stacked_rs = np.zeros((len(pop_choice), 2), dtype = 'int')
        stacked_best = np.zeros(len(pop_choice), dtype = 'int')
        for idx in range(len(pop_choice)):
            stacked_rs[idx, :] = np.random.choice(np.delete(np.arange(self.ps), pop_choice[idx]), 2, replace = False)

        for idx in range(len(pop_choice)):
            random_best = np.random.choice(np.delete(np.arange(self.ps), pop_choice[idx]), 1, replace = False)[0]
            neibor = np.argwhere(neighbor_matrix[random_best] > 0).squeeze(-1)
            niching = np.append(neibor.copy(), random_best)
            stacked_best[idx] = niching[np.argmin(val[niching])]

        r1 = stacked_rs[:, 0]
        r2 = stacked_rs[:, 1]
        v = stacked_pos + self.FF * (population[stacked_best] - stacked_pos) + self.FF * (population[r1] - population[r2])
        return v

    def act5(self, pop_choice):
        population = self.individuals['current_position'].copy()
        stacked_pos = population[pop_choice]
        stacked_rs = np.zeros((len(pop_choice), 3), dtype = 'int')
        for idx in range(len(pop_choice)):
            stacked_rs[idx, :] = np.random.choice(np.delete(np.arange(self.ps), pop_choice[idx]), 3, replace = False)

        r1 = stacked_rs[:, 0]
        r2 = stacked_rs[:, 1]
        r3 = stacked_rs[:, 2]
        v = population[r1] + self.FF * (population[r2] - population[r3])
        return v

    def cal_pr_sr(self, problem):
        raw_PR = np.zeros(5)
        raw_SR = np.zeros(5)
        solu = self.individuals['current_position'].copy()
        accuracy = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        total_pkn = problem.nopt
        for acc_level in range(5):
            nfp, _ = problem.how_many_goptima(solu, accuracy[acc_level])
            raw_PR[acc_level] = nfp / total_pkn
            if nfp >= total_pkn:
                raw_SR[acc_level] = 1
        return raw_PR, raw_SR

    # initialize GPSO environment
    def initialize_individuals(self, problem):
        rand_pos = np.random.rand(self.ps, problem.dim) * (problem.ub - problem.lb) + problem.lb
        c_cost = self.get_costs(rand_pos, problem)
        pop_dist = distance.cdist(rand_pos, rand_pos)
        neighbor_matrix = self.find_nei(pop_dist.copy())

        # find out the gbest_val
        gbest_val = np.min(c_cost)
        gbest_position = rand_pos[np.argmin(c_cost)].copy()
        local_best = [None] * self.ps
        local_best_pos = [None] * self.ps
        for idx in range(self.ps):
            neibor = np.argwhere(neighbor_matrix[idx] > 0).squeeze(-1)
            niching = np.append(neibor.copy(), idx)
            best = niching[np.argmin(c_cost[niching])]
            sub_optimal = np.min(c_cost[niching])
            local_best[idx] = sub_optimal
            local_best_pos[idx] = rand_pos[best].copy()

        # record
        self.max_cost = np.max(c_cost)
        # store all the information of the individuals
        self.individuals = {'current_position': rand_pos.copy(),  # ps, dim
                            'c_cost': c_cost.copy(),  # ps
                            'pop_dist': pop_dist.copy(),
                            'neighbor_matrix': neighbor_matrix.copy(),
                            'gbest_position': gbest_position.copy(),  # dim
                            'gbest_val': gbest_val,  # 1
                            'no_improve': 0,
                            'lbest_position': local_best_pos.copy(),
                            'lbest_val': local_best.copy(),
                            'local_no_improve': np.zeros((self.ps,)),
                            'per_no_improve': np.zeros((self.ps,))
                            }
        self.gbest_val = self.individuals['gbest_val']

    # the interface for environment reseting
    def init_population(self, problem):
        self.max_fes = problem.maxfes
        self.dim = problem.dim
        self.max_dist = np.sqrt((problem.ub - problem.lb) ** 2 * self.dim)
        self.log_interval = (problem.maxfes // self.__config.n_logpoint)

        # maintain
        self.fes = 0
        self.log_index = 1

        # initialize the population
        self.initialize_individuals(problem)
        self.cost = [self.individuals['gbest_val']]
        raw_pr, raw_sr = self.cal_pr_sr(problem)
        self.pr = [raw_pr.copy()]
        self.sr = [raw_sr.copy()]

        # get state
        state = self.observe()  # ps, 9

        if self.__config.full_meta_data:
            self.meta_X = [self.individuals['current_position'].copy()]
            self.meta_Cost = [self.individuals['c_cost'].copy()]
            raw_pr, raw_sr = self.cal_pr_sr(problem)
            self.meta_Pr = [raw_pr.copy()]
            self.meta_Sr = [raw_sr.copy()]

        # get and return the total state (population state, exploration state, exploitation state)
        return state  # ps, 9+18

    # feature encoding
    def observe(self):
        pop = self.individuals['current_position'].copy()
        val = self.individuals['c_cost'].copy()
        all_dist = self.individuals['pop_dist'].copy()
        neighbor_matrix = self.individuals['neighbor_matrix'].copy()
        bsf_pos = self.individuals['gbest_position'].copy()
        best_so_far = self.individuals['gbest_val'].copy()
        local_no_improve = self.individuals['local_no_improve'].copy()

        max_step = self.max_fes // self.ps
        states = np.zeros((self.ps, 22))
        states[:, 0] = np.average(np.sum(all_dist, -1) / (self.ps - 1)) / self.max_dist
        states[:, 1] = np.std((val) / (self.max_cost))
        states[:, 2] = (self.max_fes - self.fes) / self.max_fes
        states[:, 3] = self.individuals['no_improve'] / max_step
        states[:, 4] = np.average(val / self.max_cost)

        sub_optimal = np.zeros(self.ps)
        for idx in range(self.ps):
            neibor = np.argwhere(neighbor_matrix[idx] > 0).squeeze(-1)
            niching = np.append(neibor.copy(), idx)
            assert len(neibor) == self.k_neighbors
            best = niching[np.argmin(val[niching])]
            niching_dist = distance.cdist(pop[niching], pop[niching])
            sub_optimal[idx] = np.min(val[niching])

            states[idx, 5] = np.average(np.sum(niching_dist) / (len(niching) - 1)) / self.max_dist
            states[idx, 6] = np.std((val[niching]) / (self.max_cost))
            states[idx, 7] = local_no_improve[idx] / max_step
            states[idx, 8] = np.average(val[niching] / self.max_cost)

            states[idx, 14] = distance.euclidean(pop[idx], pop[best]) / self.max_dist
            states[idx, 15] = (val[idx] - np.min(val[niching])) / (self.max_cost)

            states[idx, 18] = np.average((val[idx] - val[neibor]) / self.max_cost)
            in_nich_dist = distance.cdist([pop[idx]], pop[neibor])[0]
            states[idx, 19] = np.average(in_nich_dist) / self.max_dist
            states[idx, 20] = np.sum(val[idx] - val) / (self.ps - 1) / self.max_cost

        sub_rank = np.argsort(sub_optimal)
        for idx in range(self.ps):
            states[idx, 9] = np.where(sub_rank == idx)[0][0] / (self.ps - 1)
        states[:, 10] = distance.cdist([pop[np.argmin(val)]], pop)[0] / self.max_dist
        states[:, 11] = distance.cdist([bsf_pos], pop)[0] / self.max_dist
        states[:, 12] = (val - best_so_far) / (self.max_cost)
        states[:, 13] = (val - np.min(val)) / (self.max_cost)

        states[:, 16] = self.individuals['per_no_improve'] / max_step
        states[:, 17] = val / self.max_cost

        states[:, 21] = np.sum(all_dist, -1) / (self.ps - 1) / self.max_dist

        assert not (True in np.isnan(states))
        return states

    def mydbscan(self, problem):
        pop = self.individuals['current_position'].copy()
        pop = (pop - problem.lb) / (problem.ub - problem.lb)
        clustering = DBSCAN(eps = self.eps, min_samples = self.min_samples).fit(pop)
        return clustering.labels_

    # direct reward function
    def cal_reward(self, problem):
        labels = self.mydbscan(problem)
        val = self.individuals['c_cost'].copy()
        rewards = 0
        for ll in range(np.max(labels) + 1):
            now_cluster = np.where(labels == ll)[0]
            minval = np.min(val[now_cluster])
            rewards += (1 - minval / self.max_cost)
        return rewards

    def update(self, action, problem):
        is_end = False

        # record the gbest_val in the begining
        pop = self.individuals['current_position'].copy()
        val = self.individuals['c_cost'].copy()
        bprimes = np.zeros((self.ps, self.dim))
        for act in range(self.n_action):
            pop_choice = np.where(action == act)[0]
            if len(pop_choice) == 0:
                continue
            if act == 0:
                bprime = self.act1(pop_choice)
            elif act == 1:
                bprime = self.act2(pop_choice)
            elif act == 2:
                bprime = self.act3(pop_choice)
            elif act == 3:
                bprime = self.act4(pop_choice)
            elif act == 4:
                bprime = self.act5(pop_choice)
            else:
                raise ValueError('invalid action')

            bprimes[pop_choice] = bprime.copy()
        jrand = np.random.randint(self.dim, size = (self.ps))
        raw_position = np.where(np.random.rand(self.ps, self.dim) < self.CR, bprimes, pop)
        raw_position[np.arange(self.ps), jrand] = bprimes[np.arange(self.ps), jrand].copy()
        new_position = np.clip(raw_position, problem.lb, problem.ub)
        # calculate the new costs
        new_cost = self.get_costs(new_position, problem)

        per_filters = new_cost < val
        pop[per_filters] = new_position[per_filters].copy()
        val[per_filters] = new_cost[per_filters].copy()

        new_pop_dist = distance.cdist(pop, pop)
        new_neighbor_matrix = self.find_nei(new_pop_dist.copy())
        gbest_position = self.individuals['gbest_position'].copy()
        gbest_val = self.individuals['gbest_val']
        no_improve = self.individuals['no_improve']
        if np.min(val) < gbest_val:
            gbest_position = pop[np.argmin(val)].copy()
            gbest_val = np.min(val)
            no_improve = 0
        else:
            no_improve += 1
        lbest_position = self.individuals['lbest_position'].copy()
        lbest_val = self.individuals['lbest_val'].copy()
        local_no_improve = self.individuals['local_no_improve'].copy()

        for idx in range(self.ps):
            neibor = np.argwhere(new_neighbor_matrix[idx] > 0).squeeze(-1)
            niching = np.append(neibor.copy(), idx)
            best = niching[np.argmin(val[niching])]
            sub_optimal = np.min(val[niching])
            if sub_optimal < lbest_val[idx]:
                lbest_val[idx] = sub_optimal
                lbest_position[idx] = pop[best].copy()
                local_no_improve[idx] = 0
            else:
                local_no_improve[idx] += 1

        per_no_improve = self.individuals['per_no_improve'].copy()
        per_no_improve[per_filters] = 0
        per_no_improve[~per_filters] += 1

        new_individuals = {'current_position': pop.copy(),
                           'c_cost': val.copy(),
                           'pop_dist': new_pop_dist.copy(),
                           'neighbor_matrix': new_neighbor_matrix.copy(),
                           'gbest_position': gbest_position.copy(),  # dim
                           'gbest_val': gbest_val,  # 1
                           'no_improve': no_improve,
                           'lbest_position': lbest_position.copy(),
                           'lbest_val': lbest_val.copy(),
                           'local_no_improve': local_no_improve.copy(),
                           'per_no_improve': per_no_improve.copy()
                           }
        self.individuals = new_individuals
        self.gbest_val = self.individuals['gbest_val']

        if self.__config.full_meta_data:
            self.meta_X.append(self.individuals['current_position'].copy())
            self.meta_Cost.append(self.individuals['c_cost'].copy())
            raw_pr, raw_sr = self.cal_pr_sr(problem)
            self.meta_Pr.append(raw_pr.copy())
            self.meta_Sr.append(raw_sr.copy())

        # see if the end condition is satisfied
        is_end = self.fes >= self.max_fes

        # cal the reward
        reward = self.cal_reward(problem)
        reward /= self.reward_scale

        # get the population next_state
        next_state = self.observe()  # ps, 9

        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.individuals['gbest_val'])
            raw_pr, raw_sr = self.cal_pr_sr(problem)
            self.pr.append(raw_pr.copy())
            self.sr.append(raw_sr.copy())

        if is_end:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.individuals['gbest_val']
                raw_pr, raw_sr = self.cal_pr_sr(problem)
                self.pr[-1] = raw_pr.copy()
                self.sr[-1] = raw_sr.copy()
            else:
                while len(self.cost) < self.__config.n_logpoint + 1:
                    self.cost.append(self.individuals['gbest_val'])
                    raw_pr, raw_sr = self.cal_pr_sr(problem)
                    self.pr.append(raw_pr.copy())
                    self.sr.append(raw_sr.copy())

        info = {}
        return next_state, reward, is_end, info