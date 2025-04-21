import numpy as np

from ..optimizer.learnable_optimizer import Learnable_Optimizer

crossover_operators = ["CR1", "CR2", "CR3"]
mutation_operators = ['DE1', 'DE2', 'DE3', 'DE4', 'DE5', 'DE6', 'DE7', 'DE8', 'DE9', 'DE10', 'DE11', 'DE12', 'DE13', 'DE14']

class select_mutation:
    def __init__(self, rng):
        # print(mutation_operators)
        operators = {}
        self.rng = rng
        for operator_name in mutation_operators:
            operators[operator_name] = eval(operator_name)(rng)
        self.mutation_operators = mutation_operators

        self.operators = operators
        self.n_operator = len(self.mutation_operators)

    def select_mutation_operator(self, mutation_operator):
        mutation_operator_name = self.mutation_operators[mutation_operator]
        operator_class = self.operators[mutation_operator_name]
        return operator_class

class select_crossover:
    def __init__(self, rng):
        self.rng = rng
        # print(crossover_operators)
        operators = {}
        for operator_name in crossover_operators:
            operators[operator_name] = eval(operator_name)(rng)
        self.crossover_operators = crossover_operators

        self.operators = operators
        self.n_operator = len(self.crossover_operators)

    def select_crossover_operator(self, crossover_operator):
        crossover_operator_name = self.crossover_operators[crossover_operator]
        operator_class = self.operators[crossover_operator_name]
        return operator_class

class RLDEAFL_Optimizer(Learnable_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.__config = config

        self.__mu_operator = 14
        self.__cr_operator = 3

        self.__n_mutation = 3
        self.__n_crossover = 2

        self.__NP = 100
        self.__dim = config.dim
        self.max_fes = config.maxFEs
        self.__reward_ratio = 1

        self.__mu_selector = None
        self.__cr_selector = None


        self.log_index = None
        self.log_interval = config.log_interval

    def __str__(self):
        return "RLDEAFL_Optimizer"

    # calculate costs of solutions
    def get_costs(self, position, problem):
        ps = position.shape[0]
        self.fes += ps
        # return problem bound
        position = np.clip(position, 0, 1)
        position = (problem.ub - problem.lb) * position + problem.lb

        if problem.optimum is None:
            cost = problem.eval(position)
        else:
            cost = problem.eval(position) - problem.optimum
        return cost

    def observe(self):
        xs = (self.current_vector - 0) / (1 - 0)
        fes = self.fes / self.max_fes
        pop = np.column_stack((xs, self.current_fitness, np.full(xs.shape[0], fes)))
        return pop

    def init_population(self, problem):
        if self.__mu_selector is None:
            self.__mu_selector = select_mutation(self.rng)
            self.__cr_selector = select_crossover(self.rng)

        self.fes = 0
        NP = self.__NP
        dim = self.__dim
        rand_vector = self.rng.uniform(low = 0, high = 1, size = (NP, dim))
        c_cost = self.get_costs(rand_vector, problem)

        self.__archive = np.array([])
        self.current_vector = rand_vector
        self.current_fitness = c_cost

        self.gbest_val = np.min(self.current_fitness)
        self.__gbest_index = np.argmin(self.current_fitness)
        self.__gbest_vector = self.current_vector[self.__gbest_index]

        self.log_index = 1
        self.cost = [self.gbest_val]
        self.__init_gbest = self.gbest_val

        if self.__config.full_meta_data:
            self.meta_X = [self.current_vector.copy() * (problem.ub - problem.lb) + problem.lb]
            self.meta_Cost = [self.current_fitness.copy()]

        return self.observe()

    def __update_archive(self, old_id):
        if self.__archive.shape[0] < self.__NP:
            self.__archive = np.append(self.__archive, self.current_vector[old_id]).reshape(-1, self.__dim)
        else:
            self.__archive[self.rng.randint(self.__archive.shape[0])] = self.current_vector[old_id]

    def update(self, action, problem):
        _, n_action = action.shape
        mutation_operator = action[:, 0]
        crossover_operator = action[:, 1]
        mutation_parameters = action[:, 2: 2 + self.__n_mutation]
        crossover_parameters = action[:, -self.__n_crossover:]

        pre_gbest = self.gbest_val

        # classification
        mu_operators_dict = {}
        for i in range(self.__mu_operator):
            indexs = np.where(mutation_operator == i)[0]
            mu_operators_dict[i] = indexs

        cr_operators_dict = {}
        for i in range(self.__cr_operator):
            indexs = np.where(crossover_operator == i)[0]
            cr_operators_dict[i] = indexs

        # apply mutation
        origin_vector = self.current_vector
        origin_fitness = self.current_fitness
        v = np.zeros_like(origin_vector)
        for de in mu_operators_dict:
            indexs = mu_operators_dict[de]
            if indexs.shape[0] == 0:
                continue
            # parametrers = mutation_parameters[indexs]
            operator = self.__mu_selector.select_mutation_operator(int(de))
            updated_sub_vector = operator.mutation(origin_vector, origin_fitness, indexs, mutation_parameters, self.__archive)
            v[indexs] = updated_sub_vector

        # bound
        v = np.where(v < 0, (origin_vector + 0) / 2, v)
        v = np.where(v > 1, (origin_vector + 1) / 2, v)

        # apply crossover
        u = np.zeros_like(v)
        for cr in cr_operators_dict:
            indexs = cr_operators_dict[cr]
            if indexs.shape[0] == 0:
                continue
            parametrers = crossover_parameters[indexs]
            operator = self.__cr_selector.select_crossover_operator(int(cr))
            updated_sub_vector = operator.crossover(origin_vector[indexs], v[indexs],
                                                    parametrers, origin_vector, origin_fitness, self.__archive)
            u[indexs] = updated_sub_vector

        # cost
        new_cost = self.get_costs(u, problem)
        optim = np.where(new_cost < self.current_fitness)[0]
        for i in optim:
            self.__update_archive(i)

        self.current_vector[optim] = u[optim]
        self.current_fitness = np.minimum(self.current_fitness, new_cost)

        self.gbest_val = np.min(self.current_fitness)
        self.__gbest_index = np.argmin(self.current_fitness)
        self.__gbest_vector = self.current_vector[self.__gbest_index]


        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.gbest_val)

        if self.__config.full_meta_data:
            self.meta_X.append(self.current_vector.copy() * (problem.ub - problem.lb) + problem.lb)
            self.meta_Cost.append(self.current_fitness.copy())

        if problem.optimum is None:
            is_done = self.fes >= self.max_fes
        else:
            is_done = self.fes >= self.max_fes

        reward = self.__reward_ratio * (pre_gbest - self.gbest_val) / self.__init_gbest

        if is_done:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.gbest_val
            else:
                while len(self.cost) < self.__config.n_logpoint + 1:
                    self.cost.append(self.gbest_val)
        return self.observe(), reward, is_done, {}


# [best/1] [best/2] [rand/1] [rand/2] [current-to-best/1] [rand-to-best/1] [current-to-rand/1] [current-to-pbest/1] [ProDE-rand/1]
# [TopoDE-rand/1] [current-to-pbest/1+archive] [HARDDE-current-to-pbest/2] [current-to-rand/1+archive] [weighted-rand-to-pbest/1]
class Basic_mutation:
    """
    This class represents a basic mutation.
    Methods:
    - get_parameters_numbers: Returns the number of parameters.
    - mutation: Performs the mutation.
    """

    # individual version
    # def mutation(self,env,individual_indice):
    #     """
    #     Perform mutation on the given individual.
    #     Parameters:
    #     - env: The environment object.
    #     - individual_indice: The index of the individual to mutate.
    #     Returns:
    #     - None
    #     """

    #     pass

    # population version
    def __init__(self, rng):
        self.rng = rng
    def mutation(self, group, cost, indexs, parameters, archive):
        pass

    def construct_random_indices(self, pop_size, indexs, x_num):
        indices = np.arange(pop_size)
        if x_num == 1:
            Indices = np.zeros(len(indexs), dtype = int)
        else:
            Indices = np.zeros((len(indexs), x_num), dtype = int)
        for i, index in enumerate(indexs):
            temp_indices = indices[indices != index]
            Indices[i] = self.rng.choice(temp_indices, x_num, replace = False)
        return Indices

    def construct_extra_random_indices(self, pop_size, indexs, x_num, extra):
        indices = np.arange(pop_size)
        if x_num == 1:
            Indices = np.zeros(len(indexs), dtype = int)
        else:
            Indices = np.zeros((len(indexs), x_num), dtype = int)
        extra_n = extra.shape[1]
        for i, index in enumerate(indexs):
            filters = indices != index
            for j in range(extra_n):
                filters = filters & (indices != extra[i, j])
            temp_indices = indices[filters]
            Indices[i] = self.rng.choice(temp_indices, x_num, replace = False)
        return Indices

    def construct_pbest(self, group, p):
        p = np.mean(p)
        pbest = group[:max(int(p * group.shape[0]), 2)]
        return pbest

# [binomial] [exponential] [p-binomial]
class Basic_crossover:
    def __init__(self, rng):
        self.rng = rng
    def crossover(self, x, v, parameters, group, cost, archive):

        pass

class DE1(Basic_mutation):

    def __init__(self, rng):
        super().__init__(rng)
    def mutation(self, group, cost, indexs, parameters, archive):
        F = parameters[indexs][:, 0]
        F = F[:, np.newaxis]
        best_index = np.argmin(cost)
        best_vector = group[best_index]
        random_indices = self.construct_random_indices(group.shape[0], indexs, 2)
        x1, x2 = group[random_indices.T]
        mutated_vector = best_vector + F * (x1 - x2)
        return mutated_vector

class DE2(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    def mutation(self, group, cost, indexs, parameters, archive):
        F = parameters[indexs][:, 0]
        F = F[:, np.newaxis]
        best_index = np.argmin(cost)
        best_vector = group[best_index]
        random_indices = self.construct_random_indices(group.shape[0], indexs, 4)
        x1, x2, x3, x4 = group[random_indices.T]
        mutated_vector = best_vector + F * (x1 - x2) + F * (x3 - x4)
        return mutated_vector

class DE3(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    def mutation(self, group, cost, indexs, parameters, archive):
        F = parameters[indexs][:, 0]
        F = F[:, np.newaxis]
        random_indices = self.construct_random_indices(group.shape[0], indexs, 3)
        x1, x2, x3 = group[random_indices.T]
        mutated_vector = x1 + F * (x2 - x3)
        return mutated_vector

class DE4(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    def mutation(self, group, cost, indexs, parameters, archive):
        F = parameters[indexs][:, 0]
        F = F[:, np.newaxis]
        random_indices = self.construct_random_indices(group.shape[0], indexs, 5)
        x1, x2, x3, x4, x5 = group[random_indices.T]
        mutated_vector = x1 + F * (x2 - x3) + F * (x4 - x5)
        return mutated_vector

class DE5(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    def mutation(self, group, cost, indexs, parameters, archive):
        F = parameters[indexs][:, 0]
        F = F[:, np.newaxis]
        best_index = np.argmin(cost)
        best_vector = group[best_index]
        current_vector = group[indexs]
        random_indices = self.construct_random_indices(group.shape[0], indexs, 2)
        x1, x2 = group[random_indices.T]
        mutated_vector = current_vector + F * (best_vector - current_vector) + F * (x1 - x2)
        return mutated_vector

class DE6(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    def mutation(self, group, cost, indexs, parameters, archive):
        F = parameters[indexs][:, 0]
        F = F[:, np.newaxis]
        best_index = np.argmin(cost)
        best_vector = group[best_index]
        random_indices = self.construct_random_indices(group.shape[0], indexs, 4)
        x1, x2, x3, x4 = group[random_indices.T]
        mutated_vector = x1 + F * (best_vector - x2) + F * (x3 - x4)
        return mutated_vector

class DE7(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    def mutation(self, group, cost, indexs, parameters, archive):
        F = parameters[indexs][:, 0]
        F = F[:, np.newaxis]
        current_vector = group[indexs]
        random_indices = self.construct_random_indices(group.shape[0], indexs, 3)
        x1, x2, x3 = group[random_indices.T]
        mutated_vector = current_vector + F * (x1 - current_vector) + F * (x2 - x3)
        return mutated_vector

class DE8(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    # current-to-pbest/1
    def mutation(self, group, cost, indexs, parameters, archive):
        # Sort
        ind = np.argsort(cost)
        temp_group = group[ind]

        ind_2 = np.argsort(ind)
        temp_indexs = ind_2[indexs]
        temp_parameters = parameters[temp_indexs]
        F = temp_parameters[:, 0]
        F = F[:, np.newaxis]
        current_vector = temp_group[temp_indexs]

        pbest = self.construct_pbest(temp_group, temp_parameters[:, 1])
        NB = pbest.shape[0]
        rb = self.rng.randint(NB, size = len(indexs))
        random_indices = self.construct_extra_random_indices(temp_group.shape[0], temp_indexs, 2, extra = rb[:, None])
        x1, x2 = temp_group[random_indices.T]
        mutated_vector = current_vector + F * (pbest[rb] - current_vector) + F * (x1 - x2)
        return mutated_vector

class DE9(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    def EuclideanDistance(self, x, y):
        return np.sqrt(np.sum(np.square(x - y)))

    def cal_R_d(self, group):
        pop_size = group.shape[0]
        R_d = np.zeros((pop_size, pop_size))
        for i in range(pop_size):
            for j in range(i + 1, pop_size):
                R_d[i, j] = self.EuclideanDistance(group[i], group[j])
                R_d[j, i] = R_d[i, j]
        return R_d

    def construct_r(self, group, indexs):
        R_d = self.cal_R_d(group)
        Sum = np.sum(R_d, axis = 0) # 1D NP
        Sum = np.where(Sum == 0, 1, Sum)
        R_p = 1 - (R_d / Sum) # NP * NP

        p = np.sum(R_p, axis = 1)
        Indices = np.zeros((len(indexs), 3), dtype = int)
        for i, index in enumerate(indexs):
            temp_p = p
            temp_p[index] = 0
            if np.sum(temp_p) == 0:
                temp_p = np.ones_like(temp_p)
                temp_p[index] = 0
            temp_p = temp_p / np.sum(temp_p)
            Indices[i] = self.rng.choice(len(temp_p), size = 3, p = temp_p, replace = False)
        return Indices

    def mutation(self, group, cost, indexs, parameters, archive):
        F = parameters[indexs][:, 0]
        F = F[:, np.newaxis]
        random_indices = self.construct_r(group, indexs)
        x1, x2, x3 = group[random_indices.T]
        matuated_vector = x1 + F * (x2 - x3)
        return matuated_vector

class DE10(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    def Topograph(self, group, cost):
        """
        generate a kNN matrix indicating the nearest neighbors of each individual
        """
        current_vector = group
        pop_size, dim = current_vector.shape
        # generate N*N distance matrix
        distance_matrix = np.zeros((pop_size, pop_size))
        for i in range(pop_size):
            for j in range(i + 1, pop_size):
                distance_matrix[i, j] = np.linalg.norm(current_vector[i] - current_vector[j])
                distance_matrix[j, i] = distance_matrix[i, j]
            distance_matrix[i, i] = np.inf
        # generate kNN matrix
        k = pop_size // 10  # number of nearest neighbors
        kNN_matrix = np.zeros((pop_size, k))
        for i in range(pop_size):
            kNN_matrix[i] = np.argsort(distance_matrix[i])[:k]
        for i in range(pop_size):
            for j in range(k):
                if cost[i] < cost[int(kNN_matrix[i, j])]:
                    kNN_matrix[i, j] = kNN_matrix[i, j]
                else:
                    kNN_matrix[i, j] = -kNN_matrix[i, j]
        return kNN_matrix

    def mutation(self, group, cost, indexs, parameters, archive):
        F = parameters[indexs][:, 0]
        F = F[:, np.newaxis]
        random_indices = self.construct_random_indices(group.shape[0], indexs, 2)
        x2, x3 = group[random_indices.T]
        knn_matrix = self.Topograph(group, cost)
        flag = np.zeros(knn_matrix.shape[0], dtype = bool)
        negative_indices = knn_matrix < 0
        purpose = np.arange(knn_matrix.shape[0])

        for i in range(group.shape[0]):
            if np.any(negative_indices[i]):
                flag[i] = True
                valid_indices = np.where(negative_indices[i])[0]
                fitness_values = cost[knn_matrix[i, valid_indices].astype(int)]
                purpose[i] = valid_indices[np.argmin(fitness_values)]

        purpose[~flag] = np.arange(knn_matrix.shape[0])[~flag]

        topu = group[purpose]
        x1 = topu[indexs]

        mutated_vector = x1 + F * (x2 - x3)
        return mutated_vector

class DE11(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    # current-to-pbest/1+archive
    def mutation(self, group, cost, indexs, parameters, archive):
        # Sort
        ind = np.argsort(cost)
        temp_group = group[ind]

        ind_2 = np.argsort(ind)
        temp_indexs = ind_2[indexs]
        temp_parameters = parameters[temp_indexs]
        F = temp_parameters[:, 0]
        F = F[:, np.newaxis]
        current_vector = temp_group[temp_indexs]

        pbest = self.construct_pbest(temp_group, temp_parameters[:, 1])
        NB = pbest.shape[0]
        NA = archive.shape[0]
        rb = self.rng.randint(NB, size = len(temp_indexs))
        r1 = self.construct_extra_random_indices(temp_group.shape[0], temp_indexs, 1, extra = rb[:, None])
        r2 = self.construct_extra_random_indices(temp_group.shape[0] + NA, temp_indexs, 1, extra = np.concatenate((rb[:, None], r1[:,None]), 1))

        xb = pbest[rb]
        x1 = temp_group[r1]
        if NA > 0:
            x2 = np.concatenate((temp_group, archive), 0)[r2]
        else:
            x2 = temp_group[r2]
        mutated_vector = current_vector + F * (xb - current_vector) + F * (x1 - x2)
        return mutated_vector

class DE12(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    def mutation(self, group, cost, indexs, parameters, archive):
        # Sort
        ind = np.argsort(cost)
        temp_group = group[ind]

        ind_2 = np.argsort(ind)
        temp_indexs = ind_2[indexs]
        temp_parameters = parameters[temp_indexs]
        F = temp_parameters[:, 0]
        F = F[:, np.newaxis]

        Fa = temp_parameters[:, 2]
        Fa = Fa[:, np.newaxis]
        current_vector = temp_group[temp_indexs]

        pbest = self.construct_pbest(temp_group, temp_parameters[:, 1])
        NB = pbest.shape[0]
        NA = archive.shape[0]
        rb = self.rng.randint(NB, size = len(temp_indexs))
        r1 = self.construct_extra_random_indices(temp_group.shape[0], temp_indexs, 1, extra = rb[:, None])
        r2 = self.construct_extra_random_indices(temp_group.shape[0] + NA, temp_indexs, 1, extra = np.concatenate((rb[:, None], r1[:,None]), 1))
        r3 = self.construct_extra_random_indices(temp_group.shape[0] + NA, temp_indexs, 1, extra = np.concatenate((rb[:, None], r1[:,None], r2[:, None]), 1))
        xb = pbest[rb]
        x1 = temp_group[r1]
        if NA > 0:
            x2 = np.concatenate((temp_group, archive), 0)[r2]
            x3 = np.concatenate((temp_group, archive), 0)[r3]
        else:
            x2 = temp_group[r2]
            x3 = temp_group[r3]
        mutated_vector = current_vector + F * (xb - current_vector) + Fa * (x1 - x2) + Fa * (x1 - x3)
        return mutated_vector

class DE13(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    def mutation(self, group, cost, indexs, parameters, archive):
        F = parameters[indexs][:, 0]
        F = F[:, np.newaxis]
        NA = archive.shape[0]
        current_vector = group[indexs]
        r1 = self.construct_random_indices(group.shape[0], indexs, 1)
        r2 = self.construct_extra_random_indices(group.shape[0] + NA, indexs, 1, extra = r1[:, None])

        x1 = group[r1]
        if NA > 0:
            x2 = np.concatenate((group, archive), 0)[r2]
        else:
            x2 = group[r2]

        mutated_vector = current_vector + F * (x1 - x2)
        return mutated_vector

class DE14(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    def mutation(self, group, cost, indexs, parameters, archive):
        # Sort
        ind = np.argsort(cost)
        temp_group = group[ind]

        ind_2 = np.argsort(ind)
        temp_indexs = ind_2[indexs]
        temp_parameters = parameters[temp_indexs]
        F = temp_parameters[:, 0]
        F = F[:, np.newaxis]

        pbest = self.construct_pbest(temp_group, temp_parameters[:, 1])
        NB = pbest.shape[0]

        Fa = temp_parameters[:, 2]
        Fa = Fa[:, np.newaxis]
        rb = self.rng.randint(NB, size = len(temp_indexs))
        random_indices = self.construct_extra_random_indices(temp_group.shape[0], temp_indexs, 2, extra = rb[:, None])

        xb = pbest[rb]
        x1, x2 = temp_group[random_indices.T]
        mutated_vector = F * x1 + F * Fa * (xb - x2)
        return mutated_vector

class CR1(Basic_crossover):
    def __init__(self, rng):
        super().__init__(rng)
    def crossover(self, x, v, parameters, group, cost, archive):
        CR = parameters[:, 0]
        NP, dim = x.shape
        jrand = self.rng.randint(dim, size = NP)
        CRs = np.repeat(CR, dim).reshape(NP, dim)
        u = np.where(self.rng.rand(NP, dim) < CRs, v, x)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        return u

class CR2(Basic_crossover):
    def __init__(self, rng):
        super().__init__(rng)
    def crossover(self, x, v, parameters, group, cost, archive):
        CR = parameters[:, 0]
        NP, dim = x.shape
        u = x.copy()
        L = self.rng.randint(dim, size = NP).repeat(dim).reshape(NP, dim)
        L = L <= np.arange(dim)
        rvs = self.rng.rand(NP, dim)
        CRs = np.repeat(CR, dim).reshape(NP, dim)
        L = np.where(rvs > CRs, L, 0)
        u = u * (1 - L) + v * L
        return u

class CR3(Basic_crossover):
    def __init__(self, rng):
        super().__init__(rng)
    def crossover(self, x, v, parameters, group, cost, archive):
        CR = parameters[:, 0]
        p = parameters[:, 1]

        p = np.mean(p)
        ind = np.argsort(cost)
        temp_group = group[ind]
        pbest = temp_group[:max(int(p * group.shape[0]), 2)]
        if archive.shape[0] > 0:
            pbest = np.concatenate((temp_group, archive), 0)[:max(int(p * (group.shape[0] + archive.shape[0])), 2)]

        NP, dim = x.shape
        cross_pbest = pbest[self.rng.randint(pbest.shape[0], size = NP)]
        jrand = self.rng.randint(dim, size = NP)
        CRs = np.repeat(CR, dim).reshape(NP, dim)
        u = np.where(self.rng.rand(NP, dim) < CRs, v, cross_pbest)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        return u
