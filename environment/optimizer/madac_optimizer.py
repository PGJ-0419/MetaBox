import copy
import functools
import numpy as np
import math
import sys
import tianshou

from operator import itemgetter
from scipy.spatial.distance import cdist
from ..optimizer.learnable_optimizer import Learnable_Optimizer
EPSILON = sys.float_info.epsilon


POSITIVE_INFINITY = float("inf")
EPSILON = sys.float_info.epsilon


class PlatypusError(Exception):
    pass


class Indicator(object):

    def __init__(self):
        super(Indicator, self).__init__()

    def __call__(self, set):
        return self.calculate(set)

    def calculate(self, set):
        raise NotImplementedError("method not implemented")


class Hypervolume(Indicator):
    # 只适用于最小化问题

    def __init__(self, reference_set=None, minimum=None, maximum=None):
        super(Hypervolume, self).__init__()
        if reference_set is not None:
            if minimum is not None or maximum is not None:
                raise ValueError("minimum and maximum must not be specified if reference_set is defined")
            self.minimum, self.maximum = normalize(reference_set)
        else:
            if minimum is None or maximum is None:
                raise ValueError("minimum and maximum must be specified when no reference_set is defined")
            self.minimum, self.maximum = minimum, maximum

    def invert(self, solution_normalized_obj: np.ndarray):
        for i in range(solution_normalized_obj.shape[1]):
            solution_normalized_obj[:, i] = 1.0 - np.clip(solution_normalized_obj[:, i], 0.0, 1.0)
        return solution_normalized_obj

    def dominates(self, solution1_obj, solution2_obj, nobjs):
        better = False
        worse = False

        for i in range(nobjs):
            if solution1_obj[i] > solution2_obj[i]:
                better = True
            else:
                worse = True
                break
        return not worse and better

    def swap(self, solutions_obj, i, j):
        solutions_obj[[i, j]] = solutions_obj[[j, i]]
        return solutions_obj

    def filter_nondominated(self, solutions_obj, nsols, nobjs):
        i = 0
        n = nsols
        while i < n:
            j = i + 1
            while j < n:
                if self.dominates(solutions_obj[i], solutions_obj[j], nobjs):
                    n -= 1
                    solutions_obj = self.swap(solutions_obj, j, n)
                elif self.dominates(solutions_obj[j], solutions_obj[i], nobjs):
                    n -= 1
                    solutions_obj = self.swap(solutions_obj, i, n)
                    i -= 1
                    break
                else:
                    j += 1
            i += 1
        return n

    def surface_unchanged_to(self, solutions_normalized_obj, nsols, obj):
        return np.min(solutions_normalized_obj[:nsols, obj])

    def reduce_set(self, solutions, nsols, obj, threshold):
        i = 0
        n = nsols
        while i < n:
            if solutions[i, obj] <= threshold:
                n -= 1
                solutions = self.swap(solutions, i, n)
            else:
                i += 1
        return n

    def calc_internal(self, solutions_obj: np.ndarray, nsols, nobjs):
        volume = 0.0
        distance = 0.0
        n = nsols

        while n > 0:
            nnondom = self.filter_nondominated(solutions_obj, n, nobjs - 1)

            if nobjs < 3:
                temp_volume = solutions_obj[0][0]
            else:
                temp_volume = self.calc_internal(solutions_obj, nnondom, nobjs - 1)

            temp_distance = self.surface_unchanged_to(solutions_obj, n, nobjs - 1)
            volume += temp_volume * (temp_distance - distance)
            distance = temp_distance
            n = self.reduce_set(solutions_obj, n, nobjs - 1, distance)

        return volume

    def calculate(self, solutions_obj: np.ndarray):

        # 对可行解进行归一化
        solutions_normalized_obj = normalize(solutions_obj, self.minimum, self.maximum)

        # 筛选出所有目标值都小于等于 1.0 的解
        valid_mask = np.all(solutions_normalized_obj <= 1.0, axis=1)
        valid_feasible = solutions_normalized_obj[valid_mask]

        if valid_feasible.size == 0:
            return 0.0

        # 对可行解进行反转操作
        inverted_feasible = self.invert(valid_feasible)

        # 计算超体积
        nobjs = inverted_feasible.shape[1]
        return self.calc_internal(inverted_feasible, len(inverted_feasible), nobjs)


class InvertedGenerationalDistance(Indicator):
    def __init__(self, reference_set, d=1.0):
        super(InvertedGenerationalDistance, self).__init__()
        self.reference_set = reference_set
        self.d = d

    def calculate(self, set):
        return math.pow(sum([math.pow(distance_to_nearest(s, set), self.d) for s in self.reference_set]),
                        1.0 / self.d) / len(self.reference_set)


def distance_to_nearest(solution_obj, set):
    if len(set) == 0:
        return POSITIVE_INFINITY

    return min([euclidean_dist(solution_obj, s) for s in set])


def euclidean_dist(x, y):
    return math.sqrt(sum([math.pow(x[i] - y[i], 2.0) for i in range(len(x))]))



def normalize(solutions_obj: np.ndarray, minimum: np.ndarray = None, maximum: np.ndarray = None) -> np.ndarray:
    """Normalizes the solution objectives.

    Normalizes the objectives of each solution within the minimum and maximum
    bounds.  If the minimum and maximum bounds are not provided, then the
    bounds are computed based on the bounds of the solutions.

    Parameters
    ----------
    solutions_obj : numpy.ndarray
        The solutions to be normalized. It should be a 2D numpy array.
    minimum : numpy.ndarray
        The minimum values used to normalize the objectives.
    maximum : numpy.ndarray
        The maximum values used to normalize the objectives.

    Returns
    -------
    numpy.ndarray
        The normalized solutions.
    """
    # 如果输入数组为空，直接返回空数组
    if len(solutions_obj) == 0:
        return solutions_obj

    # 获取目标的数量
    n_obj = solutions_obj.shape[1]

    # 如果 minimum 或 maximum 未提供，则计算它们
    if minimum is None or maximum is None:
        if minimum is None:
            minimum = np.min(solutions_obj, axis=0)
        if maximum is None:
            maximum = np.max(solutions_obj, axis=0)

    # 检查是否有目标的范围为空
    if np.any(maximum - minimum < EPSILON):
        raise ValueError("objective with empty range")

    # 进行归一化操作
    solutions_normalized_obj = (solutions_obj - minimum) / (maximum - minimum)

    return solutions_normalized_obj
    


class Operators:
    def __init__(self, rng):
        self.rng = rng

    def DE1(self, problem, parents, step_size=0.5, crossover_rate=1.0):
        """arity = 3"""
        result = copy.deepcopy(parents[0])
        jrand = self.rng.integers(problem.n_var)

        for j in range(problem.n_var):
            if self.rng.uniform() <= crossover_rate or j == jrand:
                y = parents[0][j] + step_size * (parents[1][j] - parents[2][j])
                y = np.clip(y, problem.lb[j], problem.ub[j])
                result[j] = y
        return np.array([result])
    DE1.arity = 3

    def DE2(self, problem, parents, step_size=0.5, crossover_rate=1.0):
        """arity = 5"""
        result = copy.deepcopy(parents[0])
        jrand = self.rng.integers(problem.n_var)

        for j in range(problem.n_var):
            if self.rng.uniform() <= crossover_rate or j == jrand:
                y = parents[0][j] + step_size * (parents[1][j] - parents[2][j]) + step_size * (parents[3][j] - parents[4][j])
                y = np.clip(y, problem.lb[j], problem.ub[j])
                result[j] = y
        return np.array([result])
    DE2.arity = 5

    def DE3(self, problem, parents, step_size=0.5, crossover_rate=1.0):
        """arity = 6"""
        result = copy.deepcopy(parents[0])
        jrand = self.rng.integers(problem.n_var)

        for j in range(problem.n_var):
            if self.rng.uniform() <= crossover_rate or j == jrand:
                y = parents[0][j] + step_size * (parents[0][j] - parents[1][j]) \
                    + step_size * (parents[2][j] - parents[3][j]) \
                    + step_size * (parents[4][j] - parents[5][j])
                y = np.clip(y, problem.lb[j], problem.ub[j])
                result[j] = y
        return np.array([result])
    DE3.arity = 6

    def DE4(self, problem, parents, step_size=0.5, crossover_rate=1.0):
        """arity = 4"""
        result = copy.deepcopy(parents[0])
        jrand = self.rng.integers(problem.n_var)

        for j in range(problem.n_var):
            if self.rng.uniform() <= crossover_rate or j == jrand:
                y = parents[0][j] + step_size * (parents[0][j] - parents[1][j]) + step_size * (parents[2][j] - parents[3][j])
                y = np.clip(y, problem.lb[j], problem.ub[j])
                result[j] = y
        return np.array([result])
    DE4.arity = 4
    


def chebyshev(solution_obj, ideal_point, weights, min_weight=0.0001):
    """Chebyshev (Tchebycheff) fitness of a solution with multiple objectives.

    This function is designed to only work with minimized objectives.

    Parameters
    ----------
    solution : Solution
        The solution.
    ideal_point : list of float
        The ideal point.
    weights : list of float
        The weights.
    min_weight : float
        The minimum weight allowed.
    """
    objs = solution_obj
    n_obj = objs.shape[-1]
    return max([max(weights[i], min_weight) * (objs[i] - ideal_point[i]) for i in range(n_obj)])


class MADAC_Optimizer(Learnable_Optimizer):
    def __init__(self, config):
        super.__init__()
        self.__config = config
        # Problem Related
        self.n_ref_points = 1000
        self.episode_limit = 100
        # # MDP Related
        self.reward_type = 0
        self.n_agents = 4
        self.early_stop = False
        # # MOEA/D Algorithm Related
        self.moead_neighborhood_maxsize = 30
        self.moead_delta = 0.8
        self.moead_eta = 2
        self.adaptive_open = True
        self.max_fes=config.maxFEs
        self.operators = Operators(self.rng)

    def init_population(self, problem):
        self.problem = problem
        self.n_obj = problem.n_obj
        self.n_var = problem.n_var
        self.weights = self.get_weights(self.n_obj)
        self.neighborhoods = self.get_neighborhoods()
        self.population_size = len(self.weights)
        self.population = self.rng.uniform(low=problem.lb, high=problem.ub, size=(self.population_size, problem.n_var))
        self.population_obj = problem.eval(self.population)
        self.fes = len(self.population)
        self.archive_maximum = np.max(self.population_obj, axis=0)
        self.archive_minimum = np.min(self.population_obj, axis=0)
        self.ideal_point = copy.deepcopy(self.archive_minimum)

        self._init_adaptive_weights()
        self._init_static()
        self.problem_ref_points = self.problem.get_ref_set(
            n_ref_points=self.n_ref_points)
        self.igd_calculator = InvertedGenerationalDistance(reference_set=self.problem_ref_points)
        self.inital_value = self.get_igd()
        self.best_value = self.inital_value
        self.last_value = self.inital_value
        return self.get_state()

    def get_neighborhoods(self):
        neighborhoods = []  # the i-th element save the index of the neighborhoods of it
        for i in range(len(self.weights)):
            sorted_weights = self.moead_sort_weights(
                self.weights[i], self.weights)
            neighborhoods.append(
                sorted_weights[:self.moead_neighborhood_maxsize])
        return neighborhoods

    def get_weights(self, n_obj):
        weights = None
        if n_obj == 3:
            weights = normal_boundary_weights(n_obj, 13, 0)
        elif n_obj == 6:
            weights = normal_boundary_weights(n_obj, 4, 1)
        elif n_obj == 8:
            weights = normal_boundary_weights(n_obj, 3, 2)
        else:
            weights = normal_boundary_weights(n_obj, 2, 3)
        return weights

    def moead_update_ideal(self, solution_obj):
        for i in range(solution_obj.shape[-1]):
            self.ideal_point[i] = min(
                self.ideal_point[i], solution_obj[i])

    def _init_adaptive_weights(self):
        self.EP = []
        self.EP_obj = []
        self.EP_MaxSize = int(self.population_size * 1.5)
        self.rate_update_weight = 0.05  # rate_update_weight * N = nus
        self.nus = int(
            self.rate_update_weight * self.population_size)  # maximal number of subproblems needed to be adjusted
        # adaptive iteration interval, Units are population
        self.wag = int(self.episode_limit * 0.05)
        self.adaptive_cooling_time = self.wag
        self.adaptive_end = int(self.episode_limit * self.population_size * 0.9)

    def _init_static(self):
        self.nfe = 0  # Discarded
        self.moead_generation = 0  # Discarded
        self.best_value = 1e6
        self.last_value = 1e6
        self.inital_value = None
        self.last_bonus = 0
        # The number of iterations without promotion, maximum 10 is stag_count_max
        self.stag_count = 0
        self.stag_count_max = self.episode_limit / 10

        self.hv_his = []
        self.hv_last5 = tianshou.utils.MovAvg(size=5)

        self.nds_ratio_his = []
        self.nds_ratio_last5 = tianshou.utils.MovAvg(size=5)

        self.ava_dist_his = []
        self.ava_dist_last5 = tianshou.utils.MovAvg(size=5)

        self.igd_his = []
        self.igd_last5 = tianshou.utils.MovAvg(size=5)

        self.value_his = []

        self.hv_running = tianshou.utils.RunningMeanStd()
        self.nds_ratio_running = tianshou.utils.RunningMeanStd()
        self.ava_dist_running = tianshou.utils.RunningMeanStd()
        self.igd_running = tianshou.utils.RunningMeanStd()

        self.info_reward_his = []
        self.info_obs_his = []
        self.info_igd_his = []

    def get_state(self):
        obs_ = np.zeros(22)
        obs_[0] = 1 / self.problem.n_obj
        obs_[1] = 1 / self.problem.n_var
        obs_[2] = (self.moead_generation) / self.episode_limit
        obs_[3] = self.stag_count / self.stag_count_max
        obs_[4] = self.get_hypervolume()
        obs_[5] = self.get_ratio_nondom_sol()
        obs_[6] = self.get_average_dist()
        obs_[7] = self.get_pre_k_change(1, self.hv_his)
        obs_[8] = self.get_pre_k_change(1, self.nds_ratio_his)
        obs_[9] = self.get_pre_k_change(1, self.ava_dist_his)
        obs_[10] = self.hv_last5.mean()
        obs_[11] = self.nds_ratio_last5.mean()
        obs_[12] = self.ava_dist_last5.mean()
        obs_[13] = self.hv_last5.std()
        obs_[14] = self.nds_ratio_last5.std()
        obs_[15] = self.ava_dist_last5.std()
        obs_[16] = self.hv_running.mean
        obs_[17] = self.nds_ratio_running.mean
        obs_[18] = self.ava_dist_running.mean
        obs_[19] = self.hv_running.var
        obs_[20] = self.nds_ratio_running.var
        obs_[21] = self.ava_dist_running.var
        return [obs_] * self.n_agents

    def get_action(self, action_idx, action):
        neighborsize_agent = [15, 20, 25, 30]
        os_agent = ['DE1', 'DE2', 'DE3', 'DE4']
        pc_agent = [0.4, 0.5, 0.6, 0.7]
        weight_agent = [0, 1]
        if action_idx == 0:
            return neighborsize_agent[action]
        if action_idx == 1:
            return os_agent[action]
        if action_idx == 2:
            return pc_agent[action]
        if action_idx == 3:
            return weight_agent[action]

    def update(self, action, problem):
        """
        one step update in moea/d
        inclue solution generation and solution selection
        @param action: neighboor size; operator type; operator parameter
        :return:
        """
        self.moead_neighborhood_size = self.get_action(0, action[0])
        self.os = self.get_action(1, action[1])
        self.pc = self.get_action(2, action[2])
        self.weight_adjust = self.get_action(3, action[3])

        if self.adaptive_open is False:
            action[3] = 0
        self.variator = self.operators.eval(f"{self.os}")(step_size = self.pc)
        subproblems = self.moead_get_subproblems()
        self.offspring_list = []
        self.offspring_obj_list = []
        for index in subproblems:
            mating_indices = self.moead_get_mating_indices(index)
            mating_population = [self.population[i] for i in mating_indices]
            if index in mating_indices:
                mating_indices.remove(index)

            parents = [self.population[index]] + \
                      [self.population[i] for i in
                       self.rng.choice(mating_indices, self.variator.arity - 1, replace=False)]
            offspring = self.variator.evolve(problem, parents)
            offspring_obj = problem.eval(offspring)
            self.fes += len(offspring)
            self.offspring_list.extend(offspring)
            self.offspring_obj_list.extend(offspring_obj)
            for child, child_obj in zip(offspring, offspring_obj):
                self.moead_update_ideal(child_obj)
                self.moead_update_solution(child, child_obj, mating_indices)  # selection
        
        if self.adaptive_open:
            self.update_ep()
        if action[3] > 1:
            raise Exception("action[3] > 1.")
        if action[3] == 1 and self.adaptive_cooling_time <= 0:
            self.adaptive_cooling_time = self.wag
            self.update_weight()
        self.adaptive_cooling_time -= 1
        self.moead_generation += 1
        value = self.get_igd()
        reward = self.get_reward(value)
        self.update_igd(value)
        self.obs = self.get_state()
        # if stop, then return the information
        if self.fex >= self.max_fes:
            self.done = True
            print("best_igd:{}".format(self.best_value))
        else:
            self.done = False
        
        info = {"best_igd": self.best_value, "last_igd": self.last_value}
        print(
            "generation:{},reward:{},best_igd{},last_igd{}".format(self.moead_generation, reward, self.best_value, self.last_value))
        return self.obs, [reward] * self.n_agents, self.done, info

    def update_information(self):
        index =  self.find_non_dominated_indices(self.population_obj)
        self.cost = [self.population_obj[i] for i in index] # parato front
        self.metadata = {
            "cost": self.population,
            "cost_obj": self.population_obj,
        }
        
    def find_non_dominated_indices(self, population_list):
        """
        此函数用于找出种群中的支配解
        :param population_list: 种群的目标值的列表，列表中的每个元素是一个代表单个解目标值的列表
        :return: 支配解的列表
        """
        # 将列表转换为 numpy 数组
        population = np.array(population_list)
        n_solutions = population.shape[0]
        is_dominated = np.zeros(n_solutions, dtype=bool)

        for i in range(n_solutions):
            for j in range(n_solutions):
                if i != j:
                    # 检查是否存在解 j 支配解 i
                    if np.all(population[j] <= population[i]) and np.any(population[j] < population[i]):
                        is_dominated[i] = True
                        break

        # 找出非支配解的索引
        non_dominated_indices = np.where(~is_dominated)[0]
        return non_dominated_indices

    def update_ep(self):
        """Update the current evolutional population EP
        """
        self.EP.extend(self.offspring_list)
        self.EP_obj.extend(self.offspring_obj_list)

        indices = self.find_non_dominated_indices(self.EP_obj)
        self.EP = [self.EP[i] for i in indices]
        self.EP_obj = [self.EP_obj[i] for i in indices]

        l = len(self.EP_obj)
        if l <= self.EP_MaxSize:
            return
        # Delete the overcrowded solutions in EP
        dist = cdist(
            [self.EP_obj[i] for i in range(l)],
            [self.EP_obj[i] for i in range(l)]
        )
        for i in range(l):
            dist[i][i] = np.inf
        dist.sort(axis=1)
        # find max self.EP_MaxSize item
        sub_dist = np.prod(dist[:, 0:self.n_obj], axis=1)
        idx = np.argpartition(sub_dist, - self.EP_MaxSize)[-self.EP_MaxSize:]
        self.EP = list((itemgetter(*idx)(self.EP)))
        self.EP_obj = list((itemgetter(*idx)(self.EP_obj)))

    def update_weight(self):
        # Delete the overcrowded subproblems

        # Delete the overcrowded subproblems
        l_ep = len(self.EP)
        nus = min(l_ep, self.nus)
        dist = cdist(
            [self.population_obj[i] for i in range(
                self.population_size)],
            [self.population_obj[i] for i in range(
                self.population_size)]
        )
        for i in range(self.population_size):
            dist[i][i] = np.inf
        dist.sort(axis=1)
        sub_dist = np.prod(dist[:, 0:self.n_obj], axis=1)
        idx = np.argpartition(
            sub_dist, -(self.population_size - nus))[-(self.population_size - nus):]
        self.population = list((itemgetter(*idx)(self.population)))
        self.population_obj = list((itemgetter(*idx)(self.population_obj)))
        self.weights = list((itemgetter(*idx)(self.weights)))
        # Add new subproblems
        l_p = len(self.population)
        dist = cdist(
            [self.EP_obj[i] for i in range(l_ep)],
            [self.population_obj[i] for i in range(l_p)]
        )  # shape = (l_ep, l_p)
        dist.sort(axis=1)
        sub_dist = np.prod(dist[:, 0:self.n_obj], axis=1)
        idx = np.argpartition(sub_dist, -nus)[-nus:]
        add_EP = list((itemgetter(*idx)(self.EP)))
        add_EP_obj = list((itemgetter(*idx)(self.EP_obj)))
        add_weights = []
        for e in add_EP_obj:
            ans = np.asarray(e) - np.asarray(self.ideal_point)
            ans[ans < EPSILON] = 1
            ans = 1 / ans
            ans[ans == np.inf] = 1  # when f = z
            add_weights.append((ans / np.sum(ans)).tolist())
        self.population.extend(add_EP)
        self.population_obj.extend(add_EP_obj)
        self.weights.extend(add_weights)
        # Update the neighbor
        self.neighborhoods = []  # the i-th element save the index of the neighborhoods of it
        for i in range(self.population_size):
            sorted_weights = self.moead_sort_weights(
                self.weights[i], self.weights)
            self.neighborhoods.append(
                sorted_weights[:self.moead_neighborhood_maxsize])

    def update_igd(self, value):
        self.value_his.append(value)
        if value < self.best_value:
            self.stag_count = 0
            self.best_value = value
        else:
            self.stag_count += 1
        self.last_value = value

    def moead_calculate_fitness(self, solution_obj, weights):
        return chebyshev(solution_obj, self.ideal_point, weights)

    def moead_update_solution(self, solution, solution_obj, mating_indices):
        """
        repair solution, make constraint satisfiable
        :param solution:
        :param mating_indices:
        :return:
        """

        c = 0
        self.rng.shuffle(mating_indices)

        for i in mating_indices:
            candidate = self.population[i]
            candidate_obj = self.population_obj[i]
            weights = self.weights[i]
            replace = False
            if self.moead_calculate_fitness(solution_obj, weights) < self.moead_calculate_fitness(candidate_obj,
                                                                                                  weights):
                replace = True

            if replace:
                self.population[i] = copy.deepcopy(solution)
                self.population_obj[i] = copy.deepcopy(solution_obj)
                c = c + 1

            if c >= self.moead_eta:
                break

    @staticmethod
    def moead_sort_weights(base, weights):
        """Returns the index of weights nearest to the base weight."""

        def compare(weight1, weight2):
            dist1 = math.sqrt(
                sum([math.pow(base[i] - weight1[1][i], 2.0) for i in range(len(base))]))
            dist2 = math.sqrt(
                sum([math.pow(base[i] - weight2[1][i], 2.0) for i in range(len(base))]))

            if dist1 < dist2:
                return -1
            elif dist1 > dist2:
                return 1
            else:
                return 0

        sorted_weights = sorted(
            enumerate(weights), key=functools.cmp_to_key(compare))
        return [i[0] for i in sorted_weights]

    def moead_get_subproblems(self):
        """
        Determines the subproblems to search.
        If :code:`utility_update` has been set, then this method follows the
        utility-based moea/D search.
        Otherwise, it follows the original moea/D specification.
        """
        indices = list(range(self.population_size))
        self.rng.shuffle(indices)
        return indices

    def moead_get_mating_indices(self, index):
        """Determines the mating indices.

        Returns the population members that are considered during mating.  With
        probability :code:`delta`, the neighborhood is returned.  Otherwise,
        the entire population is returned.
        """
        if self.rng.uniform(0.0, 1.0) <= self.moead_delta:
            return self.neighborhoods[index][:self.moead_neighborhood_size]
        else:
            return list(range(self.population_size))

    def get_hypervolume(self, n_samples=1e5):
        if self.problem.n_obj <= 3:
            hv_fast = False
        else:
            hv_fast = True
        if not hv_fast:
            # Calculate the exact hv value
            hyp = Hypervolume(minimum=[0 for _ in range(
                self.n_obj)], maximum=self.archive_maximum)
            hv_value = hyp.calculate(np.array(self.population_obj))
        else:
            # Estimate the hv value by Monte Carlo

            popobj = copy.deepcopy(self.population_obj)
            optimum = self.problem_ref_points
            fmin = np.clip(np.min(popobj, axis=0), np.min(popobj), 0)
            fmax = np.max(optimum, axis=0)

            popobj = (popobj - np.tile(fmin, (self.population_size, 1))) / (
                np.tile(1.1 * (fmax - fmin), (self.population_size, 1)))
            index = np.all(popobj < 1, 1).tolist()
            popobj = popobj[index]
            if popobj.shape[0] <= 1:
                hv_value = 0
                self.hv_his.append(hv_value)
                self.hv_last5.add(hv_value)
                self.hv_running.update(np.array([hv_value]))
                return hv_value
            assert np.max(popobj) < 1
            hv_maximum = np.ones([self.n_obj])
            hv_minimum = np.min(popobj, axis=0)
            n_samples_hv = int(n_samples)
            samples = np.zeros([n_samples_hv, self.n_obj])
            for i in range(self.n_obj):
                samples[:, i] = self.rng.uniform(
                    hv_minimum[i], hv_maximum[i], n_samples_hv)
            for i in range(popobj.shape[0]):
                domi = np.ones([samples.shape[0]], dtype=bool)
                m = 0
                while m < self.n_obj and any(domi):
                    domi = np.logical_and(domi, popobj[i, m] <= samples[:, m])
                    m += 1
                save_id = np.logical_not(domi)
                samples = samples[save_id, :]
            hv_value = np.prod(hv_maximum - hv_minimum) * (
                    1 - samples.shape[0] / n_samples_hv)
        self.hv_his.append(hv_value)
        self.hv_last5.add(hv_value)
        self.hv_running.update(np.array([hv_value]))
        return hv_value

    def get_igd(self):
        igd_value = self.igd_calculator.calculate(self.population_obj)
        self.igd_his.append(igd_value)
        self.igd_last5.add(igd_value)
        self.igd_running.update(np.array([igd_value]))
        return igd_value

    def get_ratio_nondom_sol(self):
        count = len(self.find_non_dominated_indices(self.population_obj))
        ratio_value = count / len(self.population_obj)
        self.nds_ratio_his.append(ratio_value)
        self.nds_ratio_last5.add(ratio_value)
        self.nds_ratio_running.update(np.array([ratio_value]))
        return ratio_value

    def get_average_dist(self):
        total_distance = cdist(
            [self.population_obj[i] for i in range(
                self.population_size)],
            [self.population_obj[i] for i in range(self.population_size)])
        if np.max(total_distance) == 0:
            ava_dist = 0
        else:
            ava_dist = np.mean(total_distance) / np.max(total_distance)
        if (np.isnan(ava_dist)):
            for i in range(self.population_size):
                print(self.population[i].objectives)
            print("total_distance:", total_distance)
            print("ava_dist is nan")
            sys.exit(0)
        self.ava_dist_his.append(ava_dist)
        self.ava_dist_last5.add(ava_dist)
        self.ava_dist_running.update(np.array([ava_dist]))
        return ava_dist

    def get_pre_k_change(self, k, value_his):
        if self.moead_generation >= k:
            return value_his[-1] - value_his[-(k + 1)]
        else:
            return 0

    def get_reward(self, value):
        """
        use the value to get reward
        value(default is igd), the smaller the better
        :return: reward based on current igd and historical igd
        """
        reward = 0
        if self.reward_type == 0:
            if value < self.best_value:
                bonus = (self.inital_value - value) / self.inital_value
                reward = (self.last_bonus + bonus) * (bonus - self.last_bonus)
            reward *= 100
        elif self.reward_type == 1:
            reward = max(self.last_value - value, 0)
        elif self.reward_type == 2:
            if value < self.best_value:
                reward = 10
            elif value < self.last_value:
                reward = 1
        elif self.reward_type == 3:
            reward = max((self.last_value - value) / value, 0)
        else:
            raise ValueError("Invaild Reward Type.")
        return reward


    def close(self):
        self.reset()


def normal_boundary_weights(nobjs, divisions_outer, divisions_inner=0):
    """Returns weights generated by the normal boundary method.

    The weights produced by this method are uniformly distributed on the
    hyperplane intersecting

        [(1, 0, ..., 0), (0, 1, ..., 0), ..., (0, 0, ..., 1)].

    Parameters
    ----------
    nobjs : int
        The number of objectives.
    divisions_outer : int
        The number of divisions along the outer set of weights.
    divisions_inner : int (optional)
        The number of divisions along the inner set of weights.
    """

    def generate_recursive(weights, weight, left, total, index):
        if index == nobjs - 1:
            weight[index] = float(left) / float(total)
            weights.append(copy.copy(weight))
        else:
            for i in range(left + 1):
                weight[index] = float(i) / float(total)
                generate_recursive(weights, weight, left - i, total, index + 1)

    def generate_weights(divisions):
        weights = []
        generate_recursive(weights, [0.0] * nobjs, divisions, divisions, 0)
        return weights

    weights = generate_weights(divisions_outer)

    if divisions_inner > 0:
        inner_weights = generate_weights(divisions_inner)

        for i in range(len(inner_weights)):
            weight = inner_weights[i]

            for j in range(len(weight)):
                weight[j] = (1.0 / nobjs + weight[j]) / 2.0

            weights.append(weight)

    return weights


if __name__ == "__main__":
    optimizer = MADAC_MOEAD_Optimizer(1)
    dtlz2 = DTLZ2()
    optimizer.init_population(dtlz2)
    optimizer.get_state()
    for i in range(100):
        first_three = np.random.randint(0, 4, 3)
        last_one = np.random.randint(0, 2, 1)
        action = np.concatenate((first_three, last_one))
        optimizer.step(action, dtlz2)
