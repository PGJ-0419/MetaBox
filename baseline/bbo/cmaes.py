import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from deap import cma
from ...environment.optimizer.basic_optimizer import Basic_Optimizer


class CMAES(Basic_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        config.NP = 50

        self.__config = config
        self.__toolbox = base.Toolbox()
        self.__creator = creator
        self.__algorithm = algorithms
        self.log_interval = config.log_interval
        self.full_meta_data = config.full_meta_data

    def __str__(self):
        return "CMAES"
    
    def run_episode(self, problem):
        self.rng_gpu = None
        self.rng_cpu = None
        self.rng = None
        np.random.seed(self.rng_seed)

        self.__creator.create("Fitnessmin", base.Fitness, weights=(-1.0,))
        self.__creator.create("Individual", list, fitness=creator.Fitnessmin)
        if self.full_meta_data:
            self.meta_Cost = []
            self.meta_X = []
            
        def problem_eval(x):
            if problem.optimum is None:
                fitness = problem.eval(x)
            else:
                fitness = problem.eval(x) - problem.optimum
            if self.full_meta_data:
                self.meta_Cost.append(fitness.copy())
                self.meta_X.append(x.copy())
            return fitness,   # return a tuple

        self.__toolbox.register("evaluate", problem_eval)
        strategy = cma.Strategy(centroid=[problem.ub] * problem.dim, sigma=0.5, lambda_=self.__config.NP)
        self.__toolbox.register("generate", strategy.generate, creator.Individual)
        self.__toolbox.register("update", strategy.update)

        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        fes = 0
        log_index = 0
        cost = []
        while True:
            _, logbook = self.__algorithm.eaGenerateUpdate(self.__toolbox, ngen=1, stats=stats, halloffame=hof, verbose=False)
            fes += len(logbook) * self.__config.NP
            while fes >= log_index * self.log_interval:
                log_index += 1
                cost.append(hof[0].fitness.values[0])
            if problem.optimum is None:
                done = fes >= self.__config.maxFEs
            else:
                done = fes >= self.__config.maxFEs
            if done:
                if len(cost) >= self.__config.n_logpoint + 1:
                    cost[-1] = hof[0].fitness.values[0]
                else:
                    while len(cost) < self.__config.n_logpoint + 1:
                        cost.append(hof[0].fitness.values[0])
                break
        results = {'cost': cost, 'fes': fes}

        if self.full_meta_data:
            metadata = {'X':self.meta_X, 'Cost':self.meta_Cost}
            results['metadata'] = metadata
        # 与agent一致，去除return，加上metadata
        return results
