import numpy as np
from deap import base
from deap import creator
from deap import tools
from ...environment.optimizer.basic_optimizer import Basic_Optimizer


class DE(Basic_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        config.NP = 50
        config.F = 0.5
        config.Cr = 0.5

        self.__config = config
        self.__toolbox = base.Toolbox()
        self.__creator = creator
        self.log_interval = config.log_interval
        self.full_meta_data = config.full_meta_data
        
    def __str__(self):
        return "DE"


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
        self.__toolbox.register("select", tools.selTournament, tournsize=3)
        self.__toolbox.register("attr_float", np.random.uniform, problem.lb, problem.ub)
        self.__toolbox.register("individual", tools.initRepeat, creator.Individual, self.__toolbox.attr_float, n=problem.dim)
        self.__toolbox.register("population", tools.initRepeat, list, self.__toolbox.individual)

        hof = tools.HallOfFame(1)

        pop = self.__toolbox.population(n=self.__config.NP)
        fitnesses = self.__toolbox.map(self.__toolbox.evaluate, pop)

        self.__FEs = self.__config.NP
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        hof.update(pop)

        log_index = 1
        self.cost = [hof[0].fitness.values[0]]

        done = False
        while not done:
            for k, agent in enumerate(pop):
                a, b, c = self.__toolbox.select(pop, 3)
                y = self.__toolbox.clone(agent)
                # mutate & crossover
                index = np.random.randint(0, problem.dim, 1)[0]
                for i, value in enumerate(agent):
                    if np.random.rand() < self.__config.Cr or i == index:
                        y[i] = a[i] + self.__config.F * (b[i] - c[i])
                        # BC
                        y[i] = max(problem.lb, min(y[i], problem.ub))
                y.fitness.values = self.__toolbox.evaluate(y)
                # selection
                if y.fitness.values[0] < agent.fitness.values[0]:
                    pop[k] = y

                hof.update(pop)
                self.__FEs += 1

                if self.__FEs >= log_index * self.log_interval:
                    log_index += 1
                    self.cost.append(hof[0].fitness.values[0])

                if problem.optimum is None:
                    done = self.__FEs >= self.__config.maxFEs
                else:
                    done = self.__FEs >= self.__config.maxFEs

                if done:
                    if len(self.cost) >= self.__config.n_logpoint + 1:
                        self.cost[-1] = hof[0].fitness.values[0]
                    else:
                        while len(self.cost) < self.__config.n_logpoint + 1:
                            self.cost.append(hof[0].fitness.values[0])
                    break
    
    
        results = {'cost': self.cost, 'fes': self.__FEs}

        if self.full_meta_data:
            metadata = {'X':self.meta_X, 'Cost':self.meta_Cost}
            results['metadata'] = metadata
        # 与agent一致，去除return，加上metadata
        return results
