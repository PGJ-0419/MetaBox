import numpy as np
from deap import base
from deap import creator
from deap import tools
from ...environment.optimizer.basic_optimizer import Basic_Optimizer


class PSO(Basic_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        config.phi1 = 2.
        config.phi2 = 2.
        config.population_size = 50

        self.__config = config
        self.__toolbox = base.Toolbox()
        self.__creator = creator
        self.log_interval = config.log_interval
        self.full_meta_data = config.full_meta_data
        
    def __str__(self):
        return "PSO"
    def run_episode(self, problem):
        self.rng_gpu = None
        self.rng_cpu = None
        self.rng = None
        np.random.seed(self.rng_seed)

        self.__creator.create("Fitnessmin", base.Fitness, weights=(-1.0,))
        self.__creator.create("Particle", np.ndarray, fitness=creator.Fitnessmin, speed=list, smin=None, smax=None, best=None)

        if self.full_meta_data:
            self.meta_Cost = []
            self.meta_X = []
        def generate(size, pmin, pmax, smin, smax):
            part = self.__creator.Particle(np.random.uniform(pmin, pmax, size))
            part.speed = np.random.uniform(smin, smax, size)
            part.smin = smin
            part.smax = smax
            return part

        def updateParticle(part, best, phi1, phi2, pmin, pmax):
            u1 = np.random.uniform(0, phi1, len(part))
            u2 = np.random.uniform(0, phi2, len(part))
            v_u1 = u1 * (part.best - part)
            v_u2 = u2 * (best - part)
            part.speed += v_u1 + v_u2
            for i, speed in enumerate(part.speed):
                if speed < part.smin:
                    part.speed[i] = part.smin
                elif speed > part.smax:
                    part.speed[i] = part.smax
            part += part.speed
            for i, value in enumerate(part):
                if value < pmin:
                    part[i] = pmin
                elif value > pmax:
                    part[i] = pmax
            return part

        def problem_eval(x):
            if problem.optimum is None:
                fitness = problem.eval(x)
            else:
                fitness = problem.eval(x) - problem.optimum
            return fitness,   # return a tuple

        pmax = problem.ub
        pmin = problem.lb
        smax = 0.5 * problem.ub
        smin = -smax

        self.__toolbox.register("particle", generate, size=problem.dim, pmin=pmin, pmax=pmax, smin=smin, smax=smax)
        self.__toolbox.register("population", tools.initRepeat, list, self.__toolbox.particle)
        self.__toolbox.register("update", updateParticle, phi1=self.__config.phi1, phi2=self.__config.phi2, pmin=pmin, pmax=pmax)
        self.__toolbox.register("evaluate", problem_eval)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # init population
        best = None  # gbest particle
        pop = self.__toolbox.population(n=self.__config.population_size)
        for part in pop:
            part.fitness.values = self.__toolbox.evaluate(part)
            part.best = self.__creator.Particle(part)
            part.best.fitness.values = part.fitness.values
            if best is None or part.fitness.values[0] < best.fitness.values[0]:
                best = self.__creator.Particle(part)
                best.fitness.values = part.fitness.values
        
        if self.full_meta_data:
            gen_meta_cost = []
            gen_meta_X = []
            for i in range(len(pop)):
                gen_meta_cost.append(pop[i].fitness.values[0])
                gen_meta_X.append(pop[i])
            self.meta_Cost.append(gen_meta_cost.copy())
            self.meta_X.append(gen_meta_X.copy())
        
        
        fes = self.__config.population_size

        log_index = 1
        cost = [best.fitness.values[0]]

        done = False
        while not done:
            for part in pop:
                self.__toolbox.update(part, best)
                part.fitness.values = self.__toolbox.evaluate(part)
                if part.fitness.values[0] < part.best.fitness.values[0]:  # update pbest
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if part.fitness.values[0] < best.fitness.values[0]:  # update gbest
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values
                fes += 1
                if fes >= log_index * self.log_interval:
                    log_index += 1
                    cost.append(best.fitness.values[0])

                if problem.optimum is None:
                    done = fes >= self.__config.maxFEs
                else:
                    done = fes >= self.__config.maxFEs 

                if done:
                    if len(cost) >= self.__config.n_logpoint + 1:
                        cost[-1] = best.fitness.values[0]
                    else:
                        while len(cost) < self.__config.n_logpoint + 1:
                            cost.append(best.fitness.values[0])
                    break
            if self.full_meta_data:
                gen_meta_cost = []
                gen_meta_X = []
                for i in range(len(pop)):
                    gen_meta_cost.append(pop[i].fitness.values[0])
                    gen_meta_X.append(pop[i])
                self.meta_Cost.append(gen_meta_cost)
                self.meta_X.append(gen_meta_X)
                
        results = {'cost': cost, 'fes': fes}

        if self.full_meta_data:
            metadata = {'X':self.meta_X, 'Cost':self.meta_Cost}
            results['metadata'] = metadata
        # 与agent一致，去除return，加上metadata
        return results
