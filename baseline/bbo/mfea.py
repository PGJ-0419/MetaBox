import numpy as np
import copy
from ...environment.optimizer.basic_optimizer import Basic_Optimizer

def SBX(parent1,parent2,n):
    pop_cnt = parent1.shape[0]
    beta = [0] * pop_cnt
    for i in range(pop_cnt):
        rand = np.random.random()
        if rand<=0.5:
            beta[i] = (2*rand) ** (1/(n+1))
        if rand>0.5:
            beta[i] = (1/(2-2*rand)) ** (1/(n+1))

    offspring1 = copy.deepcopy(parent1)
    offspring2 = copy.deepcopy(parent2)
    for i in range(pop_cnt):
        offspring1[i] = 1/2 * (parent1[i]+parent2[i]) - 1/2 * beta[i] * (parent2[i]-parent1[i])
        offspring2[i] = 1/2 * (parent1[i]+parent2[i]) + 1/2 * beta[i] * (parent2[i]-parent1[i])
        offspring1[i] = np.clip(offspring1[i],0,1)
        offspring2[i] = np.clip(offspring2[i],0,1)

    return offspring1, offspring2

def gaussian_mutation(parent, mutate_probability, sigma):
    dimension = parent.shape[0]
    offspring = copy.deepcopy(parent)
    for i in range(dimension):
        rand = np.random.random()
        if rand<mutate_probability:
            offspring[i] = np.clip(np.random.normal(parent[i],sigma),0,1)

    return offspring

def polinomial_mutation(parent, mu):
    dim = parent.shape[0]
    offspring = copy.deepcopy(parent)
    for i in range(dim):
        if np.random.random() < 0.05 :
            u = np.random.random()
            if u < 0.5:
                delta = (2 * u) ** (1 / (1 + mu)) - 1
                offspring[i] = offspring[i] + delta * offspring[i]
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / (1 + mu))
                offspring[i] = offspring[i] + delta * (1-offspring[i])
            offspring[i] = np.clip(offspring[i],0,1)
    return offspring

class Individual():
    def __init__(self, D_multitask, tasks):
        self.dim = D_multitask
        self.tasks = tasks
        self.tasks_count = len(tasks)
        self.genes = np.random.uniform(size=D_multitask)
        self.scalar_fitness = None
        self.skill_factor = None

    def update_evaluate(self):
        task = self.tasks[self.skill_factor]
        task_genes = self.genes[:task.dim]
        x = task.decode(task_genes)
        fitness = task.func(x)
        return self.skill_factor, fitness

    def first_evaluate(self):
        factorial_cost_list_j = []
        for j in range(self.tasks_count):
            task_j_genes = self.genes[:self.tasks[j].dim]
            x =  self.tasks[j].decode(task_j_genes)
            fitness = self.tasks[j].func(x)
            factorial_cost_list_j.append(fitness)

        return factorial_cost_list_j


class MFEA(Basic_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.__config = config
        self.log_interval = config.log_interval
        self.full_meta_data = config.full_meta_data

        self.cost = None
        self._fes = None
        self.log_index = None


    def __str__(self):
        return "MFEA"
    
    def run_episode(self, tasks):
        rmp = 0.3
        population_cnt = 50
        generation = 0
        mu = 2  
        self._fes = 0
        self.log_index = 1
    
        if self.full_meta_data:
            self.meta_Cost = []
            self.meta_X = []

        task_count = len(tasks)
        D = np.zeros(shape=task_count)
        for i in range(task_count):
            D[i] = tasks[i].dim
        D_multitask = int(np.max(D))

        population = np.array([Individual(D_multitask, tasks) for _ in range(2*population_cnt)])
        factorial_costs = np.full(shape=(2*population_cnt, task_count), fill_value=np.inf)
        factorial_ranks = np.empty(shape=(2*population_cnt, task_count))
        best_fitness = np.full(shape=task_count,fill_value=np.inf)
        
        for i, individual in enumerate(population[:population_cnt]):
            factorial_all_cost = individual.first_evaluate()
            factorial_costs[i] = factorial_all_cost

        for j in range(task_count):
            factorial_cost_j = factorial_costs[:, j]
            index = np.argsort(factorial_cost_j)
            for i, x in enumerate(index):
                factorial_ranks[x, j] = i + 1

        for i in range(population_cnt):
            population[i].scalar_fitness = 1 / np.min(factorial_ranks[i])
            population[i].skill_factor = np.argmin(factorial_ranks[i])
        
        if self.full_meta_data:
            list_pop = [population[i].genes for i in range(0,population_cnt)]
            self.meta_Cost.append(factorial_costs[:population_cnt])
            self.meta_X.append(list_pop)   
        
        self.cost = [best_fitness]

        done = False
        while not done:
            order = np.random.permutation(population_cnt)
            count = population_cnt
            factorial_costs[population_cnt:,:] = np.inf
            for i in range(0,population_cnt,2):
                parent1 = population[order[i]]
                parent2 = population[order[i+1]]
                offspring1 = Individual(D_multitask, tasks)
                offspring2 = Individual(D_multitask, tasks)

                if(parent1.skill_factor == parent2.skill_factor or np.random.random()<rmp):
                    offspring1.genes,offspring2.genes = SBX(parent1.genes,parent2.genes,mu)

                    rand1 = np.random.random()
                    rand2 = np.random.random()
                    if rand1 <0.5:
                        offspring1.skill_factor = parent1.skill_factor
                    else:
                        offspring1.skill_factor = parent2.skill_factor

                    if rand2 < 0.5:
                        offspring2.skill_factor = parent1.skill_factor
                    else:
                        offspring2.skill_factor = parent2.skill_factor

                else:
                    offspring1.genes = gaussian_mutation(parent1.genes,0.05,0.5)
                    offspring1.skill_factor = parent1.skill_factor
                        
                    offspring2.genes = gaussian_mutation(parent2.genes,0.05,0.5)
                    offspring2.skill_factor = parent2.skill_factor

                population[count] = offspring1
                population[count+1] = offspring2
                count+=2

            for i, individual in enumerate(population[population_cnt:]):
                j, factorial_cost = individual.update_evaluate()
                factorial_costs[population_cnt + i, j] = factorial_cost

            for j in range(task_count):
                factorial_cost_j = factorial_costs[:,j]
                index = np.argsort(factorial_cost_j)
                for i, x in enumerate(index):
                    factorial_ranks[x,j] = i+1

            for i in range(2 * population_cnt):
                population[i].scalar_fitness = 1 / np.min(factorial_ranks[i])
                population[i].skill_factor = np.argmin(factorial_ranks[i])

            scalar_fitness_list = np.array([individual.scalar_fitness for individual in population])
            select_list = np.argsort(scalar_fitness_list)[::-1]
            population = population[select_list]
            factorial_costs = factorial_costs[select_list]
            factorial_ranks = factorial_ranks[select_list]


            for j in range(task_count):
                best_j = np.argmin(factorial_costs[:, j])
                if (best_fitness[j] > factorial_costs[best_j, j]):
                    best_fitness[j] = factorial_costs[best_j, j]

            self._fes += population_cnt
            generation += 1
            if self._fes >= log_index * self.log_interval:
                log_index += 1
                self.cost.append(best_fitness)

            done = self._fes >= self.__config.maxFEs or generation >= self.__config.generation

            if done:
                if len(self.cost) >= self.__config.n_logpoint + 1:
                    self.cost[-1] = best_fitness
                else:
                    self.cost.append(best_fitness)
                break    

            if self.full_meta_data: 
                list_pop = [population[i].genes for i in range(0,population_cnt)]
                self.meta_Cost.append(factorial_costs[:population_cnt])
                self.meta_X.append(list_pop)
               
        results = {'cost': self.cost, 'fes': self._fes}

        if self.full_meta_data:
            metadata = {'X':self.meta_X, 'Cost':self.meta_Cost}
            results['metadata'] = metadata

        return results 