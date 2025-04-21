import numpy as np
from ...environment.optimizer.basic_optimizer import Basic_Optimizer


class Random_search(Basic_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.__fes=0
        self.log_index=None
        self.cost=None
        self.__max_fes=config.maxFEs
        self.__NP=100
        self.__n_logpoint = config.n_logpoint
        self.log_interval = config.log_interval
        self.full_meta_data = config.full_meta_data
    
    def __str__(self):
        return 'Random_search'
    
    def __reset(self,problem):
        self.__fes=0
        self.cost=[]
        self.__random_population(problem,init=True)
        self.cost.append(self.gbest)
        self.log_index=1
    
    def __random_population(self,problem,init):
        rand_pos=self.rng.uniform(low=problem.lb,high=problem.ub,size=(self.__NP, problem.dim))
        if problem.optimum is None:
            cost=problem.eval(rand_pos)
        else:
            cost=problem.eval(rand_pos)-problem.optimum
            
        if self.full_meta_data:
            self.meta_Cost.append(cost.copy())
            self.meta_X.append(rand_pos.copy())
        self.__fes+=self.__NP
        if init:
            self.gbest=np.min(cost)
        else:
            if self.gbest>np.min(cost):
                self.gbest=np.min(cost)

    def run_episode(self, problem):
        if self.full_meta_data:
            self.meta_Cost = []
            self.meta_X = []
        problem.reset()
        self.__reset(problem)
        is_done = False
        while not is_done:
            self.__random_population(problem,init=False)
            while self.__fes >= self.log_index * self.log_interval:
                self.log_index += 1
                self.cost.append(self.gbest)

            if problem.optimum is None:
                is_done = self.__fes>=self.__max_fes
            else:
                is_done = self.__fes>=self.__max_fes

            if is_done:
                if len(self.cost) >= self.__n_logpoint + 1:
                    self.cost[-1] = self.gbest
                else:
                    while len(self.cost) < self.__n_logpoint + 1:
                        self.cost.append(self.gbest)
                break
                
        results = {'cost': self.cost, 'fes': self.__fes}

        if self.full_meta_data:
            metadata = {'X':self.meta_X, 'Cost':self.meta_Cost}
            results['metadata'] = metadata
        # 与agent一致，去除return，加上metadata
        return results
