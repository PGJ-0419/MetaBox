# import pickle
import dill as pickle
from torch.utils.data import Dataset
from problem.basic_problem import Basic_Problem
import numpy as np
import torch
import time

class Basic_Problem_torch(Basic_Problem):
    """
    Abstract super class for problems and applications.
    """

    def reset(self):
        self.T1 = 0

    def eval(self, x):
        """
        A general version of func() with adaptation to evaluate both individual and population.
        """
        start = time.perf_counter()
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if x.dtype != torch.float64:
            x = x.type(torch.float64)
        if x.ndim == 1:  # x is a single individual
            y = self.func(x.reshape(1, -1))[0]
            end = time.perf_counter()
            self.T1 += (end - start) * 1000
            return y
        elif x.ndim == 2:  # x is a whole population
            y = self.func(x)
            end = time.perf_counter()
            self.T1 += (end - start) * 1000
            return y
        else:
            y = self.func(x.reshape(-1, x.shape[-1]))
            end = time.perf_counter()
            self.T1 += (end - start) * 1000
            return y

    def func(self, x):
        raise NotImplementedError

class GP_problem(Basic_Problem_torch):
    def __init__(self, execute, problemID, lb, ub, dim):
        self.problem = execute
        self.lb = lb
        self.ub = ub
        self.optimum = None
        self.opt = None
        self.problemID = problemID
        self.dim = dim
        self.T1 = 0
        self.FES = 0

    def func(self, x):
        return self.problem(x)

    def __call__(self, x):
        if len(x.size) == 1 and x.size[-1] == self.dim:
            x = x.reshape(1, -1)
            return self.func(x).reshape(-1)[0]
        else:
            return self.func(x)

    def get_optimal(self):
        return self.opt

    def __str__(self):
        return f'GP_Problem_{self.problemID}'

    def __name__(self):
        return f'GP_Problem_{self.problemID}'


class Symbolic_bench_Dataset_torch(Dataset):
    def __init__(self,
                 data,
                 batch_size=1):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.N = len(self.data)
        self.ptr = [i for i in range(0, self.N, batch_size)]
        self.index = np.arange(self.N)
            
    @staticmethod
    def get_datasets(upperbound = 5,
                     train_batch_size=1,
                     test_batch_size=1,
                     instance_seed=3849,
                     train_test_split = 0.8,
                     get_all = False):
        # get problem instances
        if instance_seed > 0:
            rng = np.random.default_rng(instance_seed)
            
        train_set = []
        test_set = []
        all_set = []
        with open('./problem/datafiles/SOO/self_generated/Symbolic_bench_torch/256_programs.pickle','rb')as f:
            all_functions = pickle.load(f)
            f.close()
        for program in all_functions:
            all_set.append(GP_problem(program.execute,program.problemID,lb=-upperbound,ub=upperbound,dim = program.best_dim ))
        
        if get_all:
            return Symbolic_bench_Dataset_torch(all_set,train_batch_size)
        
        nums = len(all_set)
        train_len = int(nums*train_test_split)
        # train_test_split
        rand_idx = rng.permutation(nums)

        train_set = [all_set[i] for i in rand_idx[:train_len]]
        test_set = [all_set[i] for i in rand_idx[train_len:]]
    
        return Symbolic_bench_Dataset_torch(train_set, train_batch_size), Symbolic_bench_Dataset_torch(test_set, test_batch_size)
    
    
        
    def __getitem__(self, item):
        if self.batch_size < 2:
            return self.data[self.index[item]]
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    def __len__(self):
        return self.N

    def __add__(self, other: 'Symbolic_bench_Dataset'):
        return Symbolic_bench_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)
        
    