from torch.utils.data import Dataset
import sys
import subprocess
import numpy as np
from .evox_ne import *


class NE_Dataset(Dataset):
    def __init__(self,
                 data,
                 batch_size=1):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.N = len(self.data)
        self.ptr = [i for i in range(0, self.N, batch_size)]
        self.index = np.arange(self.N)
        self.maxdim = 0
        for item in self.data:
            self.maxdim = max(self.maxdim, item.dim)

    @staticmethod
    def get_datasets(
                     train_batch_size=1,
                     test_batch_size=1,
                     difficulty='easy',
                     usr_train_list = None,
                     usr_test_list = None,
                     instance_seed=3849):
        assert difficulty in ['all','easy','difficult','user-define']
        train_set = []
        test_set = []
        if difficulty == 'all':
            for env in envs.keys():
                for depth in model_depth:
                    train_set.append(NE_Problem(env, depth, instance_seed))
                    test_set.append(NE_Problem(env, depth, instance_seed))
            return NE_Dataset(train_set, train_batch_size), NE_Dataset(test_set, test_batch_size)
            
        elif difficulty == 'easy':
            for env in envs.keys():
                for depth in model_depth:
                    if depth <=2:
                        test_set.append(NE_Problem(env, depth, instance_seed))
                    else:
                        train_set.append(NE_Problem(env, depth, instance_seed))
            return NE_Dataset(train_set, train_batch_size), NE_Dataset(test_set, test_batch_size)
        
        elif difficulty == 'difficult':
            for env in envs.keys():
                for depth in model_depth:
                    if depth <=2:
                        train_set.append(NE_Problem(env, depth, instance_seed))
                    else:
                        test_set.append(NE_Problem(env, depth, instance_seed))
            return NE_Dataset(train_set, train_batch_size), NE_Dataset(test_set, test_batch_size)
        
        elif difficulty == 'user-define':
            for env in envs.keys():
                for depth in model_depth:
                    if usr_train_list is not None and usr_test_list is not None:
                        if f'{env}-{depth}' in usr_train_list:
                            train_set.append(NE_Problem(env, depth, instance_seed))
                        if f'{env}-{depth}' in usr_test_list:
                            test_set.append(NE_Problem(env, depth, instance_seed))
                    elif usr_train_list is not None:
                        if f'{env}-{depth}' in usr_train_list:
                            train_set.append(NE_Problem(env, depth, instance_seed))
                        else:
                            test_set.append(NE_Problem(env, depth, instance_seed))
                    elif usr_test_list is not None:
                        if f'{env}-{depth}' in usr_test_list:
                            test_set.append(NE_Problem(env, depth, instance_seed))
                        else:
                            train_set.append(NE_Problem(env, depth, instance_seed))
                    else:
                        raise NotImplementedError
                
            return NE_Dataset(train_set, train_batch_size), NE_Dataset(test_set, test_batch_size)

    def __getitem__(self, item):
        
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    def __len__(self):
        return self.N

    def __add__(self, other: 'NE_Dataset'):
        return NE_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)

