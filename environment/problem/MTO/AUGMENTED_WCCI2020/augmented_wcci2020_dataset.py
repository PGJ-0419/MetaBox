from .augmented_wcci2020_numpy import Sphere, Ackley, Rosenbrock, Rastrigin, Schwefel, Griewank, Weierstrass
from .augmented_wcci2020_torch import Sphere_Torch, Ackley_Torch, Rosenbrock_Torch, Rastrigin_Torch, Schwefel_Torch, Griewank_Torch, Weierstrass_Torch
import numpy as np
from torch.utils.data import Dataset
import os
from itertools import combinations


def rotate_gen(dim):  # Generate a rotate matrix
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        mat = np.eye(dim)
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
    # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H

def get_combinations():
    numbers = list(range(1, 8))     
    all_combinations = []
    for r in range(1, len(numbers) + 1):
        all_combinations.extend(combinations(numbers, r))

    sorted_combinations = sorted(all_combinations, key=len)
    combinations_list = [list(comb) for comb in sorted_combinations]
    return combinations_list


class Augmented_WCCI2020_Dataset(Dataset):
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
    def get_datasets(version='numpy',
                     train_batch_size=1,
                     test_batch_size=1,
                     difficulty=None,
                     user_train_list=None,
                     user_test_list=None):
        if difficulty == None and user_test_list == None and user_train_list == None:
            raise ValueError('Please set difficulty or user_train_list and user_test_list.')
        if difficulty not in ['easy', 'difficult', 'all', None]:
            raise ValueError(f'{difficulty} difficulty is invalid.')

        task_cnt = 10
        dim = 50
        combinations = get_combinations()
        combination_cnt = len(combinations)
        task_set = []
        for combination in combinations:
            ub = 0
            lb = 0
            Tasks = []
            for _ in range(task_cnt):
                func_id = np.random.choice(combination)
                if func_id == 1:
                    ub = Sphere.UB
                    lb = Sphere.LB
                    shift = 0.8 * (np.random.random(dim) * (ub - lb) + lb)
                    rotate_matrix = rotate_gen(dim)
                    if version == 'numpy':
                        task = Sphere(dim, shift, rotate_matrix)
                    else:
                        task = Sphere_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 2:
                    ub = Rosenbrock.UB
                    lb = Rosenbrock.LB
                    shift = 0.8 * (np.random.random(dim) * (ub - lb) + lb)
                    rotate_matrix = rotate_gen(dim)
                    if version == 'numpy':
                         task = Rosenbrock(dim, shift, rotate_matrix)
                    else:
                        task = Rosenbrock_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 3:
                    ub = Ackley.UB
                    lb = Ackley.LB
                    shift = 0.8 * (np.random.random(dim) * (ub - lb) + lb)
                    rotate_matrix = rotate_gen(dim)
                    if version == 'numpy':
                         task = Ackley(dim, shift, rotate_matrix)
                    else:
                        task = Ackley_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 4:
                    ub = Rastrigin.UB
                    lb = Rastrigin.LB
                    shift = 0.8 * (np.random.random(dim) * (ub - lb) + lb)
                    rotate_matrix = rotate_gen(dim)
                    if version == 'numpy':
                        task = Rastrigin(dim, shift, rotate_matrix)
                    else:
                        task = Rastrigin_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 5:
                    ub = Griewank.UB
                    lb = Griewank.LB
                    shift = 0.8 * (np.random.random(dim) * (ub - lb) + lb)
                    rotate_matrix = rotate_gen(dim)
                    if version == 'numpy':
                        task = Griewank(dim, shift, rotate_matrix)
                    else:
                        task = Griewank_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 6:
                    ub = Weierstrass.UB
                    lb = Weierstrass.LB
                    shift = 0.8 * (np.random.random(dim) * (ub - lb) + lb)
                    rotate_matrix = rotate_gen(dim)
                    if version == 'numpy':
                        task = Weierstrass(dim, shift, rotate_matrix)
                    else:
                        task = Weierstrass_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 7:
                    ub = Schwefel.UB
                    lb = Schwefel.LB
                    shift = 0.8 * (np.random.random(dim) * (ub - lb) + lb)
                    rotate_matrix = rotate_gen(dim)
                    if version == 'numpy':
                        task = Schwefel(dim, shift, rotate_matrix)
                    else:
                        task = Schwefel_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)
            
            task_set.append(Tasks)

        if difficulty == 'easy':
            dataset_list = np.arange(0,combination_cnt)
            train_select_list = np.random.choice(dataset_list,size=int(combination_cnt*0.2), replace=False)
            test_select_list = dataset_list[~np.isin(dataset_list, train_select_list)]  
        elif difficulty == 'difficult':
            dataset_list = np.arange(0,combination_cnt)
            train_select_list = np.random.choice(dataset_list,size=int(combination_cnt*0.8), replace=False)
            test_select_list = dataset_list[~np.isin(dataset_list, train_select_list)]  
        elif difficulty == 'all':
            dataset_list = np.arange(0,combination_cnt)
            train_select_list = dataset_list
            test_select_list = dataset_list
        elif difficulty is None:
            train_select_list = user_train_list
            test_select_list = user_test_list

        train_set = [task_set[i] for i in train_select_list]
        test_set = [task_set[i] for i in test_select_list]

        return Augmented_WCCI2020_Dataset(train_set, train_batch_size), Augmented_WCCI2020_Dataset(test_set, test_batch_size)


    def __getitem__(self, item):
        
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    def __len__(self):
        return self.N

    def __add__(self, other: 'Augmented_WCCI2020_Dataset'):
        return Augmented_WCCI2020_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)


