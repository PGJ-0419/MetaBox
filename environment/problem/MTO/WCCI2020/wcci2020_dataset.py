from .wcci2020_numpy import Sphere, Ackley, Rosenbrock, Rastrigin, Schwefel, Griewank, Weierstrass
from .wcci2020_torch import Sphere_Torch, Ackley_Torch, Rosenbrock_Torch, Rastrigin_Torch, Schwefel_Torch, Griewank_Torch, Weierstrass_Torch
import numpy as np
from torch.utils.data import Dataset
import os

class WCCI2020_Dataset(Dataset):
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


    def __getitem__(self, item):
        
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res
    
    def __len__(self):
        return self.N

    
    def __add__(self, other: 'WCCI2020_Dataset'):
        return WCCI2020_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)

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

        func_id = [i for i in range(0, 10)]
        if difficulty == 'easy':
            train_id = [0, 1, 2, 3, 4, 5]
            test_id = [6, 7, 8, 9]
        elif difficulty == 'difficult':
            train_id = [6, 7, 8, 9]
            test_id = [0, 1, 2, 3, 4, 5]
        elif difficulty == 'all':
            train_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            test_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif difficulty is None:
            train_id = user_train_list
            test_id = user_test_list

        train_set = []
        test_set = []
        for task_ID in func_id:
            dim = 50
            task_size = 50
            choice_functions = []
            if task_ID == 0:
                choice_functions = [1]
            if task_ID == 1:
                choice_functions = [2]
            if task_ID == 2:
                choice_functions = [4]
            if task_ID == 3:
                choice_functions = [1,2,3]
            if task_ID == 4:
                choice_functions = [4,5,6]
            if task_ID == 5:
                choice_functions = [2,5,7]
            if task_ID == 6:
                choice_functions = [3,4,6]
            if task_ID == 7:
                choice_functions = [2,3,4,5,6]
            if task_ID == 8:
                choice_functions = [2,3,4,5,6,7]
            if task_ID == 9:
                choice_functions = [3,4,5,6,7]

            Tasks = []
            for task_id in range(1, task_size+1):
                id = (task_id-1) % len(choice_functions)
                func_id = choice_functions[id]

                folder_dir = os.path.join(os.path.dirname(__file__),'.','datafiles_MTO','MTO','classical','WCCI2020',f'benchmark_{task_ID+1}')
                shift_file = os.path.join(folder_dir, f'bias_{task_id}')
                rotate_file = os.path.join(folder_dir, f'matrix_{task_id}')

                rotate_matrix = np.loadtxt(rotate_file)
                shift = np.loadtxt(shift_file)

                if func_id == 1:
                    if version == 'numpy': 
                        task = Sphere(dim,shift, rotate_matrix)
                    else:
                        task = Sphere_Torch(dim,shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 2:
                    if version == 'numpy': 
                        task = Rosenbrock(dim, shift, rotate_matrix)
                    else:
                        task = Rosenbrock_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 3:
                    if version == 'numpy':
                         task = Ackley(dim, shift, rotate_matrix)
                    else:
                        task = Ackley_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 4:
                    if version == 'numpy':
                        task = Rastrigin(dim, shift, rotate_matrix)
                    else:
                        task = Rastrigin_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 5:
                    if version == 'numpy':
                        task = Griewank(dim, shift, rotate_matrix)
                    else:
                        task = Griewank_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 6:
                    if version == 'numpy':
                        task = Weierstrass(dim, shift, rotate_matrix)
                    else:
                        task = Weierstrass_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 7:
                    if version == 'numpy':
                        task = Schwefel(dim, shift, rotate_matrix)
                    else:
                        task = Schwefel_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)
            
            if task_ID in train_id:
                train_set.append(Tasks)
            if task_ID in test_id:
                test_set.append(Tasks)

        return WCCI2020_Dataset(train_set, train_batch_size), WCCI2020_Dataset(test_set, test_batch_size)
