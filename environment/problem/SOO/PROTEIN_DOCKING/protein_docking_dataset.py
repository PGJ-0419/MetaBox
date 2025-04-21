from os import path
import torch
import numpy as np
from torch.utils.data import Dataset
from ...basic_problem import Basic_Problem
import time
from .protein_docking import Protein_Docking_Torch_Problem, Protein_Docking_Numpy_Problem
class Protein_Docking_Dataset(Dataset):
    proteins_set = {'rigid': ['1AVX', '1BJ1', '1BVN', '1CGI', '1DFJ', '1EAW', '1EWY', '1EZU', '1IQD', '1JPS',
                              '1KXQ', '1MAH', '1N8O', '1PPE', '1R0R', '2B42', '2I25', '2JEL', '7CEI', '1AY7'],
                    'medium': ['1GRN', '1IJK', '1M10', '1XQS', '2HRK'],
                    'difficult': ['1ATN', '1IBR', '2C0L']
                    }
    n_start_points = 10  # top models from ZDOCK

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
    def get_datasets(version,
                     train_batch_size=1,
                     test_batch_size=1,
                     difficulty='easy',
                     dataset_seed=1035):
        # apart train set and test set
        if difficulty == 'easy':
            train_set_ratio = 0.75
        elif difficulty == 'difficult':
            train_set_ratio = 0.25
        else:
            raise ValueError
        if dataset_seed > 0:
            np.random.seed(dataset_seed)
        train_proteins_set = []
        test_proteins_set = []
        for key in Protein_Docking_Dataset.proteins_set.keys():
            permutated = np.random.permutation(Protein_Docking_Dataset.proteins_set[key])
            n_train_proteins = max(1, min(int(len(permutated) * train_set_ratio), len(permutated) - 1))
            train_proteins_set.extend(permutated[:n_train_proteins])
            test_proteins_set.extend(permutated[n_train_proteins:])
        # construct problem instances
        data = []
        base_dir = path.dirname(path.abspath(__file__))
        data_folder = path.join(base_dir, 'datafile')
        for i in train_proteins_set + test_proteins_set:
            for j in range(Protein_Docking_Dataset.n_start_points):
                problem_id = i + '_' + str(j + 1)
                data_dir = path.join(data_folder, problem_id)
                coor_init = np.loadtxt(data_dir + '/coor_init')
                q = np.loadtxt(data_dir + '/q')
                e = np.loadtxt(data_dir + '/e')
                r = np.loadtxt(data_dir + '/r')
                basis = np.loadtxt(data_dir + '/basis')
                eigval = np.loadtxt(data_dir + '/eigval')

                q = np.tile(q, (1, 1))
                e = np.tile(e, (1, 1))
                r = np.tile(r, (len(r), 1))

                q = np.matmul(q.T, q)
                e = np.sqrt(np.matmul(e.T, e))
                r = (r + r.T) / 2
                if version == 'protein':
                    data.append(Protein_Docking_Numpy_Problem(coor_init, q, e, r, basis, eigval, problem_id))
                elif version == 'protein-torch':
                    data.append(Protein_Docking_Torch_Problem(coor_init, q, e, r, basis, eigval, problem_id))
                else:
                    raise ValueError(f'{version} version is invalid or is not supported yet.')
        n_train_instances = len(train_proteins_set) * Protein_Docking_Dataset.n_start_points
        return Protein_Docking_Dataset(data[:n_train_instances], train_batch_size), Protein_Docking_Dataset(data[n_train_instances:], test_batch_size)

    def __getitem__(self, item):
        
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    def __len__(self):
        return self.N

    def __add__(self, other: 'Protein_Docking_Dataset'):
        return Protein_Docking_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)
