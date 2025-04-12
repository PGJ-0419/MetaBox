from .bbob import *
from torch.utils.data import Dataset
from .utils import *
import numpy as np

class BBOB_Dataset_torch(Dataset):
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
    def get_datasets(suit,
                     dim,
                     upperbound,
                     shifted=True,
                     rotated=True,
                     biased=True,
                     train_batch_size=1,
                     test_batch_size=1,
                     difficulty='easy',
                     instance_seed=3849):
        # get functions ID of indicated suit
        if suit == 'bbob-torch':
            func_id = [i for i in range(1, 25)]  # [1, 24]
            small_set_func_id = [1, 5, 6, 10, 15, 20]
        elif suit == 'bbob-noisy-torch':
            func_id = [i for i in range(101, 131)]  # [101, 130]
            small_set_func_id = [101, 105, 115, 116, 117, 119, 120, 125]
        else:
            raise ValueError(f'{suit} function suit is invalid or is not supported yet.')
        if difficulty != 'easy' and difficulty != 'difficult':
            raise ValueError(f'{difficulty} difficulty is invalid.')
        # get problem instances
        if instance_seed > 0:
            np.random.seed(instance_seed)
        train_set = []
        test_set = []
        assert upperbound >= 5, 'Argument upperbound must be at least 5.'
        ub = upperbound
        lb = -upperbound
        for id in func_id:
            if shifted:
                shift = 0.8 * torch.tensor(np.random.random(dim) * (ub - lb) + lb, dtype=torch.float64)
            else:
                shift = torch.zeros(dim, dtype=torch.float64)
            if rotated:
                H = rotate_gen(dim)
            else:
                H = torch.eye(dim, dtype=torch.float64)
            if biased:
                bias = torch.tensor(np.random.randint(1, 26)) * 100
            else:
                bias = 0
            instance = eval(f'F{id}')(dim=dim, shift=shift, rotate=H, bias=bias, lb=lb, ub=ub)
            if (difficulty == 'easy' and id not in small_set_func_id) or (difficulty == 'difficult' and id in small_set_func_id):
                train_set.append(instance)
            else:
                test_set.append(instance)
        return BBOB_Dataset_torch(train_set, train_batch_size), BBOB_Dataset_torch(test_set, test_batch_size)

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

    def __add__(self, other: 'BBOB_Dataset_torch'):
        return BBOB_Dataset_torch(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)
