from .kan import *
from ...basic_problem import Basic_Problem
from .bbob_numpy import *
from os import path
from torch.utils.data import Dataset
import time
import torch.nn as nn

# MLP
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.ln1 = nn.Linear(input_dim, 32)
        self.ln2 = nn.Linear(32, 64)
        self.ln3 = nn.Linear(64, 32)
        self.ln4 = nn.Linear(32, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.ln3(x)
        x = self.relu3(x)
        x = self.ln4(x)
        return x

class bbob_surrogate_model(Basic_Problem):
    def __init__(self, dim, func_id, lb, ub, shift, rotate, bias, config):
        self.dim = dim
        self.func_id = func_id

        self.instance = eval(f'F{func_id}')(dim=dim, shift=shift, rotate=rotate, bias=bias, lb=lb, ub=ub)
        self.device = config.device
        self.optimum = None

        # base_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
        base_dir = path.dirname(path.abspath(__file__))

        if dim == 2:

            if func_id in [1, 6, 8, 9, 12, 14, 19, 20, 23]:
                self.model = KAN.loadckpt(
                    path.join(base_dir, f'datafile\\Dim{dim}\\KAN\\{self.instance}\\model'))
            # elif func_id in [2, 3, 4, 5, 7, 10, 11, 13, 15, 16, 17, 18, 21, 22, 23]:
            else:
                self.model = MLP(dim)
                self.model.load_state_dict(
                    torch.load(path.join(base_dir,
                                         f'datafile\\Dim{dim}\\MLP\\{self.instance}\\model.pth'))
                )


        elif dim == 5:

            if func_id in [1, 2, 4, 6, 8, 9, 11, 12, 14, 20, 23]:
                self.model = KAN.loadckpt(
                    path.join(base_dir, f'datafile\\Dim{dim}\\KAN\\{self.instance}\\model'))
            else:
                self.model = MLP(dim)
                self.model.load_state_dict(
                    torch.load(path.join(base_dir,
                                         f'datafile\\Dim{dim}\\MLP\\{self.instance}\\model.pth'))
                )

        elif dim == 10:

            if func_id in [1, 2, 4, 6, 9, 12, 14, 23]:
                self.model = KAN.loadckpt(
                    path.join(base_dir, f'datafile\\Dim{dim}\\KAN\\{self.instance}\\model'))
            # elif func_id in [2, 5, 8, 9, 11, 16, 17, 18, 19, 20, 21, 22]:
            else:
                self.model = MLP(dim)
                self.model.load_state_dict(
                    torch.load(path.join(base_dir,
                                         f'datafile\\Dim{dim}\\MLP\\{self.instance}\\model.pth'))
                )


        else:
            raise ValueError(f'training on dim{dim} is not supported yet.')

        self.model.to(self.device)
        # KAN: 1,3,4,6,7,10,12,13,14,15,23,24  MLP:2,5,8,9,11,16,17,18,19,20,21,22

        self.ub = ub
        self.lb = lb

    def func(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).to(self.device)
            input_x = (x - self.lb) / (self.ub - self.lb)
            input_x = input_x.to(torch.float64)
            with torch.no_grad():
                y = self.model(input_x)

            return y.flatten().cpu().numpy()

        elif isinstance(x, torch.Tensor):
            input_x = (x - self.lb) / (self.ub - self.lb)
            input_x = input_x.to(torch.float64)
            with torch.no_grad():
                y = self.model(input_x)
            return y

    # return y
    def eval(self, x):
        """
        A general version of func() with adaptation to evaluate both individual and population.
        """
        start=time.perf_counter()

        if x.ndim == 1:  # x is a single individual
            y=self.func(x.reshape(1, -1))[0]
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y
        elif x.ndim == 2:  # x is a whole population
            y=self.func(x)
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y
        else:
            y=self.func(x.reshape(-1, x.shape[-1]))
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y
    def __str__(self):
        return f'Surrogate_{self.instance}'


class bbob_surrogate_Dataset(Dataset):
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
    def get_datasets(version='torch', suit='bbob-surrogate-10D',
                     train_batch_size=1,
                     test_batch_size=1, difficulty='easy',
                     user_train_list=None, user_test_list=None,
                     seed=3849, shifted=True, biased=True, rotated=True,
                     config=None, upperbound=5):

        is_train = config.is_train

        # train_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        # 			20]
        # test_id = [16, 17, 18, 19, 21, 22, 23, 24]
        if suit == 'bbob-surrogate-10D':
            dim = 10
        elif suit == 'bbob-surrogate-5D':
            dim = 5
        elif suit == 'bbob-surrogate-2D':
            dim = 2
        else:
            raise ValueError(f'{suit} is not supported.')


        if difficulty == 'easy':
            if dim == 2:
                train_id = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15,
                            20, 22]
            elif dim == 5 or dim == 10:
                train_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                            20]
        # test_id = [16, 17, 18, 19, 21, 22, 23, 24]
        elif difficulty == 'difficult':
            if dim == 2 or dim == 5:
                train_id = [1, 2, 5, 6, 10, 11, 13, 14]
            elif dim == 10:
                train_id = [1, 2, 5, 6, 10, 11, 13, 20]
        elif difficulty == None and user_train_list is not None and user_test_list is not None:
            train_id = user_train_list
            test_id = user_test_list

        elif difficulty == 'all':
            train_id = [i for i in range(1,25)]
            test_id = [i for i in range(1, 25)]
        else:
            raise ValueError(f'{difficulty} difficulty is invalid.')

        np.random.seed(seed)
        train_set = []
        test_set = []
        ub = upperbound
        lb = -upperbound

        func_id = [i for i in range(1, 25)]
        for id in func_id:
            if shifted:
                shift = 0.8 * (np.random.random(dim) * (ub - lb) + lb)
            else:
                shift = np.zeros(dim)
            if rotated:
                H = rotate_gen(dim)
            else:
                H = np.eye(dim)
            if biased:
                bias = np.random.randint(1, 26) * 100
            else:
                bias = 0

            if difficulty == 'all':
                train_instance = eval(f'F{id}')(dim = dim, shift = shift, rotate = H, bias = bias, lb = lb, ub = ub)
                test_instance = eval(f'F{id}')(dim = dim, shift = shift, rotate = H, bias = bias, lb = lb, ub = ub)
                train_set.append(train_instance)
                test_set.append(test_instance)
            else:
                if id in train_id:

                    if is_train:
                        train_instance = bbob_surrogate_model(dim, id, ub=ub, lb=lb, shift=shift, rotate=H, bias=bias,
                                                              config=config)
                    else:
                        train_instance = eval(f'F{id}')(dim=dim, shift=shift, rotate=H, bias=bias, lb=lb, ub=ub)

                    train_set.append(train_instance)

                # if id in test_id:
                else:
                    test_instance = eval(f'F{id}')(dim=dim, shift=shift, rotate=H, bias=bias, lb=lb, ub=ub)
                    test_set.append(test_instance)

        return bbob_surrogate_Dataset(train_set, train_batch_size), bbob_surrogate_Dataset(test_set, test_batch_size)

    def __len__(self):
        return self.N

    def __getitem__(self, item):

        
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    def __add__(self, other: 'bbob_surrogate_Dataset'):
        return bbob_surrogate_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = torch.randperm(self.N)
