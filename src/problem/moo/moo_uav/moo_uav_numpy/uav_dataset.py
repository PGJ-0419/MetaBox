from .uav import *
from torch.utils.data import Dataset
from ..utils import createmodel
import numpy as np
import pickle
class MOO_UAV_Dataset(Dataset):
    def __init__(self, data, batch_size = 1):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.N = len(self.data)
        self.ptr = [i for i in range(0, self.N, batch_size)]
        self.index = np.arange(self.N)

    @staticmethod
    def get_datasets(train_batch_size=1,
                     test_batch_size = 1,
                     dv = 5.0,
                     j_pen = 1e4,
                     mode = "standard",
                     seed = 3849,
                     num = 56,
                     difficulty = "easy",
                     ):
        # easy 15 diff 30
        easy_id = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54] # 28
        diff_id = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55] # 28
        # if easy |train| = 42 = 14 + 28(easy + diff)
        # if diff |train| = 42 = 28 + 14(easy + diff)
        train_set = []
        test_set = []
        np.random.seed(seed)
        if mode == "standard":
            pkl_file = "E:/Desktop/github/MetaBox/src/problem/datafiles/SOO/uav_terrain/deal.pkl"
            with open(pkl_file, 'rb') as f:
                model_data = pickle.load(f)
            func_id = range(56) # 56
            if difficulty == "easy":
                train_id = diff_id
                easy_train = list(np.random.choice(easy_id, size = 14, replace = False))
                train_id = train_id + easy_train
            else:
                train_id = easy_id
                diff_train = list(np.random.choice(diff_id, size = 14, replace = False))
                train_id = train_id + diff_train

            for id in func_id:
                terrain_data = model_data[id]
                terrain_data['n'] = dv
                terrain_data['J_pen'] = j_pen
                instance = Terrain(terrain_data, id + 1)
                if id in train_id:
                    train_set.append(instance)
                else:
                    test_set.append(instance)
        elif mode == "custom":
            instance_list = []
            for id in range(num):
                if id < 0.5 * num:
                    num_threats = 15
                else:
                    num_threats = 30
                terrain_data = createmodel(map_size = 900,
                                           r = np.random.rand() * 100,
                                           rr = np.random.rand() * 10,
                                           num_threats = num_threats)
                terrain_data['n'] = dv
                terrain_data['J_pen'] = j_pen
                instance = Terrain(terrain_data, id + 1)
                instance_list.append(instance)
            if difficulty == "easy":
                train_set = instance_list[-int(0.75 * num):]
                test_set = instance_list[:int(0.25 * num)]
            else:
                train_set = instance_list[:int(0.75 * num)]
                test_set = instance_list[-int(0.25 * num):]

        return MOO_UAV_Dataset(train_set, train_batch_size), \
               MOO_UAV_Dataset(test_set, test_batch_size)

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

    def __add__(self, other: 'MOO_UAV_Dataset'):
        return MOO_UAV_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)
