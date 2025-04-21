from os import path
import torch
import numpy as np
from torch.utils.data import Dataset
from ...basic_problem import Basic_Problem, Basic_Problem_Torch
import time


class Protein_Docking_Numpy_Problem(Basic_Problem):
    n_atoms = 100  # number of interface atoms considered for computational concern
    dim = 12
    lb = -1.5
    ub = 1.5

    def __init__(self, coor_init, q, e, r, basis, eigval, problem_id):
        self.coor_init = coor_init  # [n_atoms, 3]
        self.q = q                  # [n_atoms, n_atoms]
        self.e = e                  # [n_atoms, n_atoms]
        self.r = r                  # [n_atoms, n_atoms]
        self.basis = basis          # [dim, 3*n_atoms]
        self.eigval = eigval        # [dim]
        self.problem_id = problem_id
        self.optimum = None      # unknown, set to None

    def __str__(self):
        return self.problem_id

    def func(self, x):
        eigval = 1.0 / np.sqrt(self.eigval)
        product = np.matmul(x * eigval, self.basis)  # [NP, 3*n_atoms]
        new_coor = product.reshape((-1, self.n_atoms, 3)) + self.coor_init  # [NP, n_atoms, 3]

        p2 = np.expand_dims(np.sum(new_coor * new_coor, axis=-1), axis=-1)  # sum of squares along last dim.  [NP, n_atoms, 1]
        p3 = np.matmul(new_coor, np.transpose(new_coor, (0, 2, 1)))  # inner products among row vectors. [NP, n_atoms, n_atoms]
        pair_dis = p2 - 2 * p3 + np.transpose(p2, (0, 2, 1))
        pair_dis = np.sqrt(pair_dis + 0.01)  # [NP, n_atoms, n_atoms]

        gt0_lt7 = (pair_dis > 0.11) & (pair_dis < 7.0)
        gt7_lt9 = (pair_dis > 7.0) & (pair_dis < 9.0)

        pair_dis += np.eye(self.n_atoms)  # [NP, n_atoms, n_atoms]
        coeff = self.q / (4. * pair_dis) + np.sqrt(self.e) * ((self.r / pair_dis) ** 12 - (self.r / pair_dis) ** 6)  # [NP, n_atoms, n_atoms]

        energy = np.mean(
            np.sum(10 * gt0_lt7 * coeff + 10 * gt7_lt9 * coeff * ((9 - pair_dis) ** 2 * (-12 + 2 * pair_dis) / 8),
                   axis=1), axis=-1)  # [NP]

        return energy


class Protein_Docking_Torch_Problem(Basic_Problem_Torch):
    n_atoms = 100  # number of interface atoms considered for computational concern
    dim = 12
    lb = -1.5
    ub = 1.5

    def __init__(self, coor_init, q, e, r, basis, eigval, problem_id):
        self.coor_init = torch.as_tensor(coor_init, dtype=torch.float64)  # [n_atoms, 3]
        self.q = torch.as_tensor(q, dtype=torch.float64)  # [n_atoms, n_atoms]
        self.e = torch.as_tensor(e, dtype=torch.float64)  # [n_atoms, n_atoms]
        self.r = torch.as_tensor(r, dtype=torch.float64)  # [n_atoms, n_atoms]
        self.basis = torch.as_tensor(basis, dtype=torch.float64)    # [dim, 3*n_atoms]
        self.eigval = torch.as_tensor(eigval, dtype=torch.float64)  # [dim]
        self.problem_id = problem_id
        self.optimum = None  # unknown, set to None

    def __str__(self):
        return self.problem_id

    def eval(self, x):
        """
        A general version of func() with adaptation to evaluate both individual and population.
        """
        start=time.perf_counter()
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if x.dtype != torch.float64:
            x = x.type(torch.float64)
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

    def func(self, x):
        eigval = 1.0 / torch.sqrt(self.eigval)
        product = torch.matmul(x * eigval, self.basis)  # [NP, 3*n_atoms]
        new_coor = product.reshape((-1, self.n_atoms, 3)) + self.coor_init  # [NP, n_atoms, 3]

        p2 = torch.sum(new_coor * new_coor, dim=-1, dtype=torch.float64)[:, :,
             None]  # sum of squares along last dim.  [NP, n_atoms, 1]
        p3 = torch.matmul(new_coor,
                          new_coor.permute(0, 2, 1))  # inner products among row vectors. [NP, n_atoms, n_atoms]
        pair_dis = p2 - 2 * p3 + p2.permute(0, 2, 1)
        pair_dis = torch.sqrt(pair_dis + 0.01)  # [NP, n_atoms, n_atoms]

        gt0_lt7 = (pair_dis > 0.11) & (pair_dis < 7.0)
        gt7_lt9 = (pair_dis > 7.0) & (pair_dis < 9.0)

        pair_dis = pair_dis + torch.eye(self.n_atoms, dtype=torch.float64)  # [NP, n_atoms, n_atoms]
        coeff = self.q / (4. * pair_dis) + torch.sqrt(self.e) * (
                    (self.r / pair_dis) ** 12 - (self.r / pair_dis) ** 6)  # [NP, n_atoms, n_atoms]

        energy = torch.mean(
            torch.sum(10 * gt0_lt7 * coeff + 10 * gt7_lt9 * coeff * ((9 - pair_dis) ** 2 * (-12 + 2 * pair_dis) / 8),
                      dim=1, dtype=torch.float64), dim=-1)  # [NP]

        return energy
