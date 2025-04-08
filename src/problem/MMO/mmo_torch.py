from .mmo_basic_torch import *
import numpy as np
import math
import torch
from os import path

class F1_torch(MMO_Basic_Problem_torch): # five_uneven_peak_trap
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F1_torch, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        return 'five_uneven_peak_trap'+'_D'+str(self.dim)

    def func(self, x):
        
        if x is None:
            return None
        
        assert x.shape[1] == self.dim

        self.FES += x.shape[0]
        x = x[:, 0]
        result = torch.zeros_like(x, device = x.device, dtype=torch.float64)
        mask1 = (x >= 0) & (x < 2.50)
        mask2 = (x >= 2.5) & (x < 5)
        mask3 = (x >= 5.0) & (x < 7.5)
        mask4 = (x >= 7.5) & (x < 12.5)
        mask5 = (x >= 12.5) & (x < 17.5)
        mask6 = (x >= 17.5) & (x < 22.5)
        mask7 = (x >= 22.5) & (x < 27.5)
        mask8 = (x >= 27.5) & (x <= 30)
        
        result = mask1 * (80 * (2.5 - x)) + (~mask1) * result
        result = mask2 * (64 * (x - 2.5)) + (~mask2) * result
        result = mask3 * (64 * (7.5 - x)) + (~ mask3) * result
        result = mask4 * (28 * (x - 7.5)) + (~ mask4) * result
        result = mask5 * (28 * (17.5 - x)) + (~ mask5) * result
        result = mask6 * (32 * (x - 17.5)) + (~ mask6) * result
        result = mask7 * (32 * (27.5 - x)) + (~ mask7) * result
        result = mask8 * (80 * (x - 27.5)) + (~ mask8) * result
        
        return -result

class F2_torch(MMO_Basic_Problem_torch): # equal_maxima
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F2_torch, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        return 'equal_maxima'+'_D'+str(self.dim)

    def func(self, x):
        if x is None:
            return None

        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        return -torch.sin(5.0 * np.pi * x[:, 0]) ** 6

class F3_torch(MMO_Basic_Problem_torch): # uneven_decreasing_maxima
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F3_torch, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        return 'uneven_decreasing_maxima'+'_D'+str(self.dim)

    def func(self, x):
        if x is None:
            return None

        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        return -(
            torch.exp(-2.0 * torch.log(torch.tensor(2, device = x.device)) * ((x[:, 0] - 0.08) / 0.854) ** 2)
            * (torch.sin(torch.tensor(5 * np.pi, device = x.device) * (x[:, 0] ** 0.75 - 0.05))) ** 6
        )
        
class F4_torch(MMO_Basic_Problem_torch): # himmelblau
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F4_torch, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        return 'himmelblau'+'_D'+str(self.dim)

    def func(self, x):
        if x is None:
            return None

        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        result = 200 - (x[:, 0] ** 2 + x[:, 1] - 11) ** 2 - (x[:, 0] + x[:, 1] ** 2 - 7) ** 2
        return -result


class F5_torch(MMO_Basic_Problem_torch): # six_hump_camel_back
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F5_torch, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        return 'six_hump_camel_back'+'_D'+str(self.dim)

    def func(self, x):
        if x is None:
            return None

        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        x2 = x[:, 0] ** 2
        x4 = x[:, 0] ** 4
        y2 = x[:, 1] ** 2
        expr1 = (4.0 - 2.1 * x2 + x4 / 3.0) * x2
        expr2 = x[:, 0] * x[:, 1]
        expr3 = (4.0 * y2 - 4.0) * y2
        return -(-1.0 * (expr1 + expr2 + expr3))


class F6_torch(MMO_Basic_Problem_torch): # shubert
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F6_torch, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        return 'shubert'+'_D'+str(self.dim)

    def func(self, x):
        if x is None:
            return None

        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        n, D = x.shape
        soma = torch.zeros((n, D), device = x.device, dtype=torch.float64)
        
        for j in range(1, 6):
            soma = soma + (j * torch.cos((j + 1) * x + j))
        result = torch.prod(soma, dim = 1)


        return -(-result)



class F7_torch(MMO_Basic_Problem_torch): # vincent
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F7_torch, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        return 'vincent'+'_D'+str(self.dim)

    def func(self, x):
        if x is None:
            return None

        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        n, D = x.shape

        result = torch.sum((torch.sin(10 * torch.log(x))) / D, dim = 1)
        return -result



class F8_torch(MMO_Basic_Problem_torch): # modified_rastrigin_all
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F8_torch, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        return 'modified_rastrigin_all'+'_D'+str(self.dim)

    def func(self, x):
        if x is None:
            return None

        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        n, D = x.shape
    
        if D == 2:
            k = [3, 4]
        elif D == 8:
            k = [1, 2, 1, 2, 1, 3, 1, 4]
        elif D == 16:
            k = [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 4]

        result = torch.sum(10 + 9 * torch.cos(torch.tensor(2 * np.pi * np.array(k), device = x.device)[None, :] * x), dim=1)
        return -(-result)


class F9_torch(CFunction): # CF1
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F9_torch, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes, 6)

        # Initialize data for composition
        self._CFunction__sigma_ = np.ones(self._CFunction__nofunc_)
        self._CFunction__bias_ = np.zeros(self._CFunction__nofunc_)
        self._CFunction__weight_ = np.zeros(self._CFunction__nofunc_)
        self._CFunction__lambda_ = np.array([1.0, 1.0, 8.0, 8.0, 1.0 / 5.0, 1.0 / 5.0])

        # Load optima
        o = np.loadtxt(path.join(path.dirname(__file__), 'MMO') + "/optima.dat")
        if o.shape[1] >= dim:
            self._CFunction__O_ = o[: self._CFunction__nofunc_, :dim]
        else:  # randomly initialize
            self._CFunction__O_ = self.lb + (
                self.ub - self.lb
            ) * np.random.rand((self._CFunction__nofunc_, dim))

        # M_: Identity matrices
        self._CFunction__M_ = [np.eye(dim)] * self._CFunction__nofunc_

        # Initialize functions of the composition
        self._CFunction__function_ = {
            0: FGrienwank,
            1: FGrienwank,
            2: FWeierstrass,
            3: FWeierstrass,
            4: FSphere,
            5: FSphere,
        }

        # Calculate fmaxi
        self._CFunction__calculate_fmaxi()

    def __str__(self):
        return 'CF1'+'_D'+str(self.dim)

    def func(self, x):

        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        return -self._CFunction__evaluate_inner_(x)


class F10_torch(CFunction): # CF2
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F10_torch, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes, 8)

        # Initialize data for composition
        self._CFunction__sigma_ = np.ones(self._CFunction__nofunc_)
        self._CFunction__bias_ = np.zeros(self._CFunction__nofunc_)
        self._CFunction__weight_ = np.zeros(self._CFunction__nofunc_)
        self._CFunction__lambda_ = np.array(
            [1.0, 1.0, 10.0, 10.0, 1.0 / 10.0, 1.0 / 10.0, 1.0 / 7.0, 1.0 / 7.0]
        )

        # Load optima
        o = np.loadtxt(path.join(path.dirname(__file__), 'MMO') + "/optima.dat")
        if o.shape[1] >= dim:
            self._CFunction__O_ = o[: self._CFunction__nofunc_, :dim]
        else:  # randomly initialize
            self._CFunction__O_ = self.lb + (
                self.ub - self.lb
            ) * np.random.rand((self._CFunction__nofunc_, dim))

        # M_: Identity matrices
        self._CFunction__M_ = [np.eye(dim)] * self._CFunction__nofunc_

        # Initialize functions of the composition
        self._CFunction__function_ = {
            0: FRastrigin,
            1: FRastrigin,
            2: FWeierstrass,
            3: FWeierstrass,
            4: FGrienwank,
            5: FGrienwank,
            6: FSphere,
            7: FSphere,
        }

        # Calculate fmaxi
        self._CFunction__calculate_fmaxi()

    def __str__(self):
        return 'CF2'+'_D'+str(self.dim)

    def func(self, x):
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        return -self._CFunction__evaluate_inner_(x)


class F11_torch(CFunction): # CF3
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F11_torch, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes, 6)

        # Initialize data for composition
        self._CFunction__sigma_ = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
        self._CFunction__bias_ = np.zeros(self._CFunction__nofunc_)
        self._CFunction__weight_ = np.zeros(self._CFunction__nofunc_)
        self._CFunction__lambda_ = np.array([1.0 / 4.0, 1.0 / 10.0, 2.0, 1.0, 2.0, 5.0])

        # Load optima
        o = np.loadtxt(path.join(path.dirname(__file__), 'MMO') + "/optima.dat")
        if o.shape[1] >= dim:
            self._CFunction__O_ = o[: self._CFunction__nofunc_, :dim]
        else:  # randomly initialize
            self._CFunction__O_ = self.lb + (
                self.ub - self.lb
            ) * np.random.rand((self._CFunction__nofunc_, dim))

        # Load M_: Rotation matrices
        if dim == 2 or dim == 3 or dim == 5 or dim == 10 or dim == 20:
            fname = path.join(path.dirname(__file__), 'MMO') + "/CF3_M_D" + str(dim) + ".dat"
            self._CFunction__load_rotmat(fname)
        else:
            # M_ Identity matrices
            self._CFunction__M_ = [np.eye(dim)] * self._CFunction__nofunc_

        # Initialize functions of the composition
        self._CFunction__function_ = {
            0: FEF8F2,
            1: FEF8F2,
            2: FWeierstrass,
            3: FWeierstrass,
            4: FGrienwank,
            5: FGrienwank,
        }

        # Calculate fmaxi
        self._CFunction__calculate_fmaxi()

    def __str__(self):
        return 'CF3'+'_D'+str(self.dim)

    def func(self, x):

        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        return -self._CFunction__evaluate_inner_(x)


class F12_torch(CFunction): # CF4
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F12_torch, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes, 8)

        # Initialize data for composition
        self._CFunction__sigma_ = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        self._CFunction__bias_ = np.zeros(self._CFunction__nofunc_)
        self._CFunction__weight_ = np.zeros(self._CFunction__nofunc_)
        self._CFunction__lambda_ = np.array(
            [4.0, 1.0, 4.0, 1.0, 1.0 / 10.0, 1.0 / 5.0, 1.0 / 10.0, 1.0 / 40.0]
        )

        # Load optima
        o = np.loadtxt(path.join(path.dirname(__file__), 'MMO') + "/optima.dat")
        if o.shape[1] >= dim:
            self._CFunction__O_ = o[: self._CFunction__nofunc_, :dim]
        else:  # randomly initialize
            self._CFunction__O_ = self.lb + (
                self.ub - self.lb
            ) * np.random.rand((self._CFunction__nofunc_, dim))

        # Load M_: Rotation matrices
        if dim == 2 or dim == 3 or dim == 5 or dim == 10 or dim == 20:
            fname = path.join(path.dirname(__file__), 'MMO') + "/CF4_M_D" + str(dim) + ".dat"
            self._CFunction__load_rotmat(fname)
        else:
            # M_ Identity matrices 
            self._CFunction__M_ = [np.eye(dim)] * self._CFunction__nofunc_

        # Initialize functions of the composition
        self._CFunction__function_ = {
            0: FRastrigin,
            1: FRastrigin,
            2: FEF8F2,
            3: FEF8F2,
            4: FWeierstrass,
            5: FWeierstrass,
            6: FGrienwank,
            7: FGrienwank,
        }

        # Calculate fmaxi
        self._CFunction__calculate_fmaxi()

    def __str__(self):
        return 'CF4'+'_D'+str(self.dim)

    def func(self, x):
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        return -self._CFunction__evaluate_inner_(x)

