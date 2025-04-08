from .mmo_basic import *
import numpy as np
import math
from os import path

class F1(MMO_Basic_Problem): # five_uneven_peak_trap
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F1, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        return 'five_uneven_peak_trap'+'_D'+str(self.dim)

    def func(self, x):
        
        if x is None:
            return None
        
        x = np.asarray(x)
        assert x.shape[1] == self.dim

        self.FES += x.shape[0]
        x = x[:, 0]
        result = np.zeros_like(x)
        
        mask1 = (x >= 0) & (x < 2.50)
        mask2 = (x >= 2.5) & (x < 5)
        mask3 = (x >= 5.0) & (x < 7.5)
        mask4 = (x >= 7.5) & (x < 12.5)
        mask5 = (x >= 12.5) & (x < 17.5)
        mask6 = (x >= 17.5) & (x < 22.5)
        mask7 = (x >= 22.5) & (x < 27.5)
        mask8 = (x >= 27.5) & (x <= 30)
        
        result[mask1] = 80 * (2.5 - x[mask1])
        result[mask2] = 64 * (x[mask2] - 2.5)
        result[mask3] = 64 * (7.5 - x[mask3])
        result[mask4] = 28 * (x[mask4] - 7.5)
        result[mask5] = 28 * (17.5 - x[mask5])
        result[mask6] = 32 * (x[mask6] - 17.5)
        result[mask7] = 32 * (27.5 - x[mask7])
        result[mask8] = 80 * (x[mask8] - 27.5)
        
        return -result

class F2(MMO_Basic_Problem): # equal_maxima
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F2, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        return 'equal_maxima'+'_D'+str(self.dim)

    def func(self, x):
        if x is None:
            return None
        x = np.asarray(x)
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        return -np.sin(5.0 * np.pi * x[:, 0]) ** 6

class F3(MMO_Basic_Problem): # uneven_decreasing_maxima
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F3, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        return 'uneven_decreasing_maxima'+'_D'+str(self.dim)

    def func(self, x):
        if x is None:
            return None
        x = np.asarray(x)
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        return -(
            np.exp(-2.0 * np.log(2) * ((x[:, 0] - 0.08) / 0.854) ** 2)
            * (np.sin(5 * np.pi * (x[:, 0] ** 0.75 - 0.05))) ** 6
        )
        
class F4(MMO_Basic_Problem): # himmelblau
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F4, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        return 'himmelblau'+'_D'+str(self.dim)

    def func(self, x):
        if x is None:
            return None
        x = np.asarray(x)
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        result = 200 - (x[:, 0] ** 2 + x[:, 1] - 11) ** 2 - (x[:, 0] + x[:, 1] ** 2 - 7) ** 2
        return -result


class F5(MMO_Basic_Problem): # six_hump_camel_back
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F5, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        return 'six_hump_camel_back'+'_D'+str(self.dim)

    def func(self, x):
        if x is None:
            return None
        x = np.asarray(x)
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        x2 = x[:, 0] ** 2
        x4 = x[:, 0] ** 4
        y2 = x[:, 1] ** 2
        expr1 = (4.0 - 2.1 * x2 + x4 / 3.0) * x2
        expr2 = x[:, 0] * x[:, 1]
        expr3 = (4.0 * y2 - 4.0) * y2
        return -(-1.0 * (expr1 + expr2 + expr3))


class F6(MMO_Basic_Problem): # shubert
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F6, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        return 'shubert'+'_D'+str(self.dim)

    def func(self, x):
        if x is None:
            return None
        x = np.asarray(x)
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        n, D = x.shape
        result = np.ones(n)
        soma = np.zeros((n, D))
        
        for j in range(1, 6):
            soma = soma + (j * np.cos((j + 1) * x + j))
        result = np.prod(soma, axis = 1)

        return -(-result)



class F7(MMO_Basic_Problem): # vincent
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F7, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        return 'vincent'+'_D'+str(self.dim)

    def func(self, x):
        if x is None:
            return None
        x = np.asarray(x)
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        n, D = x.shape
        result = np.zeros(n)

        result = np.sum((np.sin(10 * np.log(x))) / D, axis = 1)
        return -result



class F8(MMO_Basic_Problem): # modified_rastrigin_all
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F8, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        return 'modified_rastrigin_all'+'_D'+str(self.dim)

    def func(self, x):
        if x is None:
            return None
        x = np.asarray(x)
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        n, D = x.shape
        
        if D == 2:
            k = [3, 4]
        elif D == 8:
            k = [1, 2, 1, 2, 1, 3, 1, 4]
        elif D == 16:
            k = [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 4]

        result = np.sum(10 + 9 * np.cos(2 * math.pi * np.array(k)[None, :] * x), axis=1)
        return -(-result)


class F9(CFunction): # CF1
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F9, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes, 6)

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
        x = np.asarray(x)
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        return -self._CFunction__evaluate_inner_(x)


class F10(CFunction): # CF2
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F10, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes, 8)

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
        x = np.asarray(x)
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        return -self._CFunction__evaluate_inner_(x)


class F11(CFunction): # CF3
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F11, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes, 6)

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
        x = np.asarray(x)
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        return -self._CFunction__evaluate_inner_(x)


class F12(CFunction): # CF4
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F12, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes, 8)

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
        x = np.asarray(x)
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        return -self._CFunction__evaluate_inner_(x)

