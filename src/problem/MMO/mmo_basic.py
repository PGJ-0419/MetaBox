from problem.basic_problem import Basic_Problem
import numpy as np
import math
MINMAX = -1 

class MMO_Basic_Problem(Basic_Problem):
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.FES = 0
        self.optimum = fopt
        self.rho = rho
        self.nopt = nopt
        self.maxfes = maxfes
    
    def func(self, x):
        raise NotImplementedError

    def how_many_goptima(self, pop, accuracy):
        NP, D = pop.shape[0], pop.shape[1]
        fits = self.eval(pop)
        order = np.argsort(fits)
        sorted_pop = pop[order, :]
        spopfits = fits[order]
        seeds_idx = self.__find_seeds_indices(sorted_pop, self.rho)
        count = 0
        goidx = []
        for idx in seeds_idx:
            seed_fitness = spopfits[idx]
            if math.fabs(seed_fitness - self.optimum) <= accuracy:
                count = count + 1
                goidx.append(idx)
            if count == self.nopt:
                break
        seeds = sorted_pop[goidx]
        return count, seeds


    def __find_seeds_indices(self, sorted_pop, radius):
        seeds = []
        seeds_idx = []
        for i, x in enumerate(sorted_pop):
            found = False
            for j, sx in enumerate(seeds):
                dist = math.sqrt(sum((x - sx) ** 2))
                if dist <= radius:
                    found = True
                    break
            if not found:
                seeds.append(x)
                seeds_idx.append(i)
        return seeds_idx


class CFunction(MMO_Basic_Problem):
    # Abstract composition function
    __nofunc_ = -1
    __C_ = 2000.0
    __lambda_ = None
    __sigma_ = None
    __bias_ = None
    __O_ = None
    __M_ = None
    __weight_ = None
    __fi_ = None
    __z_ = None
    __f_bias_ = 0
    __fmaxi_ = None
    __tmpx_ = None
    __function_ = None

    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes, nofunc):
        super(CFunction, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)
        self.__nofunc_ = nofunc

    def func(self, x):
        raise NotImplementedError

    def __evaluate_inner_(self, x):
        if self.__function_ == None:
            raise NameError("Composition functions' dict is uninitialized")
        self.__fi_ = np.zeros((x.shape[0], self.__nofunc_))

        self.__calculate_weights(x)
        for i in range(self.__nofunc_):
            self.__transform_to_z(x, i)
            self.__fi_[:, i] = self.__function_[i](self.__z_)

        tmpsum = np.zeros((x.shape[0], self.__nofunc_))
        tmpsum = self.__weight_ * (
            self.__C_ * self.__fi_ / self.__fmaxi_[None, :] + self.__bias_[None, :]
        )
        return np.sum(tmpsum, axis = 1) * MINMAX + self.__f_bias_

    def __calculate_weights(self, x):
        self.__weight_ = np.zeros((x.shape[0], self.__nofunc_))
        for i in range(self.__nofunc_):
            mysum = np.sum((x - self.__O_[i]) ** 2, -1)
            self.__weight_[:, i] = np.exp(
                -mysum / (2.0 * self.dim * self.__sigma_[i] * self.__sigma_[i])
            )
        maxw = np.max(self.__weight_, -1)

        maxw10 = maxw ** 10
        mask = (self.__weight_ != maxw[:, None])
        self.__weight_[mask] = (self.__weight_ * (1.0 - maxw10[:, None]))[mask]

        mysum = np.sum(self.__weight_, -1)
        mask1 =  (mysum == 0.0)
        self.__weight_[mask1] = 1.0 / (1.0 * self.__nofunc_)
        mask2 = (mysum != 0.0)
        self.__weight_[mask2] = self.__weight_[mask2] / mysum[:, None][mask2]

    def __calculate_fmaxi(self):
        self.__fmaxi_ = np.zeros(self.__nofunc_)
        if self.__function_ == None:
            raise NameError("Composition functions' dict is uninitialized")

        x5 = 5 * np.ones(self.dim)

        for i in range(self.__nofunc_):
            self.__transform_to_z_noshift(x5[None, :], i)
            self.__fmaxi_[i] = self.__function_[i](self.__z_)[0]

    def __transform_to_z_noshift(self, x, index):
        tmpx = np.divide(x, self.__lambda_[index])
        self.__z_ = np.dot(tmpx, self.__M_[index])

    def __transform_to_z(self, x, index):
        tmpx = np.divide((x - self.__O_[index]), self.__lambda_[index])
        self.__z_ = np.dot(tmpx, self.__M_[index])

    def __load_rotmat(self, fname):
        self.__M_ = []

        with open(fname, "r") as f:
            tmp = np.zeros((self.dim, self.dim))
            cline = 0
            ctmp = 0
            for line in f:
                line = line.split()
                if line:
                    line = [float(i) for i in line]

                    if ctmp % self.dim == 0:
                        tmp = np.zeros((self.dim, self.dim))
                        ctmp = 0

                    tmp[ctmp] = line[: self.dim]
                    if cline >= self.__nofunc_ * self.dim - 1:
                        break
                    if cline % self.dim == 0:
                        self.__M_.append(tmp)
                    ctmp = ctmp + 1
                    cline = cline + 1



# Sphere function
def FSphere(x):
    return (x ** 2).sum(axis = 1)


# Rastrigin's function
def FRastrigin(x):
    return np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x) + 10, axis=1)


# Griewank's function
def FGrienwank(x):
    i = np.sqrt(np.arange(x.shape[1]) + 1.0)
    return np.sum(x ** 2, axis = 1) / 4000.0 - np.prod(np.cos(x / i[None, :]), axis = 1) + 1.0


# Weierstrass's function
def FWeierstrass(x):
    alpha = 0.5
    beta = 3.0
    kmax = 20
    D = x.shape[1]
    exprf = 0.0

    c1 = alpha ** np.arange(kmax + 1)
    c2 = 2.0 * np.pi * beta ** np.arange(kmax + 1)
    f = np.zeros(x.shape[0])
    c = -D * np.sum(c1 * np.cos(c2 * 0.5))

    for i in range(D):
        f += np.sum(c1[None, :] * np.cos(c2[None, :] * (x[:, i:i+1] + 0.5)), axis = 1)
    return f + c


def F8F2(x):
    f2 = 100.0 * (x[:, 0] ** 2 - x[:, 1]) ** 2 + (1.0 - x[:, 0]) ** 2
    return 1.0 + (f2 ** 2) / 4000.0 - np.cos(f2)


# FEF8F2 function
def FEF8F2(x):
    D = x.shape[1]
    f = np.zeros(x.shape[0])
    for i in range(D - 1):
        f += F8F2(x[:, [i, i + 1]] + 1)
    f += F8F2(x[:, [D - 1, 0]] + 1)
    return f












