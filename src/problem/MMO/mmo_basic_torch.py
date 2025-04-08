from problem.basic_problem import Basic_Problem
import numpy as np
import math
import torch
import time
MINMAX = -1

class MMO_Basic_Problem_torch(Basic_Problem):
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.FES = 0
        self.optimum = fopt
        self.rho = rho
        self.nopt = nopt
        self.maxfes = maxfes

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
        raise NotImplementedError

    def how_many_goptima_torch(self, pop, accuracy):
        NP, D = pop.shape[0], pop.shape[1]

        fits = self.eval(torch.tensor(pop)).data.numpy()

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









class CFunction(MMO_Basic_Problem_torch):
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
        self.__fi_ = torch.tensor([], dtype = torch.float64, device = x.device)

        self.__calculate_weights(x)
        for i in range(self.__nofunc_):
            self.__transform_to_z(x, i)
            self.__fi_ = torch.cat((self.__fi_, self.__function_[i](self.__z_).reshape(x.shape[0], 1)), dim = 1)

        tmpsum = self.__weight_ * (
            self.__C_ * self.__fi_ / self.__fmaxi_[None, :].to(x.device) + torch.tensor(self.__bias_, device = x.device)[None, :]
        )
        return torch.sum(tmpsum, dim = 1) * MINMAX + self.__f_bias_

    def __calculate_weights(self, x):
        self.__weight_ = torch.tensor([], dtype = torch.float64, device = x.device)
        for i in range(self.__nofunc_):
            mysum = torch.sum((x - torch.tensor(self.__O_[i], device = x.device)) ** 2, -1)
            self.__weight_ = torch.cat((self.__weight_, torch.exp(-mysum / (2.0 * self.dim * self.__sigma_[i] * self.__sigma_[i])).reshape(x.shape[0], 1)), dim = 1)
        maxw = torch.max(self.__weight_, -1).values

        maxw10 = maxw ** 10
        mask = (self.__weight_ != maxw[:, None])
        mask_trans = (~mask)
        mask_content = (self.__weight_ * (1.0 - maxw10[:, None]))
        self.__weight_ = mask * mask_content + mask_trans * self.__weight_

        mysum = torch.sum(self.__weight_, -1)
        mask1 =  (mysum == 0.0)
        mask2 = (mysum != 0.0)
        sum_content1 = torch.ones_like(mysum, dtype = torch.float64, device = x.device)
        mysum = mask1 * sum_content1 + mask2 * mysum
        content_mask1 = torch.ones_like(self.__weight_, device = x.device, dtype = torch.float64) * (1.0 / (1.0 * self.__nofunc_))
        content_mask2 = self.__weight_ / mysum[:, None]
        self.__weight_ = mask1[:, None] * content_mask1 + mask2[:, None] * content_mask2

    def __calculate_fmaxi(self):
        self.__fmaxi_ = torch.zeros(self.__nofunc_, dtype=torch.float64)
        if self.__function_ == None:
            raise NameError("Composition functions' dict is uninitialized")

        x5 = 5 * torch.ones(self.dim, dtype = torch.float64)

        for i in range(self.__nofunc_):
            self.__transform_to_z_noshift(x5[None, :], i)
            self.__fmaxi_[i] = self.__function_[i](self.__z_)[0]

    def __transform_to_z_noshift(self, x, index):
        tmpx = torch.divide(x, torch.tensor(self.__lambda_[index], device = x.device))
        self.__z_ = torch.matmul(tmpx, torch.tensor(self.__M_[index], device = x.device))

    def __transform_to_z(self, x, index):
        tmpx = torch.divide((x - torch.tensor(self.__O_[index], device = x.device)), torch.tensor(self.__lambda_[index], device = x.device))
        self.__z_ = torch.matmul(tmpx, torch.tensor(self.__M_[index], device = x.device))

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
    return (x ** 2).sum(dim = 1)


# Rastrigin's function
def FRastrigin(x):
    return torch.sum(x ** 2 - 10.0 * torch.cos(2.0 * np.pi * x) + 10, dim=1)


# Griewank's function
def FGrienwank(x):
    i = torch.sqrt(torch.arange(x.shape[1], dtype=torch.float64, device = x.device) + 1.0)
    return torch.sum(x ** 2, dim = 1) / 4000.0 - torch.prod(torch.cos(x / i[None, :]), dim = 1) + 1.0


# Weierstrass's function
def FWeierstrass(x):
    alpha = 0.5
    beta = 3.0
    kmax = 20
    D = x.shape[1]
    exprf = 0.0

    c1 = alpha ** torch.arange(kmax + 1, dtype=torch.float64, device = x.device)
    c2 = 2.0 * np.pi * beta ** np.arange(kmax + 1)
    c2 = torch.tensor(c2, device = x.device)
    f = torch.zeros(x.shape[0], dtype=torch.float64,device = x.device)
    c = -D * torch.sum(c1 * torch.cos(c2 * 0.5))

    for i in range(D):
        f = f + torch.sum(c1[None, :] * torch.cos(c2[None, :] * (x[:, i:i+1] + 0.5)), dim = 1)
    return f + c


def F8F2(x):
    f2 = 100.0 * (x[:, 0] ** 2 - x[:, 1]) ** 2 + (1.0 - x[:, 0]) ** 2
    return 1.0 + (f2 ** 2) / 4000.0 - torch.cos(f2)


# FEF8F2 function
def FEF8F2(x):
    D = x.shape[1]
    f = torch.zeros(x.shape[0], device = x.device, dtype=torch.float64)
    for i in range(D - 1):
        f = f + F8F2(x[:, [i, i + 1]] + 1)
    f = f + F8F2(x[:, [D - 1, 0]] + 1)
    return f





