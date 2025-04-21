import geatpy as ea
# import numpy as th
import torch as th
import math
from ...basic_problem import Basic_Problem_Torch


class UF1_Torch(Basic_Problem_Torch):
    def __init__(self):
        self.n_obj = 2
        self.n_var = 30
        self.lb = th.tensor([-1] * self.n_var)
        self.lb[0] = 0
        self.ub = th.tensor([1] * self.n_var)
        self.vtype = float

    def eval(self, x):  # 目标函数
        Vars = x  # 得到决策变量矩阵
        x1 = Vars[:, [0]]
        J1 = th.tensor(list(range(2, self.n_var, 2)))
        J2 = th.tensor(list(range(1, self.n_var, 2)))
        f1 = x1 + 2 * th.mean((Vars[:, J1] - th.sin(6 * math.pi * x1 + (J1 + 1) * math.pi / self.n_var)) ** 2, 1,
                              keepdims=True)
        f2 = 1 - th.sqrt(th.abs(x1)) + 2 * th.mean(
            (Vars[:, J2] - th.sin(6 * math.pi * x1 + (J2 + 1) * math.pi / self.n_var)) ** 2, 1, keepdims=True)
        ObjV = th.hstack([f1, f2])
        return ObjV

    def get_ref_set(self,n_ref_points=1000):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = n_ref_points
        ObjV1 = th.linspace(0, 1, N)
        ObjV2 = 1 - th.sqrt(ObjV1)
        referenceObjV = th.stack([ObjV1, ObjV2]).T
        return referenceObjV





class UF2_Torch(Basic_Problem_Torch):
    def __init__(self):
        self.n_obj = 2  # 初始化（目标维数）
        self.n_var = 30 #初始化（决策变量维数）
        self.lb = th.tensor([-1] * self.n_var)
        self.lb[0] = 0
        self.ub = th.tensor([1] * self.n_var)
        self.vtype = float

    def eval(self, x):  # 目标函数
        Vars = x  # 得到决策变量矩阵
        x1 = Vars[:, [0]]
        J1 = th.tensor(list(range(2, self.n_var, 2)))
        J2 = th.tensor(list(range(1, self.n_var, 2)))
        yJ1 = Vars[:, J1] - (
                    0.3 * x1 ** 2 * th.cos(24 * math.pi * x1 + 4 * (J1 + 1) * math.pi / self.n_var) + 0.6 * x1) * th.cos(
            6 * math.pi * x1 + (J1 + 1) * math.pi / self.n_var)
        yJ2 = Vars[:, J2] - (
                    0.3 * x1 ** 2 * th.cos(24 * math.pi * x1 + 4 * (J2 + 1) * math.pi / self.n_var) + 0.6 * x1) * th.sin(
            6 * math.pi * x1 + (J2 + 1) * math.pi / self.n_var)
        f1 = x1 + 2 * th.mean((yJ1) ** 2, 1, keepdims=True)
        f2 = 1 - th.sqrt(th.abs(x1)) + 2 * th.mean((yJ2) ** 2, 1, keepdims=True)
        ObjV = th.hstack([f1, f2])
        return ObjV

    def get_ref_set(self,n_ref_points=1000):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = n_ref_points
        ObjV1 = th.linspace(0, 1, N)
        ObjV2 = 1 - th.sqrt(ObjV1)
        referenceObjV = th.stack([ObjV1, ObjV2]).T
        return referenceObjV


class UF3_Torch(Basic_Problem_Torch):  # 继承Problem的父类
    def __init__(self):
        self.n_obj = 2  # 目标维数
        self.n_var = 30  # 决策变量维数
        self.lb = th.tensor([0]*self.n_var)
        self.ub = th.tensor([1]*self.n_var)
        self.vtype = float

    def eval(self, x):  # 目标函数
        Vars = x  # 决策变量矩阵
        x1 = Vars[:, [0]]
        J1 = th.tensor(list(range(2, self.n_var, 2)))
        J2 = th.tensor(list(range(1, self.n_var, 2)))
        J = th.arange(1, 31)
        J = J[None, :]
        y = Vars - x1 ** (0.5 * (1 + (3 * (J - 2) / (self.n_var - 2))))
        yJ1 = y[:, J1]
        yJ2 = y[:, J2]
        f1 = x1 + (2 / len(J1)) * (4 * th.sum(yJ1 ** 2, 1, keepdims=True) -
                                   2 * (th.prod(th.cos((20 * yJ1 * math.pi) / (th.sqrt(J1))), 1, keepdims=True)) + 2)
        f2 = 1 - th.sqrt(x1) + (2 / len(J2)) * (4 * th.sum(yJ2 ** 2, 1, keepdims=True) -
                                                2 * (th.prod(th.cos((20 * yJ2 * math.pi) / (th.sqrt(J2))), 1,
                                                             keepdims=True)) + 2)
        ObjV = th.hstack([f1, f2])
        return ObjV

    def get_ref_set(self,n_ref_points=1000):  # 理论最优值
        N = n_ref_points
        ObjV1 = th.linspace(0, 1, N)
        ObjV2 = 1 - th.sqrt(ObjV1)
        referenceObjV = th.stack([ObjV1, ObjV2]).T
        return referenceObjV


class UF4_Torch(Basic_Problem_Torch):
    def __init__(self):
        self.n_obj = 2  # 初始化（目标维数）
        self.n_var = 30  # 初始化Dim（决策变量维数）
        self.lb = th.tensor([-2]*self.n_var)
        self.lb[0] = 0
        self.ub = th.tensor([2]*self.n_var)
        self.ub[0] = 1
        self.vtype = float

    def eval(self, x):  # 目标函数
        Vars = x  # 决策变量矩阵
        x1 = Vars[:, [0]]
        J1 = th.tensor(list(range(2, self.n_var, 2)))
        J2 = th.tensor(list(range(1, self.n_var, 2)))
        J = th.arange(1, 31)
        J = J[None, :]
        y = Vars - th.sin(6 * math.pi * x1 + (J * math.pi) / self.n_var)
        hy = th.abs(y) / (1 + th.exp(2 * (th.abs(y))))
        hy1 = hy[:, J1]
        hy2 = hy[:, J2]
        f1 = x1 + 2 * th.mean(hy1, 1, keepdims=True)
        f2 = 1 - x1 ** 2 + 2 * th.mean(hy2, 1, keepdims=True)
        ObjV = th.hstack([f1, f2])
        return ObjV

    def get_ref_set(self,n_ref_points=1000):  # 理论最优值
        N = n_ref_points
        ObjV1 = th.linspace(0, 1, N)
        ObjV2 = 1 - ObjV1 ** 2
        referenceObjV = th.stack([ObjV1, ObjV2]).T
        return referenceObjV


class UF5_Torch(Basic_Problem_Torch):
    def __init__(self):
        self.n_obj = 2  # 初始化M（目标维数）
        self.n_var = 30  # 初始化Dim（决策变量维数）
        self.lb = th.tensor([-1]*self.n_var)
        self.lb[0] = 0
        self.ub = th.tensor([1]*self.n_var)
        super().__init__(n_var=self.n_var, n_obj=self.n_obj, lb=self.lb, ub=self.ub, vtype=float)

    def eval(self, x):  # 目标函数
        Vars = x  # 决策变量矩阵
        x1 = Vars[:, [0]]
        J1 = th.tensor(list(range(2, self.n_var, 2)))
        J2 = th.tensor(list(range(1, self.n_var, 2)))
        J = th.arange(1, 31)
        J = J[None, :]
        y = Vars - th.sin(6 * math.pi * x1 + (J * math.pi) / self.n_var)
        hy = 2 * y ** 2 - th.cos(4 * math.pi * y) + 1
        # print(hy)
        hy1 = hy[:, J1]
        hy2 = hy[:, J2]
        f1 = x1 + (1 / 20 + 0.1) * th.abs(th.sin(20 * math.pi * x1)) + 2 * (th.mean(hy1, 1, keepdims=True))
        f2 = 1 - x1 + (1 / 20 + 0.1) * th.abs(th.sin(20 * math.pi * x1)) + 2 * (th.mean(hy2, 1, keepdims=True))
        ObjV = th.hstack([f1, f2])
        return ObjV

    def get_ref_set(self,n_ref_points=1000):  # 理论最优值
        N = n_ref_points  # 生成10000个参考点
        ObjV1 = th.linspace(0, 1, N)
        ObjV2 = 1 - ObjV1
        referenceObjV = th.stack([ObjV1, ObjV2]).T
        return referenceObjV


class UF6_Torch(Basic_Problem_Torch):  # 继承Problem父类
    def __init__(self):

        self.n_obj = 2  # 初始化M（目标维数）
        self.n_var = 30  # 初始化Dim（决策变量维数）
        self.lb = th.tensor([-1]*self.n_var)
        self.lb[0] = 0
        self.ub = th.tensor([1]*self.n_var)
        super().__init__(n_var=self.n_var, n_obj=self.n_obj, lb=self.lb, ub=self.ub, vtype=float)

    def eval(self, x):  # 目标函数
        Vars = x # 决策变量矩阵
        x1 = Vars[:, [0]]
        J1 = th.tensor(list(range(2, self.n_var, 2)))
        J2 = th.tensor(list(range(1, self.n_var, 2)))
        J = th.arange(1, 31)
        J = J[None, :]
        y = Vars - th.sin(6 * math.pi * x1 + (J * math.pi) / self.n_var)
        yJ1 = y[:, J1]
        yJ2 = y[:, J2]
        # hy    = 2*y**2 - th.cos(4*math.pi*y) + 1
        # print(hy)
        # hy1   = hy[:, J1]
        # hy2   = hy[:, J2]
        f1 = x1 + th.maximum(th.tensor(0), 2 * (1 / 4 + 0.1) * th.sin(4 * math.pi * x1)) + \
             (2 / len(J1)) * (4 * th.sum(yJ1 ** 2, 1, keepdims=True) - \
                              2 * (th.prod(th.cos((20 * yJ1 * math.pi) / (th.sqrt(J1))), 1, keepdims=True)) + 2)
        f2 = 1 - x1 + th.maximum(th.tensor(0), 2 * (1 / 4 + 0.1) * th.sin(4 * math.pi * x1)) + \
             (2 / len(J2)) * (4 * th.sum(yJ2 ** 2, 1, keepdims=True) - \
                              2 * (th.prod(th.cos((20 * yJ2 * math.pi) / (th.sqrt(J2))), 1, keepdims=True)) + 2)
        ObjV = th.hstack([f1, f2])
        return ObjV

    def get_ref_set(self,n_ref_points=1000):  # 理论最优值
        N = n_ref_points
        ObjV1 = th.linspace(0, 1, N)
        idx = ((ObjV1 > 0) & (ObjV1 < 1 / 4)) | ((ObjV1 > 1 / 2) & (ObjV1 < 3 / 4))
        ObjV1 = ObjV1[~idx]
        ObjV2 = 1 - ObjV1
        referenceObjV = th.stack([ObjV1, ObjV2]).T
        return referenceObjV


class UF7_Torch(Basic_Problem_Torch):  # 继承Problem父类
    def __init__(self):

        self.n_obj = 2  # 初始化M（目标维数）
        self.n_var= 30  # 初始化Dim（决策变量维数）
        self.lb = th.tensor([-1]*self.n_var)
        self.lb[0] = 0
        self.ub = th.tensor([1]*self.n_var)
        super().__init__(n_var=self.n_var, n_obj=self.n_obj, lb=self.lb, ub=self.ub, vtype=float)

    def eval(self, x):  # 目标函数
        Vars = x  # 决策变量矩阵
        x1 = Vars[:, [0]]
        J1 = th.tensor(list(range(2, self.n_var, 2)))
        J2 = th.tensor(list(range(1, self.n_var, 2)))
        J = th.arange(1, 31)
        J = J[None, :]
        y = Vars - th.sin(6 * math.pi * x1 + (J * math.pi) / self.n_var)
        yJ1 = y[:, J1]
        yJ2 = y[:, J2]
        f1 = x1 ** 0.2 + 2 * th.mean(yJ1 ** 2, 1, keepdims=True)
        f2 = 1 - x1 ** 0.2 + 2 * th.mean(yJ2 ** 2, 1, keepdims=True)
        ObjV = th.hstack([f1, f2])
        return ObjV

    def get_ref_set(self,n_ref_points=1000):  # 理论最优值
        N = n_ref_points
        ObjV1 = th.linspace(0, 1, N)
        ObjV2 = 1 - ObjV1
        referenceObjV = th.stack([ObjV1, ObjV2]).T
        return referenceObjV


class UF8_Torch(Basic_Problem_Torch):  # 继承Problem父类
    def __init__(self):
        self.n_obj = 3  # 初始化M（目标维数）
        self.n_var = 30  # 初始化Dim（决策变量维数）
        self.lb = th.tensor([0]*2+[-2]*(self.n_var-2))
        self.ub = th.tensor([1]*2+[2]*(self.n_var-2))
        super().__init__(n_var=self.n_var, n_obj=self.n_obj, lb=self.lb, ub=self.ub, vtype=float)

    def eval(self, x):  # 目标函数
        Vars = x  # 决策变量矩阵
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        J1 = th.tensor(list(range(3, self.n_var, 3)))
        J2 = th.tensor(list(range(4, self.n_var, 3)))
        J3 = th.tensor(list(range(2, self.n_var, 3)))
        # print(J1, J2, J3)
        J = th.arange(1, 31)
        J = J[None, :]
        # f    = 2*th.mean((Vars-2*x2*th.sin(2*math.pi*x1+J*math.pi/self.Dim))**2 ,1,keepdims = True)
        f = (Vars - 2 * x2 * th.sin(2 * math.pi * x1 + J * math.pi / self.n_var)) ** 2
        # print(f.shape)
        f1 = th.cos(0.5 * x1 * math.pi) * th.cos(0.5 * x2 * math.pi) + 2 * th.mean(f[:, J1], 1, keepdims=True)
        f2 = th.cos(0.5 * x1 * math.pi) * th.sin(0.5 * x2 * math.pi) + 2 * th.mean(f[:, J2], 1, keepdims=True)
        f3 = th.sin(0.5 * x1 * math.pi) + 2 * th.mean(f[:, J3], 1, keepdims=True)
        ObjV = th.hstack([f1, f2, f3])
        return ObjV

    def get_ref_set(self,n_ref_points=1000):  # 理论最优值
        N = n_ref_points
        ObjV, N = ea.crtup(self.n_obj, N)  # ObjV.shape=N,3
        ObjV = th.tensor(ObjV)
        ObjV = ObjV / th.sqrt(th.sum(ObjV ** 2, 1, keepdims=True))
        referenceObjV = ObjV
        return referenceObjV


class UF9_Torch(Basic_Problem_Torch):  # 继承Problem父类
    def __init__(self):
        self.n_obj = 3  # 初始化M（目标维数）
        self.n_var = 30  # 初始化Dim（决策变量维数）
        self.lb = th.tensor([0]*2+[-2]*(self.n_var-2))
        self.ub = th.tensor([1]*2+[2]*(self.n_var-2))
        # 调用父类构造方法完成实例化
        super().__init__(n_var=self.n_var, n_obj=self.n_obj, lb=self.lb, ub=self.ub, vtype=float)

    def eval(self, x):  # 目标函数
        Vars = x  # 决策变量矩阵
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        J1 = th.tensor(list(range(3, self.n_var, 3)))
        J2 = th.tensor(list(range(4, self.n_var, 3)))
        J3 = th.tensor(list(range(2, self.n_var, 3)))
        # print(J1, J2, J3)
        J = th.arange(1, 31)
        J = J[None, :]
        f = (Vars - 2 * x2 * th.sin(2 * math.pi * x1 + J * math.pi / self.n_var)) ** 2
        f1 = 0.5 * (th.maximum(th.tensor(0), (1.1 * (1 - 4 * (2 * x1 - 1) ** 2))) + 2 * x1) * x2 + 2 * th.mean(f[:, J1], 1,
                                                                                                    keepdims=True)
        f2 = 0.5 * (th.maximum(th.tensor(0), (1.1 * (1 - 4 * (2 * x1 - 1) ** 2))) - 2 * x1 + 2) * x2 + 2 * th.mean(f[:, J2], 1,
                                                                                                        keepdims=True)
        f3 = 1 - x2 + 2 * th.mean(f[:, J3], 1, keepdims=True)
        ObjV = th.hstack([f1, f2, f3])
        return ObjV

    def get_ref_set(self,n_ref_points=1000):  # 理论最优值
        N = n_ref_points  # 生成10000个参考点
        ObjV, N = ea.crtup(self.n_obj, N)  # ObjV.shape=N,3
        ObjV = th.tensor(ObjV)
        idx = (ObjV[:, 0] > (1 - ObjV[:, 2]) / 4) & (ObjV[:, 0] < (1 - ObjV[:, 2]) * 3 / 4)
        referenceObjV = ObjV[~idx]
        return referenceObjV

class UF10_Torch(Basic_Problem_Torch):  # 继承Problem父类
    def __init__(self):
        self.n_obj = 3  # 初始化M（目标维数）
        self.n_var = 30  # 初始化Dim（决策变量维数）
        self.lb = th.tensor([0]*2+[-2]*(self.n_var-2))
        self.ub = th.tensor([1] * 2 + [2] * (self.n_var - 2))
        super().__init__(n_var=self.n_var, n_obj=self.n_obj, lb=self.lb, ub=self.ub, vtype=float)

    def eval(self, x):  # 目标函数
        Vars = x  # 决策变量矩阵
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        J1 = th.tensor(list(range(3, self.n_var, 3)))
        J2 = th.tensor(list(range(4, self.n_var, 3)))
        J3 = th.tensor(list(range(2, self.n_var, 3)))
        J = th.arange(1, 31)
        J = J[None, :]
        y = Vars - 2 * x2 * th.sin(2 * math.pi * x1 + (J * math.pi) / self.n_var)
        f = 4 * y ** 2 - th.cos(8 * math.pi * y) + 1
        f1 = th.cos(0.5 * x1 * math.pi) * th.cos(0.5 * x2 * math.pi) + 2 * th.mean(f[:, J1], 1, keepdims=True)
        f2 = th.cos(0.5 * x1 * math.pi) * th.sin(0.5 * x2 * math.pi) + 2 * th.mean(f[:, J2], 1, keepdims=True)
        f3 = th.sin(0.5 * x1 * math.pi) + 2 * th.mean(f[:, J3], 1, keepdims=True)
        ObjV = th.hstack([f1, f2, f3])
        return ObjV
    def get_ref_set(self,n_ref_points=1000):  # 理论最优值
        N = n_ref_points  # 生成10000个参考点
        ObjV, N = ea.crtup(self.n_obj, N)  # ObjV.shape=N,3
        ObjV = th.tensor(ObjV)
        ObjV = ObjV / th.sqrt(th.sum(ObjV ** 2, 1, keepdims=True))
        referenceObjV = ObjV
        return referenceObjV

if __name__ == '__main__':
    uf1= UF1_Torch()
    uf2 = UF2_Torch()
    uf3 = UF3_Torch()
    uf4 = UF4_Torch()
    uf5 = UF5_Torch()
    uf6 = UF6_Torch()
    uf7 = UF7_Torch()
    uf8 = UF8_Torch()
    uf9 = UF9_Torch()
    uf10 = UF10_Torch()
    x = th.ones(100, 30)
    print(uf1.eval(x))
    print(uf2.eval(x))
    print(uf3.eval(x))
    print(uf4.eval(x))
    print(uf5.eval(x))
    print(uf6.eval(x))
    print(uf7.eval(x))
    print(uf8.eval(x))
    print(uf9.eval(x))
    print(uf10.eval(x))
    s1 = uf1.get_ref_set()
    s2 = uf2.get_ref_set()
    s3 = uf3.get_ref_set()
    s4 = uf4.get_ref_set()
    s5 = uf5.get_ref_set()
    s6 = uf6.get_ref_set()
    s7 = uf7.get_ref_set()
    s8 = uf8.get_ref_set()
    s9 = uf9.get_ref_set()
    s10 = uf10.get_ref_set()
