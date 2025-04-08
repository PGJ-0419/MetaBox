import geatpy as ea
import numpy as np
from problem.MOO.moo_basic import MOO_Basic_Problem


class UF1(MOO_Basic_Problem):
    def __init__(self):
        self.n_obj = 2
        self.n_var = 30
        self.lb = np.array([-1] * self.n_var)
        self.lb[0] = 0
        self.ub = np.array([1] * self.n_var)
        super().__init__(n_var = self.n_var,n_obj = self.n_obj,lb= self.lb,ub = self.ub,vtype=float)

    def eval(self, x):  # 目标函数
        Vars = x  # 得到决策变量矩阵
        x1 = Vars[:, [0]]
        J1 = np.array(list(range(2, self.n_var, 2)))
        J2 = np.array(list(range(1, self.n_var, 2)))
        f1 = x1 + 2 * np.mean((Vars[:, J1] - np.sin(6 * np.pi * x1 + (J1 + 1) * np.pi / self.n_var)) ** 2, 1,
                              keepdims=True)
        f2 = 1 - np.sqrt(np.abs(x1)) + 2 * np.mean(
            (Vars[:, J2] - np.sin(6 * np.pi * x1 + (J2 + 1) * np.pi / self.n_var)) ** 2, 1, keepdims=True)
        ObjV = np.hstack([f1, f2])
        return ObjV

    def get_ref_set(self,n_ref_points=1000):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = n_ref_points
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - np.sqrt(ObjV1)
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV





class UF2(Basic_Problem):
    def __init__(self):
        self.n_obj = 2  # 初始化（目标维数）
        self.n_var = 30 #初始化（决策变量维数）
        self.lb = np.array([-1] * self.n_var)
        self.lb[0] = 0
        self.ub = np.array([1] * self.n_var)
        super().__init__(n_var = self.n_var,n_obj = self.n_obj,lb= self.lb,ub = self.ub,vtype=float)

    def eval(self, x):  # 目标函数
        Vars = x  # 得到决策变量矩阵
        x1 = Vars[:, [0]]
        J1 = np.array(list(range(2, self.n_var, 2)))
        J2 = np.array(list(range(1, self.n_var, 2)))
        yJ1 = Vars[:, J1] - (
                    0.3 * x1 ** 2 * np.cos(24 * np.pi * x1 + 4 * (J1 + 1) * np.pi / self.n_var) + 0.6 * x1) * np.cos(
            6 * np.pi * x1 + (J1 + 1) * np.pi / self.n_var)
        yJ2 = Vars[:, J2] - (
                    0.3 * x1 ** 2 * np.cos(24 * np.pi * x1 + 4 * (J2 + 1) * np.pi / self.n_var) + 0.6 * x1) * np.sin(
            6 * np.pi * x1 + (J2 + 1) * np.pi / self.n_var)
        f1 = x1 + 2 * np.mean((yJ1) ** 2, 1, keepdims=True)
        f2 = 1 - np.sqrt(np.abs(x1)) + 2 * np.mean((yJ2) ** 2, 1, keepdims=True)
        ObjV = np.hstack([f1, f2])
        return ObjV

    def get_ref_set(self,n_ref_points=1000):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = n_ref_points
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - np.sqrt(ObjV1)
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV


class UF3(Basic_Problem):  # 继承Problem的父类
    def __init__(self):
        self.n_obj = 2  # 目标维数
        self.n_var = 30  # 决策变量维数
        self.lb = np.array([0]*self.n_var)
        self.ub = np.array([1]*self.n_var)
        super().__init__(n_var = self.n_var,n_obj = self.n_obj,lb= self.lb,ub = self.ub,vtype=float)

    def eval(self, x):  # 目标函数
        Vars = x  # 决策变量矩阵
        x1 = Vars[:, [0]]
        J1 = np.array(list(range(2, self.n_var, 2)))
        J2 = np.array(list(range(1, self.n_var, 2)))
        J = np.arange(1, 31)
        J = J[np.newaxis, :]
        y = Vars - x1 ** (0.5 * (1 + (3 * (J - 2) / (self.n_var - 2))))
        yJ1 = y[:, J1]
        yJ2 = y[:, J2]
        f1 = x1 + (2 / len(J1)) * (4 * np.sum(yJ1 ** 2, 1, keepdims=True) -
                                   2 * (np.prod(np.cos((20 * yJ1 * np.pi) / (np.sqrt(J1))), 1, keepdims=True)) + 2)
        f2 = 1 - np.sqrt(x1) + (2 / len(J2)) * (4 * np.sum(yJ2 ** 2, 1, keepdims=True) -
                                                2 * (np.prod(np.cos((20 * yJ2 * np.pi) / (np.sqrt(J2))), 1,
                                                             keepdims=True)) + 2)
        ObjV = np.hstack([f1, f2])
        return ObjV

    def get_ref_set(self,n_ref_points=1000):  # 理论最优值
        N = n_ref_points
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - np.sqrt(ObjV1)
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV


class UF4(Basic_Problem):
    def __init__(self):
        self.n_obj = 2  # 初始化（目标维数）
        self.n_var = 30  # 初始化Dim（决策变量维数）
        self.lb = np.array([-2]*self.n_var)
        self.lb[0] = 0
        self.ub = np.array([2]*self.n_var)
        self.ub[0] = 1
        super().__init__(n_var = self.n_var,n_obj = self.n_obj,lb= self.lb,ub = self.ub,vtype=float)

    def eval(self, x):  # 目标函数
        Vars = x  # 决策变量矩阵
        x1 = Vars[:, [0]]
        J1 = np.array(list(range(2, self.n_var, 2)))
        J2 = np.array(list(range(1, self.n_var, 2)))
        J = np.arange(1, 31)
        J = J[np.newaxis, :]
        y = Vars - np.sin(6 * np.pi * x1 + (J * np.pi) / self.n_var)
        hy = np.abs(y) / (1 + np.exp(2 * (np.abs(y))))
        hy1 = hy[:, J1]
        hy2 = hy[:, J2]
        f1 = x1 + 2 * np.mean(hy1, 1, keepdims=True)
        f2 = 1 - x1 ** 2 + 2 * np.mean(hy2, 1, keepdims=True)
        ObjV = np.hstack([f1, f2])
        return ObjV

    def get_ref_set(self,n_ref_points=1000):  # 理论最优值
        N = n_ref_points
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - ObjV1 ** 2
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV


class UF5(Basic_Problem):
    def __init__(self):
        self.n_obj = 2  # 初始化M（目标维数）
        self.n_var = 30  # 初始化Dim（决策变量维数）
        self.lb = np.array([-1]*self.n_var)
        self.lb[0] = 0
        self.ub = np.array([1]*self.n_var)
        super().__init__(n_var=self.n_var, n_obj=self.n_obj, lb=self.lb, ub=self.ub, vtype=float)

    def eval(self, x):  # 目标函数
        Vars = x  # 决策变量矩阵
        x1 = Vars[:, [0]]
        J1 = np.array(list(range(2, self.n_var, 2)))
        J2 = np.array(list(range(1, self.n_var, 2)))
        J = np.arange(1, 31)
        J = J[np.newaxis, :]
        y = Vars - np.sin(6 * np.pi * x1 + (J * np.pi) / self.n_var)
        hy = 2 * y ** 2 - np.cos(4 * np.pi * y) + 1
        # print(hy)
        hy1 = hy[:, J1]
        hy2 = hy[:, J2]
        f1 = x1 + (1 / 20 + 0.1) * np.abs(np.sin(20 * np.pi * x1)) + 2 * (np.mean(hy1, 1, keepdims=True))
        f2 = 1 - x1 + (1 / 20 + 0.1) * np.abs(np.sin(20 * np.pi * x1)) + 2 * (np.mean(hy2, 1, keepdims=True))
        ObjV = np.hstack([f1, f2])
        return ObjV

    def get_ref_set(self,n_ref_points=1000):  # 理论最优值
        N = n_ref_points  # 生成10000个参考点
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - ObjV1
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV


class UF6(Basic_Problem):  # 继承Problem父类
    def __init__(self):

        self.n_obj = 2  # 初始化M（目标维数）
        self.n_var = 30  # 初始化Dim（决策变量维数）
        self.lb = np.array([-1]*self.n_var)
        self.lb[0] = 0
        self.ub = np.array([1]*self.n_var)
        super().__init__(n_var=self.n_var, n_obj=self.n_obj, lb=self.lb, ub=self.ub, vtype=float)

    def eval(self, x):  # 目标函数
        Vars = x # 决策变量矩阵
        x1 = Vars[:, [0]]
        J1 = np.array(list(range(2, self.n_var, 2)))
        J2 = np.array(list(range(1, self.n_var, 2)))
        J = np.arange(1, 31)
        J = J[np.newaxis, :]
        y = Vars - np.sin(6 * np.pi * x1 + (J * np.pi) / self.n_var)
        yJ1 = y[:, J1]
        yJ2 = y[:, J2]
        # hy    = 2*y**2 - np.cos(4*np.pi*y) + 1
        # print(hy)
        # hy1   = hy[:, J1]
        # hy2   = hy[:, J2]
        f1 = x1 + np.maximum(0, 2 * (1 / 4 + 0.1) * np.sin(4 * np.pi * x1)) + \
             (2 / len(J1)) * (4 * np.sum(yJ1 ** 2, 1, keepdims=True) - \
                              2 * (np.prod(np.cos((20 * yJ1 * np.pi) / (np.sqrt(J1))), 1, keepdims=True)) + 2)
        f2 = 1 - x1 + np.maximum(0, 2 * (1 / 4 + 0.1) * np.sin(4 * np.pi * x1)) + \
             (2 / len(J2)) * (4 * np.sum(yJ2 ** 2, 1, keepdims=True) - \
                              2 * (np.prod(np.cos((20 * yJ2 * np.pi) / (np.sqrt(J2))), 1, keepdims=True)) + 2)
        ObjV = np.hstack([f1, f2])
        return ObjV

    def get_ref_set(self,n_ref_points=1000):  # 理论最优值
        N = n_ref_points
        ObjV1 = np.linspace(0, 1, N)
        idx = ((ObjV1 > 0) & (ObjV1 < 1 / 4)) | ((ObjV1 > 1 / 2) & (ObjV1 < 3 / 4))
        ObjV1 = ObjV1[~idx]
        ObjV2 = 1 - ObjV1
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV


class UF7(Basic_Problem):  # 继承Problem父类
    def __init__(self):

        self.n_obj = 2  # 初始化M（目标维数）
        self.n_var= 30  # 初始化Dim（决策变量维数）
        self.lb = np.array([-1]*self.n_var)
        self.lb[0] = 0
        self.ub = np.array([1]*self.n_var)
        super().__init__(n_var=self.n_var, n_obj=self.n_obj, lb=self.lb, ub=self.ub, vtype=float)

    def eval(self, x):  # 目标函数
        Vars = x  # 决策变量矩阵
        x1 = Vars[:, [0]]
        J1 = np.array(list(range(2, self.n_var, 2)))
        J2 = np.array(list(range(1, self.n_var, 2)))
        J = np.arange(1, 31)
        J = J[np.newaxis, :]
        y = Vars - np.sin(6 * np.pi * x1 + (J * np.pi) / self.n_var)
        yJ1 = y[:, J1]
        yJ2 = y[:, J2]
        f1 = x1 ** 0.2 + 2 * np.mean(yJ1 ** 2, 1, keepdims=True)
        f2 = 1 - x1 ** 0.2 + 2 * np.mean(yJ2 ** 2, 1, keepdims=True)
        ObjV = np.hstack([f1, f2])
        return ObjV

    def get_ref_set(self,n_ref_points=1000):  # 理论最优值
        N = n_ref_points
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - ObjV1
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV


class UF8(Basic_Problem):  # 继承Problem父类
    def __init__(self):
        self.n_obj = 3  # 初始化M（目标维数）
        self.n_var = 30  # 初始化Dim（决策变量维数）
        self.lb = np.array([0]*2+[-2]*(self.n_var-2))
        self.ub = np.array([1]*2+[2]*(self.n_var-2))
        super().__init__(n_var=self.n_var, n_obj=self.n_obj, lb=self.lb, ub=self.ub, vtype=float)

    def eval(self, x):  # 目标函数
        Vars = x  # 决策变量矩阵
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        J1 = np.array(list(range(3, self.n_var, 3)))
        J2 = np.array(list(range(4, self.n_var, 3)))
        J3 = np.array(list(range(2, self.n_var, 3)))
        # print(J1, J2, J3)
        J = np.arange(1, 31)
        J = J[np.newaxis, :]
        # f    = 2*np.mean((Vars-2*x2*np.sin(2*np.pi*x1+J*np.pi/self.Dim))**2 ,1,keepdims = True)
        f = (Vars - 2 * x2 * np.sin(2 * np.pi * x1 + J * np.pi / self.n_var)) ** 2
        # print(f.shape)
        f1 = np.cos(0.5 * x1 * np.pi) * np.cos(0.5 * x2 * np.pi) + 2 * np.mean(f[:, J1], 1, keepdims=True)
        f2 = np.cos(0.5 * x1 * np.pi) * np.sin(0.5 * x2 * np.pi) + 2 * np.mean(f[:, J2], 1, keepdims=True)
        f3 = np.sin(0.5 * x1 * np.pi) + 2 * np.mean(f[:, J3], 1, keepdims=True)
        ObjV = np.hstack([f1, f2, f3])
        return ObjV

    def get_ref_set(self,n_ref_points=1000):  # 理论最优值
        N = n_ref_points
        ObjV, N = ea.crtup(self.n_obj, N)  # ObjV.shape=N,3
        ObjV = ObjV / np.sqrt(np.sum(ObjV ** 2, 1, keepdims=True))
        referenceObjV = ObjV
        return referenceObjV


class UF9(Basic_Problem):  # 继承Problem父类
    def __init__(self):
        self.n_obj = 3  # 初始化M（目标维数）
        self.n_var = 30  # 初始化Dim（决策变量维数）
        self.lb = np.array([0]*2+[-2]*(self.n_var-2))
        self.ub = np.array([1]*2+[2]*(self.n_var-2))
        # 调用父类构造方法完成实例化
        super().__init__(n_var=self.n_var, n_obj=self.n_obj, lb=self.lb, ub=self.ub, vtype=float)

    def eval(self, x):  # 目标函数
        Vars = x  # 决策变量矩阵
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        J1 = np.array(list(range(3, self.n_var, 3)))
        J2 = np.array(list(range(4, self.n_var, 3)))
        J3 = np.array(list(range(2, self.n_var, 3)))
        # print(J1, J2, J3)
        J = np.arange(1, 31)
        J = J[np.newaxis, :]
        f = (Vars - 2 * x2 * np.sin(2 * np.pi * x1 + J * np.pi / self.n_var)) ** 2
        f1 = 0.5 * (np.maximum(0, (1.1 * (1 - 4 * (2 * x1 - 1) ** 2))) + 2 * x1) * x2 + 2 * np.mean(f[:, J1], 1,
                                                                                                    keepdims=True)
        f2 = 0.5 * (np.maximum(0, (1.1 * (1 - 4 * (2 * x1 - 1) ** 2))) - 2 * x1 + 2) * x2 + 2 * np.mean(f[:, J2], 1,
                                                                                                        keepdims=True)
        f3 = 1 - x2 + 2 * np.mean(f[:, J3], 1, keepdims=True)
        ObjV = np.hstack([f1, f2, f3])
        return ObjV

    def get_ref_set(self,n_ref_points=1000):  # 理论最优值
        N = n_ref_points  # 生成10000个参考点
        ObjV, N = ea.crtup(self.n_obj, N)  # ObjV.shape=N,3
        idx = (ObjV[:, 0] > (1 - ObjV[:, 2]) / 4) & (ObjV[:, 0] < (1 - ObjV[:, 2]) * 3 / 4)
        referenceObjV = ObjV[~idx]
        return referenceObjV

class UF10(Basic_Problem):  # 继承Problem父类
    def __init__(self):
        self.n_obj = 3  # 初始化M（目标维数）
        self.n_var = 30  # 初始化Dim（决策变量维数）
        self.lb = np.array([0]*2+[-2]*(self.n_var-2))
        self.ub = np.array([1] * 2 + [2] * (self.n_var - 2))
        super().__init__(n_var=self.n_var, n_obj=self.n_obj, lb=self.lb, ub=self.ub, vtype=float)

    def eval(self, x):  # 目标函数
        Vars = x  # 决策变量矩阵
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        J1 = np.array(list(range(3, self.n_var, 3)))
        J2 = np.array(list(range(4, self.n_var, 3)))
        J3 = np.array(list(range(2, self.n_var, 3)))
        J = np.arange(1, 31)
        J = J[np.newaxis, :]
        y = Vars - 2 * x2 * np.sin(2 * np.pi * x1 + (J * np.pi) / self.n_var)
        f = 4 * y ** 2 - np.cos(8 * np.pi * y) + 1
        f1 = np.cos(0.5 * x1 * np.pi) * np.cos(0.5 * x2 * np.pi) + 2 * np.mean(f[:, J1], 1, keepdims=True)
        f2 = np.cos(0.5 * x1 * np.pi) * np.sin(0.5 * x2 * np.pi) + 2 * np.mean(f[:, J2], 1, keepdims=True)
        f3 = np.sin(0.5 * x1 * np.pi) + 2 * np.mean(f[:, J3], 1, keepdims=True)
        ObjV = np.hstack([f1, f2, f3])
        return ObjV
    def get_ref_set(self,n_ref_points=1000):  # 理论最优值
        N = n_ref_points  # 生成10000个参考点
        ObjV, N = ea.crtup(self.n_obj, N)  # ObjV.shape=N,3
        ObjV = ObjV / np.sqrt(np.sum(ObjV ** 2, 1, keepdims=True))
        referenceObjV = ObjV
        return referenceObjV

if __name__ == '__main__':
    uf1= UF1()
    uf2 = UF2()
    uf3 = UF3()
    uf4 = UF4()
    uf5 = UF5()
    uf6 = UF6()
    uf7 = UF7()
    uf8 = UF8()
    uf9 = UF9()
    uf10 = UF10()
    x = np.random.rand(100, 30)
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
