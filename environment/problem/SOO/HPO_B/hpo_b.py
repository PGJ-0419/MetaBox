import numpy as np
import xgboost as xgb
import pickle
import os, time
from ...basic_problem import Basic_Problem


class HPOB_Problem(Basic_Problem):
    def __init__(self,bst_surrogate,dim,y_min,y_max,lb,ub,normalized=False) -> None:
        self.bst_surrogate=bst_surrogate
        self.y_min=y_min
        self.y_max=y_max
        self.dim=dim
        self.gbest=1e+10
        self.normalized = normalized
        self.collect_gbest=[]
        self.lb = lb
        self.ub = ub
        self.optimum = None
    def func(self,position):
        x_q = xgb.DMatrix(position.reshape(-1,self.dim))
        new_y = self.bst_surrogate.predict(x_q)
        cost=-self.normalize(new_y)
        self.gbest=np.minimum(self.gbest,cost)
        self.collect_gbest.append(self.gbest)
        return cost

    def normalize(self, y):
        if self.normalized:
            if self.y_min is None:
                return (y-np.min(y))/(np.max(y)-np.min(y))
            else:
                return np.clip((y-self.y_min)/(self.y_max-self.y_min),0,1)
        return y
