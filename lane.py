import numpy as np

DEFAULT_WIDTH = 0.13

class StraightLane():
    def __init__(self, p, q, w):
        self.p = np.asarray(p)
        self.q = np.asarray(q)
        self.w = w
        self.m = (self.q-self.p)/np.linalg.norm(self.q-self.p)
        self.n = np.asarray([-self.m[1], self.m[0]])
    def shifted(self, m):
        return StraightLane(self.p+self.n*self.w*m, self.q+self.n*self.w*m, self.w)