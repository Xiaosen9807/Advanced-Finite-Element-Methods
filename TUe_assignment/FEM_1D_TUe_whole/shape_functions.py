
import numpy as np
import sympy as sp
from scipy.special import roots_legendre
import scipy.linalg as la
from tools_1D import *
import matplotlib.pyplot as plt
# plt.style.use('default')
import copy
from consts import consts



class phi_func_l(shape_function):
    def __init__(self, scale, p):
        super().__init__(scale)
        self.p = p
        self.range = [0, 1]
    def expression(self, x):
        scale_up = 1/(self.scale[1]-self.scale[0]) 

        if self.p == 0:
            phi = 1-self.mapping(x)
        elif self.p == 1:
            phi = self.mapping(x) 
        else:
            raise AssertionError("p should be 0 or 1 in linear shape function, not{}".format(self.p))
        return phi #*scale_up
        
class phip_func_l(shape_function):
    def __init__(self, scale, p):
        super().__init__(scale)
        self.check_name = "phip_func_l"
        self.range = [0, 1]
        self.p = p
    def expression(self, x):
        scale_up = 1/(self.scale[1]-self.scale[0]) 
        
        if self.p == 0:
            phip =  np.zeros_like(self.mapping(x))-1
        elif self.p == 1:
            phip = np.zeros_like(self.mapping(x))+1
        else:
            raise AssertionError("p should be 0 or 1 in linear shape function, not{}".format(self.p))
        return phip *scale_up
    
class phi_func_q(shape_function):
    def __init__(self, scale, p):
        super().__init__(scale)
        self.range = [0, 1]
        self.p = p
    def expression(self, x):
        xx = self.mapping(x)
        if self.p == -1:
            phi = (xx-1)*(xx-0.5)*2
        elif self.p == 0:
            phi = -xx*(xx-1)*4
        elif self.p ==1:
            phi = xx*(xx-0.5)*2
        else:
            raise AssertionError("p should be -1, 0 or 1 in quadratic shape function, not{}".format(self.p))
        return phi
        
class phip_func_q(shape_function):
    def __init__(self, scale, p):
        super().__init__(scale)
        self.range = [0, 1]
        self.p = p
    def expression(self, x):
        scale_up = 1/(self.scale[1]-self.scale[0]) 
        xx = self.mapping(x)
        if self.p == -1:
            phip = 4*xx - 3.0
        elif self.p == 0:
            phip = 4-8*xx
        elif self.p ==1:
            phip = 4*xx - 1.0
        else:
            raise AssertionError("p should be -1, 0 or 1 in quadratic shape function, not{}".format(self.p))
        return phip*scale_up
    
def linear(scale, p):
    phis = []
    phips = []
    p = 1
    for i in range(p+1):
        new_phi = phi_func_l(scale, i)
        new_phip = phip_func_l(scale,i)
        phis.append(new_phi)
        phips.append(new_phip)
    return phis, phips
def find_region(value):
    r1 = consts["r1"]
    r2 = consts["r2"]
    r3 = consts["r3"]
    r4 = consts["r4"]

    # 判断值落在哪个区间
    if 0 <= value < r1: 
        return int(0)
    elif r1 <= value < r2:
        return int(1)
    elif r2 <= value < r3:
        return int(2)
    elif r3 <= value <= r4:
        return int(3)
    else:
        raise ValueError("Value {} is less than 0 or larger than r4".format(value))


class exact_fn():
    def __init__(self,):
        self.name = "RHS"
        self.scale = [0, 1]
        self.mu = 1
        self.mu0 = 1.257*10**-6 # H/m
        self.Jz = -1326291.1924324587
        self.B0 = -0.0001302459397568438
        self.A0 = -0.0006358641469510412

 
    def __call__(self, x):
        # 检查x是否为单个数字，如果是，将其转换为一个元素的数组
        single_value = np.isscalar(x)
        if single_value:
            x = np.array([x])

        result = np.zeros_like(x, dtype=float)
        for i in range(len(x)):
            x_ = x[i]
            region = find_region(x_)
            Param = consts["Params"][region]
            self.A0 = Param["A"]
            self.B0 = Param["B"]
            self.mu = Param["mu"]
            self.Jz = Param["Jz"]
            self.muJz = self.mu * self.mu0 * self.Jz
            if x_ != 0:
                # func1 = A0 - 1/4 * muJz * x_**2 
                func1 = self.A0 + self.B0*np.log(x_) - 1/4 * self.muJz * x_**2 
            else:
                # func1 = A0 + B0*np.log(x_) - 1/4 * muJz * x_**2 
                func1 = self.A0 - 1/4 * self.muJz * x_**2 
            result[i] = func1

        # 如果输入是单个数字，返回单个结果，否则返回数组
        return result[0] * 1e6 if single_value else result * 1e6

    def info(self):
        print("A0:", self.A0, "B0:", self.B0, "mu:", self.mu, "Jz:", self.Jz)


class rhs_fn():
    def __init__(self, a, xb):
        self.name = "RHS"
        self.a = a
        self.xb = xb
        self.scale = [0, 1]
        self.mu = 1
        self.mu0 = 1.257*10**-6 # H/m
        self.Jz = -1326291.1924324587 # A/m^2
        
    def __call__(self, x):
        muJz = self.mu * self.mu0 * self.Jz
        muJz =  self.Jz
        func1 =  - muJz * x
        return func1 *220#* 1e6
    # def B(self, x):
        # return x - self.xb

class r_r():
    def __init__(self, scale):
        self.name = "LHS"
        self.scale = scale
        self.mu = 1
    def __call__(self, x):
        return x /self.mu


if __name__ == "__main__":
    exact_func = exact_fn()
    

    print(exact_func(0))
    exact_func.info()
    