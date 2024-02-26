
import numpy as np
import sympy as sp
from scipy.special import roots_legendre
import scipy.linalg as la
from tools_1D import *
import matplotlib.pyplot as plt
# plt.style.use('default')
import copy


def Legendre(x=np.linspace(-1, 1, 100), p=5):

    if p == 0:
        return 1
    elif p == 1:
        return x

    else:
        return ((2*p-1)*x*Legendre(x, p-1)+(1-p)*Legendre(x, p-2))/p


class phi_func_l(shape_function):
    def __init__(self, scale, p):
        super().__init__(scale)
        self.p = p
        self.range = [0, 1]
    def expression(self, x):
        if self.p == 0:
            phi = 1-self.mapping(x)
        elif self.p == 1:
            phi = self.mapping(x) 
        else:
            raise AssertionError("p should be 0 or 1 in linear shape function, not{}".format(self.p))
        return phi
        
class phip_func_l(shape_function):
    def __init__(self, scale, p):
        super().__init__(scale)
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
        return phip*scale_up
    
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
    
class phi_func_h(shape_function):
    def __init__(self, scale, p):
        super().__init__(scale)
        self.p = p
    def expression(self, x):
        scale = self.scale
        i =self.p
        if i == 0:
            phi = (1-self.mapping(x))/2 
        elif i == 1:
            phi = (1+self.mapping(x))/2 
        else:
            phi = 1/np.sqrt(4*i-2)*(Legendre(self.mapping(x), i)-Legendre(self.mapping(x), i-2))
        return phi
        
class phip_func_h(shape_function):
    def __init__(self, scale, p):
        super().__init__(scale)
        self.p = p
    def expression(self, x):
        scale_up = 2/(self.scale[1]-self.scale[0]) 
        i =self.p
        
        if i == 0:
            phip =  np.zeros_like(self.mapping(x))-0.5
        elif i == 1:
            phip = np.zeros_like(self.mapping(x))+0.5
        else:
            phip = np.sqrt(i-1/2)*(Legendre(self.mapping(x), i-1))
        return phip*scale_up
    
def Hierarchical(scale, p):
    phis = []
    phips = []
    start=0
    
    for i in range(start, p+1):
        new_phi = phi_func_h(scale, i)
        new_phip = phip_func_h(scale,i)
        phis.append(new_phi)
        phips.append(new_phip)
    return phis, phips

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

def quadratic(scale, p):
    phis = []
    phips = []
    p = 1
    for i in range(-1, p+1):
        new_phi = phi_func_q(scale, i)
        new_phip = phip_func_q(scale,i)
        phis.append(new_phi)
        phips.append(new_phip)
    return phis, phips

class exact_fn():
    def __init__(self, a, xb):
        self.name = "RHS"
        self.a = a
        self.xb = xb
        self.scale = [0, 1]

    def __call__(self, x):
        A0 = 71
        muJz = 20
        func1 = A0 - 1/4 * muJz * x**2 
        # func1 = (1 - x) * (np.arctan(self.a * (x - self.xb)) + np.arctan(self.a*self.xb))
        return func1

    def derivative(self, input_value):
        x = sp.symbols('x')
        func1 = (1 - x) * (sp.atan(self.a * (x - self.xb)) + sp.atan(self.a*self.xb))
        func1_prime = sp.diff(func1, x)
        return sp.lambdify(x, func1_prime, 'numpy')(input_value)

class rhs_fn():
    def __init__(self, a, xb):
        self.name = "RHS"
        self.a = a
        self.xb = xb
        self.scale = [0, 1]
        self.mu = 20
        self.Jz = 1
        self.muJz = self.mu * self.Jz
        
    def __call__(self, x):
        muJz = self.muJz
        func1 = muJz * x
        # func1 = -2*(self.a+self.a**3*self.B(x)*(self.B(x)-x+1))/(self.a**2*self.B(x)**2+1)**2
        return func1
    # def B(self, x):
        # return x - self.xb

class r_r():
    def __init__(self, scale):
        self.name = "LHS"
        self.scale = scale
    def __call__(self, x):
        return x