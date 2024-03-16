
import numpy as np
import sympy as sp
from scipy.special import roots_legendre
import scipy.linalg as la
import matplotlib.pyplot as plt
# plt.style.use('default')
import copy

def G_integrate(u, N=3, scale=(0, 1)):

    name = u.name[0]
    N = N  # 取3个样本点
    a = scale[0]  # 积分上下限
    b = scale[1]
    x, w = roots_legendre(N)
    # print(x)

    xp = x*(b-a)/2+(b+a)/2
    wp = w*(b-a)/2

    # print('xp', xp)
    # print('wp', wp)

    s = 0
    for i in range(N):
        if name == 'LHS':
            # s += xp[i]*wp[i]*u(xp[i])
            s += wp[i]*u(xp[i])
        elif name == 'RHS':
            s += wp[i]*u(xp[i])
        # print('s', s)
        # print(xp[i]*wp[i]*u(xp[i]))
        # print('u', u(xp[i]))
            
    # print('s', s)
    return s

class shape_function:
    def __init__(self, scale=[0, 1]):
        self.name = "LHS"
        self.scale = scale
        self.x_l = scale[0]
        self.x_r = scale[1]
        self.range = [0, 1]
        
    def expression(self, x):
        return 1 - (x - self.x_l) / (self.x_r - self.x_l)
    
    def mapping(self, x):
        scale = self.scale
        range = self.range
        x_normalized = (x - scale[0]) / (scale[1] - scale[0])
        return range[0] + x_normalized * (range[1] - range[0])

    def __call__(self, x):
        x = np.asarray(x)  # convert x to a numpy array if it's not already
        expression_vectorized = np.vectorize(self.expression, otypes=['d'])
        return np.where((self.scale[0] <= x) & (x <= self.scale[-1]), expression_vectorized(x), 0)
    
class operate(shape_function):
    """
    Initializes the function object with a list of functions. If the input functions contain only one element and that element is a list or tuple, it is used as the function list. 
    """
    def __init__(self, *functions):
        try:
            self.name = [functions[0].name]
        except:
            self.name = "None"
        # 如果functions只有一个元素，并且这个元素是一个列表，那么使用这个列表作为函数列表
        if len(functions) == 1 and isinstance(functions[0], (list, tuple)):
            functions = functions[0]
        self.functions = functions
        min_scale = min((function.scale[0] for function in functions if callable(function)), default=0)
        max_scale = max((function.scale[1] for function in functions if callable(function)), default=0)
        self.scale = [min_scale, max_scale]
        
def joint_funcs(functions):
    if type(functions) != list:
        raise AssertionError("Inputs must be a list!")
    elif not callable(functions[0]):
        raise AssertionError("Elements in the list must be functions!")
    # if len(functions)%3==0: # Don't know why I had this....
    #     chunk_size = 3
    #     chunk_size = 2
    elif len(functions)%2==0:
        chunk_size = 2
    spilt_list = [functions[i:i+chunk_size] for i in range(0, len(functions), chunk_size)]
    new_lst = spilt_list[0]
    for lst in spilt_list[1:]:
        new_lst[-1] = plus(new_lst[-1], lst[0])
        new_lst+=lst[1:]

    return new_lst


class mul(operate):
    def expression(self, x):
        result = 1
        for func in self.functions:
            try:
                result *= func(x)
            except:  # func is not callable
                result *= func
        return result

class subs(operate):
    def expression(self, x):
        result = 1
        for func in self.functions:
            try:
                result -= func(x)
            except:  # func is not callable
                result -= func
        return result
class plus(operate):
    def expression(self, x):
        try:
            result = self.functions[0](x) if callable(self.functions[0]) else self.functions[0]
        except:
            result = self.functions[0]
        for i in range(1, len(self.functions)):
            func = self.functions[i]
            try :
                if callable(func) and callable(self.functions[i-1]) and func.scale[0] == self.functions[i-1].scale[-1]: 
                    func.scale[0]+=1e-10
                result += func(x) if callable(func) else func
            except:
                result += func
        return result

def assemble(*matrices):
    matrices = [np.atleast_2d(m) for m in matrices]

    # 使用第一个矩阵初始化结果矩阵
    res_matrix = matrices[0]

    # 循环将剩余的矩阵拼接到结果矩阵上
    for i in range(1, len(matrices)):
        # 获取当前矩阵的大小
        rows_num, column_num = matrices[i].shape

        # 创建一个新的空矩阵，尺寸与当前结果矩阵一样，但是多出一行和一列
        new_matrix = np.zeros((res_matrix.shape[0] + rows_num - 1, res_matrix.shape[1] + column_num - 1))

        # 将结果矩阵的值复制到新矩阵中
        new_matrix[:res_matrix.shape[0], :res_matrix.shape[1]] = res_matrix

        # 将当前矩阵的值添加到新矩阵的最后一行和最后一列
        new_matrix[-rows_num:, -column_num:] += matrices[i]

        # 更新结果矩阵
        res_matrix = new_matrix
    if res_matrix.shape[0]<=1:
        return res_matrix[0]
    return res_matrix

def cal_error(A_z_pred, A_z_exact):
    return np.mean(np.abs(np.array(A_z_pred) - np.array(A_z_exact))) * 100