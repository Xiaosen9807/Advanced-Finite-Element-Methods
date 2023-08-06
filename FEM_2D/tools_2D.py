import numpy as np
import sympy as sp
from scipy.special import roots_legendre
import scipy.linalg as la
import matplotlib.pyplot as plt
# plt.style.use('default')
import copy
import traceback
from shape_fns import *

def Gauss_points(N=3, scale_x = [-1, 1], scale_y = [-1, 1]):
    ax, bx = scale_x # Projection from [-1, 1] to scale_x
    ay, by = scale_y # Projection from [-1, 1] to scale_y
    x, wx = roots_legendre(N)
    y, wy = roots_legendre(N)
    xp = x*(bx-ax)/2+(bx+ax)/2
    wxp = wx*(bx-ax)/2
    yp = y*(by-ay)/2+(by+ay)/2
    wyp = wy*(by-ay)/2
    points = []
    Ws = []
    for i in range(N):
        for j in range(N):
            points.append([xp[j], yp[i]])
            Ws.append(wxp[j]*wyp[i]) 
    return points, Ws
    
    
def G_integrate_2D(u, N=3, scale_x = [0,1], scale_y = [0, 1]):
    
    ax, bx = scale_x # Projection from [-1, 1] to scale_x
    ay, by = scale_y # Projection from [-1, 1] to scale_y
    x, wx = roots_legendre(N)
    y, wy = roots_legendre(N)
    xp = x*(bx-ax)/2+(bx+ax)/2
    wxp = wx*(bx-ax)/2
    yp = y*(by-ay)/2+(by+ay)/2
    wyp = wy*(by-ay)/2
    integral = 0.0
    for i in range(N):
        for j in range(N):
            integral += wxp[i] * wyp[j] * u(xp[i], yp[j])
            
    return integral



class operate(shape_fns):
    def __init__(self, *functions):
        # 如果functions只有一个元素，并且这个元素是一个列表，那么使用这个列表作为函数列表
        if len(functions) == 1 and isinstance(functions[0], (list, tuple)):
            functions = functions[0]
        self.functions = functions
        min_scale_x = min((function.scale_x[0] for function in functions if callable(function)), default=0)
        max_scale_x = max((function.scale_x[1] for function in functions if callable(function)), default=0)
        self.scale_x = [min_scale_x, max_scale_x]
        
        min_scale_y = min((function.scale_y[0] for function in functions if callable(function)), default=0)
        max_scale_y = max((function.scale_y[1] for function in functions if callable(function)), default=0)
        self.scale_y = [min_scale_y, max_scale_y]


class mul(operate):
    def expression(self, x, y):
        result = 1
        for func in self.functions:
            try:
                result *= func(x, y)
            except:  # func is not callable
                result *= func
        return result

class subs(operate):
    def expression(self, x, y):
        result = 1
        for func in self.functions:
            try:
                result -= func(x, y)
            except:  # func is not callable
                result -= func
        return result
class plus(operate):
    def expression(self, x, y):
        try:
            result = self.functions[0](x, y) if callable(self.functions[0]) else self.functions[0]
        except:
            result = self.functions[0]
        for i in range(1, len(self.functions)):
            func = self.functions[i]
            try :
                if callable(func) and callable(self.functions[i-1]) and func.scale_x[0] == self.functions[i-1].scale_x[-1] and func.scale_y[0] == self.functions[i-1].scale_y[-1]: 
                    func.scale_x[0]+=1e-10
                    func.scale_y[0]+=1e-10
                result += func(x, y) if callable(func) else func
            except Exception as e:
                traceback.print_exc()
                print("Error: ", e)
                result += func
        return result
        
def joint_funcs(functions):
    if type(functions) != list:
        raise AssertionError("Inputs must be a list!")
    elif not callable(functions[0]):
        raise AssertionError("Elements in the list must be functions!")
    if len(functions)%3==0:
        chunk_size = 3
    elif len(functions)%2==0:
        chunk_size = 2
    spilt_list = [functions[i:i+chunk_size] for i in range(0, len(functions), chunk_size)]
    new_lst = spilt_list[0]
    for lst in spilt_list[1:]:
        new_lst[-1] = plus(new_lst[-1], lst[0])
        new_lst+=lst[1:]
    return new_lst

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

