import numpy as np
import sympy as sp
from scipy.special import roots_legendre
from scipy.interpolate import griddata
import scipy.linalg as la
import matplotlib.pyplot as plt
# plt.style.use('default')
import copy
import traceback
from shape_fns import *
import numpy as np
import math

class Element:
    def __init__(self, vertices):
        self.vertices = vertices

def distance(p1, p2):
    """计算两点之间的距离"""
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def triangle_area(p1, p2, p3):
    """使用海伦公式计算三角形面积"""
    a = distance(p1, p2)
    b = distance(p2, p3)
    c = distance(p1, p3)
    s = (a + b + c) / 2
    return math.sqrt(s * (s - a) * (s - b) * (s - c))

def quadrilateral_area(p1, p2, p3, p4):
    """计算四边形面积，将其分为两个三角形"""
    return triangle_area(p1, p2, p3) + triangle_area(p1, p3, p4)

def calculate_area(element):
    if len(element.vertices) == 3:
        return triangle_area(*element.vertices)
    elif len(element.vertices) == 4:
        return quadrilateral_area(*element.vertices)
    else:
        raise ValueError("Element has an unsupported number of vertices.")

def Gauss_points(element, order):
    """
    Return Gauss integration points and weights for the given shape and order using leggauss.
    
    Parameters:
    - shape: 'quad' for quadrilateral, 'triangle' for triangle
    - order: desired accuracy of integration (1, 2, 3, ...)

    Returns:
    - points: list of Gauss points
    - weights: list of Gauss weights
    """
    
    if element.shape == 'quad':
        xi, wi = np.polynomial.legendre.leggauss(order)
        points = [(x, y) for x in xi for y in xi]
        weights = [wx * wy for wx in wi for wy in wi]
        
    elif element.shape == 'triangle':
        NGP_data = {
            1: {
                'points': np.array([(1/3, 1/3)]),
                'weights': np.array([1/2])
            },
            3: {
                'points': np.array([(1/6, 1/6), (2/3, 1/6), (1/6, 2/3)]),
                'weights': np.array([1/6, 1/6, 1/6])
            },
            4: {
                'points': np.array([(1/3, 1/3), (0.6, 0.2), (0.2, 0.6), (0.2, 0.2)]),
                'weights': np.array([-27/96, 25/96, 25/96, 25/96])
            }
        }
        if order == 2:
            order = 3
        points, weights = NGP_data[order]['points'], NGP_data[order]['weights']
    else:
        raise ValueError("Shape not supported")

    return points, weights

def add_matrices(*args):
    # 检查矩阵尺寸是否相同
    first_shape = args[0].shape
    # 初始化结果矩阵为第一个矩阵
    result = np.copy(args[0])

    # 遍历其余矩阵并加到结果矩阵上
    for mat in args[1:]:
        if mat.shape != first_shape:
            raise ValueError("All matrices must have the same shape!")
        zero_indices = result == 0
        result[zero_indices] += mat[zero_indices]
    
    return result


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
def grid_to_mat(mapping, output):

    grid_x = np.linspace(mapping[:, 0].min(), mapping[:, 0].max(), 100)
    grid_y = np.linspace(mapping[:, 1].min(), mapping[:, 1].max(), 100)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)

    # Interpolate using griddata
    grid_z = griddata(mapping, output, (grid_x, grid_y), method='cubic')
    return grid_x, grid_y, grid_z
    
def principal_stresses(sigma_x, sigma_y, tau_xy):
    """
    Compute the principal stresses from stress components.
    
    Parameters:
    - sigma_x: Stress in x direction
    - sigma_y: Stress in y direction
    - tau_xy: Shear stress
    
    Returns:
    - Principal stresses (sigma1, sigma2)
    """
    
    # Calculate the average stress
    sigma_avg = (sigma_x + sigma_y) / 2
    
    # Calculate half the range of the stresses
    R = np.sqrt(((sigma_x - sigma_y) / 2)**2 + tau_xy**2)
    
    # Principal stresses
    sigma1 = sigma_avg + R
    sigma2 = sigma_avg - R
    
    return (sigma1, sigma2)

def output(input, dir, type='stress'):
    assert not (len(input) == 2 and dir not in ['x', 'y', 'norm']), "Invalid combination of input length and direction."
    assert type in ['disp', 'strain', 'stress'], "Invalid type"
    if type == 'disp':
        sigma_x, sigma_y = input
    else:
        sigma_x, sigma_y, tau_xy = input
    if dir == "x":
        return input[0]
    elif dir == "y":
        return input[1]
    elif dir == "xy":
        return input[2]
    
    elif dir == "norm":
        if len(input) ==2:
            return np.linalg.norm(input)
        else:
            sigma1, sigma2 = principal_stresses(sigma_x, sigma_y, tau_xy)
            return max(abs(sigma1), abs(sigma2))  # R
    elif dir == "von":
        result = np.sqrt(sigma_x**2-sigma_x*sigma_y+sigma_y**2+3*tau_xy**2)
        return result
