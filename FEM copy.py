from typing import Any
import numpy as np
import sympy as sp
from scipy.special import roots_legendre
import scipy.linalg as la
import matplotlib.pyplot as plt
plt.style.use('default')
import copy

def G_integrate(u, N=3, scale=(0, 1)):
    N = N  # 取3个样本点
    a = scale[0]  # 积分上下限
    b = scale[1]
    x, w = roots_legendre(N)
    # print(x)

    xp = x*(b-a)/2+(b+a)/2
    wp = w*(b-a)/2

    s = 0
    for i in range(N):
        s += wp[i]*u(xp[i])
    return s

def Legendre(x=np.linspace(-1, 1, 100), p=5):

    if p == 0:
        return 1
    elif p == 1:
        return x

    else:
        return ((2*p-1)*x*Legendre(x, p-1)+(1-p)*Legendre(x, p-2))/p
class shape_function:
    def __init__(self, scale=[-1, 1]):
        self.scale = scale
        self.x_l = scale[0]
        self.x_r = scale[1]
        self.range = [-1, 1]
        
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
    def __init__(self, *functions):
        # 如果functions只有一个元素，并且这个元素是一个列表，那么使用这个列表作为函数列表
        if len(functions) == 1 and isinstance(functions[0], (list, tuple)):
            functions = functions[0]
        self.functions = functions
        min_scale = min((function.scale[0] for function in functions if callable(function)), default=0)
        max_scale = max((function.scale[1] for function in functions if callable(function)), default=0)
        self.scale = [min_scale, max_scale]


class linear(shape_function):
    def __call__(self):
        phii = lambda x: 1 - (x - self.x_l) / (self.x_r - self.x_l)
        phij = lambda x: (x - self.x_l) / (self.x_r - self.x_l)
        phiip = lambda x: -1*(x+1e-8) / (self.x_r - self.x_l)/(x+1e-8)
        phijp = lambda x: 1*(x+1e-8) / (self.x_r - self.x_l)/(x+1e-8)
        return [phii, phij], [phiip, phijp]
class quadratic(shape_function):
    def __init__(self, scale=[0, 1]):
        super().__init__(scale)
        self.scale = np.linspace(scale[0], scale[1], 3)
        self.x_l, self.x_m, self.x_r = self.scale

    def __call__(self):
        phii = lambda x: ((x - self.x_m)*(x - self.x_r)) / ((self.x_l - self.x_m) * (self.x_l - self.x_r))  # N1
        phim = lambda x: ((x - self.x_l)*(x - self.x_r)) / ((self.x_m - self.x_l) * (self.x_m - self.x_r))  # N2
        phij = lambda x: ((x - self.x_l)*(x - self.x_m)) / ((self.x_r - self.x_l) * (self.x_r - self.x_m))  # N3
        phiip = lambda x: ((2*x - self.x_m - self.x_r)) / ((self.x_l - self.x_m) * (self.x_l - self.x_r))  # N1'
        phimp = lambda x: ((2*x - self.x_l - self.x_r)) / ((self.x_m - self.x_l) * (self.x_m - self.x_r))  # N2'
        phijp = lambda x: ((2*x - self.x_l - self.x_m)) / ((self.x_r - self.x_l) * (self.x_r - self.x_m))  # N3'
        return [phii, phim, phij], [phiip, phimp, phijp]



class exact_fn():
    def __init__(self, a, xb):
        self.a = a
        self.xb = xb
        self.scale = [0, 1]

    def __call__(self, x):
        func1 = (1 - x) * (np.arctan(self.a * (x - self.xb)) + np.arctan(self.a*self.xb))
        return func1

    def derivative(self, input_value):
        x = sp.symbols('x')
        func1 = (1 - x) * (sp.atan(self.a * (x - self.xb)) + sp.atan(self.a*self.xb))
        func1_prime = sp.diff(func1, x)
        return sp.lambdify(x, func1_prime, 'numpy')(input_value)

class rhs_fn():
    def __init__(self, a, xb):
        self.a = a
        self.xb = xb
        self.scale = [0, 1]
        
    def __call__(self, x):
        func1 = -2*(self.a+self.a**3*self.B(x)*(self.B(x)-x+1))/(self.a**2*self.B(x)**2+1)**2
        return func1
    def B(self, x):
        return x - self.xb

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

def joint_funcs(functions):
    if type(functions) != list:
        raise AssertionError("Inputs must be a list!")
    elif not callable(functions[0]):
        raise AssertionError("Elements in the list must be functions!")
    new_lst = [plus(functions[i], functions[i+1]) for i in range(1, len(functions)-1, 2)]
    new_lst.insert(0, functions[0])  # 把第一个元素插入到新列表的首位
    new_lst.append(functions[-1])
    
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

# def subs(*functions):
#     # 如果functions只有一个元素，并且这个元素是一个列表，那么使用这个列表作为函数列表
#     if len(functions) == 1 and isinstance(functions[0], (list, tuple)):
#         functions = functions[0]
#     def subtracted_func(x):
#         result = 0
#         for func in functions:
#             try :
#                 result -= func(x)
#             except:
#                 result -= func
#         return result
#     return subtracted_func
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


def h_FEM(shape_class = linear, num_elems = 3, domain = (0, 1),rhs_func = rhs_fn(a=50, xb=0.8),exact_fn_object=exact_fn(0.5,0.8), BCs = (0, 0), verbose = False):
    mesh = np.linspace(domain[0], domain[1], num_elems+1)
    phi_phip ={'phis':[], 'phips':[]} 
    for i in range(num_elems):
        scale = (mesh[i], mesh[i+1])
        phis, phips = shape_class(scale=scale)() # h-version FEM
        K_sub = np.zeros((len(phips), len(phips)))
        for indx, x in np.ndenumerate(K_sub):
            K_sub[indx] += G_integrate(mul(phips[indx[0]], phips[indx[-1]]), scale=scale)
        
        F_sub = np.zeros(len(K_sub))
        for indx in range(len(F_sub)):
            F_sub[indx] = G_integrate(mul(rhs_func, phis[indx]), scale=scale)
        if i == 0:
            K = K_sub
            F = F_sub
        else:
            K = assemble(K, K_sub)
            F = assemble(F, F_sub)
        phi_phip['phis'].append(phis)
        phi_phip['phips'].append(phips)
    K[0, 1:] = 0.0 
    K[-1, :-1] = 0.0
    F[0] = BCs[0]* K[0, 0] # -= or = ??
    F[-1] = BCs[-1] * K[-1, -1]
    # print(F)
    U = -la.solve(K, F)
    mesh = np.linspace(domain[0], domain[1], len(U))
    if verbose == True:
        print(f"Shape class: {shape_class.__name__}, Number of elements: {num_elems}, Domain: {domain}, Boundary conditions: {BCs}")
        draw(U, phi_phip, domain, exact_fn_object)
        
    return U, mesh, phi_phip

def draw(U_array,phis_phips_array,domain, exact_fn_object):
    exact_func = exact_fn_object
    exact_func_p = exact_fn_object.derivative
    draw_mash = np.linspace(domain[0], domain[1], 1000)
    exact_solution = exact_func(draw_mash)
    mesh = np.linspace(domain[0], domain[1], len(U_array))

    plt.plot(draw_mash, exact_solution, label=' Analytical solution')
    plt.plot(mesh, U_array, label='FEM solution')
    plt.legend()
    plt.title('Exact solution')
    plt.show()
    
    U_l_p = np.zeros_like(U_array)
    num_funcs = len(phis_phips_array['phips'][0])
    for i in range(len(phis_phips_array['phis'])):
        for j in range(num_funcs):
            U_l_p[(num_funcs-1)*i+j]= U_array[(num_funcs-1)*i+j]*phis_phips_array['phips'][i][j](mesh[(num_funcs-1)*i+j])
    U_exact_p = exact_func_p(draw_mash)
    plt.plot(draw_mash, U_exact_p, label='Analytical solution')
    plt.plot(mesh, U_l_p, label='FEM solution' )
    plt.title('First derivative ')
    plt.legend()
    plt.show()
    
    
def cal_energy(U_array, phis_phips_array, domain):
    energy_value = 0
    num_funcs = len(phis_phips_array['phis'][0])
    num_elements = len(phis_phips_array['phis'])
    
    mesh = np.linspace(domain[0], domain[1], len(U_array))
    # print((mesh))
    
    for i in range(num_elements):
        u_list = []
        u_prime_list = []
        # print(i)
        for j in range(num_funcs):
            new_u = lambda x, a=U_array[(num_funcs-1)*i+j], b=phis_phips_array['phis'][i][j]: a * b(x)
            u_list.append(new_u)
            new_u_prime = lambda x, a=U_array[(num_funcs-1)*i+j]*1, b=phis_phips_array['phips'][i][j]: a * b(x)
            u_prime_list.append(new_u_prime)
        u = plus(u_list)
        u_prime = plus(u_prime_list)
        # integrand = lambda x: u_prime(x)**2#+u(x)**2
        integrand = mul(u_prime, u_prime)
        
        # u = lambda x: U_array[i] * phis_phips_array['phis'][i][0](x) + U_array[i+1] * phis_phips_array['phis'][i][-1](x)
        
        # u_prime = lambda x: U_array[i] * phis_phips_array['phips'][i][0](x) + U_array[i+1] * phis_phips_array['phips'][i][-1](x)
        # u_prime = lambda x: phis_phips_array['phips'][i][0](x) + phis_phips_array['phips'][i][-1](x)
       
        # integrand = lambda x: u_prime(x)**2#+u(x)**2
        # print('u_prime(0)', u_prime(0))
        # print('u(0)', u(0))
        # print('integrand(0)', integrand(0),'\n')
        # print('-----------------------')
        energy_value += G_integrate(integrand, scale=(mesh[i], mesh[i+1]))/2
    return energy_value
if __name__ == '__main__':
    num_elems = 40
    domain = (0, 1)
    mesh = np.linspace(domain[0], domain[1], num_elems+1)

    a = 0.5
    xb = 0.8
    if a ==50:
        U_init = 1.585854059271320
    elif a == 0.5:
        U_init = 0.03559183822564316

    exact_func = exact_fn(a = a, xb=xb)
    rhs_func = rhs_fn(a=a, xb=xb)
    error_list_l = []
    DOF_l = []
    error_list_q = []
    DOF_q = []
    num_elems_list = [2, 4, 8, 16, 32]
    a = 0.5
    xb = 0.8
    if a ==50:
        U_init = 1.585854059271320
    elif a == 0.5:
        U_init = 0.03559183822564316
    exact_fn_object = exact_fn(a = a, xb=xb)
    exact_func = exact_fn_object()
    exact_func_p = exact_fn_object.derivative()
    rhs_func = rhs_fn(a=a, xb=xb)()
    shape_class = quadratic
    for num_elems in num_elems_list:
        U_l, mesh_l, phis_phips_l = h_FEM(shape_class = linear, num_elems = num_elems, domain = domain,rhs_func = rhs_func, BCs = (0, 0))
        energy_l = cal_energy(U_l, phis_phips_l, domain)
        error_l = np.sqrt(abs((energy_l-U_init))/U_init)
        error_list_l.append(error_l)
        DOF_l.append(len(U_l))
        
        U_q, mesh_q, phis_phips_q = h_FEM(shape_class = quadratic, num_elems = num_elems, domain = domain,rhs_func = rhs_func, BCs = (0, 0))
        energy_q = cal_energy(U_q, phis_phips_q, domain)
        error_q = np.sqrt(abs((energy_q-U_init))/U_init)
        error_list_q.append(error_q)
        DOF_q.append(len(U_q))
    plt.plot(np.log(num_elems_list), np.log(error_list_l),'*-', label='log error of linear shape function')
    plt.plot(np.log(num_elems_list), np.log(error_list_q),'*-', label='log error of quadratic shape function')
    plt.xlabel('Log mesh numbers')
    plt.ylabel('Log energy norm error')
    plt.title('Log-log plot for mesh numbers versus energy norm error')
    plt.legend()
    plt.show()


    plt.plot(np.log(DOF_l), np.log(error_list_l),'*-', label='log error of linear shape function')
    plt.plot(np.log(DOF_q), np.log(error_list_q),'*-', label='log error of quadratic shape function')
    plt.xlabel('log DOF')
    plt.ylabel('log energy norm error)')
    plt.title('Log-log plot for DOF versus energy norm error')
    plt.legend()
    plt.show()