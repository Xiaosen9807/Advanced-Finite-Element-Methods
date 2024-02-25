import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import minimize


class shape_fns:
    def __init__(self, scale_x = [0, 1], scale_y = [0, 1], p=0):
        self.scale_x = scale_x
        self.scale_y = scale_y
        # self.x_l = scale_x[0]
        # self.x_r = scale_x[1]
        # self.y_l = scale_y[0]
        # self.y_r = scale_y[1]
        self.p = p
    
    def gridnize(self, xi, eta):
           xi_, eta_ = np.meshgrid(xi, eta, indexing='ij')
           return xi_, eta_        
        
    def expression(self, xi, eta): 
        return 1-xi-eta

    def __call__(self, x=0, y=0):
        
        # if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        #     xi, eta = [x, y]
        # else:
        #     xi, eta = self.gridnize(x, y)
        return  self.expression(x, y)
        # return np.where((self.scale_x[0] <= xi) & (xi <= self.scale_x[1]) & (self.scale_y[0] <= eta) & (eta <= self.scale_y[1]), self.expression(xi, eta), 0)


class T3_phi(shape_fns):
    def expression(self, xi, eta): 
        if self.p == 0:
             return xi 
        elif self.p == 1:
            return eta
        elif self.p == 2:
            return 1-xi-eta
        else:
            raise ValueError("p should be 0, 1 or 2 in T3 element shape functions, not {}".format(self.p))

        
class T3_phipx(shape_fns):
    def expression(self, xi=0, eta=0):
        if self.p == 0:
             return 1
        elif self.p == 1:
            return 0
        elif self.p == 2:
            return -1
        else:
            raise ValueError("p should be 0, 1 or 2 in T3 element shape functions, not {}".format(self.p))

class T3_phipy(shape_fns):
    def expression(self, xi=0, eta=0):
        if self.p == 0:
             return 0
        elif self.p == 1:
            return 1
        elif self.p == 2:
            return -1
        else:
            raise ValueError("p should be 0, 1 or 2 in T3 element shape functions, not {}".format(self.p))


class Q4_phi(shape_fns):
    def expression(self, xi=0, eta=0):
        if self.p == 0:
            return (xi-1)*(eta-1)/4
        elif self.p == 1:
            return (1 + xi) * (1 - eta)/4
        elif self.p == 2:
            return (1 + xi) * (1 + eta)/4
        elif self.p == 3:
            return (1 - xi) * (1 + eta)/4
        else:
            raise ValueError("p should be 0, 1, 2 or 3 in Q4 element shape functions, not {}".format(self.p))

class Q4_phipx(shape_fns):
    def expression(self, xi=0, eta=0):
        if self.p == 0:
             return (eta - 1)/4
        elif self.p == 1:
            return (1 - eta)/4
        elif self.p == 2:
            return (1 + eta)/4
        elif self.p == 3:
            return -(1 + eta)/4
        else:
            raise ValueError("p should be 0, 1, 2 or 3 in Q4 element shape functions, not {}".format(self.p))       


class Q4_phipy(shape_fns):
    def expression(self, xi=0, eta=0):
        if self.p == 0:
            return (xi - 1)/4
        elif self.p == 1:
            return -(xi + 1)/4
        elif self.p == 2:
            return (1 + xi)/4
        elif self.p == 3:
            return (1 - xi)/4
        else:
            raise ValueError("p should be 0, 1, 2 or 3 in Q4 element shape functions, not {}".format(self.p))
def exact_fn(x, y,a_b, type='stress'):

    E, nu = 200e3, 0.3
    D = E / (1 - nu**2)* np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2]
            ])
    b = 20
    a = b*a_b
    sigma_0 = 50 #Mpa
    lambda_ = 1/2*(x**2+y**2-a**2-b**2+((x**2+y**2-a**2+b**2)**2+4*(a**2-b**2) * y**2)**0.5)

            # Define rho_a and rho_b
    rho_a = a /np.sqrt(a**2 + lambda_)
    rho_b = b / np.sqrt(b**2 + lambda_)

    # Define n_x and n_y
    n_x = x * (b**2 + lambda_) / np.sqrt(x**2 * (b**2+lambda_)**2 + y**2 * (a**2 + lambda_)**2)
    n_y = y * (a**2 + lambda_) / np.sqrt(x**2 * (b**2+lambda_)**2 + y**2 * (a**2 + lambda_)**2)

    # Define H_1 to H_5
    H_1 = (a**2 * rho_a**2 * rho_b**2 + b**2 * rho_a**2 + a * b * rho_a * rho_b) / (a * rho_b + b * rho_a)**2 - rho_b**2 * n_x**2 - rho_a**2 * n_y**2 + (5 * rho_a**2 + 5 * rho_b**2 - 4 * rho_a**2 * n_x**2 - 4 * rho_b**2 * n_y**2 - 4) * n_x**2 * n_y**2
    H_2 = (rho_b * a * (a * rho_b + b * rho_a + 2 * b * rho_a * rho_b**2 + a * rho_b**3)) / (a * rho_b + b * rho_a)**2 + n_y**2 * (2 - 6 * rho_b**2 + (rho_a**2 + 9 * rho_b**2 - 4 * rho_a**2 * n_x**2 - 4 * rho_b**2 * n_y**2 - 4) * n_y**2)
    H_3 = n_x * n_y * (1 - 3 * rho_b**2 + (3 * rho_a**2 + 7 * rho_b**2 - 4 * rho_a**2 * n_x**2 - 4 * rho_b**2 * n_y**2 - 4) * n_y**2)
    H_4 = (rho_a * b * (a * rho_b + b * rho_a + 2 * a * rho_a**2 * rho_b + b * rho_a**3)) / (a * rho_b + b * rho_a)**2 + n_x**2 * (2 - 6 * rho_a**2 + (9 * rho_a**2 + rho_b**2 - 4 * rho_a**2 * n_x**2 - 4 * rho_b**2 * n_y**2 - 4) * n_x**2)
    H_5 = n_x * n_y * (1 - 3 * rho_a**2 + (7 * rho_a**2 + 3 * rho_b**2 - 4 * rho_a**2 * n_x**2 - 4 * rho_b**2 * n_y**2 - 4) * n_x**2)
    
    sigma_x = sigma_0*(1-rho_a*rho_b*(H_1/2 -(b/a+0.5)*H_4))
    sigma_y = sigma_0*(-rho_a*rho_b*(H_2/2 -(b/a+0.5)*H_1))
    tau_xy = sigma_0*(-rho_a*rho_b*(H_3/2 -(b/a+0.5)*H_5))
    
    stress_vector = np.array([sigma_x, sigma_y, tau_xy])
    strain_vector = np.linalg.inv(D) @ stress_vector.T

    if type == 'stress':
        return stress_vector
    elif type == 'strain':
        return strain_vector
    else:
        raise ValueError('Unknown type: {}'.format(type))
 
class rhs_fn(shape_fns):
    def __init__(self, scale_x=[0, 40], scale_y=[0, 40]):
        self.scale_x = scale_x
        self.scale_y = scale_y
    def expression(self, x=0, y=0):
        pass


def posterior_energy(energy_list_array, DOFs_array, slope):
    if len(energy_list_array)<3:
        raise AssertionError("The value of energy should be greater than three!")
    elif len(energy_list_array)!= len(DOFs_array):
        raise AssertionError("The number of energy values should be equal to the number of DOFs!")

    Bh = abs(slope)
    i = 0
    U_list = []
    while i+3 < len(energy_list_array):
        U0, U1, U2 = energy_list_array[i:i+3]
        h0, h1, h2 = 1/np.sqrt(DOFs_array[i:i+3])
        Q = np.log((h0/h1))/np.log((h1/h2))
        lhs = lambda U: np.log(abs((U-U0)/(U-U1)))/np.log(abs((U-U1)/(U-U2)))
        initial_guess = np.mean(energy_list_array[1:])
        result = minimize(lhs, initial_guess)
        U_list.append(result.x)
        i+=1
    return np.mean(U_list)

        
if __name__=="__main__":
 
    vertices = [[0, 0], [2, 1], [0.5, 2]]
    vertices = [[1, 0.2], [2, 1.3], [0.5, 1.7]]
    vertices = [[0, 0], [1, 0], [0, 1]]
    Node_list = []
    
    t3_phi = T3_phi(p=0)

    # ????????????
    x0, x1 = [0, 1]
    y0, y1 = [0, 2]
    xi = np.linspace(x0, x1, 100)
    eta = np.linspace(y0, y1, 100)
    #    xi = 0.1
    #    eta = 0.2
    
    # ??expression???????
    output_1 = t3_phi(xi, eta)
    print('output', output_1)
    plt.imshow(output_1, origin='lower', extent=[x0, x1, y0, y1], cmap='jet')
    plt.colorbar()
    plt.title('Shape Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

   
