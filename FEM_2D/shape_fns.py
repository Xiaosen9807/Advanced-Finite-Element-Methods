import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

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
        
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            xi, eta = [x, y]
        else:
            xi, eta = self.gridnize(x, y)
        return  self.expression(xi, eta)
        # return np.where((self.scale_x[0] <= xi) & (xi <= self.scale_x[1]) & (self.scale_y[0] <= eta) & (eta <= self.scale_y[1]), self.expression(xi, eta), 0)


class T3_phi(shape_fns):
    def expression(self, xi, eta): 
        if self.p == 0:
             return 1-xi-eta
        elif self.p == 1:
            return xi
        elif self.p == 2:
            return eta
        else:
            raise ValueError("p should be 0, 1 or 2 in T3 element shape functions, not {}".format(self.p))

        
class T3_phipx(shape_fns):
    def expression(self, xi=0, eta=0):
        if self.p == 0:
             return -1+np.zeros_like(xi)
        elif self.p == 1:
            return 1+np.zeros_like(xi)
        elif self.p == 2:
            return 0+np.zeros_like(xi)
        else:
            raise ValueError("p should be 0, 1 or 2 in T3 element shape functions, not {}".format(self.p))

class T3_phipy(shape_fns):
    def expression(self, xi=0, eta=0):
        if self.p == 0:
             return  -1 +np.zeros_like(eta)
        elif self.p == 1:
            return 0+np.zeros_like(eta)
        elif self.p == 2:
            return 1+np.zeros_like(eta)
        else:
            raise ValueError("p should be 0, 1 or 2 in T3 element shape functions, not {}".format(self.p))


class Q4_phi(shape_fns):
    def expression(self, xi=0, eta=0):
        if self.p == 0:
             return (1 - xi) * (1 - eta)/4
        elif self.p == 1:
            return (1 + xi) * (1 - eta)/4
        elif self.p == 2:
            return (1 + xi) * (1 + eta)/4
        elif self.p == 3:
            return (1 - xi) * (1 + eta)/4
        else:
            raise ValueError("p should be 0, 1, 2 or 3 in Q4 element shape functions, not {}".format(self.p))

class Q4_phipx(shape_fns):
    def expression(self, xi, eta):
        if self.p == 0:
             return (eta - 1)/4
        elif self.p == 1:
            return (1 - eta)/4
        elif self.p == 2:
            return (1 + eta)/4
        elif self.p == 3:
            return (-1 - eta)/4
        else:
            raise ValueError("p should be 0, 1, 2 or 3 in Q4 element shape functions, not {}".format(self.p))       


class Q4_phipy(shape_fns):
    def expression(self, xi=0, eta=0):
        if self.p == 0:
            return (xi - 1)/4
        elif self.p == 1:
            return -(xi - 1)/4
        elif self.p == 2:
            return (1 + xi)/4
        elif self.p == 3:
            return (1 - xi)/4
        else:
            raise ValueError("p should be 0, 1, 2 or 3 in Q4 element shape functions, not {}".format(self.p))


class exact_fn(shape_fns):
    def __init__(self, a=20, b=20, scale_x=[0, 40], scale_y=[0, 40], dire='xx'):
        #unit: mm
        assert dire not in ['xx', 'yy', 'xy'], "The stress  direction should be 'xx', 'yy' or 'xy'"
        self.dire = dire
        self.a = a
        self.b = b
        self.scale_x = scale_x
        self.scale_y = scale_y
    def expression(self, x, y):
        a = self.a
        b = self.b
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

        if self.dire == 'xx':
            return  sigma_0*(1-rho_a*rho_b*(H_1/2 -(b/a+0.5)*H_4))
        elif self.dire == 'yy':
            return sigma_0*(-rho_a*rho_b*(H_2/2 -(b/a+0.5)*H_1))
        elif self.dire == 'xy':
            return sigma_0*(-rho_a*rho_b*(H_3/2 -(b/a+0.5)*H_5))
        
class rhs_fn(shape_fns):
    def __init__(self, scale_x=[0, 40], scale_y=[0, 40]):
        self.scale_x = scale_x
        self.scale_y = scale_y
    def expression(self, x=0, y=0):
        pass




        
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
    output = t3_phi(xi, eta)
    print('output', output)
    plt.imshow(output, origin='lower', extent=[x0, x1, y0, y1], cmap='jet')
    plt.colorbar()
    plt.title('Shape Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

   
