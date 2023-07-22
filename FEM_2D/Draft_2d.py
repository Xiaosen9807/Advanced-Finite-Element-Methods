import numpy as np
print('hello world!')
import sympy as sym
import numpy as np
from sympy.matrices import Matrix

# ?????????????? (x0, y0), (x1, y1), (x2, y2)
x0, x1, x2, y0, y1, y2 = sym.symbols(['x_0', 'x_1', 'x_2', 'y_0', 'y_1', 'y_2'])

# ????????????
J = Matrix([[x1-x0, y1-y0], 
            [x2-x0, y2-y0]])

# ??????????? (eta, zeta)
eta, zeta = sym.symbols(['\eta', '\zeta'])
u0 = 1 - eta - zeta
u1 = eta
u2 = zeta

# ??????? [du/deta, du/dzeta]
du = Matrix([[sym.diff(u0, eta),sym.diff(u1, eta),sym.diff(u2, eta)],
[sym.diff(u0, zeta), sym.diff(u1, zeta), sym.diff(u2, zeta)]])

# ???????? du/dx, du/dy = J^{-1} * [du/deta, du/dzeta]
def deriv_u(): 
    return sym.simplify(J.inv() * du)

# ??? e ???? 9 ?????????????? 1/2|J|
Ke = sym.zeros(3,3)
du_ =  deriv_u() * J.det()
for i in range(3):
    for j in range(3):
        Ke[i, j] = du_[:, i].dot(du_[:, j])
print(Ke)        

def integrate_ukf(k=0):
    f0, f1, f2 = sym.symbols(['f_0', 'f_1', 'f_2'])
    f = u0 * f0 + u1 * f1 + u2 * f2
    u = [u0, u1, u2]
    feta = sym.integrate(u[k] * f, (zeta, 0, 1-eta))
    return sym.integrate(feta, (eta, 0, 1))

F = integrate_ukf(k=0)
print(F)
