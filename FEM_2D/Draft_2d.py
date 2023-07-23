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

def point_in_polygon(polygon, point):
    count = 0
    n = len(polygon)
    for i in range(n):
        p1, p2 = polygon[i], polygon[(i + 1) % n]
        
        # ???????
        if np.all(p1 == point) or np.all(p2 == point):
            return True

        # ????????
        if p1[1] == p2[1]:
            if point[1] == p1[1] and min(p1[0], p2[0]) <= point[0] <= max(p1[0], p2[0]):
                return True
        # ??????????
        elif min(p1[1], p2[1]) <= point[1] < max(p1[1], p2[1]):
            x = (point[1] - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]  # ??? x ??
            # ??????
            if x == point[0]:
                return True
            # ???????
            elif x > point[0]:
                count += 1
    # ???????????? True????? False
    return count % 2 == 1

# ??????????
A = np.array([0, 0])
B = np.array([1, 0])
C = np.array([1, 1])
D = np.array([0.5, 1.5])
E = np.array([0, 1])
P = np.array([0, -1])
points_list = np.array([A, B, C, D, E, P])


print(point_in_polygon([A, B, C, D, E], P))  # True
import numpy as np

# ?????????
import sympy as sp

x1, x2, x3, x4, y1, y2, y3, y4, eta, xi = sp.symbols('x1, x2, x3, x4, y1, y2, y3, y4, eta, xi')
X = np.array([x1, x2, x3, x4])
Y = np.array([y1, y2, y3, y4])

# ??Q4????????ksi?eta????
dN_dksi = np.array([-0.25*(1-eta), 0.25*(1-eta), 0.25*(1+eta), -0.25*(1+eta)])
dN_deta = np.array([-0.25*(1-xi), -0.25*(1+xi), 0.25*(1+xi), 0.25*(1-xi)])

# ??Jacobian??
J = np.array([[np.dot(dN_dksi, X), np.dot(dN_deta, X)], 
              [np.dot(dN_dksi, Y), np.dot(dN_deta, Y)]])
