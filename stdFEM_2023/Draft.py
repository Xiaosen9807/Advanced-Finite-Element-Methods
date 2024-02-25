import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_legendre



def G_integrate_2D(u, N=3, scale_x=(0, 1), scale_y=(0, 1)):

    ax, bx = scale_x
    ay, by = scale_y


    x, wx = roots_legendre(N)
    y, wy = roots_legendre(N)


    x = x * (bx - ax) / 2 + (bx + ax) / 2
    y = y * (by - ay) / 2 + (by + ay) / 2


    integral = 0.0
    for i in range(N):
        for j in range(N):
            integral += wx[i] * wy[j] * u(x[i], y[j])

    integral *= (bx - ax) * (by - ay) / 4

    return integral

X = np.array([[0, 0],
              [1, 0],
              [0, 1]])

dN = np.array([[-1, -1],
               [1, 0],
               [0, 1]])
def jacobian(X, dN):
    # compute Jacobian matrix
    J = np.dot(np.transpose(X),dN)
    # print(np.transpose(X))
    print(J)
    return np.linalg.inv(J),np.linalg.det(J)

X = [[0.4, 0.5]]
dN = [[0,1]]
J_inv, detJ = jacobian(X, dN)
print(np.transpose(X))
print( np.dot(np.transpose(X),dN))
print(J_inv, detJ)

