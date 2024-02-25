# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import sympy as sp


def fem1d_quadratic(f, d2f, n_num=11):

    #  Define the interval.
    #
    a = 0.0
    b = 1.0
#
#  Define the mesh, N_NUM evenly spaced points between A and B.
#  Because we are using quadratic elements, we need N_NUM to be odd!
#

    x = np.linspace(a, b, n_num)
    e_num = int((n_num - 1) / 2)
    print('e_num', e_num)


#
#  Set a 3 point quadrature rule on the reference interval [-1,1].
#
    q_num = 3

    xg = np.array((
        -0.774596669241483377035853079956,
        0.0,
        0.774596669241483377035853079956))

    wg = np.array((
        5.0 / 9.0,
        8.0 / 9.0,
        5.0 / 9.0))

#
#  Set a 3 point quadrature rule on the reference interval [0,1].
#
    # q_num = 3

    # xg = np.array((
    #     0.112701665379258311482073460022,
    #     0.5,
    #     0.887298334620741688517926539978))

    # wg = np.array((
    #     5.0 / 18.0,
    #     8.0 / 18.0,
    #     5.0 / 18.0))
#
#  Compute the system matrix A and right hand side RHS.
#
    A = np.zeros((n_num, n_num))
    rhs = np.zeros(n_num)
#
#  Look at element E: (0, 1, 2, ..., E_NUM-1).
#
    for e in range(0, e_num):
        print('e', e)

        l = 2 * e
        m = 2 * e + 1
        r = 2 * e + 2

        xl = x[l]
        xm = x[m]
        xr = x[r]
        print('l, m, r', l, m, r)
        print('xl, xm, xr', xl, xm, xr)
#
#  Consider quadrature point Q: (0, 1, 2 ) in element E.
#
        for q in range(0, q_num):
            #
            #  Map XG and WG from [-1,1] to
            #      XQ and WQ in [XL,XM,XR].
            #
            # xq = xl + (xg[q] + 1.0) * (xr - xl) / 2.0
            # wq = wg[q] * (xr - xl) / 2.0
            xq = ((1.0 - xg[q]) * xl + (1.0 + xg[q]) * xr) / 2.0
            wq = wg[q] * (xr - xl) / 2.0
#
#  Evaluate PHI(L), PHI(M) and PHI(R), and their derivatives at XQ.
#
#  It must be true that PHI(L) is 1 at XL and 0 at XM and XR,
#  with similar requirements for PHI(M) and PHI(R).
#
            phil = (xq - xm) * (xq - xr) / \
                ((xl - xm) * (xl - xr))
            philp = ((xq - xr) + (xq - xm)) / \
                    ((xl - xm) * (xl - xr))

            phim = (xq - xl) * (xq - xr) / \
                ((xm - xl) * (xm - xr))
            phimp = ((xq - xr) + (xq - xl)) / \
                    ((xm - xl) * (xm - xr))

            phir = (xq - xl) * (xq - xm) / \
                ((xr - xl) * (xr - xm))
            phirp = ((xq - xm) + (xq - xl)) / \
                    ((xr - xl) * (xr - xm))

            fxq = -d2f(xq)
#
#  Add the terms from this element to the matrix.
#
            A[l][l] = A[l][l] + wq * philp * philp
            A[l][m] = A[l][m] + wq * philp * phimp
            A[l][r] = A[l][r] + wq * philp * phirp
            rhs[l] = rhs[l] + wq * phil * fxq

            A[m][l] = A[m][l] + wq * phimp * philp
            A[m][m] = A[m][m] + wq * phimp * phimp
            A[m][r] = A[m][r] + wq * phimp * phirp
            rhs[m] = rhs[m] + wq * phim * fxq

            A[r][l] = A[r][l] + wq * phirp * philp
            A[r][m] = A[r][m] + wq * phirp * phimp
            A[r][r] = A[r][r] + wq * phirp * phirp
            rhs[r] = rhs[r] + wq * phir * fxq
  
#
#  Modify the linear system to enforce the left boundary condition.
#
    A[0, 0] = 1.0
    A[0, 1:n_num-1] = 0.0
    rhs[0] = f(x[0])
#
#  Modify the linear system to enforce the right boundary condition.
#
    A[n_num-1, n_num-1] = 1.0
    A[n_num-1, 0:n_num-1] = 0.0
    rhs[n_num-1] = f(x[n_num-1])
    print('AAAAAAA', A)

    #r8vec_print ( n_num, rhs, '  RHS' )
#
#  Solve the linear system.
#
    #u = la.solve(A, rhs)
    u = la.solve(A, rhs)
    
    #print('xxxxxxx', u)
#
#  Evaluate the exact solution at the nodes.
#
    uex = np.zeros(n_num)
    for i in range(0, n_num):
        uex[i] = f(x[i])
#
#  Compare the solution and the error at the nodes.
# #
#     print("")
#     print("  Node          Ucomp           Uexact          Error")
#     print("")
    err_tot = []
    for i in range(0, n_num):
        err = abs(uex[i] - u[i])
        err_tot.append(err)
#         print("  %4d  %14.6g  %14.6g  %14.6g" % (i, u[i], uex[i], err))
#
#  Plot the computed solution and the exact solution.
#  Evaluate the exact solution at enough points that the curve will look smooth.
#
    npp = 51
    xp = np.linspace(a, b, npp)
    up = np.zeros(npp)
    for i in range(0, npp):
        up[i] = f(xp[i])

    plt.plot(x, u, 'bo-', label='u')
    plt.plot(xp, up, 'r.', label='up')
    plt.title('h_quadratic')
    plt.legend()
    # plt.show()

    return err_tot, u, up


def exact_fn(x):
    #
    # EXACT_FN evaluates the exact solution.
    #

    value = (1 - x) * (np.arctan(a * (x - xb)) + np.arctan(a*xb))

    return value


def rhs_fn(x):
    #
    # RHS_FN evaluates the right hand side.
    #
    B = x-xb
    value = -2*(a+a**3*B*(B-x+1))/(a**2*B**2+1)**2
    return value


#
#  If this script is called directly, then run it as a program.
#
if (__name__ == '__main__'):
    a = 50
    xb = 0.8
    mesh_list = [5]
    for i in mesh_list:
        err, u, up = fem1d_quadratic(exact_fn, rhs_fn, i)
    # print(err)
    # plt.plot(err)
    # %%
