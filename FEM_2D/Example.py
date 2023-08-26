from tools_2D import *
from shape_fns import *
from Elements import *
from Energy import *
from Mesh import create_mesh, Boundary
#!/usr/bin/env python
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# Copyright 2022 Jorgen Bergstrom
import sys
import numpy as np
import math
from matplotlib import pyplot as plt
# Input file syntax:
# *Node
##    1, 0.0, 0.0
##    2, 0.0, 1.0
##    3, 1.0, 1.0
##    4, 1.0, 0.0
# *Element
##    1, 1, 2, 3, 4
# *Step
# *Boundary
# 1, 1, 2, 0.0          # nodeId, dof1, dof2, value
##    2, 1, 1, 0.0
##    3, 1, 1, 0.01
##    4, 1, 1, 0.01
##    4, 2, 2, 0.0


def shape(xi):
    """Shape functions for a 4-node, isoparametric element
            N_i(xi,eta) where i=[1,2,3,4]
            Input: 1x2,  Output: 1x4"""
    xi, eta = tuple(xi)
    N = [(1.0-xi)*(1.0-eta), (1.0+xi)*(1.0-eta),
         (1.0+xi)*(1.0+eta), (1.0-xi)*(1.0+eta)]
    return 0.25 * np.array(N)


def gradshape(xi):
    """Gradient of the shape functions for a 4-node, isoparametric element.
            dN_i(xi,eta)/dxi and dN_i(xi,eta)/deta
            Input: 1x2,  Output: 2x4"""
    xi, eta = tuple(xi)
    dN = [[-(1.0-eta),  (1.0-eta), (1.0+eta), -(1.0+eta)],
          [-(1.0-xi), -(1.0+xi), (1.0+xi),  (1.0-xi)]]
    return 0.25 * np.array(dN)


def local_error(str):
    print("*** ERROR ***")
    print(str)
    sys.exit(3)


def read_inp_file(inpFileName, nodes, conn, boundary):
    print('\n** Read input file')
    path = '/Users/xusenqin/Desktop/Advanced-Finite-Element-Methods/FEM_2D/Dogbone_Tension.input'
    # inpFile = open(inpFileName, 'r')
    inpFile = open(path, 'r')
    lines = inpFile.readlines()
    inpFile.close()
    state = 0
    for line in lines:
        line = line.strip()
        if len(line) <= 0:
            continue
        if line[0] == '*':
            state = 0
        if line.lower() == "*node":
            state = 1
            continue
        if line.lower() == "*element":
            state = 2
            continue
        if line.lower() == "*boundary":
            state = 3
            continue
        if state == 0:
            continue
        if state == 1:
            # read nodes
            values = line.split(",")
            if len(values) != 3:
                local_error("A node definition needs 3 values")
            nodeNr = int(values[0]) - 1  # zero indexed
            xx = float(values[1])
            yy = float(values[2])
            nodes.append([xx, yy])   # assume the nodes are ordered 1, 2, 3...
            continue
        if state == 2:
            # read elements
            values = line.split(",")
            if len(values) != 5:
                local_error("An element definition needs 5 values")
            elemNr = int(values[0])
            n1 = int(values[1]) - 1  # zero indexed
            n2 = int(values[2]) - 1
            n3 = int(values[3]) - 1
            n4 = int(values[4]) - 1
            # conn.append([n1, n2, n3, n4]) # assume elements ordered 1, 2, 3
            conn.append([n1, n4, n3, n2])  # assume elements ordered 1, 2, 3
            continue
        if state == 3:
            # read displacement boundary conditions
            values = line.split(",")
            if len(values) != 4:
                local_error("A displacement boundary condition needs 4 values")
            nodeNr = int(values[0]) - 1  # zero indexed
            dof1 = int(values[1])
            dof2 = int(values[2])
            val = float(values[3])
            if dof1 == 1:
                boundary.append([nodeNr, 1, val])
            if dof2 == 2:
                boundary.append([nodeNr, 2, val])
            continue


def main():
    ##
    # Main Program
    ##
    a_b = 1
    mesh_size = 2
    mesh_shape = 1
    nodes = []
    conn = []
    boundary = []
    # if len(sys.argv) <= 1: local_error('No input file provided.')
    # print('Input file:', sys.argv[1])
    # read_inp_file(sys.argv[1], nodes, conn, boundary)
    nodes_coord,  element_nodes = create_mesh(a_b, mesh_shape, mesh_size)
# with open("Coords/saved_data.pkl", "rb") as file:
#    nodes_coord,  element_nodes = pickle.load(file)
    # nodes_list = Boundary(nodes_coord, a_b)
    element_list = []
    if mesh_shape == 0:
        element_nodes = element_nodes.reshape(-1, 3)
    elif mesh_shape == 1:
        element_nodes = element_nodes.reshape(-1, 4)
    nodes = np.array(nodes_coord)
    num_nodes = len(nodes)
    conn = np.array(element_nodes, dtype=int)
    for i, node_i in enumerate(nodes):
        if abs(node_i[0]-0) <= 1e-3:
            nn = i
            dof = 1
            val = 0
            boundary.append([nn, dof, val])
        if abs(node_i[1]-0) <= 1e-3:
            nn = i
            dof = 2
            val = 0
            boundary.append([nn, dof, val])
        if abs(node_i[0]-40) <= 1e-3:
            nn = i
            dof = 0
            val = 50
            boundary.append([nn, dof, val])
    boundary = np.array(boundary)
    print('   number of nodes:', len(nodes))
    print('   number of elements:', len(conn))
    print('   number of displacement boundary conditions:', len(boundary))

    ###############################
    # Plane-strain material tangent (see Bathe p. 194)
    # C is 3x3
    E = 200000.0
    nu = 0.3
    C = E / (1 - nu**2) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2]
    ])  # Plain Stress
    ###############################
    # Make stiffness matrix
    # if N is the number of DOF, then K is NxN
    K = np.zeros((2*num_nodes, 2*num_nodes))    # square zero matrix
    # 2x2 Gauss Quadrature (4 Gauss points)
    # q4 is 4x2
    q4 = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]]) / math.sqrt(3.0)
    print('\n** Assemble stiffness matrix')
    # strain in an element: [strain] = B    U
    #                        3x1     = 3x8  8x1
    #
    # strain11 = B11 U1 + B12 U2 + B13 U3 + B14 U4 + B15 U5 + B16 U6 + B17 U7 + B18 U8
    #          = B11 u1          + B13 u1          + B15 u1          + B17 u1
    #          = dN1/dx u1       + dN2/dx u1       + dN3/dx u1       + dN4/dx u1
    B = np.zeros((3, 8))
    # conn[0] is node numbers of the element
    for c in conn:     # loop through each element
        # coordinates of each node in the element
        # shape = 4x2
        # for example:
        #    nodePts = [[0.0,   0.0],
        #               [0.033, 0.0],
        #               [0.033, 0.066],
        #               [0.0,   0.066]]
        nodePts = nodes[c, :]
        Ke = np.zeros((8, 8))  # element stiffness matrix is 8x8
        for q in q4:			# for each Gauss point
            # q is 1x2, N(xi,eta)
            dN = gradshape(q)       # partial derivative of N wrt (xi,eta): 2x4
            J = np.dot(dN, nodePts).T  # J is 2x2
            # print(J, dN)
            # partial derivative of N wrt (x,y): 2x4
            dN = np.dot(np.linalg.inv(J), dN)
            # assemble B matrix  [3x8]
            B[0, 0::2] = dN[0, :]
            B[1, 1::2] = dN[1, :]
            B[2, 0::2] = dN[1, :]
            B[2, 1::2] = dN[0, :]
            # print(B)
            # element stiffness matrix
            Ke += np.dot(np.dot(B.T, C), B) * np.linalg.det(J)
            # print('np.linalg.det(J)', np.linalg.det(J))
        # print(Ke)
        # Scatter operation
        for i, I in enumerate(c):
            for j, J in enumerate(c):
                I, J = int(I), int(J)
                K[2*I, 2*J] += Ke[2*i, 2*j]
                K[2*I+1, 2*J] += Ke[2*i+1, 2*j]
                K[2*I+1, 2*J+1] += Ke[2*i+1, 2*j+1]
                K[2*I, 2*J+1] += Ke[2*i, 2*j+1]
    ###############################
    # Assign nodal forces and boundary conditions
    #    if N is the number of nodes, then f is 2xN
    f = np.zeros((2*num_nodes))          # initialize to 0 forces
    print(boundary)
    # print(conn)
    for elem in conn:
        # exit()
        for node_id in elem:
            # print(node_id)
            # print(boundary[:, 0])
            if node_id in boundary[:, 0]:
                # exit()
                idx = np.where(boundary[:, 0] == node_id)[0][0]
                _id = int(node_id)
                # print(idx)
                # print(boundary[idx])
                if boundary[idx][1] == 0:
                    f[_id*2] += 50

    # print(f)
    # How about displacement boundary conditions:
    #    [k11 k12 k13] [u1] = [f1]
    #    [k21 k22 k23] [u2]   [f2]
    #    [k31 k32 k33] [u3]   [f3]
    #
    #    if u3=x then
    #       [k11 k12 k13] [u1] = [f1]
    #       [k21 k22 k23] [u2]   [f2]
    #       [k31 k32 k33] [ x]   [f3]
    #   =>
    #       [k11 k12 k13] [u1] = [f1]
    #       [k21 k22 k23] [u2]   [f2]
    #       [  0   0   1] [u3]   [ x]
    #   the reaction force is
    #       f3 = [k31 k32 k33] * [u1 u2 u3]
    for i in range(len(boundary)):  # apply all boundary displacements
        nn = boundary[i][0]
        dof = boundary[i][1]
        val = boundary[i][2]
        j = 2*nn
        if dof == 2 or dof == 1:
            j = j + dof - 1
            K[j, :] = 0.0
            K[j, j] = 1.0
            f[j] = 0
    print(K)

    ###############################
    # [K] = 2N x 2N, [f] = 2N x 1, [u] = 2N x 1
    print('\n** Solve linear system: Ku = f')
    u = np.linalg.solve(K, f)
    ###############################
    print('\n** Post process the data')
    # (pre-allocate space for nodal stress and strain)
    node_strain = []
    node_stress = []
    for ni in range(len(nodes)):
        node_strain.append([0.0, 0.0, 0.0])
        node_stress.append([0.0, 0.0, 0.0])
    node_strain = np.array(node_strain)
    node_stress = np.array(node_stress)

    print(
        f'   min displacements: u1={min(u[0::2]):.4g}, u2={min(u[1::2]):.4g}')
    print(
        f'   max displacements: u1={max(u[0::2]):.4g}, u2={max(u[1::2]):.4g}')
    emin = np.array([9.0e9,  9.0e9,  9.0e9])
    emax = np.array([-9.0e9, -9.0e9, -9.0e9])
    smin = np.array([9.0e9,  9.0e9,  9.0e9])
    smax = np.array([-9.0e9, -9.0e9, -9.0e9])
    for c in conn:  # for each element (conn is Nx4)
        print(c)
        # c is like [2,5,22,53]
        # 4x2, eg: [[1.1,0.2], [1.2,0.3], [1.3,0.4], [1.4, 0.5]]
        nodePts = nodes[c, :]
        for q in q4:					# for each integration pt, eg: [-0.7,-0.7]
            dN = gradshape(q)					# 2x4
            J = np.dot(dN, nodePts).T			# 2x2
            dN = np.dot(np.linalg.inv(J), dN)  # 2x4
            B[0, 0::2] = dN[0, :]					# 3x8
            B[1, 1::2] = dN[1, :]
            B[2, 0::2] = dN[1, :]
            B[2, 1::2] = dN[0, :]

            UU = np.zeros((8, 1))				# 8x1
            UU[0] = u[2*c[0]]
            UU[1] = u[2*c[0] + 1]
            UU[2] = u[2*c[1]]
            UU[3] = u[2*c[1] + 1]
            UU[4] = u[2*c[2]]
            UU[5] = u[2*c[2] + 1]
            UU[6] = u[2*c[3]]
            UU[7] = u[2*c[3] + 1]
            # get the strain and stress at the integration point
            strain = B @ UU		# (B is 3x8) (UU is 8x1) 		=> (strain is 3x1)
            # (C is 3x3) (strain is 3x1) 	=> (stress is 3x1)
            stress = C @ strain
            emin[0] = min(emin[0], strain[0][0])
            emin[1] = min(emin[1], strain[1][0])
            emin[2] = min(emin[2], strain[2][0])
            emax[0] = max(emax[0], strain[0][0])
            emax[1] = max(emax[1], strain[1][0])
            emax[2] = max(emax[2], strain[2][0])

            node_strain[c[0]][:] = strain.T[0]
            node_strain[c[1]][:] = strain.T[0]
            node_strain[c[2]][:] = strain.T[0]
            node_strain[c[3]][:] = strain.T[0]
            node_stress[c[0]][:] = stress.T[0]
            node_stress[c[1]][:] = stress.T[0]
            node_stress[c[2]][:] = stress.T[0]
            node_stress[c[3]][:] = stress.T[0]
            smax[0] = max(smax[0], stress[0][0])
            smax[1] = max(smax[1], stress[1][0])
            smax[2] = max(smax[2], stress[2][0])
            smin[0] = min(smin[0], stress[0][0])
            smin[1] = min(smin[1], stress[1][0])
            smin[2] = min(smin[2], stress[2][0])
    print(
        f'   min strains: e11={emin[0]:.4g}, e22={emin[1]:.4g}, e12={emin[2]:.4g}')
    print(
        f'   max strains: e11={emax[0]:.4g}, e22={emax[1]:.4g}, e12={emax[2]:.4g}')
    print(
        f'   min stress:  s11={smin[0]:.4g}, s22={smin[1]:.4g}, s12={smin[2]:.4g}')
    print(
        f'   max stress:  s11={smax[0]:.4g}, s22={smax[1]:.4g}, s12={smax[2]:.4g}')
    ###############################
    print('\n** Plot displacement')
    xvec = []
    yvec = []
    res = []
    plot_type = 'u1'
    for ni, pt in enumerate(nodes):
        xvec.append(pt[0] + u[2*ni])
        yvec.append(pt[1] + u[2*ni+1])
        if plot_type == 'u1':
            res.append(u[2*ni])				# x-disp
        if plot_type == 'u2':
            res.append(u[2*ni+1])				# y-disp
        if plot_type == 's11':
            res.append(node_stress[ni][0])		# s11
        if plot_type == 's22':
            res.append(node_stress[ni][1])		# s22
        if plot_type == 's12':
            res.append(node_stress[ni][2])		# s12
        if plot_type == 'e11':
            res.append(node_strain[ni][0])		# e11
        if plot_type == 'e22':
            res.append(node_strain[ni][1])		# e22
        if plot_type == 'e12':
            res.append(node_strain[ni][2])		# e12
    tri = []
    for c in conn:
        tri.append([c[0], c[1], c[2]])
        tri.append([c[0], c[2], c[3]])
    t = plt.tricontourf(xvec, yvec, res, triangles=tri,
                        levels=14, cmap=plt.cm.jet)
    # plt.scatter(xvec, yvec, marker='o', c='b', s=0.5) # (plot the nodes)
    plt.grid()
    plt.colorbar(t)
    plt.title(plot_type)
    plt.axis('equal')
    plt.show()
    print('Done.')


if __name__=='__main__':
    main()
