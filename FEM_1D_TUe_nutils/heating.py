## Standard library imports
import numpy as np
import timeit

## FEniCSx related imports
import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner, nabla_grad

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

### ------ Problem domain description (with the example of cantilever problem)-------------
## Geometry description
x0 = 0.0; y0 = 0.0 
x1 = 2.0; y1 = 1.0
nx = 100; ny = 50
h = 0.10 # half length on right edge with load
trac0 = -1.0 # applied traction

## Material parameters
E = 1.0
nu = 0.3
lambda_ = E*nu/(1+nu)/(1-2*nu)
mu = E/(2*(1+nu))
## ---------------------------------------------


### ------ Function for the calculation of stress -----
def epsilon(v):
    return ufl.sym(ufl.grad(v)) 
def sigma(v):
    """Return an expression for the stress \sigma given a displacement field v """
    return lambda_ * ufl.nabla_div(v) * ufl.Identity(len(v)) + 2*mu*epsilon(v)
## ---------------------------------------------


### ------- Mesh generation and boundary condition facets -----------------
msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                                points=((x0, y0), (x1, y1)), n = (nx, ny),
                                cell_type=mesh.CellType.quadrilateral,) 

## ---------------------------------------------------


### ------- Function spaces and functions for density and displacement fields -------------
## Function space for displacement
V_disp = fem.VectorFunctionSpace(msh, ("Lagrange", 1))
## Function space for density field
V_density = fem.FunctionSpace(msh, ("DG", 0))

## Test and trial function for displacement solution
u, du = ufl.TestFunction(V_disp), ufl.TrialFunction(V_disp)
u_sol = fem.Function(V_disp)

## -----------------------------------------------------

### --------- Setup variational problem -----------------
## Apply dirichlet boundary condition
dofs_dirchlet_zero = fem.locate_dofs_geometrical(V_disp, marker = lambda x: np.isclose(x[0], x0))
bc = fem.dirichletbc(value= np.array([0,0], dtype=ScalarType), dofs= dofs_dirchlet_zero, V= V_disp)

## Neumann bc
T = fem.Constant(msh, ScalarType((0, trac0))) # traction bc
dofs_neumann = fem.locate_dofs_geometrical(V_disp, marker = lambda x: np.logical_and(np.isclose(x[0], x1), 
                                                        np.logical_and( x[1]<= y1/2 + h, x[1] >= y1/2 - h)))
#print(dofs_neumann)

meshtags_neumann = mesh.meshtags(mesh = msh, dim = (msh.topology.dim - 1), entities = dofs_neumann, values= 1)
ds = ufl.Measure("ds", subdomain_data = meshtags_neumann) #  
print(meshtags_neumann.indices)
print(meshtags_neumann.values)

## Variational problem
a = ufl.inner(sigma(u), epsilon(du)) * dx
L =  ufl.dot(T, du) * ds(1)

problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_sol = problem.solve()

## ---------------------------------------------------

with io.XDMFFile(msh.comm, "data/check_elasticity_sol.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(u_sol)