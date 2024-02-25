import sys
from math import pi, cos
import numpy as np

import collections, itertools
import numpy as np


Quadrature = collections.namedtuple("Quadrature", "wgt xi")

def grule(n):
    """Code used to compute a Gaussian integration rule of n points
    
    The code was adapted to Python from Fortran code found in the book
    "Methods of Numerical Integration", by P. J. Davis, P. Rabinowitz
    and W Rheinbolt, page 487"""
    
    assert n>0, "ERROR: Cannot get Gauss rule of {} points"
    
    xi = np.zeros(n)
    wgt = np.zeros(n)
    
    m = (n+1)//2
    e1 = n*(n+1)
    for i in range(m):
        t = (4*i+3)*pi / (4*n+2)
        x0 = (1-(1-1/n)/(8*n*n))*cos(t)
        pkm1 = 1
        pk = x0
        for k in range(2,n+1):
            t1 = x0*pk
            pkp1 = t1-pkm1-(t1-pkm1)/k+t1
            pkm1=pk
            pk=pkp1

        den=1-x0*x0
        d1=n*(pkm1-x0*pk)
        dpn=d1/den
        d2pn=(2*x0*dpn - e1*pk)/den
        d3pn=(4*x0*d2pn+(2-e1)*dpn)/den
        d4pn=(6*x0*d3pn+(6-e1)*d2pn)/den
        u=pk/dpn
        v=d2pn/dpn
        h=-u*(1+0.5*u*(v+u*(v*v-d3pn/(3*dpn))))
        p=pk+h*(dpn+0.5*h*(d2pn+h/3*(d3pn+0.25*h*d4pn)))
        dp=dpn+h*(d2pn+0.5*h*(d3pn+h*d4pn/3))
        h=h-p/dp
        xi[i]=-x0-h
        fx=d1-h*e1*(pk+0.5*h*(dpn+h/3*(d2pn+0.25*h*(d3pn+0.2*h*d4pn))))
        wgt[i]=2*(1-xi[i]*xi[i])/(fx*fx)

    # copy values to the rest of the arrays
    xi[-1:-m-1:-1] = -xi[:m]
    wgt[-1:-m-1:-1] =  wgt[:m]

    if m+m > n:
        xi[m-1] = 0

    return Quadrature(wgt, xi)
    

def cartesian2D(wgt, xi):
    return [i*j for i in wgt for j in wgt], list(itertools.product(xi, xi))

def quadrature(etype, gpts):
    if etype == 2:
        return grule(gpts)
    elif etype == 3:
        return eval('tri_{}gp'.format(gpts))()
    elif etype == 4:
        rule = grule(gpts)
        return Quadrature(*cartesian2D(rule.wgt, rule.xi))

# triangle integration rules

def tri_1gp():
    xi   = [(1/3., 1/3.)]
    wgt  = [1/2.]
    return Quadrature(wgt, xi)

def tri_3gp():
    xi = [(2/3., 1/6.), (1/6., 1/6.), (1/6., 2/3.)]
    wgt  = [1/6., 1/6., 1/6.]
    return Quadrature(wgt, xi)
