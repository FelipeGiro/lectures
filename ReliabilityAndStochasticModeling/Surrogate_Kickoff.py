# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 08:44:39 2020

@author: Felipe GIRO

POLYNOMIAL SURROGATE MODEL - KickOff
====================================

general investigation for surrogate model

Remarks:
- Metamodel = surrogate model
- mimics the relationship between input and output
- quadrature: int{pdf*f(x)*dx} = sum{w*f(x)}
- lambda-node Gauss quadrature for a PDF:
    - Gauss-Legendre quadrature
    - Gauss-Hermite quadrature
- Least mean square aproximate by means of Gaussian quadrature:
    min 0.5 sum{w*|g(x) - sum{c(x)^alpha}|^2}
    min 0.5*(y - [M]*c).T * [W]*[y - [M]*c]
- Solution: method of normal equations, and solve by Cholesky factorization:
    [M].T*[W]*[M]*c = [M].T*[W]*y # normal equation
    - note, Cholesky factorization: positive matrix. product leads to a 
    upper triangular matrix. and it can be solver by back-substitution
- numerical sensitive. soluion: normalization
"""

#%%
import numpy as np
import scipy
import matplotlib.pyplot as plt

#%% simple implementation

p = 14
_lambda = p + 1

# function to be studied
def g(x):
    return 1.0/(1.0 + 25.0*x**2.)

# comuting the  
x, w = np.polynomial.legendre.leggauss(_lambda)

# normal equation
M = np.array([x,]*_lambda).T
for j in range(len(M)):
    M[:, j] = np.power(M[:, j], j)
y = g(np.array(x)).T
W = np.diag(w)

# Cholesk factorization
G = np.dot(np.dot(M.T, W), M)
R = np.linalg.cholesky(G).T # lower triangular. in MatLab gives upper triangular

# solvind for d
d = np.linalg.solve(
    a = R.T, 
    b= np.dot(np.dot(M.T, W), y))

# solving for c
c = np.linalg.solve(
    a = R,
    b = d)

x_show = np.linspace(-1.0, 1.0, num=100)
y_new = np.polyval(
    p=np.flip(c),
    x=x_show)

# plotting comparisons
plt.title('p={}'.format(p))
plt.plot(x_show, g(x_show), label='g(x)')
plt.plot(x_show, y_new, label='Polynomial regression', c='red', ls=':')
plt.scatter(x, y, c='magenta')

plt.grid()
plt.legend()



