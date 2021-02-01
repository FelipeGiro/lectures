# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:07:30 2020

@author: flg12


EXAMPLE OF RELIABILITY CLASS
============================
"""
import numpy as np


#%% Function definition
def g(x, y):
    return x + y**2 + x*y

g_0 = 0

def g_x(x):
    return x

def g_y(y):
    return y**2

def g_xy(x, y):
    return x*y

#%% Sensitivity indeces computation

size = 1500
X_axis = np.random.uniform(-1, 1, size=size)
Y_axis = np.random.uniform(-1, 1, size=size)

X, Y = np.meshgrid(X_axis, Y_axis)

Z = g(X, Y)

var_Z = np.var(Z)

s_x = np.var(np.mean(Z, axis=0))
s_y = np.var(np.mean(Z, axis=1))
s_xy = var_Z - s_x - s_y

print('\nV{E{Z|X}}')
print('s_x:', s_x/var_Z*100 ,'%')
print('s_y:', s_y/var_Z*100,'%')
print('s_xy:',s_xy/var_Z*100 ,'%')

#%% Monte Carlo integration

left = g(X, Y) - np.mean(g(X, Y))

s_x = np.mean(left*(g_x(X) - np.mean(g_x(X))))
s_y = np.mean(left*(g_y(Y) - np.mean(g_y(Y))))
# s_xy = var_Z - s_x - s_y
s_xy = np.mean(left*(g_xy(X, Y) - np.mean(g_xy(X, Y))))
print('\nMonte Carlo integration')
print('s_x:', s_x/var_Z*100 ,'%')
print('s_y:', s_y/var_Z*100,'%')
print('s_xy:',s_xy/var_Z*100 ,'%')