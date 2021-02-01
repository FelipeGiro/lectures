# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 18:48:55 2020

@author: flg12
"""

import chaospy
import numpy
#%% Probability distributions
length = 3

dist_a0    = chaospy.Iid(chaospy.Exponential(0.1603), length)
dist_logCa = chaospy.Iid(chaospy.Normal(mu= -27.6302, sigma=0.4599), length)
dist_S     = chaospy.Iid(chaospy.Normal(mu= 7.3463, sigma=0.3*7.3463), length)

#%% generate orthogonal expansion
orthogonal_expansion = chaospy.generate_expansion(1, dist_logCa)
print(orthogonal_expansion)

#%%
test = chaospy.monomial(1, 4, dimensions=3, cross_truncation=numpy.inf)
print(test)