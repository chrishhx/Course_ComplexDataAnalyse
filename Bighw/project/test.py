# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 19:43:00 2017

@author: chris
"""

import time
import numpy as np
from scipy import stats

X = np.zeros([100000,2])
MU = np.array([0,0])
Sigma = np.matrix([[1,0],[0,1]])

t0 = time.clock()
for i in range(len(X)):
    stats.multivariate_normal.pdf(X[i],MU,Sigma)
print (time.clock()-t0)

t0 = time.clock()
stats.multivariate_normal.pdf(X,MU,Sigma)
print (time.clock()-t0)