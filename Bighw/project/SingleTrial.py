# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 08:51:24 2017

@author: chris
"""

from mymodule.GMM import GMM
import numpy as np
from scipy import stats

g = 3
N = 1000
P = [0.5,
     0.3,
     0.2]
MU = [[-3,0],
      [0,0],
      [0,3]]
SIGMA = [[[1,0],[0,1]],
         [[1,0.2],[0.2,1]],
         [[1,0.7],[0.7,1]]]

# single trial
np.random.seed(123)
group = np.random.multinomial(N,P)
Y = np.random.multivariate_normal(MU[0],SIGMA[0],group[0])
for i in range(1,len(P)):
    Y = np.append(Y,np.random.multivariate_normal(MU[i],SIGMA[i],group[i]),axis=0)

model = GMM(Y,g,True)
#model = GMM(Y,g,False)

# plot for dimension = 2
from matplotlib import pyplot as plt
right=0
colors = ['tab:blue','tab:orange','tab:olive']
fig = plt.figure()
for i in range(g):
    left = right
    right += group[i]
    plt.scatter(Y[left:right,0],Y[left:right,1],alpha=0.6,color=colors[i])
    a,b = np.arange(min(Y[:,0]),max(Y[:,0]),0.1),np.arange(min(Y[:,1]),max(Y[:,1]),0.1)
    A,B = np.meshgrid(a,b)
    C = np.zeros(shape=A.shape)
    for j in range(len(A)):
        for k in range(len(A[j])):
            C[j,k] = stats.multivariate_normal.pdf([A[j,k],B[j,k]],model.mu[i],model.sigma[i])
    plt.contour(A,B,C)
fig.savefig("E:/workspace/abc.png",dpi=200)
