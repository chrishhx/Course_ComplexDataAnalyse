# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 08:51:24 2017

@author: chris
"""
import os
os.chdir("E:\\workspace\\project")

from mymodule.GMM import GMM
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

def mixture(x,p,mu,sigma):
    mix = p[0]*stats.multivariate_normal.pdf(x,mu[0],sigma[0])
    for i in range(1,len(p)):
        mix += p[i]*stats.multivariate_normal.pdf(x,mu[i],sigma[i])
    return mix

def scatter_contour(y,model,G,devide=True):
    right=0
    colors = ['tab:blue','tab:orange','tab:olive']
    fig = plt.figure()
    # scatter
    for i in range(G):
        left = right
        right += group[i]
        plt.scatter(y[left:right,0],y[left:right,1],alpha=0.6,color=colors[i])
    # contour
    if (devide):
        for i in range(model.g):
            A,B = np.meshgrid(np.arange(min(y[:,0]),max(y[:,0]),0.1),np.arange(min(y[:,1]),max(y[:,1]),0.1))
            tmp = np.append(A.reshape(A.shape[0]*A.shape[1],1),B.reshape(B.shape[0]*B.shape[1],1),1)
            C = stats.multivariate_normal.pdf(tmp,model.mu[i],model.sigma[i]).reshape(A.shape)
            plt.contour(A,B,C,5)
    else:
        A,B = np.meshgrid(np.arange(min(Y[:,0]),max(Y[:,0]),0.1),np.arange(min(Y[:,1]),max(Y[:,1]),0.1))
        tmp = np.append(A.reshape(A.shape[0]*A.shape[1],1),B.reshape(B.shape[0]*B.shape[1],1),1)
        C = mixture(tmp,model.p,model.mu,model.sigma).reshape(A.shape)
        plt.contour(A,B,C,10)
    return fig

G = 3
N = 1000
P = [0.5,
     0.3,
     0.2]
MU = [[-3,0],
      [0,0],
      [0,3]]
SIGMA = [[[1,-0.2],[-0.2,1]],
         [[1,0],[0,1]],
         [[1,0.7],[0.7,1]]]

# single trial
np.random.seed(0)
group = np.random.multinomial(N,P)
Y = np.random.multivariate_normal(MU[0],SIGMA[0],group[0])
for i in range(1,len(P)):
    Y = np.append(Y,np.random.multivariate_normal(MU[i],SIGMA[i],group[i]),axis=0)

# train model
model = GMM(Y,G,True,1e-3)
# plot hetero_model_hetero_data.png
fig = scatter_contour(Y,model,G,True)
fig.savefig("hetero_model_hetero_data.png",dpi=120)
# plot hetero_model_hetero_data_mixcontour.png
fig = scatter_contour(Y,model,G,False)
fig.savefig("hetero_model_hetero_data_mixcontour.png",dpi=120)

# train model
model = GMM(Y,G,False,1e-3)
# plot homo_model_hetero_data.png
fig = scatter_contour(Y,model,G,True)
fig.savefig("homo_model_hetero_data.png",dpi=120)
# plot homo_model_hetero_data.png
fig = scatter_contour(Y,model,G,False)
fig.savefig("homo_model_hetero_data_mixcontour.png",dpi=120)

# plot for g=1,2,3,4
for g in [1,2,3]:
    model = GMM(Y,g,True,1e-3)
    fig = scatter_contour(Y,model,G,True)
    fig.savefig("hetero_model_hetero_data(g=%d).png"%g,dpi=120)
    fig = scatter_contour(Y,model,G,False)
    fig.savefig("hetero_model_hetero_data_mixcontour(g=%d).png"%g,dpi=120)

g = 4
for i in [1,2]:    
    model = GMM(Y,g,True,1e-3)
    fig = scatter_contour(Y,model,G,True)
    fig.savefig("hetero_model_hetero_data(g=%d,%d).png"%(g,i),dpi=120)
    fig = scatter_contour(Y,model,G,False)
    fig.savefig("hetero_model_hetero_data_mixcontour(g=%d,%d).png"%(g,i),dpi=120)