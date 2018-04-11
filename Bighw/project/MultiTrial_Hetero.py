# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 10:16:28 2017

@author: chris
"""
import os
os.chdir("E:\\workspace\\project")

from multiprocessing import Pool
from multiprocessing import cpu_count
from mymodule.GMM import GMM
import numpy as np
import pickle
import os

def worker(args):
    return GMM(args[0],args[1],args[2])

np.random.seed(0)
G = 3
N = 1000
#N = 2000
trials = 1000

P = [0.5, 0.3, 0.2]
#P = [0.42, 0.33, 0.25]
MU = [[-3,0],
      [0,0],
      [0,3]]
SIGMA = [[[1,-0.2],[-0.2,1]],
         [[1,0],[0,1]],
         [[1,0.7],[0.7,1]]]

# Multi trial
Ys = []
for i in range(trials):
    group = np.random.multinomial(N,P)
    Y = np.random.multivariate_normal(MU[0],SIGMA[0],group[0])
    for i in range(1,len(P)):
        Y = np.append(Y,np.random.multivariate_normal(MU[i],SIGMA[i],group[i]),axis=0)
    Ys.append([Y,G,True,1e-3])


if __name__ == '__main__':    
    p = Pool(cpu_count()-1)
    print("Detect %d Threads of CPU ,Running at %d Threads"%(cpu_count(),cpu_count()-1))
    results = p.map(worker,Ys)
    output = open('multitrial_Hetero(N=%d).pkl'%N,'wb')
    #output = open('multitrial_Hetero(N=%d,pi_modify).pkl'%N,'wb')
    data = []
    for model in results:
        data.append([model.p,model.mu,model.sigma,model.p_sd,model.mu_sd,model.sigma_sd])
    pickle.dump(data,output)
    output.close()
    print("Outputs in %s/multitrial_Hetero(N=%d).pkl"%(os.getcwd(),N))