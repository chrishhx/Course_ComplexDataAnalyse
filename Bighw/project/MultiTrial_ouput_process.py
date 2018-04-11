# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 12:34:02 2017

@author: chris
"""
import os
os.chdir("E:\\workspace\\project")

import pickle
import numpy as np

c = 2
g = 3
P = [0.5, 0.3, 0.2]
#P = [0.42, 0.33, 0.25]
MU = [[-3,0],
      [0,0],
      [0,3]]
# Hetero
SIGMA = [[[1,-0.2],[-0.2,1]],
         [[1,0],[0,1]],
         [[1,0.7],[0.7,1]]]
file = open('multitrial_Hetero(N=1000).pkl','rb')
#file = open('multitrial_Hetero(N=1000,pi_modify).pkl','rb')
#file = open('multitrial_Hetero(N=2000).pkl','rb')

# Homo
'''
SIGMA = [[[1,0.2],[0.2,1]],
         [[1,0.2],[0.2,1]],
         [[1,0.2],[0.2,1]]]
'''
#file = open('multitrial_Homo.pkl','rb')

data = pickle.load(file)
file.close()

rep = len(data)

p = np.zeros([len(data),g])
mu = np.zeros([len(data),g,c])
sigma = np.zeros([len(data),g,c,c])
p_sd = np.zeros([len(data),g])
mu_sd = np.zeros([len(data),g,c])
sigma_sd = np.zeros([len(data),g,c,c])

for i in range(rep):
    p[i], mu[i], sigma[i], p_sd[i], mu_sd[i], sigma_sd[i] = data[i]
    ## sort by p
    for j in range(g):
        for k in range(j+1,g):
            if (p[i][j] < p[i][k]):
                p[i][j],p[i][k] = p[i][k].copy(),p[i][j].copy()
                mu[i][j],mu[i][k] = mu[i][k].copy(),mu[i][j].copy()
                sigma[i][j],sigma[i][k] = sigma[i][k].copy(),sigma[i][j].copy()
                p_sd[i][j],p_sd[i][k] = p_sd[i][k].copy(),p_sd[i][j].copy()
                mu_sd[i][j],mu_sd[i][k] = mu_sd[i][k].copy(),mu_sd[i][j].copy()
                sigma_sd[i][j],sigma_sd[i][k] = sigma_sd[i][k].copy(),sigma_sd[i][j].copy()

## without outlier (delete whole outlier models)
validcases=[]
for k in range(rep):
    outlier = False
    for i in range(g):
        for r in range(c):
            loc = np.mean(mu[:,i,r])
            std = np.std(mu[:,i,r])
            if (outlier or (abs(mu[k,i,r]-loc) > 3*std) or (mu_sd[k,i,r] > 1)):
                outlier=True
                break
            for s in range(r,c):
                loc = np.mean(sigma[:,i,r,s])
                std = np.std(sigma[:,i,r,s])
                if (outlier or (abs(sigma[k,i,r,s]-loc) > 3*std) or (sigma_sd[k,i,r,s] > 1)):
                    outlier=True
                    break
    if (not outlier):
        validcases.append(k)

print ('| $\pi$ | Real | Avg Est | Bias |  SE  |  SEE  | CP(95%)  | valid cases |')
print ('|:------|:----:|:------:|:------:|:-----:|:-----:|:----:|:-----------:|')
for i in range(g):
    cases = []
    for k in validcases:
        if ((p_sd[k,i] != 0) and (p_sd[k,i]<1)):
            cases.append(k)
    print ('|$\pi_{%d}$'%(i+1),end='')
    print ('| %.1f'%(P[i]),end='')
    print ('| %.3f '%(np.mean(p[cases,i])),end='')
    print ('| %.3f '%(np.mean(p[cases,i])-P[i]),end='')
    SE = np.std(p[cases,i]-P[i])
    print ('| %.3f '%SE,end='')
    SEE = np.mean(p_sd[cases,i])
    print ('| %.3f '%SEE,end='')
    CP = len(cases)
    for k in cases:
        if (P[i] < p[k,i] - 1.96*p_sd[k,i]): CP -= 1
        if (P[i] > p[k,i] + 1.96*p_sd[k,i]): CP -= 1
    CP = CP/len(cases)
    print ('| %.2f '%CP,end='')
    print ('| %d |'%len(cases))
    
print ('| $\mu$ | Real | Avg Est | Bias |  SE  |  SEE  | CP(95%)  | valid cases |')
print ('|:------|:----:|:------:|:------:|:-----:|:-----:|:----:|:-----------:|')
for i in range(g):
    for j in range(c):
        cases = validcases
        print ('| $\mu_{%d%d}$ '%(i+1,j+1),end='')
        print ('| %.1f'%(MU[i][j]),end='')
        print ('| %.3f '%(np.mean(mu[cases,i,j])),end='')
        print ('| %.3f '%(np.mean(mu[cases,i,j])-MU[i][j]),end='')
        SE = np.std(mu[cases,i,j]-MU[i][j])
        print ('| %.3f '%SE,end='')
        SEE = np.mean(mu_sd[cases,i,j])
        print ('| %.3f '%SEE,end='')
        CP = len(cases)
        for k in cases:
            if (MU[i][j] < mu[k,i,j] - 1.96*mu_sd[k,i,j]): CP -= 1
            if (MU[i][j] > mu[k,i,j] + 1.96*mu_sd[k,i,j]): CP -= 1
        CP = CP/len(cases)
        print ('| %.2f '%CP,end='')
        print ('| %d |'%len(cases))
        
print ('| $\Sigma$ | Real | Avg Est | Bias |  SE  |  SEE  | CP(95%) | valid cases |')
print ('|:--------|:----:|:-------:|:-----:|:----:|:----:|:----:|:-----------:|')
for i in range(g):
    for r in range(c):
        for s in range(r,c):
            cases = validcases
            print ('| $\Sigma_{%d,%d%d}$'%(i+1,r+1,s+1),end='')
            print ('| %.1f'%(SIGMA[i][r][s]),end='')
            print ('| %.3f '%(np.mean(sigma[cases,i,r,s])),end='')
            print ('| %.3f '%(np.mean(sigma[cases,i,r,s])-SIGMA[i][r][s]),end='')
            SE = np.std(sigma[cases,i,r,s]-SIGMA[i][r][s])
            print ('| %.3f '%SE,end='')
            SEE = np.mean(sigma_sd[cases,i,r,s])
            print ('| %.3f '%SEE,end='')
            CP = len(cases)
            for k in range(len(cases)):
                if (SIGMA[i][r][s] < sigma[k,i,r,s] - 1.96*sigma_sd[k,i,r,s]): CP -= 1
                if (SIGMA[i][r][s] > sigma[k,i,r,s] + 1.96*sigma_sd[k,i,r,s]): CP -= 1
            CP = CP/len(cases)
            print ('| %.2f '%CP,end='')
            print ('| %d |'%len(cases))

'''
## without outlier (only delete outlier parameters)
print ('| $\pi$ | Real | Avg Est | Bias |  SE  |  SEE  | CP(95%)  | valid cases |')
print ('|:------|:----:|:------:|:------:|:-----:|:-----:|:----:|:-----------:|')
for i in range(g):
    cases = []
    for k in range(rep):
        if ((p_sd[k,i] != 0) and (p_sd[k,i]<1)):
            cases.append(k)
    print ('|$\pi_{%d}$'%(i+1),end='')
    print ('| %.1f'%(P[i]),end='')
    print ('| %.3f '%(np.mean(p[cases,i])),end='')
    print ('| %.3f '%(np.mean(p[cases,i])-P[i]),end='')
    SE = np.std(p[cases,i]-P[i])
    print ('| %.3f '%SE,end='')
    SEE = np.mean(p_sd[cases,i])
    print ('| %.3f '%SEE,end='')
    CP = len(cases)
    for k in cases:
        if (P[i] < p[k,i] - 1.96*p_sd[k,i]): CP -= 1
        if (P[i] > p[k,i] + 1.96*p_sd[k,i]): CP -= 1
    CP = CP/len(cases)
    print ('| %.2f '%CP,end='')
    print ('| %d |'%len(cases))


print ('| $\mu$ | Real | Avg Est | Bias |  SE  |  SEE  | CP(95%)  | valid cases |')
print ('|:------|:----:|:------:|:------:|:-----:|:-----:|:----:|:-----------:|')
for i in range(g):
    for j in range(c):
        cases = []
        loc = np.mean(mu[:,i,j])
        std = np.std(mu[:,i,j])
        for k in range(rep):
            if ((abs(mu[k,i,j]-loc) < 3*std) and (mu_sd[k,i,j] < 1)):
                cases.append(k)
        print ('| $\mu_{%d%d}$ '%(i+1,j+1),end='')
        print ('| %.1f'%(MU[i][j]),end='')
        print ('| %.3f '%(np.mean(mu[cases,i,j])),end='')
        print ('| %.3f '%(np.mean(mu[cases,i,j])-MU[i][j]),end='')
        SE = np.std(mu[cases,i,j]-MU[i][j])
        print ('| %.3f '%SE,end='')
        SEE = np.mean(mu_sd[cases,i,j])
        print ('| %.3f '%SEE,end='')
        CP = len(cases)
        for k in cases:
            if (MU[i][j] < mu[k,i,j] - 1.96*mu_sd[k,i,j]): CP -= 1
            if (MU[i][j] > mu[k,i,j] + 1.96*mu_sd[k,i,j]): CP -= 1
        CP = CP/len(cases)
        print ('| %.2f '%CP,end='')
        print ('| %d |'%len(cases))


print ('| $\Sigma$ | Real | Avg Est | Bias |  SE  |  SEE  | CP(95%) | valid cases |')
print ('|:--------|:----:|:-------:|:-----:|:----:|:----:|:----:|:-----------:|')
for i in range(g):
    for r in range(c):
        for s in range(r,c):
            cases = []
            loc = np.mean(sigma[:,i,r,s])
            std = np.std(sigma[:,i,r,s])
            for k in range(rep):
                if ((abs(sigma[k,i,r,s]-loc) < 3*std) and (sigma_sd[k,i,r,s] < 1)):
                    cases.append(k)
            print ('| $\Sigma_{%d,%d%d}$'%(i+1,r+1,s+1),end='')
            print ('| %.1f'%(SIGMA[i][r][s]),end='')
            print ('| %.3f '%(np.mean(sigma[cases,i,r,s])),end='')
            print ('| %.3f '%(np.mean(sigma[cases,i,r,s])-SIGMA[i][r][s]),end='')
            SE = np.std(sigma[cases,i,r,s]-SIGMA[i][r][s])
            print ('| %.3f '%SE,end='')
            SEE = np.mean(sigma_sd[cases,i,r,s])
            print ('| %.3f '%SEE,end='')
            CP = len(cases)
            for k in range(len(cases)):
                if (SIGMA[i][r][s] < sigma[k,i,r,s] - 1.96*sigma_sd[k,i,r,s]): CP -= 1
                if (SIGMA[i][r][s] > sigma[k,i,r,s] + 1.96*sigma_sd[k,i,r,s]): CP -= 1
            CP = CP/len(cases)
            print ('| %.2f '%CP,end='')
            print ('| %d |'%len(cases))
'''        