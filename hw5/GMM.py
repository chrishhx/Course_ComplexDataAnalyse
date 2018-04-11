# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 11:43:10 2017

@author: chris
"""

import numpy as np

def f(x,mu,sigma):
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-(x - mu)**2 / (2*sigma**2))

def mixture(x,p,mu,sigma):
    result = p[0]*f(x,mu[0],sigma[0]) # compatible with vector and scalar
    for h in range(1,len(p)):
        result += p[h]*f(x,mu[h],sigma[h])
    return result

def EM(y,p,mu,sigma):
    n = len(y)
    c = len(p)
    w = np.zeros([c,n])
    re_p = np.zeros(c)
    re_mu = np.zeros(c)
    re_sigma = np.zeros(c)
    re_logL = 0
    # E-step
    for j in range(n):
        tmp = mixture(y[j],p,mu,sigma)
        for i in range(c):
            w[i,j] = p[i] * f(y[j],mu[i],sigma[i]) / tmp
            re_logL += w[i,j] * (np.log(p[i]*f(y[j],mu[i],sigma[i])))
            # w[i,j] = p[i] * f(y[j],mu[i],sigma[i]) / mixture(y[j],p,mu,sigma)
            # use tmp to reduce computational complexity
    # M-step
    for i in range(c):
        re_p[i] = np.mean(w[i])
        re_mu[i] = sum(w[i]*y) / sum(w[i])
        re_sigma[i] = np.sqrt(sum(w[i] * (y-re_mu[i])**2) / sum(w[i]))
    return re_p,re_mu,re_sigma,re_logL

## simulation
np.random.seed(0)
N = 400
#Y = np.append(np.append( np.random.normal(loc=-2,scale=1,size=N//4) , np.random.normal(loc=2,scale=1,size=N//2)) , np.random.normal(loc=6,scale=1,size=N//4))
Y = np.append( np.random.beta(4,1,size=N//2) , np.random.beta(1,4,size=N//2))


## hyper-parameter
g = 5

## EM control parameter
cache_size = 1000
tol = 1e-4

## init
em_p = np.zeros([cache_size,g])
em_mu = np.zeros([cache_size,g])
em_sigma = np.zeros([cache_size,g])
em_l = np.zeros([cache_size])
em_la = np.zeros([cache_size])

em_p[0] = np.ones(g)/g
em_mu[0] = np.random.normal(loc=np.mean(Y),scale=np.std(Y),size=g)
em_sigma[0] = np.ones(g) * np.std(Y)
em_p[0],em_mu[0],em_sigma[0],em_l[0] = EM(Y,em_p[0],em_mu[0],em_sigma[0]) ## calculate logL via EM

t=0
while (t+1 < cache_size):
    em_p[t+1],em_mu[t+1],em_sigma[t+1],em_l[t+1] = EM(Y,em_p[t],em_mu[t],em_sigma[t])
    if (t>0):
        a = (em_l[t+1] - em_l[t]) / (em_l[t] - em_l[t-1])
        em_la[t+1] = em_l[t] + (em_l[t+1]-em_l[t])/(1-a)
        if (abs(em_la[t+1] - em_la[t]) < tol or np.isnan(em_la[t+1])):
            break;
    t += 1

from matplotlib import pyplot as plt
from scipy import stats

# example 1
n,bins,patches = plt.hist(Y,bins=30,normed=1,color=[0.4,0.8,0.8])
plt.plot(bins,mixture(bins,em_p[t-1],em_mu[t-1],em_sigma[t-1]),ls='--',color=[0.4,0.2,0.9])
plt.plot(bins,mixture(bins,[0.25,0.5,0.25],[-2,2,6],[1,1,1]),ls='--',color=[0.8,0.3,0.2])
plt.legend(('mixture model','real distribution'))
plt.title("g=%d , convergence after %d iterations"%(g,t))
plt.savefig(filename='E://workspace//ex1g%d.png'%g,dpi=100)

# example 2
n,bins,patches = plt.hist(Y,bins=30,normed=1,color=[0.4,0.8,0.8])
plt.plot(bins,mixture(bins,em_p[t-1],em_mu[t-1],em_sigma[t-1]),ls='--',color=[0.4,0.2,0.9])
plt.plot(bins,0.5*stats.beta.pdf(bins, 1, 4) + 0.5*stats.beta.pdf(bins, 4, 1), 'r-')
#plt.plot(bins,0.5*stats.beta.pdf(bins, 4, 1), 'g-')
plt.legend(('mixture model','real distribution'))
plt.title("g=%d , convergence after %d iterations"%(g,t))
plt.savefig(filename='E://workspace//ex2g%d.png'%g,dpi=100)