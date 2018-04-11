# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 09:00:11 2017

Problem 8.16 implement

@author: chris
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)

# actual values
_p = 0.5
_mu0 = -1
_mu1 = 2
_sigma = 1

# randomly generate data
n = 200
r = np.random.binomial(200,0.5)
X = np.random.binomial(1,_p,size=n)
Y = np.zeros(n)
Y[np.where(X==0)] = np.random.normal(loc=_mu0,scale=_sigma,size=Y[np.where(X==0)].size)
Y[np.where(X==1)] = np.random.normal(loc=_mu1,scale=_sigma,size=Y[np.where(X==1)].size)

# generate monotone missing pattern
# since it is generate at random, simplily remove the last n-r values will be fine

# EM's control parameters
cache_size = 1000 # record the last $cache_size$ iteration's results
delta = np.inf    # difference between (p,mu0,mu1,sigma)[t] and (p,mu0,mu1,sigma)[t+1]
c_limit = 1e-6    # stop iteration if delta < $c_limit$

# f(y|x=1;p,mu0,mu1,sigma)
def f(y,loc,scale):
    return np.exp(-(y-loc)**2/(2*scale**2))/(np.sqrt(2*np.pi)*scale)

# EM
w = np.zeros(n)
w[0:r] = X[0:r]
p = np.zeros(cache_size)
mu0 = np.zeros(cache_size)
mu1 = np.zeros(cache_size)
sigma = np.zeros(cache_size)
t = 0
p[0] = np.mean(X)
mu0[0] = np.mean(Y[np.intersect1d(np.where(X==0),range(r))])
mu1[0] = np.mean(Y[np.intersect1d(np.where(X==1),range(r))])
sigma[0] = np.sqrt(np.mean(Y[0:r]*Y[0:r]))
while (delta > c_limit and t<cache_size):
    # E-step
    w[r:n] = (f(Y[r:n],mu1[t],sigma[t])*p[t]) / (f(Y[r:n],mu1[t],sigma[t])*p[t] + f(Y[r:n],mu0[t],sigma[t])*(1-p[t]))
    # M-step
    p[t+1] = np.mean(w)
    mu0[t+1] = sum((1-w)*Y) / sum(1-w)
    mu1[t+1] = sum(w*Y) / sum(w)
    sigma[t+1] = np.sqrt((sum((1-w)*(Y-mu0[t+1])**2) + sum(w*(Y-mu1[t+1])**2))/n)
    delta = max([p[t+1]-p[t],mu0[t+1]-mu0[t],mu1[t+1]-mu1[t],sigma[t+1]-sigma[t]])
    t += 1
 
print('|   p  | mu_0 | mu_1 | sigma|')
print('|------|------|------|------|')
for i in range(t+1):
    print('|%.4f|%.4f|%.4f|%.4f|'%(p[i],mu0[i],mu1[i],sigma[i]))

index = np.array(range(t+1))
fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.plot(index,p[index],'r')
ax1.set_ylabel('p',rotation='horizontal')
ax1.set_xlabel('iteration')
ax2.plot(index,mu0[index],'g')
ax2.set_ylabel('μ0',rotation='horizontal')
ax2.set_xlabel('iteration')
ax3.plot(index,mu1[index],'b')
ax3.set_ylabel('μ1',rotation='horizontal')
ax3.set_xlabel('iteration')
ax4.plot(index,sigma[index],'k')
ax4.set_ylabel('σ',rotation='horizontal')
ax4.set_xlabel('iteration')
fig.savefig(filename="convergence.png",dpi=100)
