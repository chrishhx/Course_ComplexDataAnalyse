# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 16:35:24 2017

@author: chris
"""

import numpy as np

# one step EM
def EM(y,mu,sigma):
    r,c = y.shape
    if (mu.shape != (c,) or sigma.shape!=(c,c)):
        raise( NameError('the data and parameter not fit'))
    x = y.copy()
    # E-step
    Covmis = np.zeros([r,c,c])
    for i in range(r):
        if (np.all(np.isnan(x[i])==False)):
            continue;
        mis = np.where(np.isnan(x[i]))[0]
        obs = np.where(np.isnan(x[i])==False)[0]
        # re-arange the cov matirx
        Smis = np.matrix(sigma[mis].transpose()[mis].transpose())
        Sobs = np.matrix(sigma[obs].transpose()[obs].transpose())
        Smvo = np.matrix(sigma[mis].transpose()[obs].transpose())
        x[i][mis] = mu[mis] + np.array(Smvo*np.linalg.inv(Sobs)*np.matrix(x[i][obs]-mu[obs]).transpose()).reshape(1,len(mis))
        tmp = Smis - Smvo*np.linalg.inv(Sobs)* Smvo.transpose()
        for j in range(len(mis)):
            for k in range(len(mis)):
                Covmis[i,mis[j],mis[k]] = tmp[j,k]
    # M-step
    re_mu = np.mean(x,axis=0)
    re_sigma = np.zeros([c,c])
    for j in range(c):
        for k in range(c):
            tmp = 0
            for i in range(r):
                tmp += (x[i,j]-re_mu[j])*(x[i,k]-re_mu[k]) + Covmis[i][j][k]
            re_sigma[j][k] = tmp/r
    return re_mu,re_sigma

def getInfoMat(y,sigma):
    r,c = y.shape
    Jmu = np.matrix(np.zeros([c,c]))
    Jsigma = np.matrix(np.zeros([c*(c+1)//2,c*(c+1)//2]))
    psi = np.zeros([r,c,c])
    for i in range(r):
        obs = np.where(np.isnan(y[i])==False)[0]
        Sobs = np.matrix(sigma[obs].transpose()[obs].transpose())
        Sobs_inv = np.linalg.inv(Sobs)
        for j in range(len(obs)):
            for k in range(len(obs)):
                psi[i,obs[j],obs[k]] = Sobs_inv[j,k]
    for j in range(c):
        for k in range(c):
            Jmu[j,k] = sum(psi[0:r,j,k])
    t1=0
    for l in range(c):
        for m in range(l,c):
            t2=0
            for v in range(c):
                for s in range(v,c):
                    Jsigma[t1,t2] = sum( psi[0:r,l,v]*psi[0:r,m,s] + psi[0:r,l,s]*psi[0:r,m,v] )
                    if (l==m and v==s):
                        Jsigma[t1,t2] /=4
                    elif (not (l!=m and v!=s)):
                        Jsigma[t1,t2] /=2
                    t2 += 1
            t1 += 1
    return Jmu,Jsigma

def printvec(vec):
    print('(',end='')
    for i in range(len(vec)):
        print ('%.2f'%vec[i],end='')
        if (i+1 < len(vec)):
            print (',',end='')
    print(')')
        
def printmat(mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            print ('%.2f'%mat[i,j],end='')
            if (j+1<mat.shape[1]):
                print ('&',end='')
        print('')

###########################################################
# simulation
def simulation(R,C,mu,sigma,output=False):
    # generate multivariate normal sample
    ORIGIN_Y = np.random.multivariate_normal(mu,sigma,size=R)
    # randomly remove some obsevations
    missing_rate = 0.4
    MISS = np.random.binomial(1,missing_rate,size=[R,C])
    # minor fix for those entirely deleted obsevations
    for i in range(100):
        if (np.all(MISS[i] == 1)):
            MISS[i,np.random.randint(0,C)] = 0
            
    Y = ORIGIN_Y.copy()
    Y[np.where(MISS==1)] = np.nan
    # Data generate completed

    # EM's control parameters
    cache_size = 1000 # record the last $cache_size$ iteration's results
    delta = np.inf    # difference between theta[t] and theta[t+1]
    c_limit = 1e-6    # stop iteration if delta < $c_limit$
    
    r_mu = np.zeros([cache_size,C])
    r_sigma = np.zeros([cache_size,C,C])
    
    # access the complete cases
    complete = []
    for i in range(R):
        if (np.all(np.isnan(Y[i])==False)):
            complete.append(i)
    
    # use complete case mean and covariance matrix as start point
    r_mu[0] = np.mean(Y[complete],axis=0)
    r_sigma[0] = np.cov(Y[complete],rowvar=False,bias=False)
    
    # get ML estimate via EM
    t=0
    while (delta > c_limit and t < cache_size-1):
        r_mu[t+1],r_sigma[t+1] = EM(Y,r_mu[t],r_sigma[t])
        delta = max(max(abs(r_mu[t+1] - r_mu[t])),max(abs(r_sigma[t+1]-r_sigma[t]).reshape(C*C)))
        t += 1
    
    # get standard error
    Jmu,Jsigma = getInfoMat(Y,r_sigma[t])
    mu_sd = np.diag(np.linalg.inv(Jmu))
    sigma_sd = np.zeros([C,C])
    tmp = np.diag(np.linalg.inv(Jsigma))
    t = 0
    for i in range(C):
        for j in range(i,C):
            sigma_sd[i,j] = tmp[t]
            sigma_sd[j,i] = tmp[t]
            t+=1
    
    # output
    if (output==True):
        print ('mean:')
        printvec(MU)
        print ('origin sample mean:')
        printvec(np.mean(ORIGIN_Y,axis=0))
        print ('EM\'s MLE mean:')
        printvec(r_mu[t])
        print ('complete case MLE mean:')
        printvec(r_mu[0])
        print("Real cov")
        printmat(SIGMA)
        print("origin Y cov")
        printmat(np.cov(ORIGIN_Y,rowvar=False,bias=False))
        print("EM's MLE cov")
        printmat(r_sigma[t])
        print("CC's MLE cov")
        printmat(r_sigma[0])
        print("standard error of Mu")
        printvec(np.sqrt(mu_sd))
        print("standard error of Sigma")
        printmat(np.sqrt(sigma_sd))
    
    return r_mu[t],r_sigma[t],mu_sd,sigma_sd


# change seed to get different MU and SIGMA
np.random.seed(0)
R = 200 # sample size
C = 4   # number of random variables
# generate random mean MU and covariance matrix SIGMA
MU = np.random.randint(-4,4,size=C)
# generate a semidefinite matrix as covariance matrix
A = np.matrix(np.random.normal(0,1,size=[C,C]))
SIGMA = A*A.transpose()
simulation(R,C,MU,SIGMA,True)
sr_mu = np.zeros([100,C])
sr_sigma = np.zeros([100,C,C])
sr_mu_sd = np.zeros([100,C])
sr_sigma_sd = np.zeros([100,C,C])

for i in range(100):
    sr_mu[i] , sr_sigma[i] , sr_mu_sd[i] , sr_sigma_sd[i] = simulation(R,C,MU,SIGMA,False)
    print (i)

from matplotlib import pyplot as plt

fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(111)
ax1.boxplot(sr_mu_sd)
ax1.set_xlabel('s.e. of μ')
fig.savefig(fname="D://workspace//box1.png",dpi=100)
