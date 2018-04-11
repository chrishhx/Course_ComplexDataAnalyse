# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 21:47:30 2017

@author: chris
"""

import numpy as np
from scipy import stats

mean = [0,0]
cov  = [[1,1],[1,2]]

N = 20
Repeat = 1000

SigCases   = 0
inSigCases = 0
CI_1 = np.zeros(2000).reshape(1000,2)
CI_2 = np.zeros(2000).reshape(1000,2)
CI_3 = np.zeros(2000).reshape(1000,2)

np.random.seed(0)
for i in range(Repeat):
    # generate Y1,Y2
    Y1,Y2 = np.random.multivariate_normal(mean,cov,N).T
    tmp = np.random.randint(1,100,N)
    # generate M
    M = np.zeros(N)
    M[np.intersect1d(np.where(Y1<0),np.where(tmp<=20))] = 1
    M[np.intersect1d(np.where(Y1>=0),np.where(tmp<=80))] = 1
    Y1obs = Y1[np.where(M==0)]
    Y1mis = Y1[np.where(M==1)]
    Y2obs = Y2[np.where(M==0)]
    # carry out a t-test between Y1obs and Y1mis
    test = stats.ttest_ind(Y1obs,Y1mis)
    if (test[1] < 0.05):
        SigCases = SigCases + 1
    else:
        inSigCases = inSigCases + 1
    # 95% confidence interval
    r = len(Y2obs)
    # (1) the data before values were deleted
    mu2 = np.mean(Y2)
    [[s11,s12],[s21,s22]] = np.cov([Y1,Y2],bias=1)
    sigma221 = s22-s12**2/s11
    rho = s12/np.sqrt(s11*s22)
    var = sigma221 * (1/N + (rho**2)/(N*(1-rho**2)))
    CI_1[i][0],CI_1[i][1] = (stats.norm.interval(0.95,loc=mu2,scale=var))
    # (2) the complete cases
    mu2 = np.mean(Y2obs)
    [[s11,s12],[s21,s22]] = np.cov([Y1obs,Y2obs],bias=1)
    sigma221 = s22-s12**2/s11
    rho = s12/np.sqrt(s11*s22)
    var = sigma221 * (1/r + (rho**2)/(r*(1-rho**2)))
    CI_2[i][0],CI_2[i][1] = (stats.norm.interval(0.95,loc=mu2,scale=var))
    # (3) t-approximation in (2) of Table 7.2
    mu1 = np.mean(Y1)
    sigma11 = np.var(Y1)
    [[s11,s12],[s21,s22]] = np.cov([Y1obs,Y2obs],bias=1)
    beta211 = s12/s11
    beta201 = np.mean(Y2obs) - beta211*np.mean(Y1obs)
    sigma221 = s22-s12**2/s11
    mu2 = np.mean(Y2obs) + beta211*(mu1-np.mean(Y1obs))
    sigma22 = s22 + (beta211**2)*(sigma11-s11)
    rho = (s12/s11) * np.sqrt(sigma11/sigma22) #rou = s12/np.sqrt(s11*s22)*np.sqrt(sigma11/s11)*np.sqrt(s22/sigma22)
    var = sigma221 * (1/r + (rho**2)/(N*(1-rho**2)) + ((np.mean(Y1obs)-mu1)**2/(r*s11)) )
    CI_3[i][0],CI_3[i][1] = (stats.t.interval(0.95,df=r-1,loc=mu2,scale=var))

print(SigCases,inSigCases)
np.mean(CI_1[:,0]) , np.mean(CI_1[:,1]) , np.mean(CI_1[:,1]) - np.mean(CI_1[:,0])
np.mean(CI_2[:,0]) , np.mean(CI_2[:,1]) , np.mean(CI_2[:,1]) - np.mean(CI_2[:,0])
np.mean(CI_3[:,0]) , np.mean(CI_3[:,1]) , np.mean(CI_3[:,1]) - np.mean(CI_3[:,0])


# if the machanism is MCAR
SigCases   = 0
inSigCases = 0
np.random.seed(0)
for i in range(Repeat):
    # generate Y1,Y2
    Y1,Y2 = np.random.multivariate_normal(mean,cov,N).T
    tmp = np.random.randint(1,100,N)
    # generate M
    M = np.zeros(N)
    M[np.where(tmp<=50)] = 1
    Y1obs = Y1[np.where(M==0)]
    Y1mis = Y1[np.where(M==1)]
    if (stats.ttest_ind(Y1obs,Y1mis)[1] < 0.05):
        SigCases = SigCases + 1
    else:
        inSigCases = inSigCases +1

print(SigCases,inSigCases)

