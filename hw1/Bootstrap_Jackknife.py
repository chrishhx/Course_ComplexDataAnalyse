# -*- coding: utf-8 -*-
"""
2017-9-14
Bootstrap & Jackknife
@author: chris
"""

import numpy as np

np.random.seed(0)

mu = 7
sigma = 2
N = 100
B = 500

x = mu + sigma*np.random.randn(100)

# Moment
est_mu = np.mean(x)
est_mu_var = np.var(x) / N
est_sigma2 = np.var(x)
est_sigma2_var = ( np.mean(np.power(x-np.mean(x),4)) - np.power(np.var(x),2) ) / N

# Bootstrap
sample = np.random.randint(0,99,size=(B,80))
b_mu = np.zeros(B)
b_sigma = np.zeros(B)

for i in range(B):
    b_mu[i] = np.mean(x[sample[i]])
    b_sigma[i] = np.var(x[sample[i]])
    
b_est_mu = np.mean(b_mu)
b_est_mu_var = B*np.var(b_mu) / (B-1)
b_est_sigma2 = np.mean(b_sigma)
b_est_sigma2_var = B*np.var(b_sigma) / (B-1)

# Jackknife
j_mu = np.zeros(N)
j_sigma2 = np.zeros(N)
for i in range(N):
    j_mu[i] = (sum(x)-x[i]) / (N-1)
    j_sigma2[i] = (sum(np.power(x-j_mu[i],2)) - np.power(x[i]-j_mu[i],2)) / (N-1)
j_est_mu = est_mu + (N-1)*(est_mu-np.mean(j_mu))
j_est_mu_var = ((N-1)/N) * sum(np.power(j_mu-np.mean(j_mu),2))
j_est_sigma2 = est_sigma2 + (N-1)*(est_sigma2-np.mean(j_sigma2))
j_est_sigma2_var = ((N-1)/N) * sum(np.power(j_sigma2-np.mean(j_sigma2),2))

# np.var calculate var by devide N , not N-1
print("         \test_mu  \test_mu_var\test_sigma2\test_sigma2_var")
print("Moment   \t%.8f\t%.8f\t%.8f\t%.8f"%(est_mu,est_mu_var,est_sigma2,est_sigma2_var))
print("Bootstrap\t%.8f\t%.8f\t%.8f\t%.8f"%(b_est_mu,b_est_mu_var,b_est_sigma2,b_est_sigma2_var))
print("Jackknife\t%.8f\t%.8f\t%.8f\t%.8f"%(j_est_mu,j_est_mu_var,j_est_sigma2,j_est_sigma2_var))
