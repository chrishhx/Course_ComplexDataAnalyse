# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 20:46:01 2017

@author: chris
"""

import time
import numpy as np
from scipy import stats

## Class GMM
class GMM(object):
    
    def mixture(self,x,p,mu,sigma):
        mix = p[0]*stats.multivariate_normal.pdf(x,mu[0],sigma[0])
        for i in range(1,len(p)):
            mix += p[i]*stats.multivariate_normal.pdf(x,mu[i],sigma[i])
        return mix

    def EM(self,p,mu,sigma):
        n, c = len(self.y), len(self.y[0])
        w,re_p,re_mu,re_sigma = np.zeros([self.g,n]),np.zeros(self.g),np.zeros([self.g,c]),np.zeros([self.g,c,c])
        re_logL = 0
        # obtain w[i,j] aka tau[i,j]
        # calculate w[i,j] = f[y[j],mu[i],sigma[i]] / mixture(y[j],mu,sigma) as vectors to accelate
        tmp_mixture = self.mixture(self.y,p,mu,sigma)
        for i in range(self.g):
            tmp_mltnorm = stats.multivariate_normal.pdf(self.y,mu[i],sigma[i])
            w[i] = p[i] * tmp_mltnorm / tmp_mixture
            re_logL += sum(w[i] * np.log(p[i]*p[i]*tmp_mltnorm))
        if (not self.dif_cov): sigma_tmp = np.matrix(np.zeros([c,c]))
        # Formula from Page 82,83 of <Finite Mixture Models>
        for i in range(self.g):
            T1 = sum(w[i])
            T2 = np.zeros(c)
            T3 = np.matrix(np.zeros([c,c]))
            for j in range(n):
                T2 += w[i,j]*self.y[j]
                T3 += w[i,j]*self.y[j].reshape([c,1])*self.y[j]
            re_p[i] = T1/n
            re_mu[i] = T2/T1
            re_sigma[i] = (T1*T3 - T2.reshape([c,1])*T2) / T1**2
            if (not self.dif_cov): sigma_tmp += T1*re_sigma[i]
        # homoskedastic
        if (not self.dif_cov):
            sigma_tmp /= n
            for i in range(self.g):
                re_sigma[i] = sigma_tmp
        
        return re_p,re_mu,re_sigma,re_logL
    
    def fit(self):
        c = len(self.y[0])
        # EM's control parameters
        cache_size = 500
        while (True):
            em_p, em_mu, em_sigma = np.zeros([cache_size,self.g]), np.zeros([cache_size,self.g,c]), np.zeros([cache_size,self.g,c,c])
            em_l, em_la = np.zeros([cache_size]), np.zeros([cache_size])
            # init values
            em_p[0] = np.ones(self.g) / self.g
            em_mu[0] = np.random.multivariate_normal(mean=np.mean(self.y,0),cov=np.cov(self.y,rowvar=False),size=self.g)
            em_sigma[0] = np.cov(self.y,rowvar=False)
            ## make the first 2 step
            em_p[0],em_mu[0],em_sigma[0],em_l[0] = self.EM(em_p[0],em_mu[0],em_sigma[0])
            em_p[1],em_mu[1],em_sigma[1],em_l[1] = self.EM(em_p[0],em_mu[0],em_sigma[0])
            # run EM until convergence
            t=1
            while (t+1 < cache_size):
                em_p[t+1],em_mu[t+1],em_sigma[t+1],em_l[t+1] = self.EM(em_p[t],em_mu[t],em_sigma[t])
                if ((em_l[t] == em_l[t-1])): break
                a = (em_l[t+1] - em_l[t]) / (em_l[t] - em_l[t-1])
                em_la[t+1] = em_l[t] + (em_l[t+1]-em_l[t])/(1-a)
                t+=1
                if (abs(em_la[t] - em_la[t-1]) < self.tol or np.isnan(em_la[t])): break
                if (t==499): print("cache too small")
            # encounter log(0) error indicate bad convergence, choose another start and run EM again
            if (not np.isnan(em_la[t]) and t<499): break
        return em_p[t],em_mu[t],em_sigma[t],em_l[t]
    
    def standard_error(self):
        n, c = len(self.y), len(self.y[0])
        w = np.zeros([self.g,n])
        # obtain w[i,j] aka tau[i,j]
        # calculate w[i,j] = f[y[j],mu[i],sigma[i]] / mixture(y[j],mu,sigma) as vectors to accelate
        tmp_mixture = self.mixture(self.y,self.p,self.mu,self.sigma)
        for i in range(self.g):
            tmp_mltnorm = stats.multivariate_normal.pdf(self.y,self.mu[i],self.sigma[i])
            w[i] = self.p[i] * tmp_mltnorm / tmp_mixture
        # obtain inv(sigma)
        invcov = np.zeros([self.g,c,c])
        for i in range(self.g):
            invcov[i] = np.linalg.inv(np.matrix(self.sigma[i]))
        # Fomula from Page 84 of <Finite Mixture Models>
        if (self.dif_cov):
            I = np.zeros([self.g-1 + self.g*c + self.g*c*(c+1)//2,self.g-1 + self.g*c + self.g*c*(c+1)//2])
        else:
            I = np.zeros([self.g-1 + self.g*c + c*(c+1)//2,self.g-1 + self.g*c + c*(c+1)//2])
        for j in range(n):
            Dp = np.zeros(self.g-1)
            Dmu = np.zeros([self.g,c])
            Dsigma = np.zeros([self.g,c*(c+1)//2])
            for i in range(self.g-1): Dp[i] = w[i,j]/self.p[i] - w[self.g-1,j]/self.p[self.g-1]
            for i in range(self.g):
                Dmu[i] = w[i,j]*(np.matrix(invcov[i])*np.matrix(self.y[j]-self.mu[i]).transpose()).transpose()
                k = 0
                for r in range(c):
                    for s in range(r,c):
                        Dsigma[i][k] = w[i,j] * (np.matrix(self.y[j]-self.mu[i]) * np.matrix(invcov[i])[:,r] * np.matrix(self.y[j]-self.mu[i]) * np.matrix(invcov[i])[:,s] - invcov[i,r,s])
                        if (r==s): Dsigma[i][k] *= 0.5
                        k += 1
            if (self.dif_cov):
                S = np.matrix(np.append(np.append(Dp,Dmu),Dsigma))
            else:
                S = np.matrix(np.append(np.append(Dp,Dmu),Dsigma[0]))
            I = I + S.transpose()*S
            
        sd = np.sqrt(np.diag(np.linalg.inv(I)))
        p_sd = np.append(sd[0:self.g-1],0)
        mu_sd = sd[self.g-1:self.g*c+self.g-1].reshape([self.g,c])
        sigma_sd = np.zeros([self.g,c,c])
        # transform upper tringle of sigma back to matrix sigma
        k = self.g*c+self.g-1
        for i in range(self.g):
            if (not self.dif_cov): k = self.g*c+self.g-1
            for r in range(c):
                for s in range(r,c):
                    sigma_sd[i][r][s] = sigma_sd[i][s][r] = sd[k]
                k+=1
        return p_sd,mu_sd,sigma_sd
    
    def AIC(self):
        # -2logL + 2d
        c = len(self.y[0])
        if (self.dif_cov):
            return (-2*self.logL + 2*(self.g-1 + self.g*c + self.g*c*(c+1)/2))
        else:
            return (-2*self.logL + 2*(self.g-1 + self.g*c + c*(c+1)/2))
        
    def BIC(self):
        # -2logL + d log n
        c = len(self.y[0])
        if (self.dif_cov):
            return (-2*self.logL + np.log(len(self.y))*(self.g-1 + self.g*c + self.g*c*(c+1)/2))
        else:
            return (-2*self.logL + np.log(len(self.y))*(self.g-1 + self.g*c + c*(c+1)/2))
    
    def __init__(self,input_y,input_g,input_dif_cov,input_tol=1e-3):
        self.y = input_y.copy()
        self.g = input_g
        self.dif_cov = input_dif_cov
        self.tol = input_tol
        print("job start at ",time.clock())
        self.p, self.mu, self.sigma, self.logL = self.fit()
        self.p_sd, self.mu_sd, self.sigma_sd = self.standard_error()
## END of Class GMM