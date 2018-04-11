# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 09:17:13 2017

@author: chris
"""

import numpy as np
import random as rd

rd.seed(0)
np.random.seed(0)

_n=18 ## '_' prefix marks const
_r=12

Y1 = np.array([8,6,11,22,14,17,18,24,19,23,26,40,
               4,4,5,6,8,10])
Y2 = np.array([59,58,56,53,50,45,43,42,39,38,30,27,
               np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])

#######################################################

def EM(r,n,y1,y2,pa):
    [m1,s11,m2,s22,rho] = pa
    # Estep
    s12 = rho*np.sqrt(s11*s22)
    b211 = s12/s11
    b201 = m2 - s12*m1/s11
    s221 = s22 - s12**2/s11
    Ey2    = sum(y2[0:r]) + sum(b211*y1[r:n]+b201)
    Ey2sq  = sum(y2[0:r]**2) + sum((b211*y1[r:n]+b201)**2 + s221)
    Ey1y2  = sum(y1[0:r]*y2[0:r]) + sum(y1[r:n]*(b211*y1[r:n]+b201))
    # Mstep
    re_mu2 = Ey2 / n
    # in Meng and Rubin 1991 , they update sigma22 and rho asynchronously
    re_sigma22 = (Ey2sq - 2*re_mu2*Ey2 + n*re_mu2**2) / n
    re_rho = (Ey1y2 - re_mu2*n*m1 - m1*Ey2 + n*m1*re_mu2) / np.sqrt(s11*re_sigma22) / n
    # I also tried to update sigma22 and rho asynchronously
    # re_sigma22 = (Ey2sq - 2*m2*Ey2 + n*m2**2) / n
    # re_rho = (Ey1y2 - m2*n*m1 - m1*Ey2 + n*m1*m2) / np.sqrt(s11*s22) / n
    return [re_mu2,re_sigma22,re_rho]

def trans(pa):
    re1 = pa[0]
    re2 = np.log(pa[1])
    re3 = 0.5 * (np.log((1+pa[2])/(1-pa[2])))
    return [re1,re2,re3]

# EM's control parameters
cache_size = 1000 # record the last $cache_size$ iteration's results
delta = np.inf    # difference between theta[t] and theta[t+1]
c_limit = 1e-8    # stop iteration if delta < $c_limit$

mu1 = np.mean(Y1)
sigma11 = np.var(Y1)
para = np.zeros([cache_size,3])
# para[:][0] -> mu2 , para[:][1] -> sigma22 , para[:][2] -> rho
theta = np.zeros([cache_size,3])
# theta[:][0] -> mu2 , theta[:][1] -> ln(sigma22) , theta[:][2] -> Zrho
para[0] = [np.mean(Y2[0:_r]),np.var(Y2[0:_r]),np.cov(Y1[0:_r],Y2[0:_r],bias=True)[0][1]/np.sqrt(sigma11*np.var(Y2[0:_r]))]
theta[0]= trans(para[0])
t = 0
while (delta > c_limit and t < cache_size):
    tmp_pa = [mu1,sigma11,para[t][0],para[t][1],para[t][2]]
    para[t+1] = EM(_r,_n,Y1,Y2,tmp_pa)
    theta[t+1]= trans(para[t+1])
    delta = max(abs(theta[t+1] - theta[t]))
    t += 1

# collect the results
mu2 = round(para[t][0],8)
sigma22 = round(para[t][1],8)
rho = round(para[t][2],8)
print ("|mu2\t|lns22\t|Zrho\t|")
print ("|%.2f\t|%.2f\t|%.2f\t|"%(theta[t][0],theta[t][1],theta[t][2]))
_theta=[theta[t][0],theta[t][1],theta[t][2]]


######################################################################
# SEM
# EM's control parameters
cache_size = 1000 # record the last $cache_size$ iteration's results
delta = np.inf    # difference between theta[t] and theta[t+1]
c_limit = 1e-4    # stop iteration if delta < $c_limit$
# initialize
para = np.zeros([cache_size,3])
theta = np.zeros([cache_size,3])
theta_t = np.zeros([3,3])
R = np.zeros([cache_size,3,3])

para[0] = [np.mean(Y2[0:_r]),np.var(Y2[0:_r]),np.cov(Y1[0:_r],Y2[0:_r],bias=True)[0][1]/np.sqrt(sigma11*np.var(Y2[0:_r]))]
theta[0]= trans(para[0])

t=0
cnt = 0 # cnt > 3 indicates that DM is stable
while (cnt < 3 and t < cache_size-1):
    # usual EM
    tmp_pa = [mu1,sigma11,para[t][0],para[t][1],para[t][2]]
    para[t+1]  = EM(_r,_n,Y1,Y2,tmp_pa)
    theta[t+1] = trans(para[t+1])
    # fix parameter
    for i in range(3):
        tmp_pa = [mu1,sigma11,mu2,sigma22,rho]
        tmp_pa[i+2] = para[t][i]
        [t1,t2,t3] = EM(_r,_n,Y1,Y2,tmp_pa)
        theta_t[i] = [t1,np.log(t2),0.5*np.log((1+t3)/(1-t3))]
        R[t+1][i] = (theta_t[i] - _theta) / (theta[t][i] - _theta[i])
    delta = max((R[t+1]-R[t]).reshape(9))
    if (delta<c_limit):
        cnt += 1
    else:
        cnt = 0
    t += 1

DM = np.matrix(R[t])

G1 = np.matrix([[4.9741,0],[0,0.1111]])
G2 = np.matrix([[-5.0387,0,0],[0,0.0890,-0.0497]])
G3 = np.matrix([[6.3719,0,0],[0,0.1111,-0.0497],[0,-0.0497,0.0556]])
deltaV = (G3 - G2.transpose() * np.linalg.inv(G1) * G2) * DM * np.linalg.inv(np.matrix(np.eye(3))-DM)

for i in range(3):
    print('%.4f & %.4f & %.4f'%(deltaV[i,0],deltaV[i,1],deltaV[i,2]))

for i in range(3):
    print('%.5f & %.5f & %.5f'%(DM[i,0],DM[i,1],DM[i,2]))

######################################################################
# bootstrap B times
    
# EM's control parameters
cache_size = 1000 # record the last $cache_size$ iteration's results
delta = np.inf    # difference between theta[t] and theta[t+1]
c_limit = 1e-4    # stop iteration if delta < $c_limit$
B = 500

b_record = np.zeros([B,3])
for T in range(B):
    n = round(_n*4)
    index = np.random.randint(0,_n,n)
    y1 = Y1[index]
    y2 = Y2[index]
    r = n - sum(np.isnan(y2))
    for i in range(n):
        if (np.isnan(y2[i])):
            j=n-1
            while (np.isnan(y2[j]) and j > i):j -= 1
            y1[i],y1[j] = y1[j],y1[i]
            y2[i],y2[j] = y2[j],y2[i]
    # EM initialize
    delta = np.inf
    b_mu1 = np.mean(y1)
    b_sigma11 = np.var(y1)
    b_para    = np.zeros([cache_size,3])
    b_theta   = np.zeros([cache_size,3])
    b_para[0] = [np.mean(y2[0:r]),np.var(y2[0:r]),np.cov(y1[0:r],y2[0:r],bias=True)[0][1]/np.sqrt(np.var(y1[0:r])*np.var(y2[0:r]))]
    b_theta[0]  = trans(para[0])
    t = 0
    while (delta > c_limit and t+1 < cache_size):
        tmp_pa = [b_mu1,b_sigma11,b_para[t][0],b_para[t][1],b_para[t][2]]
        b_para[t+1] = EM(r,n,y1,y2,tmp_pa)
        b_theta[t+1]= trans(b_para[t+1])
        delta = max(abs(b_theta[t+1] - b_theta[t]))
        t += 1
    b_record[T] = b_theta[t]

print('|mu2|lns22|Zp|')
print('|%.2f|%.2f|%.2f|'%(np.mean(b_record[0:B,0]),np.mean(b_record[0:B,1]),np.mean(b_record[0:B,2])))

semu2   = np.sqrt(np.var(b_record[0:B,0]) * B / (B-1))
selns22 = np.sqrt(np.var(b_record[0:B,1]) * B / (B-1))
seZp    = np.sqrt(np.var(b_record[0:B,2]) * B / (B-1))
print('|%.2f|%.2f|%.2f|'%(semu2,selns22,seZp))
