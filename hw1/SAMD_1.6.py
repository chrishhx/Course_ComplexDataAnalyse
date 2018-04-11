# -*- coding: utf-8 -*-
"""
2017-9-7
<Statistical Analysis with Missing Data>
Problems 1.6
Page 23
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

N = 100
np.random.seed(0)
z = np.random.randn(N,4)

#(1)
a = 0 # 0 , 2 , 0
b = 2 # 0 , 0 , 2
y = np.zeros((N,3))
u = np.zeros(N)

y[:,1] = 1 + z[:,1]
y[:,2] = 5 + 2*z[:,1] + z[:,2]
u = a*(y[:,1]-1) + b*(y[:,2]-5) + z[:,3]

m = np.where(u<0,1,0)

fig = plt.figure(figsize=(9,3))

ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.hist(y[:,1],bins=10,normed=1,histtype='stepfilled',range=(-2,4),
         cumulative=False,color='c')
ax1.set_title("Y1 (origin)")
ax2.hist(y[np.where(m==0),1][0],bins=10,normed=1,histtype='stepfilled',range=(-2,4),
           cumulative=False,color='c')
ax2.set_title("Y1 (Y1 observed, Y2 observed)")
ax3.hist(y[np.where(m==1),1][0],bins=10,normed=1,histtype='stepfilled',range=(-2,4),
           cumulative=False,color='c')
ax3.set_title("Y1 (Y1 observed, Y2 missing)")

fig.tight_layout(pad=1.2)
fig.subplots_adjust(top=0.8)
fig.suptitle("a=0,b=2",fontsize=16)
fig.savefig(filename="a0b2.png",dpi=120)

'''
a=0,b=0
f(M|Y1,Y2,U) = F(M|Z3)
Z3 ~ N(0,1) independent
the machanism is MCAR

a=2,b=0
f(M|Y1,Y2,U) = F(M|Y1,Z3)
the machanism is MAR

a=0,b=2
f(M|Y1,Y2,U) = F(M|Y2,Z3)
the machanism is NMAR
'''

#(2)
N = 100
np.random.seed(0)
z = np.random.randn(N,4)

#(1)
a = 0 # 0 , 2 , 0
b = 0 # 0 , 0 , 2
y = np.zeros((N,3))
u = np.zeros(N)

y[:,1] = 1 + z[:,1]
y[:,2] = 5 + 2*z[:,1] + z[:,2]
u = a*(y[:,1]-1) + b*(y[:,2]-5) + z[:,3]

m = np.where(u<0,1,0)

Y1origin = y[:,1]
Y1observe = y[np.where(m==0),1][0]
Y1missing = y[np.where(m==1),1][0]
print ("Y1(origin) : mean =%.6e, std=%.6e"%(np.mean(Y1origin),np.std(Y1origin)))
print ("Y1(obs)    : mean =%.6e, std=%.6e"%(np.mean(Y1observe),np.std(Y1observe)))
print ("Y1(mis)    : mean =%.6e, std=%.6e"%(np.mean(Y1missing),np.std(Y1missing)))
originVmis = stats.ttest_ind(Y1origin,Y1missing)
obsVmis = stats.ttest_ind(Y1observe,Y1missing)
print ("origin Vs mis : t =%.4e, pvalue =%.4e"%(originVmis[0],originVmis[1]))
print ("obs Vs mis    : t =%.4e, pvalue =%.4e"%(obsVmis[0],obsVmis[1]))