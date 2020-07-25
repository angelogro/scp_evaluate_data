#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:58:50 2020

@author: angelo
"""

from plotting import *
import numpy as np
from matplotlib import pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#497
np.random.seed(245)
size=1000
var=0.1
cov = np.array([[1,0.1],[-0.1,0.3]])*var
means=np.array([[1.5,1.5],[0.7,0.7]])
fac=0.8
alpha = 0.2

X_1 = np.random.multivariate_normal(means[0],cov=cov,size=size)/fac
X_2 = np.random.multivariate_normal(means[1],cov=cov,size=size)/fac
Y = np.concatenate((np.zeros(size),np.ones(size)))
X = np.concatenate((X_1,X_2))
#plt.scatter()
#plt.scatter(list(zip(*X_2))[0],list(zip(*X_2))[1],c='r')

fig, ax = plt.subplots(figsize=(6,6))


ax.scatter(list(zip(*X_1))[0],list(zip(*X_1))[1],s=7,marker='o',alpha=alpha)
ax.scatter(list(zip(*X_2))[0],list(zip(*X_2))[1],s=7,marker='o',c='r',alpha=alpha)
label = ax.set_xlabel(r'${X}_{1}$', fontsize = 9)
label = ax.set_ylabel(r'${X}_{2}$', fontsize = 9,rotation='horizontal')

m1 = np.mean(X_1,axis=0)
m2 = np.mean(X_2,axis=0)

add_mean(ax,fig,m1,m2)




clf = LinearDiscriminantAnalysis()
clf.fit(X,Y)
#normalize weight vector
w = clf.coef_/np.linalg.norm(clf.coef_)
w = -w if np.all(w<0) else w

add_separating_line(ax,fig,m1,m2,w[0],linestyle= ':')

w_arr=add_w_arrow(ax,fig,w[0],'w',offset=(m1+m2)/2)

ax.text(0.9,0.82,'Separating hyperplane: '+r'$\mathbf{w}^{T}\mathbf{x}-b=0$',
        rotation=-13.6,fontsize=10)
#ortho_lines = add_orthogonal_lines(ax,fig,w[0],X_1,c='c')
#ortho_lines = add_orthogonal_lines(ax,fig,w[0],X_2,c='r')

format_plot(ax,fig)
plt.show()