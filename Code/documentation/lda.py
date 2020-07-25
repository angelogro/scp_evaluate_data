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
size=10
var=1
fac=1.5

X_1 = np.random.multivariate_normal((2.,2.),cov=np.array([[var,0],[0,var]]),size=size)/fac
X_2 = np.random.multivariate_normal((1.5,1.5),cov=np.array([[var,0],[0,var]]),size=size)/fac
Y = np.concatenate((np.zeros(size),np.ones(size)))
X = np.concatenate((X_1,X_2))
#plt.scatter()
#plt.scatter(list(zip(*X_2))[0],list(zip(*X_2))[1],c='r')

fig, ax = plt.subplots(figsize=(6,6))


ax.scatter(list(zip(*X_1))[0],list(zip(*X_1))[1],s=7,marker='o')
ax.scatter(list(zip(*X_2))[0],list(zip(*X_2))[1],s=7,marker='o',c='r')
label = ax.set_xlabel(r'${X}_{1}$', fontsize = 9)
label = ax.set_ylabel(r'${X}_{2}$', fontsize = 9,rotation='horizontal')






clf = LinearDiscriminantAnalysis()
clf.fit(X,Y)
#normalize weight vector
w = clf.coef_/np.linalg.norm(clf.coef_)
w = -w if np.all(w<0) else w

w_arr=add_w_arrow(ax,fig,w[0],'w')

ortho_lines = add_orthogonal_lines(ax,fig,w[0],X_1,c='c')
ortho_lines = add_orthogonal_lines(ax,fig,w[0],X_2,c='r')

format_plot(ax,fig)
plt.show()