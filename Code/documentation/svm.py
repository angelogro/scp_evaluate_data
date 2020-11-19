# -*- coding: utf-8 -*-

from plotting import *
import numpy as np
from matplotlib import pyplot as plt

from sklearn.svm import LinearSVC,SVC
#497
np.random.seed(245)
size=10
var=0.5
fac=1.5

X_1 = np.random.multivariate_normal((2.,2.),cov=np.array([[var,0],[0,var]]),size=size)/fac
X_2 = np.random.multivariate_normal((0.7,0.7),cov=np.array([[var,0],[0,var]]),size=size)/fac
Y = np.concatenate((np.zeros(size),np.ones(size)))
X = np.concatenate((X_1,X_2))
"""
#plt.scatter()
#plt.scatter(list(zip(*X_2))[0],list(zip(*X_2))[1],c='r')

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

fig, ax = plt.subplots(figsize=(6,6))


ax.scatter(list(zip(*X_1))[0],list(zip(*X_1))[1],s=7,marker='o')
ax.scatter(list(zip(*X_2))[0],list(zip(*X_2))[1],s=7,marker='o',c='r')
label = ax.set_xlabel(r'${X}_{1}$', fontsize = 9)
label = ax.set_ylabel(r'${X}_{2}$', fontsize = 9,rotation='horizontal')

clf = LinearSVC(max_iter=100000,C=1)
clf.fit(X,Y)

w=clf.coef_[0]

plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
"""
clfs = (SVC(kernel='linear',C=10000),
        SVC(kernel='rbf',C=10000,degree=4))
for clf in clfs:
    plt.figure()
    clf.fit(X, Y)
    
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=30, cmap=plt.cm.Paired)
    
    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    
    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')
    
    label = ax.set_xlabel(r'${X}_{1}$', fontsize = 9)
    label = ax.set_ylabel(r'${X}_{2}$', fontsize = 9,rotation='horizontal')
    plt.show()