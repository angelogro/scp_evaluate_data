#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 20:38:56 2020

@author: angelo
"""
from matplotlib import pyplot as plt
from matplotlib import cm
from bci_plotFuncs import forceAspect
import numpy as np

art_labels = ['Press Feet','Lift Tongue','Clinch Teeth','Push Breath','Wrinkle Nose','Swallow']

FP = [[10.7,7.8],
      [0,0],
      [0,0],
      [3.6,1.5],
      [2.9,9.8],
      [0.1,0]]

FN = [[34,18.8],
      [17.1,3.2],
      [11.5,4],
      [13.9,8.8],
      [32.3,26.3],
      [34.5,15.3]]

def plotty():
    max_ = max(max(FN+FP))
    
    plt.figure(figsize=(8,8))
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    plt.imshow(FP,cmap=cm.Reds)
    plt.ylabel('Artifact')
    plt.yticks(np.arange(6),art_labels)
    plt.clim(0,max_)
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(8,8))
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    plt.imshow(FN,cmap=cm.Reds)
    plt.ylabel('Artifact')
    plt.yticks(np.arange(6),art_labels)
    plt.clim(0,max_)
    plt.colorbar()
  
    plt.show()
    
