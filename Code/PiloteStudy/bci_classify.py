#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 17:20:01 2020

@author: angelo
"""

from scipy import signal
import numpy as np
import os
import bci_minitoolbox as bci
from bci_classifiers import crossvalidation,train_LDAshrink,train_LDA,loss_weighted_error,crossvalidationDetailedLoss

def convertEpoToFeature(epo,epo_t,mrk_class,ivals):
    '''
    Description:
        Converts data to features of mean amplitude in specified intervals.
    Usage:
        feat, mrk_class = convertEpoToFeature(epo,epo_t,mrk_class,ivals)
    Parameters:
        epo: 3D array of segmented signals (samples x channels x epochs)
        epo_t: 1D array of time points of epochs relative to marker (in ms)
        mrk_class: 1D array with class information for each marker
        ivals: 1D array with lists containing start and endtime of each interval
                e.g. [[0,100],[150,180],[330,400]]
    Returns:
        meanAmplFeature: a 2D array (features x epochs)
        mrk_class: a 1D array that assigns markers to classes (0, 1)
    '''
    assert epo.shape[2]==len(mrk_class)
    assert epo.shape[0]==len(epo_t)
    
    sampleIndices = [np.where((epo_t>=ivals[i][0])&(epo_t<=ivals[i][1])) for i in range (len(ivals))]
    meanAmpl=np.zeros((len(ivals),epo.shape[1],epo.shape[2]))
    for i in range(len(ivals)):
       meanAmpl[i]=np.mean(epo[sampleIndices[i],:,:].squeeze(),axis=0)
       
    meanAmplFeature = meanAmpl.reshape((len(ivals)*epo.shape[1]),-1)
    return meanAmplFeature,mrk_class
    

def calcAndValLDATemp(epo,epo_t,mrk_class,ival):
    '''
    Description:
        Crossvalidates a LDA and Shrinked-LDA classifier on temporal features. 
    Usage:
        feat, mrk_class = calcAndValLDATemp(epo,epo_t,mrk_class,ivals)
    Parameters:
        epo: 3D array of segmented signals (samples x channels x epochs)
        epo_t: 1D array of time points of epochs relative to marker (in ms)
        mrk_class: 1D array with class information for each marker
        ival: 1D array with lists containing start and endtime of each interval
                e.g. [[0,100],[150,180],[330,400]]
    Returns:
        None
    '''
    #Convert time intervals to list of sample point number within an epoch recording
    sampleIndices = [np.where((epo_t>=ival[i][0])&(epo_t<=ival[i][1])) for i in range (len(ival))]
    
    meanAmpl=np.zeros((len(ival),epo.shape[1],epo.shape[2]))
    for i in range(len(ival)):
       meanAmpl[i]=np.mean(epo[sampleIndices[i],:,:].squeeze(),axis=0)
       print('***Interval '+str(ival[i])+' ms :')
       print('  Using common LDA   : ' +str(crossvalidation(train_LDA,meanAmpl[i],mrk_class,3,False) [0] ))
       print('  Using shrinked LDA : ' +str(crossvalidation(train_LDAshrink,meanAmpl[i],mrk_class,3,False)[0]))

def calcAndValLDATempSpat(epo, epo_t, mrk_class, ivals,returnParams=False):  
    '''
    Description:
        Crossvalidates LDA and Shrinked-LDA classifier on spatio-temporal
        features returning the calculated Shrinked-LDA parameters
    Usage:
        paramsLDA = calcAndValLDATempSpat(epo,epo_t,mrk_class,ivals,returnParams)
    Parameters:
        epo: 3D array of segmented signals (samples x channels x epochs)
        epo_t: 1D array of time points of epochs relative to marker (in ms)
        mrk_class: 1D array with class information for each marker
        ivals: 1D array with lists containing start and endtime of each interval
                e.g. [[0,100],[150,180],[330,400]]
        returnParams: boolean indicating if the parameters shall be returned
    Returns:
        paramsLDA: {optional} 1D array with tuples (weigth vector, bias) of all
        calculated classifiers
    '''
    meanAmplFeature,mrk_class = convertEpoToFeature(epo,epo_t,mrk_class,ivals)
    print(meanAmplFeature.shape)
    print('**** All features in one vector: ****')
    print('  Using common LDA   : {:5.1f}'.format(crossvalidation(train_LDA,meanAmplFeature,mrk_class,3,False,False)[0]  ))
    errTeFP,errTeFN,paramsLDA = crossvalidationDetailedLoss(train_LDAshrink,meanAmplFeature,mrk_class,3,False,returnParams)
    print('  Using shrinked LDA : FP: {:5.1f} , FN: {:5.1f}'.format(errTeFP,errTeFN))
    if returnParams:
        return paramsLDA
 
def calcAndValLDACSP(epo, epo_t, mrk_class, ivals,returnParams=False,verbose=False):  
    '''
    Description:
        Crossvalidates LDA and Shrinked-LDA classifier on spatio-temporal
        features returning the calculated Shrinked-LDA parameters
    Usage:
        paramsLDA = calcAndValLDATempSpat(epo,epo_t,mrk_class,ivals,returnParams)
    Parameters:
        epo: 3D array of segmented signals (samples x channels x epochs)
        epo_t: 1D array of time points of epochs relative to marker (in ms)
        mrk_class: 1D array with class information for each marker
        ivals: 1D array with lists containing start and endtime of each interval
                e.g. [[0,100],[150,180],[330,400]]
        returnParams: boolean indicating if the parameters shall be returned
    Returns:
        paramsLDA: {optional} 1D array with tuples (weigth vector, bias) of all
        calculated classifiers
    '''
    meanAmplFeature,mrk_class = convertCSPToFeature(epo,epo_t,mrk_class,ivals)

    print('**** All features in one vector: ****')
    print('  Using common LDA   : {:5.1f}'.format(crossvalidationDetailedLoss(train_LDA,meanAmplFeature,mrk_class,3,verbose=verbose,returnParams=False)[0]  ))
    errTeFP,errTeFN,paramsLDA = crossvalidationDetailedLoss(train_LDAshrink,meanAmplFeature,mrk_class,3,verbose=verbose,returnParams=True)
    print('  Using shrinked LDA : FP: {:5.1f} , FN: {:5.1f}'.format(errTeFP,errTeFN))
    if returnParams:
        return paramsLDA
    

def convertCSPToFeature(epo,epo_t,mrk_class,ivals):
    '''
    Description:
        Converts data to features of mean amplitude in specified intervals.
    Usage:
        feat, mrk_class = convertEpoToFeature(epo,epo_t,mrk_class,ivals)
    Parameters:
        epo: 3D array of segmented signals (samples x channels x epochs)
        epo_t: 1D array of time points of epochs relative to marker (in ms)
        mrk_class: 1D array with class information for each marker
        ivals: 1D array with lists containing start and endtime of each interval
                e.g. [[0,100],[150,180],[330,400]]
    Returns:
        meanAmplFeature: a 2D array (features x epochs)
        mrk_class: a 1D array that assigns markers to classes (0, 1)
    '''
    assert epo.shape[2]==len(mrk_class)
    assert epo.shape[0]==len(epo_t)
    
    sampleIndices = [np.where((epo_t>=ivals[i][0])&(epo_t<=ivals[i][1])) for i in range (len(ivals))]
    logVarAmpl=np.zeros((len(ivals),epo.shape[1],epo.shape[2]))
    for i in range(len(ivals)):
       logVarAmpl[i]=np.log(np.var(epo[sampleIndices[i],:,:].squeeze(),axis=0))
       
    meanAmplFeature = logVarAmpl.reshape((len(ivals)*epo.shape[1]),-1)
    return meanAmplFeature,mrk_class



def validateLDA(params,epo,epo_t,mrk_class,ivals):         
    '''
    Description:
        Validates artifact data with LDA parameters.
    Usage:
        meanError = validateLDA(params,epo,epo_t,mrk_class,ival)
    Parameters:
        params: 1D array with tuples (weigth vector, bias) of all calculated classifiers
        epo: 3D array of segmented signals (samples x channels x epochs)
        epo_t: 1D array of time points of epochs relative to marker (in ms)
        mrk_class: 1D array with class information for each marker
        ivals: 1D array with lists containing start and endtime of each interval
                e.g. [[0,100],[150,180],[330,400]]
        returnParams: boolean indicating if the parameters shall be returned
    Returns:
        meanError: averaged classification error over all set of parameters (in %)
    '''
    X,y = convertEpoToFeature(epo,epo_t,mrk_class,ivals)
    errTe = []
    for w,b in params:
        out = w.T.dot(X) - b
        errTe.append(np.sum(out>0)/y.shape*100)
    return np.mean(errTe)