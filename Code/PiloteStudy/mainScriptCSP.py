#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 16:37:49 2020

@author: angelo
"""
import os
projectFolder = '/home/angelo/Master_Arbeit/'
import sys
sys.path.insert(1,os.path.join(projectFolder,'Code/libs'))
from wyrm import io

import bci_minitoolbox as bci
from bci_dataPrep import *
from bci_plotFuncs import *
from bci_classify import *
from bci_processing import *
import global_vars

import numpy as np
from scipy import signal

measurementFolder = os.path.join(projectFolder,'Data/Pilot_Studie/Temp_20_01_21/')
dataFolder = os.path.join(projectFolder,'Data/')


#%%
highpass = 0
lowpass = 0
bandpass = [20,50]
ival = [0,700]

rejectChannels = np.array(['Fp1', 'Fp2', 'F3', 'F4', 'P3', 'P4', 'O1', 'O2', 'F7',
       'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Pz', 'Oz', 'FC1', 'FC2',
       'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POz'])
#%%
data_list = load_multiple_data([os.path.join(measurementFolder,'2020-MSc-Angelo-1__oddballTemp03.vhdr'),os.path.join(measurementFolder,'2020-MSc-Angelo-1__oddballTemp02.vhdr'),
                                                                      os.path.join(measurementFolder,'2020-MSc-Angelo-1__oddballTemp.vhdr')],                                                                      
                                                                     dattype='oddball')
#%%
epo_o,epo_t_o,mrk_class_o,clab,mnt = prepareData(data_list,downsample_factor=10,highPassCutOff=highpass,lowPassCutOff=lowpass,ival=ival,
                                                                     ref_ival=[-100,0],reject_voltage=1000,nRemovePCA = 0
                                                                     )  
#%%
data_list_art = load_multiple_data([os.path.join(measurementFolder,'2020-MSc-Angelo-1__artifactTemp.vhdr'),
                                         os.path.join(measurementFolder,'2020-MSc-Angelo-1__artifactTemp02.vhdr'),
                                         os.path.join(measurementFolder,'2020-MSc-Angelo-1__artifactTemp03.vhdr'),
                                         os.path.join(measurementFolder,'2020-MSc-Angelo-1__artifactTemp04.vhdr')])
#%%
epo_a,epo_t_a,mrk_class_a,clab,mnt = prepareData(data_list_art,downsample_factor=10,highPassCutOff=highpass,lowPassCutOff=lowpass,ival=ival,
                                                                     ref_ival=[-100,0],reject_voltage=10000,nRemovePCA = 0
                                                                     )
#%%
""" Classify artifact vs target
"""
epo_o_target = epo_o[:,:,mrk_class_o==0]
for artifact in np.unique(mrk_class_a):
    epo_a_art = epo_a[:,:,mrk_class_a==artifact]
    
    epo = np.concatenate([epo_o_target,epo_a_art],axis=2)
    mrk_class = np.concatenate([np.zeros(epo_o_target.shape[2]),np.ones(epo_a_art.shape[2])],axis=0)
    epo,mrk_class = balanceClasses(epo, mrk_class)
    numIterations = 1
    print("Artifact: "+str(artifactDict[artifact]))
    for i in range(numIterations):
        params=calcAndValLDATempSpat(epo, epo_t_o, mrk_class, ivals,returnParams=True,n_folds=21)
 #%%
CSPintervals=np.array([[[300,400],[400,500],[500,600]], #Press Feet
                   [[100,200],[200,300],[300,400],[400,500]], #Lift Tongue
                   [[100,200],[200,300],[300,400],[400,500]], #Clinch Teeth
                   [[200,300],[300,400],[400,500]], #Push Breath
                   [[300,400],[400,500],[500,600]], #Wrinkle Nose
                   [[100,200],[200,300],[300,400]]])  #Swallow
                   
#%% Calculate CSP

epo_o_ = epo_o[:,:,mrk_class_o==0]  #Extract only target trials
numCSP = 3
for idx,artifact in enumerate(np.unique(mrk_class_a)):
    epo_a_art = epo_a[:,:,mrk_class_a==artifact]
    epo = np.concatenate([epo_o_,epo_a_art],axis=2)
    mrk_class = np.concatenate([mrk_class_o[mrk_class_o==0],np.ones(len(mrk_class_a[mrk_class_a==artifact]))],axis=0)
    epo,mrk_class = balanceClasses(epo, mrk_class)
    W,_,d = calculate_csp(epo, mrk_class)
    """
    plt.figure()
    plt.plot(d,'.',label='right')
    plt.plot(1-d,'.',label='left')
    plt.xlabel('Rank number of Eigenvector [#]')
    plt.ylabel('Generalized Eigenvalue')
    plt.legend()
    """
    # Calculate maximum difference in eigenvalues of the two classes
    ddif = np.abs(d-(1-d))
    
    # Get the indexes of the 6 maximum eigenvalues   
    eigidx = np.argsort(ddif)[-numCSP:]
    
    # Spatial filtering of signals
    CSPs = apply_spatial_filter(epo,W[:,eigidx])
    
    AllCSP = apply_spatial_filter(epo,W)
    A = np.dot(np.dot(np.cov(epo.reshape(epo.shape[1],-1)),W),np.linalg.inv(np.cov(AllCSP.reshape(epo_o.shape[1],-1))))
    """
    plt.figure(figsize=(8,8))
    n=221
    for i in range(eigidx.shape[0]):
        plt.subplot(n+i)
        plt.title('CSP filter '+str(4-i))
        bci.scalpmap(mnt,A[:,eigidx[i]],clim='sym',cb_label='[a.u.]')
    """
    
    epoCSP = np.abs(signal.hilbert(CSPs,axis = 0))
    
    aveCSPsT = np.mean(epoCSP[:,:,mrk_class==0],axis=2)
    aveCSPsArtifact = np.mean(epoCSP[:,:,mrk_class==1],axis=2) 
    
    plt.figure(figsize=(10,8)) 
    n=221
    for i in range(eigidx.shape[0]):
        plt.subplot(n+i)
        plt.title('Averaged ERD, EigNum =  '+str(eigidx[i]))
        plt.plot(epo_t_o,aveCSPsT[:,i],label='target')
        plt.plot(epo_t_o,aveCSPsArtifact[:,i],label=str(artifactDict[artifact]))
        plt.ylabel('Amplitude [uV]')
        plt.xlabel('time [ms]')
        plt.legend()
    
    
    
    print("Artifact: "+str(artifactDict[artifact]))
    for i in range(3):
        params=calcAndValLDACSP(CSPs, epo_t_o, mrk_class, CSPintervals[idx],returnParams=True,verbose=False)
    
    
#%%
