#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:00:30 2020

@author: angelo
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score,cross_val_predict,cross_validate
import os
projectFolder = '/home/angelo/Daten/Master_Arbeit/Master_Arbeit'
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
from scipy.signal import savgol_filter

measurementFolder = os.path.join(projectFolder,'Data/Pilot_Studie/Temp_20_01_21/')
dataFolder = os.path.join(projectFolder,'Data/')


#%%
highpass = 0
lowpass = 0
bandpass = [20,50]
ival = [-100,1000]

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
"""
Time intervals of epo data from which features are extracted
"""
ivals = [[160, 200], [200,240],[240,280],[280,300],[300,320],[320,340],[340,360],
        [360,380],[380, 400], [400,440],[440,480],[480, 520]]

#%%
epo_o_target = epo_o[:,:,mrk_class_o==0]

epo = np.concatenate([epo_o_target,epo_a],axis=2)
mrk_class = np.concatenate([np.zeros(epo_o_target.shape[2]),mrk_class_a],axis=0)
#%%
meanAmplFeature,mrk_class = convertEpoToFeature(epo,epo_t_o,mrk_class,ivals)
clf = LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
#%%
"""
scores = cross_validate(clf, meanAmplFeature.T, mrk_class, cv=20,return_estimator=True)

for estimator in scores['estimator']:
    ld.predict_proba(meanAmplFeature.T)
    # CHECK what percentage of FP / FN per class

"""
#%%

for artifact_class in np.unique(mrk_class_a):
    meanArtifact=np.mean(epo_a[:,:,mrk_class_a==artifact_class],axis=2)
    smooth_meanArtifact=savgol_filter(meanArtifact,11,3,axis=0)
    np.savetxt(str(artifact_class),meanArtifact)



