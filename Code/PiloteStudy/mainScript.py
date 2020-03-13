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
data_list = load_multiple_data([os.path.join(baseFolder,'2020-MSc-Angelo-1__oddballTemp03.vhdr'),os.path.join(baseFolder,'2020-MSc-Angelo-1__oddballTemp02.vhdr'),
                                                                      os.path.join(baseFolder,'2020-MSc-Angelo-1__oddballTemp.vhdr')],                                                                      
                                                                     dattype='oddball')

#%%
data_list_art = load_multiple_data([os.path.join(baseFolder,'2020-MSc-Angelo-1__artifactTemp.vhdr'),
                                         os.path.join(baseFolder,'2020-MSc-Angelo-1__artifactTemp02.vhdr'),
                                         os.path.join(baseFolder,'2020-MSc-Angelo-1__artifactTemp03.vhdr'),
                                         os.path.join(baseFolder,'2020-MSc-Angelo-1__artifactTemp04.vhdr')])
epo_a,epo_t_a,mrk_class_a,clab,mnt = prepareData(data_list_art,downsample_factor=10,highPassCutOff=0,lowPassCutOff=0,ival=[-100,1000],
                                                                     ref_ival=[-100,0],reject_voltage=1000,nRemovePCA = 0)
#%%
for highPassCutOff in [0]:
    for lowPassCutOff in [0]:
        for remove_pca in [0,1]:
            for reject_voltage in [1000]:
            
                epo_o,epo_t_o,mrk_class_o,clab,mnt = prepareData(data_list,downsample_factor=10,highPassCutOff=highPassCutOff,lowPassCutOff=lowPassCutOff,ival=[-100,1000],
                                                                         ref_ival=[-100,0],reject_channels=None,
                                                                         reject_voltage=reject_voltage,nRemovePCA = remove_pca)
                plotMeanCurves(epo_o,epo_t_o,clab,['Cz','FC1','FC2','CP1','CP2','C3','C4'],mrk_class_o)
#%%
rej_channels = ['P7','TP10','TP9','T8','T7','F8','O2','O1','P3','FP2','Oz']
epo_o,epo_t_o,mrk_class_o,clab,mnt = prepareData(data_list,downsample_factor=10,highPassCutOff=0,lowPassCutOff=0,ival=[-100,1000],
                                                                     ref_ival=[-100,0],reject_channels=['P7','TP10','TP9'],reject_voltage=1000,
                                                                     nRemovePCA = 1)    
#%%
epo_o,epo_t_o,mrk_class_o,clab,mnt = prepareData(data_list,downsample_factor=10,highPassCutOff=0,lowPassCutOff=0,ival=[-100,1000],
                                                                     ref_ival=[-100,0],reject_voltage=1000,
                                                                     nRemovePCA = 0)         
#%%
plotMeanCurves(epo_o,epo_t_o,clab,['Cz','FC1','FC2','CP1','CP2','C3','C4'],mrk_class_o) #['Cz','FC1','FC2','CP1','CP2']
#%%
intervals =[[100,120],[150,160],[200,210],[250,260],[300,310],[350,360],[400,410]]
#%%
plotScalpmapsOddball(epo_o, epo_t_o, clab, mrk_class_o, intervals, mnt)
#%%
plotPrincipalComponents(cnt_o, mnt, clab)

#%%
plotChannelwiseCovarianceMatrix(epo_o, epo_t_o, clab, mrk_class_o, 400)
#%%
calcAndPlotR2(epo_o, epo_t_o, mrk_class_o, clab)
#%%
"""
Time intervals of epo data from which features are extracted
"""
ivals = [[160, 200], [200,240],[240,280],[280,300],[300,320],[320,340],[340,360],
        [360,380],[380, 400], [400,440],[440,480],[480, 520]]
 
#%%
"""
Fix amount of iterations.
Create classifiers in each iterations and test how many of each class of artifacts
would be classified as target.
"""
numIterations = 10
#testErrors = np.zeros((np.unique(mrk_class_a).shape[0],numIterations))
for i in range(numIterations):
    params=calcAndValLDATempSpat(epo_o, epo_t_o, mrk_class_o, ivals,returnParams=True)
    #for idx,artifact in enumerate(np.unique(mrk_class_a)):
    #    testErrors[idx,i]= validateLDA(params,epo_a[:,:,mrk_class_a==artifact],epo_t_a,np.ones(sum(mrk_class_a==artifact)),ivals)

    
#%%
selectedChannels = ['Oz','Pz','Cz']
#%%
plotMeanCurvesArtifact(epo_a,epo_t_a,clab,selectedChannels,mrk_class_a)

#%%
selectedChannels = ['Oz','Pz','Cz']
epo_sel = selectChannelFromEpo(epo_o, clab, selectedChannels)
plotMeanCurves(epo_sel,epo_t_o,selectedChannels,mrk_class_o)