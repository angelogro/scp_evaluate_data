#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:00:30 2020

@author: angelo
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score,cross_val_predict,cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from sklearn.multiclass import OneVsOneClassifier
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
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import signal
from scipy.signal import savgol_filter

measurementFolder = os.path.join(projectFolder,'Data/Pilot_Studie/Temp_20_01_21/')
dataFolder = os.path.join(projectFolder,'Data/')


#%%
highpass = 0.1
lowpass = 0
bandpass = [20,50]
ival = [-100,1000]
ref_ival=[-100,0]

rejectChannels = np.array(['Fp1', 'Fp2', 'F3', 'F4', 'P3', 'P4', 'O1', 'O2', 'F7',
       'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Pz', 'Oz', 'FC1', 'FC2',
       'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POz'])


#%%
data_list = load_multiple_data([os.path.join(measurementFolder,'2020-MSc-Angelo-1__oddballTemp03.vhdr'),os.path.join(measurementFolder,'2020-MSc-Angelo-1__oddballTemp02.vhdr'),
                                                                      os.path.join(measurementFolder,'2020-MSc-Angelo-1__oddballTemp.vhdr')],                                                                      
                                                                     dattype='oddball')
#%%
epo_o,epo_t_o,mrk_class_o,clab,mnt = prepareData(data_list,downsample_factor=10,highPassCutOff=highpass,lowPassCutOff=lowpass,ival=ival,
                                                                     ref_ival=ref_ival,reject_voltage=1000,nRemovePCA = 0,
                                                                     performpca=False)  
#%%
data_list_art = load_multiple_data([os.path.join(measurementFolder,'2020-MSc-Angelo-1__artifactTemp.vhdr'),
                                         os.path.join(measurementFolder,'2020-MSc-Angelo-1__artifactTemp02.vhdr'),
                                         os.path.join(measurementFolder,'2020-MSc-Angelo-1__artifactTemp03.vhdr'),
                                         os.path.join(measurementFolder,'2020-MSc-Angelo-1__artifactTemp04.vhdr')])
#%%
epo_a,epo_t_a,mrk_class_a,clab,mnt = prepareData(data_list_art,downsample_factor=10,highPassCutOff=highpass,lowPassCutOff=lowpass,ival=ival,
                                                                     ref_ival=ref_ival,reject_voltage=1000,nRemovePCA = 0,performpca=False
                                                                     )
#%%
"""
Time intervals of epo data from which features are extracted
"""
ivals = [[160, 190], [200,230],[240,270],[280,290],[300,310],[320,330],[340,350],
        [360,370],[380, 390], [400,430],[440,470],[480, 520]]

#%%h
epo_o_target = epo_o[:,:,mrk_class_o==0]

epo = np.concatenate([epo_o_target,epo_a],axis=2)
mrk_class = np.concatenate([np.zeros(epo_o_target.shape[2]),mrk_class_a],axis=0)
#%%
meanAmplFeature,mrk_class = convertEpoToFeature(epo,epo_t_o,mrk_class,ivals)
clf = LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
#%%

scores = cross_validate(clf, meanAmplFeature.T, mrk_class, cv=20,return_estimator=True)

for estimator in scores['estimator']:
    x=estimator.predict_proba(meanAmplFeature.T)
    print(x)
    # CHECK what percentage of FP / FN per class


#%%
# LDA Classifier: Artifact vs Target
clf = LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
meanconf,stdconf = classifyTargetVsArtifact(clf,mrk_class_a,epo_o_target,epo_a,ivals,epo_t_o)
#%%
xaxis = [artifactDict[x] for x in np.unique(mrk_class_a)]
plt.figure(figsize=(10,6))
df = pd.DataFrame(zip(xaxis*2,["TP"]*6+["TN"]*6,np.concatenate((meanconf[:,0,0],meanconf[:,1,1])),
                      np.concatenate((stdconf[:,0,0],stdconf[:,1,1]))),columns=["Artifact Type", "Type","Average Rate","std"])
grouped_barplot(df, "Artifact Type", "Type", "Average Rate", "std",colors=plotcolours,
                title="LDA with shrinkage - Classification Accuracy")   
#%%
# Multiclass
clf = LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
meanconf,stdconf = classifyTargetVsArtifacts(clf,mrk_class,epo,ivals,epo_t_o)

disp=ConfusionMatrixDisplay(meanconf, list(map(lambda x:abbrDict[x],np.unique(mrk_class))))
plt.figure(figsize=(10,10))
disp.plot(values_format='.2f',cmap='Blues')
#%%
# Extract relevant data (T = target, NT = NonTarget)
chans=['Cz','Pz']
time=300
plt.figure(figsize=(10,10))
for mrk_c in np.unique(mrk_class):
    first=epo[np.where(epo_t_o==time),clab==chans[0],mrk_class==mrk_c]
    second=epo[np.where(epo_t_o==time),clab==chans[1],mrk_class==mrk_c]
    plt.scatter(first,second,s=2,label=artifactDict[mrk_c])
plt.legend()
plt.xlim((-10,10))                  
#%%
# LDA Classifier: Artifact vs Target
clf = LinearDiscriminantAnalysis(solver='eigen')
meanconf,stdconf = classifyTargetVsArtifact(clf,mrk_class_a,epo_o_target,epo_a,ivals,epo_t_o)
#%%
xaxis = [artifactDict[x] for x in np.unique(mrk_class_a)]
plt.figure(figsize=(10,6))
df = pd.DataFrame(zip(xaxis*2,["TP"]*6+["TN"]*6,np.concatenate((meanconf[:,0,0],meanconf[:,1,1])),
                      np.concatenate((stdconf[:,0,0],stdconf[:,1,1]))),columns=["Artifact Type", "Type","Average Rate","std"])
grouped_barplot(df, "Artifact Type", "Type", "Average Rate", "std",colors=plotcolours,
                title="LDA - Classification Accuracy") 
#%%
# Multiclass
clf = LinearDiscriminantAnalysis(solver='eigen')
meanconf,stdconf = classifyTargetVsArtifacts(clf,mrk_class,epo,ivals,epo_t_o)

disp=ConfusionMatrixDisplay(meanconf, list(map(lambda x:abbrDict[x],np.unique(mrk_class))))
plt.figure(figsize=(10,10))
disp.plot(values_format='.2f',cmap='Blues')

#%%
# Prepare CSP data
for art in np.unique(mrk_class_a):
    plotPSDfromEpoMulticlass(epo,mrk_class,clab,'Cz',[0,art],'artifact',100,frange=None)

frequency_bands=[[2,3.5],[3.5,10],[10.5,12],[13,15],[16,50]]    
#%%
epo_fr=[]
for band in frequency_bands:
    epo_o,epo_t_o,mrk_class_o,clab,mnt = prepareData(data_list,downsample_factor=10,highPassCutOff=highpass,lowPassCutOff=lowpass,ival=ival,
                                                                         ref_ival=ref_ival,reject_voltage=1000,nRemovePCA = 0,bandpass=band,
                                                                         performpca=False) 
    epo_a,epo_t_a,mrk_class_a,clab,mnt = prepareData(data_list_art,downsample_factor=10,highPassCutOff=highpass,lowPassCutOff=lowpass,ival=ival,
                                                                     ref_ival=ref_ival,reject_voltage=1000,nRemovePCA = 0,bandpass=band,performpca=False,
                                                                     )
    epo_o_target = epo_o[:,:,mrk_class_o==0]

    epo_fr.append(np.concatenate([epo_o_target,epo_a],axis=2))
    mrk_class = np.concatenate([np.zeros(epo_o_target.shape[2]),mrk_class_a],axis=0)
#%%
numCSP=4
mrk_class = np.concatenate([np.zeros(epo_o_target.shape[2]),mrk_class_a],axis=0)
clf = LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
meanconf,stdconf=[],[]
for idx,artifact in enumerate(np.unique(mrk_class_a)):
    meanAmplFeatures = []
    mrk_class_ = np.concatenate([np.zeros(epo_o_target.shape[2]),mrk_class_a[mrk_class_a==artifact]],axis=0)
    for fr_idx,epo_ in enumerate(epo_fr):
        print(fr_idx)
        title=str(frequency_bands[fr_idx])+artifactDict[artifact]
        epo = epo_
        epo = epo[:,:,((mrk_class==0) | (mrk_class==artifact))]
        
        W,_,d = calculate_csp(epo, mrk_class_)
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
            plt.suptitle(title)
        """
        epoCSP = np.abs(signal.hilbert(CSPs,axis = 0))
        
        aveCSPsT = np.mean(epoCSP[:,:,mrk_class_==0],axis=2)
        aveCSPsArtifact = np.mean(epoCSP[:,:,mrk_class_==artifact],axis=2) 
        """
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
        """
        meanAmplFeature,mrk_class_ = convertCSPToFeature(epo,epo_t_o,mrk_class_,ivals)
        meanAmplFeatures.append(meanAmplFeature)
    meanAmplFeatures = np.array(meanAmplFeatures)
    meanAmplFeatures=meanAmplFeatures.reshape(-1,meanAmplFeatures.shape[2])
    confusion_matrices=[]
    scores = cross_validate(clf, meanAmplFeatures.T, mrk_class_, cv=20,return_estimator=True)
    for estimator in scores['estimator']:
        conf_mat = confusion_matrix(mrk_class_,estimator.predict(meanAmplFeatures.T),normalize='true')
        confusion_matrices.append(conf_mat)
    confusion_matrices = np.array(confusion_matrices)   
    meanconf.append(np.mean(confusion_matrices,axis=0))
    stdconf.append(np.std(confusion_matrices,axis=0))
meanconf,stdconf=np.array(meanconf),np.array(stdconf)

#%%
meanconf,stdconf=np.array(meanconf),np.array(stdconf)
xaxis = [artifactDict[x] for x in np.unique(mrk_class_a)]
plt.figure(figsize=(10,6))
df = pd.DataFrame(zip(xaxis*2,["TP"]*6+["TN"]*6,np.concatenate((meanconf[:,0,0],meanconf[:,1,1])),
                      np.concatenate((stdconf[:,0,0],stdconf[:,1,1]))),columns=["Artifact Type", "Type","Average Rate","std"])
grouped_barplot(df, "Artifact Type", "Type", "Average Rate", "std",colors=plotcolours,
                title="CSP - Classification Accuracy",legend=False) 
#%%


# Generation of Mean epochs of artifacts
saveMeanArtifactSignals(epo_a,epo_t_a,mrk_class_a,n_batches=8)
#plotMeanArtifactSignals(epo_a,epo_t_a,clab,['Cz','Pz'],mrk_class_a)

#%%
# Generation of scalp map plots of artifacts
arti_ivals =[[-100,0],[0,100],[100,200],[200,300],[300,400],[400,500],[500,600],[600,700]]
plotScalpmapsArtifact(epo_a, epo_t_a, clab, mrk_class_a, arti_ivals, mnt)

#%%
# Generation of Mean epochs of target/non-target

plotMeanOddballSignals(epo_o,epo_t_o,clab,['Fz','Cz','Pz'],mrk_class_o)

#%%

plotScalpmapsArtifact(epo_o, epo_t_o, clab, mrk_class_o, arti_ivals, mnt)

