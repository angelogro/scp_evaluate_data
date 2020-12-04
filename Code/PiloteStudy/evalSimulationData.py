#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 21:07:13 2020

@author: angelo
"""


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
simDataFolder = os.path.join(dataFolder,'simulation')


#%%
highpass = 0.1
lowpass = 0
bandpass = [20,50]
ival = [-100,1000]
ref_ival=[-100,0]

clab = np.array(['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3',
    'P4', 'O1', 'O2', 'F7','F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2',
    'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POz'])

cindex={j:i for i,j in enumerate(clab)}

clabs={31:clab,
       27:np.array(['F3', 'F4', 'C3', 'C4', 'P3',
    'P4', 'O1', 'O2', 'F7','F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2',
    'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'POz']),
       23:np.array(['F3', 'F4', 'C3', 'C4','O1', 'O2', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2',
    'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'POz']),
       19:np.array(['F3', 'F4', 'C3', 'C4','O1', 'O2', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz',
    'FC5', 'FC6', 'CP5', 'CP6', 'POz']),
       15:np.array(['F3', 'F4', 'C3', 'C4','T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz',
     'CP5', 'CP6', 'POz']),
       12:np.array(['F3', 'F4', 'C3', 'C4','T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz']),
       9:np.array(['C3', 'C4','T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']),
       7:np.array(['C3', 'C4','T7', 'T8', 'Cz', 'Fz','Pz']),
       5:np.array(['C3', 'C4','Cz','Fz', 'Pz']),
       3:np.array(['Cz','Fz', 'Pz']),
       1:np.array(['Cz'])
    }
cindexes={k:[cindex[c] for c in v] for k,v in clabs.items()}
ivals = [[160, 190]]
n_samples=100
epo_t= np.linspace(-100, 1000,111)
datadict={}
#%% DATA PREPARATION

for n_source in ['5','10','20']:
    for art in ['1','2','4','6','7','8']:
        dat=np.genfromtxt(os.path.join(simDataFolder,art+'_1',n_source+'.csv'),delimiter=',')
        datadict[(art+'_1',n_source)]=np.reshape(dat,(31,111,125),order='F').swapaxes(0,1)[:,:,:n_samples]
#%%
for n_source in ['5','10','20']:        
    for stddev in [0,0.2,0.4,0.6,0.8,1]:
        dat=np.genfromtxt(os.path.join(simDataFolder,'stddev'+str(stddev),n_source+'.csv'),delimiter=',')
        datadict[('t'+str(stddev),n_source)]=np.reshape(dat,(31,111,1000),order='F').swapaxes(0,1)[:,:,:n_samples]
#%%
mrk_class_a=[]        
for art in ['1','2','4','6','7','8']:
    mrk_class_a.append(np.ones(n_samples)*int(art))
mrk_class_a=np.array(mrk_class_a).ravel()
        
for n_source in ['5','10','20']:
    locals()['art'+n_source]=np.concatenate((datadict[('1'+'_1',n_source)],datadict[('2'+'_1',n_source)],
                                             datadict[('4'+'_1',n_source)],datadict[('6'+'_1',n_source)],
                                             datadict[('7'+'_1',n_source)],datadict[('8'+'_1',n_source)]),axis=2)
mrk_class_o=np.zeros(n_samples)
#%%
mrk_class=np.concatenate((mrk_class_o,mrk_class_a))

#%%
for stddev in [0,0.2,0.4,0.6,0.8,1]: 
    epo5=np.concatenate((datadict[('t'+str(stddev),'5')],art5),axis=2)
    epo10=np.concatenate((datadict[('t'+str(stddev),'10')],art10),axis=2)
    epo20=np.concatenate((datadict[('t'+str(stddev),'20')],art20),axis=2)

#%%
# LDA Classifier: Artifact vs Target
clf = LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
meanconf,stdconf = classifyTargetVsArtifact(clf,mrk_class_a,datadict[('t1','20')],art20,ivals,epo_t,cv=2)
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
meanconf,stdconf = classifyTargetVsArtifacts(clf,mrk_class,epo20_,ivals,epo_t,cv=5)

disp=ConfusionMatrixDisplay(meanconf, list(map(lambda x:abbrDict[x],np.unique(mrk_class))))
plt.figure(figsize=(10,10))
disp.plot(values_format='.2f',cmap='Blues')
