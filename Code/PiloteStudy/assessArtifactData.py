# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
import sys
sys.path.insert(1, '/home/angelo/Master_Arbeit/Code/libs')
from wyrm import io

import bci_minitoolbox as bci
from bci_dataPrep import *
from bci_plotFuncs import *
from bci_classify import *
import global_vars

import numpy as np
import os


baseFolder = '/home/angelo/Master_Arbeit/Data/Pilot_Studie/Temp_20_01_21/'
dataFolder = '/home/angelo/Master_Arbeit/Data/'

#%%

dat = io.load_brain_vision_data(os.path.join(baseFolder,'2020-MSc-Angelo-1__artifactTemp.vhdr'))

#%%
epo,epo_t,mrk_class = load_multiple_artifact_data([os.path.join(baseFolder,'2020-MSc-Angelo-1__artifactTemp.vhdr'),
                                         os.path.join(baseFolder,'2020-MSc-Angelo-1__artifactTemp02.vhdr'),
                                         os.path.join(baseFolder,'2020-MSc-Angelo-1__artifactTemp03.vhdr'),
                                         os.path.join(baseFolder,'2020-MSc-Angelo-1__artifactTemp04.vhdr')])
#%%


#%%
ivals = [[160, 200], [200,240],[240,280],[280,300],[300,320],[320,340],[340,360],
         [360,380],[380, 400], [400,440],[440,480],[480, 520]]
 
#%%
    
for artifact in np.unique(mrk_class):
    print(validateLDA(params,epo[:,:,mrk_class==artifact],epo_t,np.ones(sum(mrk_class==artifact)),ivals))

#%%
    
calcAndPlotR2(epo,epo_t,mrk_class,clab,ival=[0,600])
#%%
cnt, fs, clab, mnt, mrk_pos, mrk_class = load_data(dat,dattype='artifact')
#%%
# CHeck for power line artifacts
signal_extract = extractRawChannel(cnt,clab,'Fz')
plotPSD(signal_extract, fs,frange=[2,50])

#%%
"""
Notch filter signals
"""
cnt = notchFilter(cnt)
cnt = hpFilter(cnt)
#%%
d,v = plotPrincipalComponents(cnt, mnt,clab,n=2)

#%%
# Store given information in variables. Subsequent code should only refer to these variables and not
# contain the constants.
ival= [-100, 1000]
ref_ival= [-100, 0]

# Segment continuous data into epochs:
epo, epo_t = bci.makeepochs(cnt, fs, mrk_pos, ival)


#%%
# Baseline correction:
epo = bci.baseline(epo, epo_t, ref_ival)
#%%
#Downsampling the signal
epo,epo_t = downsampleEpo(epo, epo_t)
#%
#%%


selectedChannels = ['Oz','Pz','Cz']
epo_sel = selectChannelFromEpo(epo, clab, selectedChannels)
plotMeanCurvesArtifact(epo_sel,epo_t,selectedChannels,mrk_class)