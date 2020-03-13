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

import numpy as np
import os


baseFolder = '/home/angelo/Master_Arbeit/Data/Pilot_Studie/Temp_20_01_21/'
dataFolder = '/home/angelo/Master_Arbeit/Data/'

factor_downsample = 10
#%%

dat = io.load_brain_vision_data(os.path.join(baseFolder,'2020-MSc-Angelo-1__oddballTemp03.vhdr'))
#%%

cnt, fs, clab, mnt, mrk_pos, mrk_class = load_data(dat)
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
X = cnt
Xmean = np.mean(cnt,axis=1)

Xm = X-Xmean[:,np.newaxis]

S = np.dot(v.T,Xm)
Seeg = S[2:]
cnt = np.dot(vecs[:,2:],Seeg)

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


selectedChannels = ['P7','FC6']
epo_sel = selectChannelFromEpo(epo, clab, selectedChannels)
plotMeanCurves(epo_sel,epo_t,selectedChannels,mrk_class)
ival = [[160, 200], [200, 220], [300, 320], [330, 370], [380, 430], [480, 520]]
#%%
    
calcAndValLDATemp(epo_sel,epo_t, mrk_class, ival)    
#%%
calcAndValLDATempSpat(epo, epo_t, mrk_class, ival)

#%%
    
calcAndPlotR2(epo,epo_t,mrk_class,clab,ival=[0,600])