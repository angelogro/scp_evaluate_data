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

#%%

epo,epo_t,mrk_class = load_multiple_artifact_data([os.path.join(baseFolder,'2020-MSc-Angelo-1__oddballTemp.vhdr'),
                                                                      os.path.join(baseFolder,'2020-MSc-Angelo-1__oddballTemp02.vhdr'),
                                                                      os.path.join(baseFolder,'2020-MSc-Angelo-1__oddballTemp03.vhdr')],
                                                                     dattype='oddball')

#%%
ivals = [[160, 200], [200,240],[240,280],[280,300],[300,320],[320,340],[340,360],
         [360,380],[380, 400], [400,440],[440,480],[480, 520]]
 
#%%
params=calcAndValLDATempSpat(epo, epo_t, mrk_class, ivals,returnParams=True)

#%%
    
calcAndPlotR2(epo,epo_t,mrk_class,clab,ival=[-100,1000])

