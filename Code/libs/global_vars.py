#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 22:56:06 2020

@author: angelo
"""
import numpy as np

artifactDict={0:'Target',
              1:'Press feet',
              2:'Lift tongue',
              3:'Forced breath',
              4:'Clinch teeth',
              5:'Move eyes',
              6:'Push breath',
              7:'Wrinkle nose',
              8:'Swallow',
              9:'Relax'}

abbrDict={0:'T',
              1:'PF',
              2:'LT',
              3:'Forced breath',
              4:'CT',
              5:'Move eyes',
              6:'PB',
              7:'WN',
              8:'SW',
              9:'Relax'}

plotcolours=['powderblue','lightskyblue']

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