#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:07:42 2020

@author: angelo
"""
from calcElectrodePositions import ElectrodePositions
from scipy import signal
from wyrm import io
import bci_minitoolbox as bci
import numpy as np
import os

def load_data(dat,dattype='oddball'):
    clab = dat.axes[1]
    fs = int(dat.fs)
    erpmarkers = list(filter(lambda x: int(x[1][1:])<200,dat.markers))
    mrk_pos = np.array(erpmarkers)[:,0].astype(np.float).astype(np.uint32)
    cnt =  dat.data.T
    e = ElectrodePositions(os.path.join('/home/angelo/Master_Arbeit/Data/','elec32_pos.csv'))
    
    mnt = np.array(list(map(lambda x:e.getCoord(x),clab)))
    if dattype == 'artifact':
        mrk_class = np.array(list(map(lambda x: int(x[1:]),np.array(erpmarkers)[:,1])))
        return cnt, fs, clab, mnt, mrk_pos, mrk_class
        
    mrk_class = 1-np.array(list(map(lambda x: int(x[1:])<100,np.array(erpmarkers)[:,1])),np.int)  
    # In our data the target markers will be a '1' while as the non-target markers will be a '0'.       
    return cnt, fs, clab, mnt, mrk_pos, mrk_class

def load_multiple_data(list_vhdr_files,dattype='artifact'):   
    data_list=[]
    for vhdr_file in list_vhdr_files:
        dat = io.load_brain_vision_data(vhdr_file)
        cnt, fs, clab, mnt, mrk_pos, mrk_class = load_data(dat,dattype)
        data_list.append((cnt, fs, clab, mnt, mrk_pos, mrk_class))
    return data_list
        
    
    
def prepareData(data_list,notch50Hz=True,highPassCutOff=0.1,lowPassCutOff=100,ival= [-100, 1000],ref_ival= [-100, 0],
                downsample_factor=10,reject_channels= None,reject_voltage = 20,nRemovePCA = 0,bandpass = None):
    
    epo_list=[]
    mrk_class_list=[]
    for data in data_list:
        
        cnt, fs, clab, mnt, mrk_pos, mrk_class = data
        clab_idx = np.ones(len(clab)).astype('bool')
        if reject_channels != None:
            for reject_chanel in reject_channels:
                clab_idx = clab_idx & (clab!=reject_chanel)
            cnt = cnt[clab_idx,:]
            clab = clab[clab_idx]
        if notch50Hz:
            cnt = notchFilter(cnt)
        if highPassCutOff > 0 :
            cnt = hpFilter(cnt,fc=highPassCutOff)
        if lowPassCutOff > 0 :
            cnt = lpFilter(cnt,fc=lowPassCutOff)
        if bandpass != None:         
            b,a = signal.butter(8,[bandpass[0]/(fs/2),bandpass[1]/(fs/2)], btype = 'bandpass')
            cnt = signal.lfilter(b,a,cnt,axis=0)
        if nRemovePCA > 0:
            cnt = removePCASignal(cnt,n=nRemovePCA)
            
        if downsample_factor>0:
            cnt,fs = downSampleCnt(cnt, fs, downsample_factor)
        # Segment continuous data into epochs:
        epo, epo_t = bci.makeepochs(cnt, fs, (mrk_pos/downsample_factor).astype(int), ival)
        epo = bci.baseline(epo, epo_t, ref_ival)
        """
        if downsample_factor>1:
            epo,epo_t = downsampleEpo(epo, epo_t,downsample_factor)
        """
        epo,mrk_class = reject(epo,mrk_class,reject_voltage)
        epo_list.append(epo)
        mrk_class_list.append(mrk_class)
        
    if len(epo_list)>1:
        epo_ = np.concatenate(epo_list,axis=2)
        mrk_class_ = np.concatenate(mrk_class_list,axis=0)
        return epo_,epo_t,mrk_class_,clab,mnt
    else:
        return epo_list[0],epo_t,mrk_class,clab,mnt
    
def load_multiple_data_raw(list_vhdr_files,dattype='artifact',notch50Hz=True,highPassCutOff=0.1):
    cnt_list=[]
    mrk_class_list=[]
    for vhdr_file in list_vhdr_files:
        dat = io.load_brain_vision_data(vhdr_file)
        cnt, fs, clab, mnt, mrk_pos, mrk_class = load_data(dat,dattype)
        if notch50Hz:
            cnt = notchFilter(cnt)
        if highPassCutOff>0:
            cnt = hpFilter(cnt,highPassCutOff)
        
        cnt_list.append(cnt)
        mrk_class_list.append(mrk_class)
    
    if len(cnt_list)>1:
        cnt_ = np.concatenate(cnt_list,axis=1)
        mrk_class_ = np.concatenate(mrk_class_list,axis=0)
        return cnt_,mrk_class_,clab,mnt
    else:
        return cnt_list[0],mrk_class[0],clab,mnt

def reject(epo,mrk_class,reject_voltage):
    idxes = (np.max(epo,axis=(0,1))-np.min(epo,axis=(0,1)))>reject_voltage
    epo = epo[:,:,idxes==False]
    mrk_class = mrk_class[idxes==False]
    return epo,mrk_class

def extractRawChannel(cnt,clab,channel):
    return cnt[list(clab).index('Cz'),:].flatten('F')

def notchFilter(cnt,f=50,fs=1000,Q=10):
    b,a = signal.iirnotch(f, Q,fs=fs)
    return signal.filtfilt(b,a,cnt)

def hpFilter(cnt,fc=50,fs=1000,Q=30):
    sos = signal.butter(Q,fc, btype = 'highpass',output='sos',fs=fs)
    return signal.sosfilt(sos,cnt,axis=1)

def lpFilter(cnt,fc=50,fs=1000,Q=10):
    sos = signal.butter(Q,fc, btype = 'lowpass',output='sos',fs=fs)
    return signal.sosfilt(sos,cnt,axis=1)

def selectChannelFromEpo(epo,clab,selectedChannels):
    indices = np.flip(np.where(np.in1d(clab,selectedChannels))[0])

    return epo[:,indices,:]

def downsampleEpo(epo,epo_t,ds_factor=10):
    epo = signal.decimate(epo,ds_factor,axis=0)
    epo_t = epo_t[0::ds_factor]
    return epo,epo_t

def downSampleCnt(cnt,fs,ds_factor):
    cnt = signal.decimate(cnt,ds_factor,axis=1)
    return cnt,fs/ds_factor

def removePCASignal(X,n=1):
    Xmean = np.mean(X,axis=1)
    
    Xm = X-Xmean[:,np.newaxis]
    d,v=np.linalg.eig(np.cov(X))
    S = np.dot(v.T,Xm)
    Seeg = S[n:]
    return np.dot(v[:,n:],Seeg)

