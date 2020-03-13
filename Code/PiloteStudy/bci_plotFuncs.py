#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:41:46 2020

@author: angelo
"""
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy import signal
import numpy as np
import os
import bci_minitoolbox as bci
from global_vars import *

def plotPSD(signal_,fs,frange=None):
    if frange == None:
        frange = [0,fs/2]
    f, psd = signal.welch(signal_, fs=1000,nperseg=4096)
    plt.figure()
    idx_range = np.where((f>=frange[0])&(f<=frange[1]))
    plt.plot(f[idx_range],psd[idx_range])
    plt.title(''.join(['Spectral density @fs=',str(fs),' Hz']))
    plt.xlabel('frequency / Hz')
    plt.ylabel('PSD')
    
def plotPSDfromEpoMultichannel(epo,mrk_class,clab,sel_channels,sel_mrk_class,fs,frange=None):
    epo = epo[:,:,mrk_class==sel_mrk_class]
    epo = epo.swapaxes(0,1)
    epo = epo.swapaxes(2,1)
    epo = epo.reshape(len(clab),-1)
    if frange == None:
        frange = [0,fs/2]
    
    plt.figure()
    for channel in sel_channels:
        f, psd = signal.welch(epo[np.where(clab==channel)[0][0],:], fs=fs,nperseg=4096)
        idx_range = np.where((f>=frange[0])&(f<=frange[1]))
        plt.plot(f[idx_range],psd[idx_range],label=channel)

    plt.title(''.join(['Spectral density @fs=',str(fs),' Hz @class ',artifactDict[sel_mrk_class]]))
    plt.xlabel('frequency / Hz')
    plt.ylabel('PSD')
    plt.legend()


    
def plotPSDfromEpoMulticlass(epo,mrk_class,clab,sel_channel,sel_mrk_classes,dattype,fs,frange=None):
    if frange == None:
        frange = [0,fs/2]
    plt.figure()
    
    for clss in sel_mrk_classes:
        epo_ = epo[:,:,mrk_class==clss]
        epo_ = epo_.swapaxes(0,1)
        epo_ = epo_.swapaxes(2,1)
        epo_ = epo_.reshape(len(clab),-1)
        f, psd = signal.welch(epo_[np.where(clab==sel_channel)[0][0],:], fs=1000,nperseg=4096)
        if dattype == 'artifact':
            lab = artifactDict[clss]
        else:
            if clss==0:
                lab='Target'
            else:
                lab='Non-Target'
        idx_range = np.where((f>=frange[0])&(f<=frange[1]))
        plt.plot(f[idx_range],psd[idx_range],label=lab)
        plt.title(''.join(['Spectral density @fs=',str(fs),' Hz @channel ',sel_channel]))
        plt.xlabel('frequency / Hz')
        plt.ylabel('PSD')
        plt.ylim(0,0.5)
        plt.legend()
    
def plotPrincipalComponents(cnt,mnt,clab,n=3):
    X=cnt
    Xmean= np.mean(X,axis=1)
    X = X- Xmean[:,np.newaxis]
    d,v=np.linalg.eig(np.cov(X))
    
    for i in range(n):
        plt.figure(figsize=(6,6))
        plt.title(''.join(['Component #',str(i+1),', std:',str(np.sqrt(d[i]))]))
        bci.scalpmap(mnt,v[:,i],clim='sym',cb_label='[a.u.]',clab=clab)
        
    return d,v

def plotScalpmapsOddball(epo,epo_t,clab,mrk_class,intervals,mnt):
    
    lim_min, lim_max = 100,-100
    for mrk_cl in np.unique(mrk_class):
        epo_cls = np.mean(epo[:,:,mrk_class == mrk_cl],axis=2)
        for ival in intervals:   
            idx_range = np.where((epo_t>=ival[0])&(epo_t<=ival[1]))
            epo_cls_ival = np.mean(epo_cls[idx_range,:],axis=0)
            lim_min = np.min(epo_cls_ival) if np.min(epo_cls_ival)<lim_min else lim_min
            lim_max = np.max(epo_cls_ival) if np.max(epo_cls_ival)>lim_max else lim_max
            
    for mrk_cl in np.unique(mrk_class):
       label = 'Target' if mrk_cl == 0 else 'Non-Target'
       epo_cls = np.mean(epo[:,:,mrk_class == mrk_cl],axis=2)
       print(epo_cls.shape)
       for ival in intervals:
           plt.figure(figsize=(6,6))
           idx_range = np.where((epo_t>=ival[0])&(epo_t<=ival[1]))
           print(epo_cls[idx_range,:].shape)
           epo_cls_ival = np.mean(epo_cls[idx_range,:].squeeze(),axis=0)
           print(epo_cls_ival.shape)
           plt.title(''.join([label,', Interval ',str(ival[0]),' - ',str(ival[1])]))
           bci.scalpmap(mnt,epo_cls_ival,clim=[lim_min,lim_max],cb_label='µV',clab=clab)      

            


def plotMeanCurves(epo,epo_t,clab,selectedChannels,mrk_class):
    # Sums up the data of all epochs of the same class for each channel
    epoMeanTarget = np.mean(epo[:,:,mrk_class==0],axis=2)
    nTarSamples = epo[:,:,mrk_class==0].shape[2]
    epoMeanNonTarget = np.mean(epo[:,:,mrk_class==1],axis=2)
    nNonTarSamples = epo[:,:,mrk_class==1].shape[2]
    print(epoMeanTarget.shape)
    clab = list(clab)
    # Plots the data...
    for chan in selectedChannels:
        plt.figure()
        plt.plot(epo_t,epoMeanTarget[:,clab.index(chan)],label="Target")
        plt.plot(epo_t,epoMeanNonTarget[:,clab.index(chan)],label="Non-Target")
        plt.title('Mean of epochs for '+chan+', tSamples='+str(nTarSamples)+', ntSamples='+str(nNonTarSamples))
        plt.xlabel('time [ms]'); plt.ylabel('uV');
        plt.legend()
        
def plotMeanCurvesArtifact(epo,epo_t,clab,selectedChannels,mrk_class):
    # Sums up the data of all epochs of the same class for each channel
    clab = list(clab)
    meanArtifacts = [np.mean(epo[:,:,mrk_class==artifact_id],axis=2) for artifact_id in np.unique(mrk_class)]
    epoMean = np.stack(meanArtifacts,axis=2)
    print(epoMean.shape)
    for chan in selectedChannels:
        plt.figure()
        for idx,num in enumerate(np.unique(mrk_class)):
            plt.plot(epo_t,epoMean[:,clab.index(chan),idx],label=artifactDict[num])
        plt.title('Mean of epochs for '+chan)
        plt.xlabel('time [ms]'); plt.ylabel('uV');
        plt.legend()
        
def signed_r_square(epo, y):
    '''
    Synopsis:
        epo_r = signed_r_square(epo, y)
    Arguments:
        epo:    3D array of segmented signals (time x channels x epochs), 
                see makeepochs
        y:      labels with values 0 and 1 (1 x epochs)
    Output:
        epo_r:  2D array of signed r^2 values (time x channels)
    '''
    N1,N2 = len(y[y==True]),len(y[y==False])
    prefactor = 1.0*N1*N2/(N1+N2)**2
    u1,u2 =np.mean(epo[:,:,y==0],axis=2),np.mean(epo[:,:,y==1],axis=2)
    epo_r = prefactor* (u1-u2)**2/np.var(epo,axis=2)
    return np.sign(u1-u2)*epo_r

def calcAndPlotR2(epo,epo_t,mrk_class,clab,ival=[-100,1000]):
    epoForr2 = epo[np.where((epo_t>=ival[0])&(epo_t<=ival[1]))]
    signedr2data=signed_r_square(epoForr2, mrk_class)
    
    fig=plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    plt.imshow(np.swapaxes(signedr2data,0,1))
    plt.title('Signed r^2, time vs channels')
    plt.xlabel('sample point'); plt.ylabel('channel')
    forceAspect(ax,aspect=1)
    plt.colorbar(cmap=cm.bwr)
    plt.yticks(range(31),clab)

    
    
def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
        
def plotChannelwiseCovarianceMatrix(epo,epo_t,clab,mrk_class,ref_time):
    dataT = epo[np.where(epo_t==ref_time),:,mrk_class==0]
    dataNT = epo[np.where(epo_t==ref_time),:,mrk_class==1]
    
    
    
    covT = np.cov(dataT[0,:].T)
    covNT = np.cov(dataNT[0,:].T)
    covs= (covT,covNT)
    
    titles = ('Target','Non-Target')
    
    plt.figure(figsize=(12,5))
    plt.suptitle('Covariance between channels @ '+str(ref_time)+' ms', fontsize=20)
    plt.gca().set_aspect('equal')
    n=121
    for i in range(len(covs)):
        plt.subplot(n+i)
        plt.title(titles[i])
        plt.imshow(covs[i],cmap='bwr')
        ax = plt.gca()
        ax.set_xticklabels(clab[0::5])
        ax.set_xticks(np.arange(len(clab))[0::5])
        ax.set_yticklabels(clab[0::5])
        ax.set_yticks(np.arange(len(clab))[0::5])
        plt.colorbar(shrink=.5, label='[uV]')