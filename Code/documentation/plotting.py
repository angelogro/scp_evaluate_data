#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:24:37 2020

@author: angelo
"""
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D 
import matplotlib.ticker as plticker
import numpy as np
from helpers import *

def format_plot(ax,fig):
    #
    ax.axes.set_aspect('equal')
    # set the x-spine
    ax.spines['left'].set_position('zero')
    
    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()
    
    # set the y-spine
    ax.spines['bottom'].set_position('zero')
    
    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()
    
    loc = plticker.MultipleLocator(base=0.5)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    
    xmin, xmax = ax.get_xlim() 
    ymin, ymax = ax.get_ylim()
    
    # get width and height of axes object to compute 
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height
    
     
    # manual arrowhead width and length
    hw = 1./30.*(ymax-ymin) 
    hl = 1./30.*(xmax-xmin)
    lw = 0.1 # axis line width
    ohg = 0.3 # arrow overhang
     
    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height
    
    ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw, 
             head_width=hw, head_length=hl, overhang = ohg, 
             length_includes_head= True, clip_on = False) 
     
    ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw, 
             head_width=yhw, head_length=yhl, overhang = ohg, 
             length_includes_head= True, clip_on = False) 
    x_tick_labels = ax.get_xticks()

    ax.set_xticklabels(list(map(lambda x:'' if x ==0 else x,x_tick_labels)))
    y_tick_labels = ax.get_yticks()

    ax.set_yticklabels(list(map(lambda x:'' if x ==0 else x,y_tick_labels)))
    
    ax.xaxis.set_label_coords(1,0.0)
    ax.yaxis.set_label_coords( 0.1,1)

def add_w_arrow(ax,fig,w,label,offset=np.array([0,0])):
    scale = 3
    warrow= ax.arrow(offset[0], offset[1], w[0], w[1], fc='k', ec='k', lw = 1, 
             head_width=0.1, head_length=0.15, overhang = 0.1, 
             length_includes_head= True, clip_on = False)
    ax.text(w[0]-0.4+offset[0],w[1]-0.5+offset[1],r'$'+label+'$', fontsize = 16)
    #ax.add_line(Line2D([0,w[0]*scale],[0,w[1]*scale],color='k',linewidth=0.5))
    return warrow

def add_orthogonal_lines(ax,fig,w,X,c='b'):
    scale = 3
    for datapoint in X:
        x,y=intersection(line((0,0),w), line(datapoint,(datapoint[0]+w[1],datapoint[1]-w[0])))
        ax.add_line(Line2D([datapoint[0],x],[datapoint[1],y],color=c,linewidth=0.5))
        
def add_mean(ax,fig,m1,m2 ,offset=0.1):
   
    ax.scatter(m1[0],m1[1],s=20,c='k',)
    ax.text(m1[0]+offset,m1[1],r'$\mu_1$',c='k',fontsize=16)
    ax.scatter(m2[0],m2[1],s=20,c='k',)  
    ax.text(m2[0]+offset,m2[1],r'$\mu_2$',c='k',fontsize=16)      
    
def add_separating_line(ax,fig,m1,m2,w,scale=5,linestyle='-'):
    mid = (m1+m2)/2
    ax.add_line(Line2D([(mid[0]-w[1]*scale),(mid[0]+w[1]*scale)],
                        [(mid[1]+w[0]*scale),(mid[1]-w[0]*scale)],
                         color='k',linewidth=1,ls=linestyle))
    