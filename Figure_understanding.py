#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import torch, torchvision, gc, imageio
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFile
from tqdm import tqdm
from collections import Counter
import scipy, pickle, random, os
import scipy.stats


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
import clean_cornets, ds2

dim = 224
img_classes = 5
img_classes2 = 5
wrd_classes = 5
batch_size = 20
num_train_items = 100
num_val_items = 10
num_workers = 0
max_epochs = 10
learning_rate = 0.001
in_wrd=49
hid_num=512
pre_epochs=50
lit_epochs=30
param = [img_classes, wrd_classes, batch_size, num_train_items, num_val_items, num_workers, max_epochs, learning_rate]

restore_path = 'save/save_lit_80_full_nomir.pth.tar'
test_path = 'mini_images/val/'
output_path = 'activations/'
layer= 'decoder'
sublayer= 'linear'
use_gpu = True


def printSparsity(act_codes):
    l,c = np.shape(act_codes)
    counts= np.zeros(l)
    for line in range(l):
          zero_indices = np.where(act_codes[line] < 0.0001)[0]
          counts = len(zero_indices)
    sparsity = 100*np.mean(counts)/float(c)
    print ('sparsity:', sparsity)

# Sparsity
def plotSparsity(mode=None, epoch=None, legend=False, show=False, density=True, nbins=100):
    print('loading full codes')
    act_codes = np.load('../save/all_words/act_codes_'+ mode + '_' + str(epoch) + '.npy')
    printSparsity(act_codes)

    print('loading cutoff codes')
    act_codes_cutoff = np.load('../save/all_words/act_codes_cutoff_'+ mode + '_' + str(epoch) + '.npy')
    printSparsity(act_codes_cutoff)

    plt.hist(np.ravel(act_codes), density=density, color='blue', alpha=0.5, label='original', bins=nbins)
    plt.hist(np.ravel(act_codes_cutoff), density=density, color='green', alpha=0.5, label='cutoff', bins=nbins)
    if legend:
        plt.legend(frameon=False)
    if show:
        plt.show()
    return 'done sparsity'

# Sparsity 2
def plotSparsity2(mode=None, epoch=79, legend=False, save=False, show=False):
    print('loading full codes')
    act_codes = np.load('../save/all_words/act_codes_'+ mode + '_' + str(epoch) + '.npy')
    printSparsity(act_codes)

    print('loading cutoff codes')
    act_codes_cutoff = np.load('../save/all_words/act_codes_cutoff_'+ mode + '_' + str(epoch) + '.npy')
    printSparsity(act_codes_cutoff)

    print('loading score for increasing cutoffs')
    #current_path = os.getcwd()
    #scores = np.load(current_path+'/reconstruction/all_scores.npy')
    scores = np.load('all_cutoff_scores_'+str(epoch)+'.npy')
    if mode == 'lit_no_bias':
        plt.plot(scores[1,0], color='black', lw=2)
        plt.plot(scores[1,1], color='gray', lw=2, ls='--')
        plt.xticks(np.linspace(0,100,11), [str(int(i)) for i in np.linspace(0,100,11)])
        plt.xlabel('Sorted responses cut-off (%)', size=13)
        plt.ylabel('Accuracy on words (%)', size=13)
        
    if mode == 'lit_bias':
        plt.plot(scores[0,0], label='ascending', color='black', lw=2)
        plt.plot(scores[0,1], label='descending', color='gray', lw=2, ls='--')
        plt.xticks(np.linspace(0,100,11), [str(int(i)) for i in np.linspace(0,100,11)])
        plt.yticks(np.linspace(0,100,6), ['','','','','',''])
        plt.xlabel('Sorted responses cut-off (%)', size=13)
        plt.ylabel('Accuracy on words (%)', size=13)
        plt.legend(frameon=False)
        
    if save:
        plt.savefig('cutoffs_'+str(epoch)+'.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return 'done sparsity'


def plotSparsity3(mode=None, epoch=79, legend=False, save=False, show=False):
    print('loading full codes')
    act_codes = np.load('../save/all_words/act_codes_'+ mode + '_' + str(epoch) + '.npy')
    printSparsity(act_codes)

    print('loading cutoff codes')
    act_codes_cutoff = np.load('../save/all_words/act_codes_cutoff_'+ mode + '_' + str(epoch) + '.npy')
    printSparsity(act_codes_cutoff)

    print('loading score for increasing cutoffs')
    #current_path = os.getcwd()
    #scores = np.load(current_path+'/reconstruction/all_scores.npy')
    mean_scores = np.load('mean_scores_cutoffs.npy')
    sem_scores = np.load('sem_scores_cutoffs.npy')
    x = np.linspace(0,100,101)
    if mode == 'lit_no_bias':
        plt.errorbar(x, mean_scores[1,0], yerr = sem_scores[1,0], color='black', lw=2)
        plt.errorbar(x, mean_scores[1,1], yerr = sem_scores[1,1], color='gray', lw=2, ls='--')
        plt.xticks(np.linspace(0,100,11), [str(int(i)) for i in np.linspace(0,100,11)])
        plt.xlabel('Responses suppressed (%)', size=13)
        plt.ylabel('Accuracy on words (%)', size=13)
        
    if mode == 'lit_bias':
        plt.errorbar(x, mean_scores[0,0], yerr = sem_scores[1,0], label = 'Lowest', color='black', lw=2)
        plt.errorbar(x, mean_scores[0,1], yerr = sem_scores[1,1], label = 'Highest', color='gray', lw=2, ls='--')
        plt.xticks(np.linspace(0,100,11), [str(int(i)) for i in np.linspace(0,100,11)])
        #plt.yticks(np.linspace(0,100,6), ['','','','','',''])
        plt.xlabel('Responses suppressed (%)', size=13)
        plt.ylabel('Accuracy on words (%)', size=13)
        plt.legend(frameon=False)
        
    if save:
        plt.savefig('cutoffs_'+str(epoch)+'.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return 'done sparsity'

# Barcodes
def plotBarcodes(mode=None, epoch=None, show=False):
    barcodes = imageio.imread('barcodes_sample2_'+ mode + '_' + str(epoch) + '.png')
    barcodes = Image.fromarray(barcodes)
    plt.imshow(barcodes, aspect='auto')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    
    if show:
        plt.show()
    return 'done barcodes'

def plotBarcodes2(mode=None, epoch=None, show=False):
    barcodes = imageio.imread('barcodes_sample2_'+ mode + '_' + str(epoch) + '.png')
    barcodes = Image.fromarray(barcodes)
    units = np.load('../save/all_words/'+mode+'_'+str(epoch)+'_words_selective.npy')
    if mode == 'lit_bias':
        units = units[4:]
    plt.imshow(barcodes, aspect='auto')
    x = range(len(units))
    plt.xticks(x, ['' for i in units], rotation=40)
    ax = plt.gca()
    ax.axes.yaxis.set_visible(False)
    if show:
        plt.show()
    return 'done barcodes'

def Barcodes(mode='lit_bias', epoch=79, ind1=907, ind2=962, string1='tenir', string2='venir', save=True, show=False):
    units = np.load('../save/all_words/'+mode+'_'+str(epoch)+'_words_selective.npy')
    if mode == 'lit_bias':
        units = units[4:]
    codes = np.load('../save/all_words/act_codes_cutoff_'+ mode + '_' + str(epoch) + '.npy')
    code1, code2 = np.expand_dims(codes[ind1],axis=0), np.expand_dims(codes[ind2],axis=0)

    plt.subplot(2,1,1)
    x = range(len(units))
    plt.imshow(code1, aspect='auto')
    plt.xticks(x, ['' for i in units], rotation=90)
    plt.yticks([])
    plt.ylabel(string1)
    #plt.colorbar(orientation = 'vertical')

    plt.subplot(2,1,2)
    plt.imshow(code2, aspect='auto')
    plt.xticks(x, [str(i) for i in units], rotation=60, size=8)
    plt.yticks([])
    plt.ylabel(string2)
    #plt.colorbar(orientation = 'vertical')
    plt.subplots_adjust(left = 0.06, bottom = 0.10, right = 0.94, top = 0.90, wspace = 0.2, hspace = 0.3)
    
    if show:
        plt.show()
    if save:
        plt.savefig('barcodes_sample2_'+mode+"_"+str(epoch)+".png", dpi=300, bbox_inches='tight')
    plt.clf()
    return 'done barcodes'

def plotBarcodes3(mode='lit_bias', epoch=79, indices=[732, 733, 735, 736], strings=['porte', 'poste', 'porter', 'poster'], legend=False, save=True, show=False):
    units = np.load('../save/all_words/'+mode+'_'+str(epoch)+'_words_selective.npy')
    if mode == 'lit_bias':
        units = units[4:]
    codes = np.load('../save/all_words/act_codes_cutoff_'+ mode + '_' + str(epoch) + '.npy'); l,c = np.shape(codes)

    ls = len(strings)
    codes_with_nan = np.nan*np.zeros((2*ls - 1, c))
    strings_with_tab = []
    rev_strings = list(reversed(strings)); rev_indices = list(reversed(indices))
    for i, (ind, s) in enumerate(zip(rev_indices, rev_strings)):
        #codes_with_nan = np.expand_dims(codes[ind1],axis=0)
        codes_with_nan[2*i] = codes[ind]
        if i != ls:
            strings_with_tab += [s, '']
        else:
            strings_with_tab += [s]
    x = range(len(units)); y = range(len(strings_with_tab))
    #codes_with_nan = np.expand_dims(codes_with_nan, axis=0)
    plt.imshow(codes_with_nan, aspect='auto')
    plt.xticks(x, [str(i) for i in units], rotation=60, size=4)
    plt.xlabel('Unit number', size=13)
    #plt.yticks(y, strings_with_tab)
    plt.yticks([2*i for i in range(ls)], rev_strings, size=13)
    #plt.setp(plt.get_yticklabels(), visible=False)
    #plt.colorbar(orientation = 'vertical')
    
    if show:
        plt.show()
    if save:
        plt.savefig('barcodes_sample2_'+mode+"_"+str(epoch)+".png", dpi=300, bbox_inches='tight')
    
    return 'done barcodes'

# Length
def plotLength(mode=None, norm=True, legend=False, show=False):
    ws_data = np.load('length/ws_length_'+mode+'_data.npy')
    nws_data = np.load('length/nws_length_'+mode+'_data.npy')
    lws,cws = np.shape(ws_data)
    print ('np.shape(ws_means)', lws, cws)
    lnws,cnws = np.shape(nws_data)
    print ('np.shape(nws_means)', lnws, cnws)
    if norm:
        for i in range(cws):
            if np.max(ws_data[:,i]) != 0:
                ws_data[:,i] = ws_data[:,i]/np.max(ws_data[:,i])
        for i in range(cnws):
            if np.max(nws_data[:,i]) != 0:
                nws_data[:,i] = nws_data[:,i]/np.max(nws_data[:,i])
        print ('np.min(ws_data)', np.min(ws_data))
        print ('np.max(ws_data)', np.max(ws_data))
        print ('np.min(nws_data)', np.min(nws_data))
        print ('np.max(nws_data)', np.max(nws_data))
        plt.ylabel('Normalized activation', size=13)
        plt.ylim(0, 1.05)
    else:
        plt.ylabel('Average activation', size=13)
    #plt.plot(np.mean(ws_data, axis=1), label='word-selective units', lw=2, marker='o')
    #plt.plot(np.mean(nws_data, axis=1), label='non word-selective units', lw=2, marker='v')
    x = [0, 1, 2, 3, 4, 5]
    y = np.mean(ws_data, axis=1)
    e = scipy.stats.sem(ws_data, axis=1)
    plt.errorbar(x, y, yerr = e, label='word-selective units', lw=2, marker='o')

    y = np.mean(nws_data, axis=1)
    e = scipy.stats.sem(nws_data, axis=1)
    plt.errorbar(x, y, yerr = e, label='non word-selective units', lw=2, marker='v')
    plt.xlabel('Word length', size=13)
    plt.xticks(x, [str(k) for k in np.array(x)+3])
    if legend:
        lg = plt.legend(frameon=False)
    if show:
        plt.show()
    return 'done length'

# Stats on entities coded for
def parse_rosetta(rosetta= None, stats='t-test', mode='lit_bias'):
    if rosetta is None:
        path_out = '../save/rosetta_'+mode+'_'+stats+'.npy'
        rosetta = np.load(path_out, allow_pickle=True)
    print ("rosetta[0]", rosetta[0])
    l, c = np.shape(rosetta)
    total = np.zeros(l)
    granula = np.zeros(l)
    pos_inv = np.zeros(l)
    all_pos_inv = []
    all_entities = []
    for i in range(l):
        # number of entities coded for by a unit
        total[i] = len(rosetta[i, 1])
        all_entities += rosetta[i, 1]
        
        # number of entities coded for in a position invariant way
        pos_inv[i] = len(rosetta[i, 2])

        # grouping all PI entities
        all_pos_inv += rosetta[i, 2]
        
        # granularity of entities coded for by a unit
        grain = [len(k)-1 for k in rosetta[i, 1] if k[-1] in ['12345678']] + [len(k) for k in rosetta[i, 1] if k[-1] not in ['12345678']]
        granula[i] = np.mean(grain)

    T = np.mean(total); T_sems = scipy.stats.sem(total)
    PI = np.mean(pos_inv); PI_sems = scipy.stats.sem(pos_inv)
    G = np.mean(granula); G_sems = scipy.stats.sem(granula)
    PI_stats = Counter(all_pos_inv)
    all_stats = Counter(all_entities)
    U = np.mean([num for _, num in PI_stats.most_common()])
    E = np.mean([num for _, num in all_stats.most_common()])
    
    print ('Average number of entities coded by a unit:', np.round(T, decimals=2))
    print ('Average number of position-invariant entities coded by a unit:', np.round(PI, decimals=2))
    print ('Average granularity of entities coded by a unit:', np.round(G, decimals=2))
    print ('Reality check: the total number of PI entities (with repeats) coded for across all units, is', len(all_pos_inv))
    print ('Average number of units coding for a given entity:', np.round(E, decimals=2))
    print ('Average number of units coding for a given position invariant entity:', np.round(U, decimals=2))
    
    means = np.array([T, PI, G])
    sems = np.array([T_sems, PI_sems, G_sems])
    return means, sems

# Entities
def plotEntities(mode=None, legend=False, show=False):
    means, sems = parse_rosetta(stats='sems', mode=mode)
    bplot = plt.bar([1, 2, 3], means, width=0.8, color='0.5', edgecolor='k', yerr=[np.zeros(3), sems], linewidth = 2, capsize=10, label='placeholder')
    plt.xticks([1, 2, 3], ["multi.", "pos. inv.", "gran."])
    plt.ylabel('# of entities', size=13)
    plt.ylim(0,16)
    if legend:
        lg = plt.legend(frameon=False)
    if show:
        plt.show()
    return 'done entities'

# Entities
def plotEntities2(mode=None, legend=False, show=False):
    means, sems = parse_rosetta(mode=mode)

    ax1 = plt.gca()
    bplot1 = ax1.bar([1, 2], means[:2], width=0.8, color='0.5', edgecolor='k', yerr=[np.zeros(2), sems[:2]], linewidth = 2, capsize=10, label='placeholder')
    plt.xticks([1, 2, 3], ["multi.", "pos. inv.", "gran."])
    ax1.set_ylim(0,16)
    if mode == 'lit_no_bias':
        ax1.set_ylabel('# of entities coded (per unit)', size=13)
    if mode == 'lit_bias':
        ax1.set_yticks([])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    #bplot2 = ax2.bar([3], means[2], width=0.8, color='0.5', edgecolor='k', yerr=[np.zeros(1), sems[2]], linewidth = 2, capsize=10, label='placeholder')
    #color = 'tab:cyan'
    color = 'blue'
    bplot2 = ax2.bar([1, 2, 3], [0, 0, means[2]], width=0.8, color=color, alpha= 0.5, yerr=[np.zeros(3), np.array([0, 0, sems[2]])], linewidth = 2, capsize=10, edgecolor='k', hatch="/")
    ax2.set_ylim(0,4)
    if mode == 'lit_bias':
        ax2.set_ylabel('Granularity', color=color)
        ax2.set_yticks([0, 1, 2, 3, 4], ['0', '1', '2', '3', '4'])
        ax2.tick_params(axis='y', labelcolor=color)
    if mode == 'lit_no_bias':
        ax2.set_yticks([])
        
    if legend:
        lg = plt.legend(frameon=False)
    if show:
        plt.show()
    return 'done entities'

"""
 --------------------------------------------
   -----------------Figure8----------------
 --------------------------------------------
"""
#Model-Data comparison for:
#A. Sparsity (unbiased model)                       B. Sparsity (biased model)
#C. Barcodes (unbiased model)                       D. Barcodes (biased model)
#E. Length (unbiased model)                         F. Length (biased model)
#G. Entities coded for (unbiased model)             H. Entities coded for (biased model)
def Figure8(show=0):
    l=4;c=2
    fig, axgen = plt.subplots(frameon=False,figsize = (9,16))
    fig.patch.set_visible(False)
    axgen.axis('off')
    ax = np.zeros(l*c).tolist()
    x,y,x_shift,y_shift = -0.10, 1.05,1.05,.173
    S = 18
    #axgen.annotate("Unbiased network",xy=(0., 0.), xycoords = "axes fraction", xytext=(.05,1.05), size = S,color='black',fontweight="bold")
    #axgen.annotate("Biased network",xy=(0., 0.), xycoords = "axes fraction", xytext=(.60,1.05), size = S,color='black',fontweight="bold")
    
    for i in range(l):
        for j in range(c):
            n = c*i + j
            print ("n",n)
            ax[n] = fig.add_subplot(l,c,n+1)
            ax[n].xaxis.grid(False)
            ax[n].yaxis.grid(False)

            if n == 0:
                #Sparsity
                ax[n].annotate("A",xy=(0., 0.), xycoords = "axes fraction", xytext=(x,y), size = S,color='black',fontweight="bold")
                plt.title('Unbiased network', size=18, y=1.1, fontweight="bold")
                #plotSparsity(mode='lit_no_bias', epoch=79, nbins=40, legend=False)
                plotSparsity3(mode='lit_no_bias')

            if n == 1:
                #Sparsity
                ax[n].annotate("B",xy=(0., 0.), xycoords = "axes fraction", xytext=(1.03,y), size = S,color='black',fontweight="bold")
                plt.title('Biased network', size=18, y=1.1, fontweight="bold")
                #plotSparsity(mode='lit_no_bias', epoch=79, nbins=40, legend=True)
                plotSparsity3(mode='lit_bias')
                ax[n].yaxis.tick_right()
                ax[n].yaxis.set_label_position('right')
                
            if n == 2:
                #Barcodes
                ax[n].annotate("C",xy=(0., 0.), xycoords = "axes fraction", xytext=(x,y), size = S,color='black',fontweight="bold")
                #plt.imshow(np.random.randn(20,20))
                #Barcodes(mode='lit_bias', epoch=79)
                plotBarcodes3(mode='lit_no_bias', epoch=79)

            if n == 3:
                #Barcodes
                ax[n].annotate("D",xy=(0., 0.), xycoords = "axes fraction", xytext=(1.03,y), size = S,color='black',fontweight="bold")
                #plt.imshow(np.random.randn(20,20))
                #Barcodes(mode='lit_bias', epoch=79)
                plotBarcodes3(mode='lit_bias', epoch=79)
                ax[n].yaxis.tick_right()
                ax[n].yaxis.set_label_position('right')

            if n == 4:
                #Length
                ax[n].annotate("E",xy=(0., 0.), xycoords = "axes fraction", xytext=(x,y), size = S,color='black',fontweight="bold")
                plotLength(mode='lit_no_bias', norm=True, legend=False, show=False)
                                           

            if n == 5:
                #Length
                ax[n].annotate("F",xy=(0., 0.), xycoords = "axes fraction", xytext=(1.03,y), size = S,color='black',fontweight="bold")
                #plt.imshow(np.random.randn(20,20))
                plotLength(mode='lit_bias', norm=True, legend=True, show=False)
                ax[n].yaxis.tick_right()
                ax[n].yaxis.set_label_position('right')
                              
            if n == 6:
                #Entities
                ax[n].annotate("G",xy=(0., 0.), xycoords = "axes fraction", xytext=(x,y), size = S,color='black',fontweight="bold")
                plotEntities2(mode='lit_no_bias', legend=False, show=False)

            if n == 7:
                #Entities
                ax[n].annotate("H",xy=(0., 0.), xycoords = "axes fraction", xytext=(1.03,y), size = S,color='black',fontweight="bold")
                plotEntities2(mode='lit_bias', legend=True, show=False)                
                
                
    if show == 1:
        #plt.tight_layout()
        #plt.subplots_adjust(left= 0.08, right= 0.91, bottom = 0.04, top = .96, wspace = 0.35, hspace = 0.30)
        
        plt.show()

    plt.subplots_adjust(bottom = 0.04, top = .96, hspace = 0.30)
    fig.savefig("Figure_understanding.jpg",dpi=300)
        
    del ax, fig
    gc.collect()
    return 'done'
