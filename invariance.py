#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import torch, torchvision, gc
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFile
from tqdm import tqdm
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



normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.ToTensor(),
                normalize,
            ])


transform2 = torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                normalize,
            ])

"""
****************************************************************************************
quantify invariance to various transforms as correlation indices between conditions
****************************************************************************************
"""

def gen2(text = 'text', index=1, mirror=False, invert=False, fontname='arial', fontpath='../save/ttf_fonts/', savepath='', W = 500, H = 500, size=24, xshift=0, yshift=0, upper=0, show=None):
    if upper:
        text = text.upper()
    if invert:
        text = text[::-1]
    img = Image.new("RGB", (W,H), color = (255, 255, 255))
    #fnt = ImageFont.truetype('/Library/Fonts/'+fontname+'.ttf', size) #size in pixels
    fnt = ImageFont.truetype(fontpath+fontname+'.ttf', size)
    draw = ImageDraw.Draw(img)
    w, h = fnt.getsize(text)
    draw.text((xshift + (W-w)/2, yshift + (H-h)/2), text, font=fnt, fill='black')

    if mirror:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if savepath != '':
        img.save(savepath+text+str(index)+'.jpg')
    if show:
        img.save('plots/'+text+str(index)+'.jpg')
        img.show()

    return img

def generate_stim_shuffle(path_in='../save/training_wordlist.npy', num_words=100):
    #define words, sizes, fonts
    words = np.load(path_in)

    x,y = 0,0
    ys = [-100, 0, 100]
    s = 60
    w,h = 224,224
    f = 'arial'
    case = True

    data1 = np.zeros((num_words, 3, w, h))
    data2 = np.zeros((num_words, 3, w, h))
    for i in tqdm(range(num_words), miniters=20):

        y1, y2 = random.randint(-100,100), random.randint(-100,100)
        
        stim = gen2(text=words[i], index=0, fontname=f, size=s, xshift=x, yshift=y1, upper=case)
        stim = transform(stim)
        data1[i] = stim.numpy()
        del stim

        wshuf = ''.join(random.sample(words[i],len(words[i])))
        stim = gen2(text=wshuf, index=0, fontname=f, size=s, xshift=x, yshift=y2, upper=case)
        stim = transform(stim)
        data2[i] = stim.numpy()
        del stim

        gc.collect()
            
    #save to numpy file
    path_out = '../save/shuffleStims/data1.npy'
    np.save(path_out, data1)            
    path_out = '../save/shuffleStims/data2.npy'
    np.save(path_out, data2)  
    print( 'shuffled stimuli generated and saved')

def generate_stim_mirror(path_in='../save/training_wordlist.npy', num_words=100):
    #define words, sizes, fonts
    words = np.load(path_in)

    x,y = 0,0
    ys = [-100, 0, 100]
    s = 60
    w,h = 224,224
    f = 'arial'
    case = True

    data1 = np.zeros((num_words, 3, w, h))
    data2 = np.zeros((num_words, 3, w, h))
    for i in tqdm(range(num_words), miniters=20):

        y1, y2 = random.randint(-100,100), random.randint(-100,100)

        stim = gen2(text=words[i], index=0, fontname=f, size=s, xshift=x, yshift=y1, upper=case, mirror= False)
        stim = transform(stim)
        data1[i] = stim.numpy()
        del stim

        stim = gen2(text=words[i], index=0, fontname=f, size=s, xshift=x, yshift=y2, upper=case, mirror= True)
        stim = transform(stim)
        data2[i] = stim.numpy()
        del stim

        gc.collect()
            
    #save to numpy file
    path_out = '../save/mirrorStims/data1.npy'
    np.save(path_out, data1)            
    path_out = '../save/mirrorStims/data2.npy'
    np.save(path_out, data2)  
    print( 'mirror stimuli generated and saved')

def generate_stim_font(path_in='../save/training_wordlist.npy', num_words=100):
    #define words, sizes, fonts
    words = np.load(path_in)

    x,y = 0,0
    ys = [-100, 0, 100]
    s = 60
    w,h = 224,224
    #fonts = ['arial', 'times', 'calibri', 'courier', 'comicsans']
    fonts = ['arial', 'tahoma']
    case = True

    data1 = np.zeros((num_words, 3, w, h))
    data2 = np.zeros((num_words, 3, w, h))
    for i in tqdm(range(num_words), miniters=20):

        
        random.shuffle(fonts)
        f1, f2 = fonts[:2]

        y1, y2 = random.randint(-100,100), random.randint(-100,100)

        stim = gen2(text=words[i], index=0, fontname=f1, size=s, xshift=x, yshift=y1, upper=case)
        stim = transform(stim)
        data1[i] = stim.numpy()
        del stim

        stim = gen2(text=words[i], index=0, fontname=f2, size=s, xshift=x, yshift=y2, upper=case)
        stim = transform(stim)
        data2[i] = stim.numpy()
        del stim

        gc.collect()
            
    #save to numpy file
    path_out = '../save/fontStims/data1.npy'
    np.save(path_out, data1)            
    path_out = '../save/fontStims/data2.npy'
    np.save(path_out, data2)  
    print( 'font stimuli generated and saved')

def generate_stim_case(path_in='../save/training_wordlist.npy', num_words=100):
    #define words, sizes, fonts
    words = np.load(path_in)

    x,y = 0,0
    ys = [-100, 0, 100]
    s = 60
    w,h = 224,224
    f = 'arial'
    cases = [True, False]

    data1 = np.zeros((num_words, 3, w, h))
    data2 = np.zeros((num_words, 3, w, h))
    for i in tqdm(range(num_words), miniters=20):
        
        random.shuffle(cases)
        c1, c2 = cases[:2]

        y1, y2 = random.randint(-100,100), random.randint(-100,100)

        stim = gen2(text=words[i], index=0, fontname=f, size=s, xshift=x, yshift=y1, upper=c1)
        stim = transform(stim)
        data1[i] = stim.numpy()
        del stim

        stim = gen2(text=words[i], index=0, fontname=f, size=s, xshift=x, yshift=y2, upper=c2)
        stim = transform(stim)
        data2[i] = stim.numpy()
        del stim

        gc.collect()
            
    #save to numpy file
    path_out = '../save/caseStims/data1.npy'
    np.save(path_out, data1)            
    path_out = '../save/caseStims/data2.npy'
    np.save(path_out, data2)  
    print( 'case stimuli generated and saved')

def generate_stim_size(path_in='../save/training_wordlist.npy', num_words=100):
    #define words, sizes, fonts
    words = np.load(path_in)

    x,y = 0,0
    ys = [-100, 0, 100]
    sizes = [40, 50, 60, 70, 80]
    w,h = 224,224
    f = 'arial'
    case = True

    data1 = np.zeros((num_words, 3, w, h))
    data2 = np.zeros((num_words, 3, w, h))
    for i in tqdm(range(num_words), miniters=20):

        random.shuffle(sizes)
        s1, s2 = sizes[:2]

        y1, y2 = random.randint(-100,100), random.randint(-100,100)
        
        stim = gen2(text=words[i], index=0, fontname=f, size=s1, xshift=x, yshift=y1, upper=case)
        stim = transform(stim)
        data1[i] = stim.numpy()
        del stim

        stim = gen2(text=words[i], index=0, fontname=f, size=s2, xshift=x, yshift=y2, upper=case)
        stim = transform(stim)
        data2[i] = stim.numpy()
        del stim

        gc.collect()
            
    #save to numpy file
    path_out = '../save/sizeStims/data1.npy'
    np.save(path_out, data1)            
    path_out = '../save/sizeStims/data2.npy'
    np.save(path_out, data2)  
    print( 'size stimuli generated and saved')

def compute_cc(model, layer=None, transform=None):
    # retrieve data
    Data1 = np.load('../save/'+transform+"Stims/data1.npy")
    Data2 = np.load('../save/'+transform+"Stims/data2.npy")

    # torchify data
    Data1 = torch.Tensor(Data1)
    Data1 = Variable(Data1)
    Data2 = torch.Tensor(Data2)
    Data2 = Variable(Data2)

    # process data
##    h1, out1 = model(Data1)
##    h2, out2 = model(Data2)
    v1_1, v2_1, v4_1, it1, h1, out1 = model(Data1)
    v1_2, v2_2, v4_2, it2, h2, out2 = model(Data2)

    # select target units and untorchify
    if layer == 'OUTPUT':
        target1 = out1.detach().numpy()
        target2 = out2.detach().numpy()

    if layer == 'VWFA':
        target1 = h1.detach().numpy()[:,-in_wrd:]
        target2 = h2.detach().numpy()[:,-in_wrd:]

    if layer == 'DENSE':
        target1 = h1.detach().numpy()[:,:in_wrd]
        target2 = h2.detach().numpy()[:,:in_wrd]

    if layer == 'IT':
        target1 = h1.detach().numpy()
        target2 = h2.detach().numpy()

    if layer == 'V4':
        target1 = v4_1.detach().numpy()
        target2 = v4_2.detach().numpy()

    if layer == 'V2':
        target1 = v2_1.detach().numpy()
        target2 = v2_2.detach().numpy()

    if layer == 'V1':
        target1 = v1_1.detach().numpy()
        target2 = v1_2.detach().numpy()    

    # compute r scores or whatever for each unit across the conditions
    l1,c1 = np.shape(target1)
    l2,c2 = np.shape(target2)
    c = np.min([c1, c2])
##    print ('l1,c1', l1,c1)
##    print ('l2,c2', l2,c2)
##    print ('c', c)
    
    corel = []
    for i in range(c):
        corr = np.corrcoef(target1[:, i], target2[:, i])[0,1]
        if not np.isnan(corr):      #possibly important choice: we only compute correlation coefficients for non-silent units.
            corel += [corr]
            
    return np.array(corel)

def load_model(mode = 'lit_bias', epoch = 79):
    print ('epoch', epoch)
    if (epoch < 50) or (mode == 'illit'):
        model_path = '../save/networks/save_'+mode+'_z_'+str(epoch)+'_full_nomir.pth.tar'
        #load model
        print ('loading model', model_path)
        ckpt_data = torch.load(model_path, map_location='cpu')
        keys = list(ckpt_data['state_dict'].keys())
        if 'module.' in keys[0]:
            for name in keys:
                ckpt_data['state_dict'][name[7:]] = ckpt_data['state_dict'].pop(name)
        model = clean_cornets.CORnet_Z_tweak()
        model.load_state_dict(ckpt_data['state_dict'])
        print ('model loaded')
        
    else:
        if mode == 'lit_bias':
            model_path = '../save/networks/save_'+mode+'_z_'+str(epoch)+'_full_nomir.pth.tar'
            #load model
            print ('loading model', model_path)
            ckpt_data = torch.load(model_path, map_location='cpu')
            keys = list(ckpt_data['state_dict'].keys())
            if 'module.' in keys[0]:
                for name in keys:
                    ckpt_data['state_dict'][name[7:]] = ckpt_data['state_dict'].pop(name)
            model = clean_cornets.CORNet_Z_biased_words()
            model.load_state_dict(ckpt_data['state_dict'])


        if mode == 'lit_no_bias':
            model_path = '../save/networks/save_'+mode+'_z_'+str(epoch)+'_full_nomir.pth.tar'
            #load model
            print ('loading model', model_path)
            ckpt_data = torch.load(model_path, map_location='cpu')
            keys = list(ckpt_data['state_dict'].keys())
            if 'module.' in keys[0]:
                for name in keys:
                    ckpt_data['state_dict'][name[7:]] = ckpt_data['state_dict'].pop(name)
            model = clean_cornets.CORNet_Z_nonbiased_words()
            model.load_state_dict(ckpt_data['state_dict'])
  
        print ('model loaded')
    
    return model

def warpup_inv(mode = 'lit_bias', gen = True, epoch=79):
    #load model once and for all
    model = load_model(mode=mode, epoch=epoch)

    #for each layer, for each transform, compute and save
    layers = ['V1', 'V2', 'V4', 'IT', 'DENSE', 'VWFA', 'OUTPUT']
    layers.reverse()
    transforms = ['size', 'font', 'case', 'mirror', 'shuffle']
    l=len(layers);c=len(transforms)
    
    for i in range(c):
        print ('transform', transforms[i])
        if gen:
            if i == 0:
                # generate stimuli for size
                generate_stim_size()

            if i == 1:
                # generate stimuli for font
                generate_stim_font()

            if i == 2:
                # generate stimuli for case
                generate_stim_case()

            if i == 3:
                # generate stimuli for mirroring
                generate_stim_mirror()

            if i == 4:
                # generate stimuli for shuffling
                generate_stim_shuffle()
                
        for j in range(l):
            print ('layer',layers[j])
            # compute correlation data for each unit in the target layer
            data = compute_cc(model, layer = layers[j], transform = transforms[i])
            
            # save data
            path = '../save/invariance/' + mode + '_' + layers[j] + '_' + transforms[i] + '.npy'
            np.save(path, data)

    return 'done'

def warpup():
##    warpup_inv(mode = 'pre', gen = True, epoch=49)
##    warpup_inv(mode = 'illit', gen = False, epoch=79)
    warpup_inv(mode = 'lit_bias', gen = False, epoch=79)
    warpup_inv(mode = 'lit_no_bias', gen = False, epoch=79)
    FigureInvariance(mode='pre')
    FigureInvariance(mode='illit')
    FigureInvariance(mode='lit_bias')
    FigureInvariance(mode='lit_no_bias')
    return 'done'

"""
****************************************************************************************
****************************************************************************************
****************************************************************************************
"""

# overview of responses to words
def FigureInvariance(mode='pre'):

    fig, axgen = plt.subplots(frameon=False,figsize = (8,8))
    fig.patch.set_visible(False)
    axgen.axis('off')

    layers = ['V1', 'V2', 'V4', 'IT', 'DENSE', 'VWFA', 'OUTPUT']
    layers.reverse()
    transforms = ['size', 'font', 'case', 'mirror', 'shuffle']
    colors = ['#006600', '#336666', '#00CCCC', '#0033FF', '#000099', '#660099', '#990066']
    colors.reverse()
    l=len(layers);c=len(transforms)
    ax = np.zeros(l*c).tolist()
    
    x,y,x_shift,y_shift = -0.04, 1.05,1.05,.173
    S = 25

    for i in range(l):
        for j in range(c):
            n = c*i + j
            print ("n",n)
            ax[n] = fig.add_subplot(l,c,n+1)
            ax[n].xaxis.grid(False)
            ax[n].yaxis.grid(False)

            # load data
            path = '../save/invariance/' + mode + '_' + layers[i] + '_' + transforms[j] + '.npy'
            data = np.ravel(np.load(path))
##            print ('np.shape(data)', np.shape(data))
##            print ('np.min(data)', np.min(data))
##            print ('np.max(data)', np.max(data))
   
            # plot data as a histogram
            ld = len(data)
            bins = np.max([20, int(ld/1000.0)])
##            print ('bins', bins)
##            print ('description',scipy.stats.describe(data))
            weights = 100*np.ones_like(data)/float(ld)
            ax[n].hist(data, bins=20, weights=weights, color=colors[i])
            plt.xlim(-1.01, 1.01); plt.ylim(0, 50)

            # annotations
            #ax[n].annotate("a",xy=(0., 0.), xycoords = "axes fraction", xytext=(-.14,1.01), size = S,color='black',fontweight="bold")

            # possible y axis labels
            if j == 0:
                plt.ylabel(layers[i])

            # possible titles
            if i == 0:
                plt.title(transforms[j])

    if mode == 'pre':
        plt.suptitle('Distribution of invariance across units \nPre-literate network', size=15)
    if mode == 'lit_bias':
        plt.suptitle('Distribution of invariance across units \nLiterate network with bias', size=15)
    if mode == 'lit_no_bias':
        plt.suptitle('Distribution of invariance across units \nLiterate network without bias', size=15)
    if mode == 'illit':
        plt.suptitle('Distribution of invariance across units \nIlliterate network', size=15)
        
    plt.subplots_adjust(left= 0.08, right= 0.91, bottom = 0.03, top = .9, wspace = 0.35, hspace = 0.30)
    plt.show()
    
    fig.savefig('Invariance_'+mode+'.jpg',dpi=300)
    del fig
    return "done"

# overview of responses to words
# transforms = ['mirror', 'shuffle']
def FigureInvariance2(transforms = ['size', 'font', 'case']):
    c=len(transforms)
    fig, axgen = plt.subplots(frameon=False,figsize = (2*(c+1),8), constrained_layout=True)
    plt.subplots_adjust(bottom=0.2)
    fig.patch.set_visible(False)
    axgen.axis('off')

    layers = ['V1', 'V2', 'V4', 'IT', 'DENSE', 'VWFA', 'OUTPUT']
    layers.reverse()
    layers_plot = ['V1', 'V2', 'V4', 'IT', 'DENSE - VWFA', 'VWFA', 'OUTPUT']
    layers_plot.reverse()
    modes = ['pre', 'illit', 'lit_bias', 'lit_no_bias']
    modes_plot = ['pre', 'illit', 'bias', 'no bias']
    colors = ['#006600', '#336666', '#00CCCC', '#0033FF', '#000099', '#660099', '#990066']
    colors.reverse()
    l=len(layers);m=len(modes)
    ax = np.zeros(l*(c*m + 2)).tolist()
    
    x,y,x_shift,y_shift = -0.04, 1.05,1.05,.173
    S = 25

    for i in range(l):
        for j in range(c*m + 2):
            if j in [4, 9]:
                n = (c*m + 2)*i + j
                ax[n] = fig.add_subplot(l, c*m + 2, n+1)
                ax[n].axis('off')
                
            else:
                print ("j",j)
                print ("(m+1)",(m+1))
                print ("j%(m+1)",j%(m+1))
                mode, mode_plot = modes[j%(m+1)], modes_plot[j%(m+1)]
                n = (c*m + 2)*i + j
                print ("n",n)
                ax[n] = fig.add_subplot(l,c*m + 2, n+1)
                ax[n].xaxis.grid(False)
                ax[n].yaxis.grid(False)
                if j != 0:
                    plt.yticks([0, 25, 50], ['', '', ''])
                if i != l-1:
                    plt.xticks([-1, 0, 1], ['', '', ''])

                # load data (with dirty tricks, apologies)
                index = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2][j]
                path = '../save/invariance/' + mode + '_' + layers[i] + '_' + transforms[index] + '.npy'
                data = np.ravel(np.load(path))
       
                # plot data as a histogram
                ld = len(data); med = np.median(data)
                bins = np.max([20, int(ld/1000.0)])
                weights = 100*np.ones_like(data)/float(ld)
                ax[n].hist(data, bins=20, weights=weights, color=colors[i])
                plt.axvline(x=med, c='red', ls= '--', lw=0.8)
                plt.xlim(-1.01, 1.01); plt.ylim(0, 50)

                # possible y axis labels
                if j == 0:
                    plt.ylabel(layers_plot[i])

                # possible titles
                if i == 0:
                    print (mode, transforms[index])
                    plt.title(mode_plot, size=10)

                # possible subtitle
                if (i == l-1) & (j == 6) & (c == 3):
                    plt.xlabel('\t   Distribution of correlation coefficients'.expandtabs())
                if (i == l-1) & (j == 3) & (c == 2):
                    plt.xlabel('\t\t   Distribution of correlation coefficients'.expandtabs())
                    
    #suptitle = transforms[0] + '                      ' + transforms[1] + '                      ' + transforms[2]
    if c == 3:
        suptitle = transforms[0] + '\t\t\t\t' + transforms[1] + '\t\t\t\t' + transforms[2]
        plt.subplots_adjust(left= 0.08, right= 0.91, bottom = 0.06, top = .9, wspace = 0.35, hspace = 0.30)
    if c == 2:
        suptitle = transforms[0] + '\t\t\t\t' + transforms[1]
        plt.subplots_adjust(left= 0.1, right= 0.97, bottom = 0.06, top = .9, wspace = 0.35, hspace = 0.30)
    plt.suptitle(suptitle.expandtabs(), size=15, fontweight='bold')        

    plt.show()

    joined = '_'.join(transforms)
    fig.savefig('Invariance_'+joined+'.jpg',dpi=300)
    del fig
    return "done"

# overview of responses to words
# transforms = ['mirror', 'shuffle']
def FigureInvariance3(transforms = ['size', 'font', 'case']):
    c=len(transforms)
    fig, axgen = plt.subplots(frameon=False,figsize = (4*(c+1),4), constrained_layout=True)
    plt.subplots_adjust(bottom=0.28, wspace=0.3)
    fig.patch.set_visible(False)
    axgen.axis('off')

    layers = ['V1', 'V2', 'V4', 'IT', 'DENSE', 'VWFA', 'OUTPUT']
    layers_plot = ['V1', 'V2', 'V4', 'IT', 'DENSE-VWFA', 'VWFA', 'OUTPUT']
    modes = ['pre', 'illit', 'lit_bias', 'lit_no_bias']
    modes_plot = ['pre', 'illit', 'bias', 'no bias']
    colors = ['#006600', '#336666', '#00CCCC', '#0033FF', '#000099', '#660099', '#990066']
    colors.reverse()
    l=len(layers);m=len(modes)
    ax = np.zeros(m).tolist()
    
    x,y,x_shift,y_shift = -0.04, 1.05,1.05,.173
    S = 25

    for i, transform in enumerate(transforms):
        print ("i+1", i+1)

        ax[i] = fig.add_subplot(1, c, i+1)
        if i == 0:
            plt.ylabel('Invariance index \n (Median of distribution)')
        if i != 0:
            plt.yticks()

        plt.ylim(-0.05,1.05)
        plt.xticks(list(range(l)), layers_plot, rotation = 60, ha='right', size=10)
        #plt.xlabel('model units')
        plt.title(transform, size=15)
        
        for j, mode in enumerate(modes):
            meds = []
            for k, layer in enumerate(layers):
            
                # load data
                path = '../save/invariance/' + mode + '_' + layer + '_' + transform + '.npy'
                data = np.ravel(np.load(path))
                med = np.median(data)
                meds += [med]

            # plot data as a histogram
            if i != c - 1:
                ax[i].plot(meds)

            if i == c - 1:
                ax[i].plot(meds, label=modes_plot[j])
                
    plt.legend()
    plt.show()

    joined = '_'.join(transforms)
    fig.savefig('Median Invariance_'+joined+'.jpg',dpi=300)
    del fig
    return "done"

def round_up_to_odd(f):
    f = int(np.ceil(f))
    return f + 1 if f % 2 == 0 else f

# overview of responses to word transforms
def FigureInvariance4(transforms = ['size', 'font', 'case']):
    import seaborn as sns
    c=len(transforms)
    fig, axgen = plt.subplots(frameon=False,figsize = (2*(c+1),8), constrained_layout=True)
    plt.subplots_adjust(bottom=0.10, top=0.95, hspace=0.4)
    fig.patch.set_visible(False)
    axgen.axis('off')

    layers = ['V1', 'V2', 'V4', 'IT', 'DENSE', 'VWFA', 'OUTPUT']
    layers.reverse()
    layers_plot = ['V1', 'V2', 'V4', 'IT', 'DENSE-', 'DENSE+', 'OUTPUT']
    layers_plot.reverse()
    modes = ['pre', 'illit', 'lit_no_bias', 'lit_bias']
    modes_plot = ['pre', 'illit', 'no bias', 'bias']
    colors = ['#006600', '#336666', '#00CCCC', '#0033FF', '#000099', '#660099', '#990066']
    colors.reverse()
    four_colors = ['blue', 'orange', 'green', 'red']
    l=len(layers);m=len(modes)
    ax = np.zeros((l+1)*c).tolist()
    
    x,y,x_shift,y_shift = -0.04, 1.05,1.05,.173
    S = 25

    #present distribution curves for each layer (line) and transform (column)
    for i, layer in enumerate(layers):

        for j, transform in enumerate(transforms):

            n = i*c + j
            ax[n] = fig.add_subplot(l+2, c, n+1)
            
            ax[n].xaxis.grid(False)
            ax[n].yaxis.grid(False)
            if j != 0:
                plt.yticks([0, 0.5, 1], ['', '', ''])
            if i != l-1:
                plt.xticks([-1, 0, 1], ['', '', ''])
            

            if i == j == 0:
                ax[n].annotate("A",xy=(0., 0.), xycoords = "axes fraction", xytext=(x-0.4,y+0.05), size = 20, color='black', fontweight="bold")
            
            # possible y axis labels
            if j == 0:
                plt.ylabel(layers_plot[i]+'\ndensity')

            # possible titles
            if i == 0:
                plt.title(transforms[j], size=15)

            # possible x axis labels
            if i == l-1:
                plt.xlabel('Correlation coefficient', labelpad=-2)

            for k, mode in enumerate(modes):
                # load data
                path = '../save/invariance/' + mode + '_' + layer + '_' + transform + '.npy'
                data = np.ravel(np.load(path))
                #print ('\nnp.mean(data)', np.mean(data))
                #print ('\nnp.median(data)', np.median(data))
                if (mode in ['pre', 'illit']) and (layer == 'OUTPUT'):
                    print ('\n')
                    print (mode, layer, transform)
                    t_data, p_data = scipy.stats.ttest_1samp(data, 0)
                    print ("1-sampled t-test of distribution against mean zero hypothesis:")
                    print ("mean, t,p", np.round(np.mean(data), decimals=3), np.round(t_data, decimals=3), np.round(p_data, decimals=3))


                # plot data as a histogram
                ld = len(data); med = np.median(data)
                #bins = np.max([21, int(ld/1000.0)])
                bins = 25
                ext = np.max([np.abs(np.min(data)), np.abs(np.max(data))])
                weights = 100*np.ones_like(data)/float(ld)
                #ax[n].hist(data, bins=20, weights=weights, color=colors[i])
                data, x_hist = np.histogram(data, bins=bins, range=(-1, 1))
                #data, x_hist = np.histogram(data, bins=31, range=(-1, 1))
                #data, x_hist = np.histogram(data, bins=bins, range=(-ext, ext))
                #print ('bin_edges', x_hist)
                #x_bins = np.linspace(-1, 1, len(data))
                #print ('bin_edges', bin_edges)
                #print ('data', data)
                #ax[n] = sns.kdeplot(np.array(data), bw=0.1, color=four_colors[k])
                #transforming the axes a little.
                #x_hist = bin_edges[:-1]; l_hist = 2*len(x_hist)/(np.max(x_hist) - np.min(x_hist))
                #x_hist = np.linspace(-1, 1, l_hist); min_hist = ; max_hist = 
                #y_hist = np.zeros(l_hist); y_hist[min_hist, max_hist] = data;
                #x_hist = bin_edges[:-1]
                y_hist = np.concatenate((data, np.zeros(1)))
                y_hist = y_hist/np.sum(y_hist)
                ax[n].plot(x_hist, y_hist, color=four_colors[k], alpha = 0.5)
                plt.xlim(-1.01, 1.01); plt.ylim(0, 1)

##                # possible subtitle
##                if (i == l-1) & (j == 6) & (c == 3):
##                    plt.xlabel('\t   Distribution of correlation coefficients'.expandtabs())
##                if (i == l-1) & (j == 3) & (c == 2):
##                    plt.xlabel('\t\t   Distribution of correlation coefficients'.expandtabs())

                
    #present invariance indices, across layers
    layers.reverse()
    layers_plot.reverse()
    
    for i, transform in enumerate(transforms):
        print ("i+1", i+1)

        ax[n+i+1] = fig.add_subplot(l+1, c, n+i+2)
        if i == 0:
            #plt.ylabel('Invariance index \n (Median of distribution)')
            plt.ylabel('Invariance index')
        if i != 0:
            plt.yticks()

        plt.ylim(-0.05,1.05)
        plt.xticks(list(range(l)), layers_plot, rotation = 60, ha='right', size=10)
        #plt.xlabel('model units')
        #plt.title(transform, size=15)
        
        for j, mode in enumerate(modes):
            if i == j == 0:
                ax[n+i+1].annotate("B",xy=(0., 0.), xycoords = "axes fraction", xytext=(x-0.4,y+0.05), size = 20, color='black', fontweight="bold")
                
            meds = []
            for k, layer in enumerate(layers):
            
                # load data
                path = '../save/invariance/' + mode + '_' + layer + '_' + transform + '.npy'
                data = np.ravel(np.load(path))
                med = np.median(data)
                meds += [med]

            # plot data as a histogram
            if i != 2:
                ax[n+i+1].plot(meds)

            if i == 2:
                ax[n+i+1].plot(meds, label=modes_plot[j])
                
    plt.legend(ncol=2, frameon=False, loc=(-1.2, 1.0))
    plt.show()

    joined = '_'.join(transforms)
    fig.savefig('Invariance4_'+joined+'.jpg',dpi=300)
    del fig
    return "done"
