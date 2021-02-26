import torch, torchvision, imageio, copy, gc
from torch.autograd import Variable
import numpy as np
from PIL import Image
from tqdm import tqdm
import scipy, pickle, random, os
import scipy.stats
from scipy import ndimage
from scipy.stats import sem

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib import gridspec


import sys
sys.path.append('../')
import clean_cornets   #custom networks based on the CORnet family from di carlo lab

normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.ToTensor(),
                normalize,
            ])

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def Read_process_and_store(model, proj=None, cat='tools', image_path = '../stimuli/', epoch=None, mode=None, clip=False):

    #load images
    images = [item for item in os.listdir(image_path+cat)]
    try:
        images.remove('.DS_Store')
    except ValueError:
        pass
    
    l = len(images)

    #go through all images in folder and store
    data = np.zeros((l, 3, 224, 224))
    for i in range(l):
        path = image_path+cat+'/'+images[i]
        f = open(path, 'rb')
        img = Image.open(f)
##        print (images[i])
        if 'L' in img.getbands():
            img = img.convert('RGB')
        img = transform(img)
        arr = img.numpy()
        if arr.shape[0] == 1:
            arr = np.reshape(3*[arr], (3, arr.shape[1], arr.shape[2]))
        data[i] = arr[:, :224, :224]
        f.close()
        

    #torchify data
    data = torch.Tensor(data)
    data = Variable(data)

    #process images by convnet
    v1, v2, v4, it, h, out = model(data, clip=clip)

    #turn into dim 512 codes
    """
    out:    torch.Size([1, 1000])
    h:      torch.Size([1, 512])
    it:     torch.Size([1, 512, 8, 8])
    v4:     torch.Size([1, 256, 15, 15])
    v2:     torch.Size([1, 128, 29, 29])
    v1:     torch.Size([1, 64, 57, 57])
    """
    v1 = v1.detach().numpy()#; v1 = np.dot(v1,proj)
    v2 = v2.detach().numpy()#; v2 = np.dot(v2,proj)
    v4 = v4.detach().numpy()#; v4 = np.dot(v4,proj)
    it = it.detach().numpy()#; it = np.dot(it,proj)
    h = h.detach().numpy()
    vwfa = h[:,-49:]
    h_excluding_vwfa = h[:,:-49]

    #save codes
    np.save('../save/'+ cat + mode+ '_' +'v1_codes_'+ str(epoch) +'.npy', v1)
    np.save('../save/'+ cat + mode+ '_' +'v2_codes_'+ str(epoch) +'.npy', v2)
    np.save('../save/'+ cat + mode+ '_' +'v4_codes_'+ str(epoch) +'.npy', v4)
    np.save('../save/'+ cat + mode+ '_' +'it_codes_'+ str(epoch) +'.npy', it)
    np.save('../save/'+ cat + mode+ '_' +'h_codes_'+ str(epoch) +'.npy', h)
    np.save('../save/'+ cat + mode+ '_' +'h_excluding_vwfa_codes_'+ str(epoch) +'.npy', h_excluding_vwfa)
    np.save('../save/'+ cat + mode+ '_' +'vwfa_codes_'+ str(epoch) +'.npy', vwfa)

    return 'done'


def warp_process_store(epoch=79, mode='lit_bias', clip=False):
##    proj = np.load('save/random_matrix_25088_512.npy') 
##    for epoch in tqdm(range(81)):
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

    #paths = ['false_fonts/', 'infreq_letters/', 'freq_letters/', 'freq_bigrams/', 'freq_quadrigrams/', 'words/']
    paths = ['faces/', 'bodies/', 'houses/', 'tools/', 'false_fonts/', 'infreq_letters/', 'freq_letters/', 'freq_bigrams/', 'freq_quadrigrams/', 'words/'] 
    for cat in tqdm(paths):
        Read_process_and_store(model, cat=cat, image_path = '../stimuli/', epoch=epoch, mode=mode, clip=clip)

    gc.collect()
     
    return 'done'


def analyze_units(target_cat = 'words', baseline_cats = ['faces', 'bodies', 'houses', 'tools'], mode = 'lit_bias', epoch = 79, crit=3, verbose=False, show=False):

    dim = 512
    n_ex = 100
    n_cat = 10
    categories = ['faces', 'bodies', 'houses', 'tools', 'false_fonts', 'infreq_letters', 'freq_letters', 'freq_bigrams', 'freq_quadrigrams', 'words']
    readable = ['faces', 'bodies', 'houses', 'tools', 'fonts', 'inf.let.', 'fre.let.', 'bigrams', 'quad.', 'words']
    colors = ['red', 'violet', 'yellow', 'blue', 'aquamarine', 'mediumaquamarine', 'lightgreen', 'lime', 'forestgreen', 'darkgreen']

    # retrieve indices of target and bottomline
    target_index = categories.index(target_cat)
    rest_indices = []
    for cat in baseline_cats:
        rest_indices += [categories.index(cat)]
    print ('target_index', target_index)
    print ('rest_indices', rest_indices)
    
    # retrieve patterns in dense layer for each category
    codes = np.zeros((n_cat, n_ex, dim))
    for i in range(n_cat):
        print (categories[i]) 
        code = np.load('../save/'+ categories[i]+ '/' + mode+ '_h_codes_'+str(epoch)+'.npy')[:n_ex]
        print ('np.shape(code)', np.shape(code))
        codes[i] = code

    # scan responses from each unit to see if it is target-category selective
    target_selective = []
    for unit in range(dim):
        codes[:,:,unit] = codes[:,:,unit]/np.max(codes[:,:,unit])   #normalize by the maximum unit response over all categories

        # unit i is target category selective if mean over target category is superior to mean over all other catrgories + crit*stdev
        mean_target = np.mean(codes[target_index,:,unit])
        mean_rest = np.mean(codes[rest_indices,:,unit])
        stdev_rest = np.var(codes[rest_indices,:,unit])**0.5

        if mean_target >= mean_rest + crit*stdev_rest:
            if verbose:
                print ('unit', unit, 'is ',target_cat[:-1],'selective')
                print ('mean_target', mean_target)
                print ('mean_rest', mean_rest)
                print ('stdev_rest', stdev_rest)
            target_selective += [unit]

    print (len(target_selective), target_cat, '-selective units have been found.')
    plot_select_dense_unit_responses(units = target_selective, mode = mode, epoch = epoch, save=True, show=show)

    cat_selective = Initial_Tuning(word_units = target_selective, mode = mode, epoch = epoch, crit=2)
    word_grabs2(cat_selective, mode = mode, show=show)
    np.save('../save/all_words/'+mode+'_'+str(epoch)+'_'+target_cat+'_selective.npy', target_selective)
    return target_selective, cat_selective

def warpup(gen=False, show=False):
    if gen:
        warp_process_store(epoch=49, mode='pre', clip=False)
        warp_process_store(epoch=79, mode='illit', clip=False)
        warp_process_store(epoch=79, mode='lit_no_bias', clip=False)
        warp_process_store(epoch=79, mode='lit_bias', clip=False)
    analyze_units(mode='illit', epoch=79, show=show)
    analyze_units(mode='lit_no_bias', epoch=79, show=show)
    analyze_units(mode='lit_bias', epoch=79, show=show)
    return 'done'

def Initial_Tuning(word_units = [], mode = 'bias', epoch = 79, crit=3, verbose=False):
    ##   for each word unit, compute its selectivity before training on words.
    dim = 512
    n_ex = 100
    n_cat = 10
    categories = ['faces', 'bodies', 'houses', 'tools', 'false_fonts', 'infreq_letters', 'freq_letters', 'freq_bigrams', 'freq_quadrigrams', 'words']
    readable = ['faces', 'bodies', 'houses', 'tools', 'fonts', 'inf.let.', 'fre.let.', 'bigrams', 'quad.', 'words']
    colors = ['red', 'violet', 'yellow', 'blue', 'aquamarine', 'mediumaquamarine', 'lightgreen', 'lime', 'forestgreen', 'darkgreen']

    # retrieve patterns in dense layer for each category at the last epoch before words (49)
    codes = np.zeros((n_cat, n_ex, dim))
    for i in range(n_cat):
        print (categories[i]) 
        code = np.load('../save/'+ categories[i]+ '/'+mode+'_h_codes_'+str(epoch)+'.npy')[:n_ex]
        print ('np.shape(code)', np.shape(code))
        codes[i] = code

    # compute selectivity for the predetermined word-selective units
    cat_selective = np.zeros(n_cat+1)
    for unit in word_units:
        #normalize by the maximum unit response over all categories
        codes[:,:,unit] = codes[:,:,unit]/np.max(codes[:,:,unit])   

        # find the maximum caegorical response
        mean_cats = np.mean(codes[:,:,unit], axis=1)    
        max_index = np.argmax(mean_cats)
        rest_indices = list(set(range(n_cat)) - set([max_index]))
##        print ('max_index', max_index)
##        print ('rest_indices', rest_indices)
        
        # unit i is then target category selective if mean over target category is superior to mean over all other categories + crit*stdev
        mean_max = np.mean(codes[max_index,:,unit])
        mean_rest = np.mean(codes[rest_indices,:,unit])
        stdev_rest = np.var(codes[rest_indices,:,unit])**0.5

        if mean_max >= mean_rest + crit*stdev_rest:
            if verbose:
                print ('unit', unit, 'is ',categories[max_index],'selective')
                print ('mean_target', mean_target)
                print ('mean_rest', mean_rest)
                print ('stdev_rest', stdev_rest)
            cat_selective[max_index] += 1
        else:
            cat_selective[-1] += 1
    
    # plot
    return cat_selective

"""
PLOTS
"""
def plot_select_dense_unit_responses(units = [], mode = 'lit_no_bias', epoch = 79, save=True, show=True):
    if units == []:
        units = np.load('../save/all_words/'+mode+'_'+str(epoch)+'_words_selective.npy')
    dim = 512
    n_ex = 100
    n_cat = 10
    if mode == 'pre':
        suptext = 'Profile of word-selective units \n (n ='+str(len(units))+' units, pre-literate model)'
        epoch = 49
    if mode == 'lit_no_bias':
        suptext = 'Profile of word-selective units \n (n ='+str(len(units))+' units, literate model without bias)'
    if mode == 'lit_bias':
        suptext = 'Profile of word-selective units \n (n ='+str(len(units))+' units, literate model with bias)'
    if mode == 'illit':
        suptext = 'Profile of word-selective units \n (n ='+str(len(units))+' units, illiterate model)'
    categories = ['faces', 'bodies', 'houses', 'tools', 'false_fonts', 'infreq_letters', 'freq_letters', 'freq_bigrams', 'freq_quadrigrams', 'words']
    readable = ['faces', 'bodies', 'houses', 'tools', 'fonts', 'inf.let.', 'fre.let.', 'bigrams', 'quad.', 'words']
    legendary = ['faces', 'bodies', 'houses', 'tools', 'false fonts', 'letters \n(low F)', 'letters \n(high F)', 'bigrams', 'quadrigrams', 'words']
    colors = ['red', 'violet', 'yellow', 'blue', 'aquamarine', 'mediumaquamarine', 'lightgreen', 'lime', 'forestgreen', 'darkgreen']
    
    # retrieve patterns in dense layer for each category
    codes = np.zeros((n_cat, n_ex, dim))
    for i in range(n_cat):
        print (categories[i]) 
        code = np.load('../save/'+ categories[i]+ '/' + mode+ '_h_codes_'+str(epoch)+'.npy')[:n_ex]
        print ('np.shape(code)', np.shape(code))
        codes[i] = code
        
    # boxplots on average over units
    print ('plotting average over units')

    # create figure
    f = plt.figure(figsize=(8,6))
    #subplots_adjust(left = 0.06, bottom = 0.10, right = 0.94, top = 0.90, wspace = 0.1, hspace = 0.2)
           
    # plot
    print ('codes.shape', codes.shape)
    codes = np.mean(codes[:,:,units], axis=2)                       #average over all selected units
    codes = codes/np.max(codes)    #normalize by the maximum mean response over all categories
    data = [codes[0],
            codes[1],
            codes[2],
            codes[3],
            codes[4],
            codes[5],
            codes[6],
            codes[7],
            codes[8],
            codes[9]]

    bplot = plt.boxplot(data, showfliers=False, patch_artist=True)
        
    plt.title(suptext, size=15)
    plt.ylim(0,1)
    
    # fill with colors
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    plt.xticks(range(1,11), readable, rotation = 60, ha='right', size=10)
    plt.ylabel('normalized average activation', size=10)
    
    # save figure
    if save:
        plt.savefig('average_profile_'+mode+'_'+str(epoch)+'.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    return 'done'


def plot_four_select_dense_unit_responses(save=True, show=True):

    fig = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(hspace=0.3)
    categories = ['faces', 'bodies', 'houses', 'tools', 'false_fonts', 'infreq_letters', 'freq_letters', 'freq_bigrams', 'freq_quadrigrams', 'words']
    readable = ['faces', 'bodies', 'houses', 'tools', 'fonts', 'inf.let.', 'fre.let.', 'bigrams', 'quad.', 'words']
    legendary = ['faces', 'bodies', 'houses', 'tools', 'false fonts', 'letters \n(low F)', 'letters \n(high F)', 'bigrams', 'quadrigrams', 'words']
    colors = ['red', 'violet', 'yellow', 'blue', 'aquamarine', 'mediumaquamarine', 'lightgreen', 'lime', 'forestgreen', 'darkgreen']
    modes = ['pre', 'illit', 'lit_no_bias', 'lit_bias']
    epochs = [49, 79, 79, 79]
    ax = np.zeros(4).tolist()
    for n, (mode, epoch) in enumerate(zip(modes, epochs)):
        units = np.load('../save/all_words/'+mode+'_'+str(epoch)+'_words_selective.npy')
        dim = 512
        n_ex = 100
        n_cat = 10
        if mode == 'pre':
            suptext = 'Pre-literate ('+str(len(units))+' units)'
        if mode == 'lit_no_bias':
            suptext = 'Unbiased literate ('+str(len(units))+' units)'
        if mode == 'lit_bias':
            suptext = 'Biased literate ('+str(len(units))+' units)'
        if mode == 'illit':
            suptext = 'Illiterate ('+str(len(units))+' units)'
    
        # retrieve patterns in dense layer for each category
        codes = np.zeros((n_cat, n_ex, dim))
        for i in range(n_cat):
            print (categories[i]) 
            code = np.load('../save/'+ categories[i]+ '/' + mode+ '_h_codes_'+str(epoch)+'.npy')[:n_ex]
            print ('np.shape(code)', np.shape(code))
            codes[i] = code
            
        # boxplots on average over units
        print ('plotting average over units')
               
        # plot
        print ('codes.shape', codes.shape)
        codes = np.mean(codes[:,:,units], axis=2)                       #average over all selected units
        codes = codes/np.max(codes)    #normalize by the maximum mean response over all categories
        data = [codes[0],
                codes[1],
                codes[2],
                codes[3],
                codes[4],
                codes[5],
                codes[6],
                codes[7],
                codes[8],
                codes[9]]

        ax[n] = fig.add_subplot(2, 2, n+1)
        bplot = ax[n].boxplot(data, showfliers=False, patch_artist=True)
            
        plt.title(suptext, size=13)
        ax[n].annotate('ABCD'[n], xy=(0, 0), xytext=(0.5, 0.5), size=20, weight='extra bold')
        #plt.annotate('ABCD'[n], xy=(0, 0), xytext=(0.5, 0.5), size=20, weight='extra bold')
        #plt.ylim(0,1)
        
        # fill with colors
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        # maybe put axes labels
        if n < 2:
            plt.xticks(range(1,11), 10*[''])
        if n >= 2:
            plt.xticks(range(1,11), readable, rotation = 60, ha='right', size=10)
        if n in [0,2]:
            plt.ylabel('normalized average activation', size=12)

    plt.suptitle('Average response profile of word-selective units', size=15)
    # save figure
    if save:
        #plt.savefig('average_profile_all_modes.png', dpi=300, bbox_inches='tight')
        plt.savefig('average_profile_all_modes.png')
    if show:
        plt.show()

    return 'done'

def word_grabs2(means, mode='no_bias', show=True, save=True, verbose=True):

    if mode == 'pre':
        suptext = 'Initial tuning of word-selective units \n (n ='+str(int(np.sum(means)))+' units, pre-literate model)'
    if mode == 'lit_no_bias':
        suptext = 'Initial tuning of word-selective units \n (n ='+str(int(np.sum(means)))+' units, literate model without bias)'
    if mode == 'lit_bias':
        suptext = 'Initial tuning of word-selective units \n (n ='+str(int(np.sum(means)))+' units, literate model with bias)'
    if mode == 'illit':
        suptext = 'Initial tuning of word-selective units \n (n ='+str(int(np.sum(means)))+' units, illiterate model)'

    f = plt.figure(figsize=(8,6))
    plt.subplots_adjust(bottom = 0.2)
    
    #Plot as a histogram with just means
    categories = ['faces', 'bodies', 'houses', 'tools', 'false_fonts', 'infreq_letters', 'freq_letters', 'freq_bigrams', 'freq_quadrigrams', 'words', 'uncommited']
    readable = ['faces', 'bodies', 'houses', 'tools', 'false fonts', 'frequent letters', 'infrequent letters', 'bigrams', 'quadrigrams', 'words', 'labile']
    legendary = ['faces', 'bodies', 'houses', 'tools', 'false fonts', 'letters \n(low F)', 'letters \n(high F)', 'bigrams', 'quadrigrams', 'words', 'labile']
    colors = ['red', 'violet', 'yellow', 'blue', 'aquamarine', 'mediumaquamarine', 'lightgreen', 'lime', 'forestgreen', 'darkgreen', 'gray']

    l = len(categories)
    print ('means',means)
    x = np.array(range(l))
    bars = plt.bar(x+0.5, means, width=0.5)
    for i in range(l):
        bars[i].set_facecolor(colors[i])
    #plt.title('Initial tuning of word-selective units ('+str(num_exp)+' maps)',size=20)
    plt.title(suptext,size=15)
    plt.xticks(x+0.5, readable, rotation = 60, ha='right', size=10)
    plt.ylabel('number of units',size=15)
    if save:
        plt.savefig('initial_tuning_'+mode+'.png', dpi=300)
    if show:
        plt.show()

    return 'done'


def plot_all_dense_unit_responses(mode = 'lit_no_bias', epoch = 79, save=True, show=True):

    dim = 512
    n_ex = 100
    n_cat = 10
    if mode == 'pre':
        suptext = 'Pre-literate model'
    if mode == 'lit_no_bias':
        suptext = 'Literate model without bias'
    if mode == 'lit_bias':
        suptext = 'Literate model with bias'
    if mode == 'illit':
        suptext = 'Illiterate model'
    categories = ['faces', 'bodies', 'houses', 'tools', 'false_fonts', 'infreq_letters', 'freq_letters', 'freq_bigrams', 'freq_quadrigrams', 'words']
    readable = ['faces', 'bodies', 'houses', 'tools', 'fonts', 'inf.let.', 'fre.let.', 'bigrams', 'quad.', 'words']
    legendary = ['faces', 'bodies', 'houses', 'tools', 'false fonts', 'letters \n(low F)', 'letters \n(high F)', 'bigrams', 'quadrigrams', 'words']
    colors = ['red', 'violet', 'yellow', 'blue', 'aquamarine', 'mediumaquamarine', 'lightgreen', 'lime', 'forestgreen', 'darkgreen']
    
    # retrieve patterns in dense layer for each category
    codes = np.zeros((n_cat, n_ex, dim))
    for i in range(n_cat):
        print (categories[i]) 
        code = np.load('../save/'+ categories[i]+ '/' + mode+ '_h_codes_'+str(epoch)+'.npy')[:n_ex]
        print ('np.shape(code)', np.shape(code))
        codes[i] = code
        
    # boxplots in 8 figures of 64 windows each
    # elegant ? no. Readable ? yes.
    ranges = [range(64), range(64,128), range(128,192), range(192, 256), range(256, 320), range(320,384), range(384,448), range(448, 512)]
    strings = ['0-64', '64-128', '128-192', '192-256', '256-320', '320-384', '384-448', '448-512']
##    ranges = [range(448, 512)]
##    strings = ['448-512']
    cf = 0 
    for units in ranges:
        print ('plotting units ', strings[cf])

        # create figure
        f = plt.figure(figsize=(14, 8))
        plt.suptitle(suptext+', '+'epoch' +str(epoch), size=17)
        
        gs0 = gridspec.GridSpec(2, 1, figure=f, height_ratios=[1, 20], left = 0.06, bottom = 0.10, right = 0.94, top = 0.90, wspace = 0.1, hspace = 0.01)
        
        # show custom legend
        gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0])
        ax = f.add_subplot(gs00[0,0])
        plt.axis('off')
        for i in range(n_cat):
##                xshift, yshift = 0.03, 1.1
##                xtshift, ytshift = 0.022, 0.01
                xshift, yshift = 0.065, 1.
                xtshift, ytshift = 0.0565, 0.1
                ratio = i/12.0
                ratiot = i/12.0
                # Create a Rectangle patch
                rect = patches.Rectangle((xshift + ratio, yshift), 0.01, 1, linewidth=15, edgecolor=colors[i], facecolor=colors[i])

                # Add the patch to the Axes
                ax.add_patch(rect)

                # Annotate
                ax.annotate(legendary[i], (0, 0), (xtshift + ratiot, yshift + ytshift))

        if mode == 'bias':
                ratio = (i+1)/12.0
                ratiot = (i+1)/12.0
                # Create a Rectangle patch
                rect = patches.Rectangle((xshift + ratio, yshift), 0.01, 20, linewidth=5, edgecolor='red', facecolor=colors[i], fill = False)

                # Add the patch to the Axes
                ax.add_patch(rect)

                # Annotate
                ax.annotate('biased units', (0, 0), (xtshift + ratiot, yshift + ytshift))
        
        # create subplot for each unit
        gs01 = gs0[1].subgridspec(8, 8)
        #subplots_adjust(left = 0.06, bottom = 0.10, right = 0.94, top = 0.90, wspace = 0.1, hspace = 0.2)
        cu = 0
        for unit in units:
               
            # plot
            axi = f.add_subplot(gs01[cu//8, cu%8])
            
            cu += 1
            
            codes[:,:,unit] = codes[:,:,unit]/np.max(codes[:,:,unit])   #normalize by the maximum unit response over all categories
            data = [codes[0,:,unit], codes[1,:,unit], codes[2,:,unit], codes[3,:,unit],codes[4,:,unit], codes[5,:,unit],codes[6,:,unit], codes[7,:,unit], codes[8,:,unit], codes[9,:,unit]]

            bplot = axi.boxplot(data, showfliers=False, patch_artist=True)
                
            plt.ylim(0,1)
            axi.set_facecolor('none')

            # in bias condition, color the frame of vwfa unit plots in red
            if (mode == 'bias') and (unit >= 463):
                plt.setp(axi.spines.values(), color='red')
                plt.setp([axi.get_xticklines(), axi.get_yticklines()], color='red')
            
            # fill with colors
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            if cu <= 56:
                plt.xticks([])
            if cu > 56:
                plt.xticks(range(1,11), readable, rotation = 60, ha='right', size=10)
            if cu % 8 - 1 != 0:
                plt.yticks([])

            # add unit number
            leg1 = Rectangle((0, 0), 0, 0, alpha=0.0)
            lg2 = plt.legend([leg1], ['unit '+str(unit)], handlelength=0, loc=2, ncol=1,prop={'size':10}, framealpha=0, bbox_to_anchor=(-0.15,1.15))
            lg2.get_frame().set_linewidth(0)

        # add 
        # save figure
        if save:
            plt.savefig('profiles_'+mode+'_'+str(epoch)+'_'+strings[cf]+'.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.clf()
        del f
        cf += 1

    eyeballed_no_bias_string_selective_units = [4, 9, 29, 38, 41, 59, 83, 89, 92, 109, 110, 113, 115, 131, 142, 150, 152, 161, 165, 166, 170, 194, 197, 203, 204, 210, 212, 222, 226, 229, 232, 237, 242, 246, 250, 252, 254, 265, 270, 278, 282, 283, 291, 293, 294, 302, 308, 311, 315, 318, 323, 351, 362, 367, 372, 377, 385, 389, 392, 407, 410, 426, 432, 434, 436, 442, 451, 483, 487, 507]    
    return eyeballed_no_bias_string_selective_units
