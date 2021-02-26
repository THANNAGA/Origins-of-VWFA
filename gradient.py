import torch, torchvision, imageio, copy, gc
from torch.autograd import Variable
import numpy as np
from PIL import Image
from tqdm import tqdm
import scipy, pickle, random, os
import scipy.stats

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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
    vwfa = h.detach().numpy()[:,-49:]
    h = h.detach().numpy()[:,:-49]

    #save codes
    np.save('../save/'+ cat + mode+ '_' +'v1_codes_'+ str(epoch) +'.npy', v1)
    np.save('../save/'+ cat + mode+ '_' +'v2_codes_'+ str(epoch) +'.npy', v2)
    np.save('../save/'+ cat + mode+ '_' +'v4_codes_'+ str(epoch) +'.npy', v4)
    np.save('../save/'+ cat + mode+ '_' +'it_codes_'+ str(epoch) +'.npy', it)
    np.save('../save/'+ cat + mode+ '_' +'h_codes_'+ str(epoch) +'.npy', h)
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

    paths = ['false_fonts/', 'infreq_letters/', 'freq_letters/', 'freq_bigrams/', 'freq_quadrigrams/', 'words/']
    #paths = ['faces/', 'bodies/', 'houses/', 'tools/', 'false_fonts/', 'infreq_letters/', 'freq_letters/', 'freq_bigrams/', 'freq_quadrigrams/', 'words/'] 
    for cat in tqdm(paths):
        Read_process_and_store(model, cat=cat, image_path = '../stimuli/', epoch=epoch, mode=mode, clip=clip)

    gc.collect()
     
    return 'done'



def plot_gradient(epoch=79, mode='lit_bias', norm=False, show=False):

    plt.subplots(figsize=(14,4))
    plt.subplots_adjust(left = 0.06, bottom = 0.15, right = 0.94, top = 0.85, wspace = 0.36, hspace = 0.20)
    
    categories = ['false_fonts', 'infreq_letters', 'freq_letters', 'freq_bigrams', 'freq_quadrigrams', 'words'];n = len(categories)
    titles = ['false fonts', 'infrequent letters', 'frequent letters', 'frequent bigrams', 'frequent quadrigrams', 'words']
    layers = ['v1', 'v2', 'v4', 'it', 'h', 'vwfa']
    xtickslabs = ['v1', 'v2', 'v4', 'it', 'dense', 'vwfa']
    l = len(layers)
    
    for i in range(n-1, -1, -1):
        #load codes
        means, sems = np.zeros(l), np.zeros(l)
        for j in range(l):
            codes = np.load('../save/'+ categories[i]+ '/' + mode+ '_' +layers[j]+'_codes_'+str(epoch)+'.npy')
            means[j] = np.mean(codes)
            sems[j] = np.mean(scipy.stats.sem(codes))

        if norm & (i == n-1):
            word_means = means
            
        #plot in subplot
        plt.subplot(1,n,i+1)
        x = list(range(1,l+1))
        if norm:
            y = means/word_means
            plt.ylim(0.6,1.2)
        if not norm:
            y = means
            plt.ylim(0.0,2.0)
        yerr_upper = sems
        yerr_lower=np.zeros(l)
        plt.bar(x, y, width=0.8, color='0.5', edgecolor='k', yerr=[yerr_lower,yerr_upper], linewidth = 2, capsize=10)
        plt.xticks(x, xtickslabs, rotation = 60, ha='right')

        if i == 0:
            plt.ylabel('mean activation')
        plt.title(titles[i])

    if mode == 'pre':
        plt.suptitle('Pre-literate network', size=15)
    if mode == 'lit_bias':
        plt.suptitle('Literate network with bias', size=15)
    if mode == 'lit_no_bias':
        plt.suptitle('Literate network without bias', size=15)
    if mode == 'illit':
        plt.suptitle('Illiterate network', size=15)
  

    if norm:
        plt.savefig('gradient_with_word_norm_'+mode+'.png')
    else:
        plt.savefig('gradient_without_word_norm_'+mode+'.png')

    if show:
        plt.show()
    
    return 'done'

def plot_gradient2(norm=False, show=False):
    fig, axgen = plt.subplots(frameon=False,figsize = (8,8))
    fig.patch.set_visible(False)
    axgen.axis('off')

    layers = ['V1', 'V2', 'V4', 'IT', 'DENSE-', 'DENSE+']
    layers.reverse()
    layers_call = ['v1', 'v2', 'v4', 'it', 'h', 'vwfa']
    layers_call.reverse()
    conditions = ['Pre-literate', 'Illiterate', 'Literate without bias', 'Literate with bias']
    conditions_call = ['pre', 'illit', 'lit_no_bias', 'lit_bias']
    epoch_call = ['49', '79', '79', '79']
    layer_colors = ['#006600', '#336666', '#00CCCC', '#0033FF', '#000099', '#660099', '#990066']
    layer_colors.reverse()
    categories = ['false_fonts', 'infreq_letters', 'freq_letters', 'freq_bigrams', 'freq_quadrigrams', 'words']
    xtickslabs = ['ff', 'il', 'fl', 'fb', 'fq', 'w']
    cat_colors = ['aquamarine', 'mediumaquamarine', 'lightgreen', 'lime', 'forestgreen', 'darkgreen']


    l=len(layers); c=len(conditions); lcat=len(categories)
    ax = np.zeros(l*c).tolist()
    
    x,y,x_shift,y_shift = -0.04, 1.05,1.05,.173
    S = 25

    for i in range(l):          # layers
        for j in range(c):      # network conditions
            n = c*i + j
            print ("n",n)
            ax[n] = fig.add_subplot(l,c,n+1)
            ax[n].xaxis.grid(False)
            ax[n].yaxis.grid(False)

            # load data
            means, sems = np.zeros(lcat), np.zeros(lcat)
            for k in range(lcat):
                codes = np.load('../save/'+ categories[k]+ '/' + conditions_call[j]+ '_' +layers_call[i]+'_codes_'+epoch_call[j]+'.npy')
                means[k] = np.mean(codes)
                sems[k] = np.mean(scipy.stats.sem(codes))
   
            # plot data
            x = list(range(1,lcat+1))
            if norm:
                y = means/word_means
                plt.ylim(0.6,1.2)
            if not norm:
                y = means
                if (layers_call[i] != 'vwfa') or (conditions_call[j] != 'lit_bias'):
                    plt.ylim(0.0,3.0)
                    
            yerr_upper = sems
            yerr_lower=np.zeros(lcat)
            plt.bar(x, y, width=0.5, color=cat_colors, edgecolor='k', yerr=[yerr_lower,yerr_upper], linewidth = 1, capsize=5)
            

            # possible y axis labels
            if j == 0:
                plt.ylabel(layers[i], color=layer_colors[i], fontweight='bold')

            # possible titles
            if i == 0:
                plt.title(conditions[j])
            if i == l-1:
                plt.xticks(x, xtickslabs)
            if i < l-1:
                plt.xticks(x, ['', '', '', '', '', ''])
        
    plt.subplots_adjust(left= 0.08, right= 0.91, bottom = 0.03, top = .9, wspace = 0.35, hspace = 0.30)
    plt.suptitle('Mean activations in the networks', fontweight='bold')
    plt.show()
    
    fig.savefig('Gradients.jpg',dpi=300)
    del fig    
    return 'done'

def Stats_on_levels(epoch='79', layer='vwfa', condition='lit_bias', verbose=False):
    #print ('One-way Anova on '+condition+' network, epoch'+epoch+', layer '+layer)
    layers_call = ['v1', 'v2', 'v4', 'it', 'h', 'vwfa']
    conditions_call = ['pre', 'illit', 'lit_no_bias', 'lit_bias']
    epoch_call = ['49', '79', '79', '79']
    categories_call = ['false_fonts', 'infreq_letters', 'freq_letters', 'freq_bigrams', 'freq_quadrigrams', 'words']
    categories = ['ff', 'il', 'fl', 'fb', 'fq', 'w']
    l=len(layers_call); c=len(conditions_call); lcat=len(categories)

    # load data
    data = [[]]*lcat
    for k in range(lcat):
        codes = np.load('../save/'+ categories_call[k]+ '/' + condition+ '_' +layer+'_codes_'+epoch+'.npy')
        data[k] = np.mean(codes, axis=1)        #average actiations across units at this level

    # overall difference between the 6 categories of stimuli    
    F_1_6, p_1_6 = scipy.stats.f_oneway(*data)    #data is a list (stimulus categories) of numpy arrays (array of mean activation for a stimulus across all units of all feature maps at this level in the network)
    F_1_6, p_1_6 = format_for_stats(F_1_6, p_1_6)
    if verbose:
        print ('\n overall difference between the 6 categories of stimuli ')
        print ('F-value', F_1_6)
        print ('p-value', p_1_6)

    # difference between 'ff' and 'il' categories of stimuli
    F_1_2, p_1_2 = scipy.stats.f_oneway(*data[:2])    #data is a list (stimulus categories) of numpy arrays (array of mean activation for a stimulus across all units of all feature maps at this level in the network)
    F_1_2, p_1_2 = format_for_stats(F_1_2, p_1_2)
    if verbose:
        print ('\n difference between "ff" and "il"')
        print ('F-value', F_1_2)
        print ('p-value', p_1_2)

    # difference between 'fl', 'fb', 'fq', 'w' categories of stimuli
    F_3_6, p_3_6 = scipy.stats.f_oneway(*data[2:])    #data is a list (stimulus categories) of numpy arrays (array of mean activation for a stimulus across all units of all feature maps at this level in the network)
    F_3_6, p_3_6 = format_for_stats(F_3_6, p_3_6)
    if verbose:
        print ('\n difference between "fl", "fb", "fq", "w"')
        print ('F-value', F_3_6)
        print ('p-value', p_3_6)
    return F_1_6, p_1_6, F_1_2, p_1_2, F_3_6, p_3_6

def format_for_stats(F, p):
    F_dec = 2; p_dec =3
    F = 'F='+str(np.round(F, decimals=F_dec))
    if p < 0.0001:
        p ='p<0.0001'
    else:
        p = 'p='+str(np.round(p, decimals=p_dec))
    return F, p

def warp_stats_and_print_tableS1():
    layers = ['v1', 'v2', 'v4', 'it', 'h', 'vwfa'];layers.reverse()
    models = ['pre', 'illit', 'lit_no_bias', 'lit_bias']
    for i, layer in enumerate(layers):
        pre_F_1_6, pre_p_1_6, pre_F_1_2, pre_p_1_2, pre_F_3_6, pre_p_3_6 = Stats_on_levels(epoch='49', layer=layer, condition='pre', verbose=False)
        ill_F_1_6, ill_p_1_6, ill_F_1_2, ill_p_1_2, ill_F_3_6, ill_p_3_6 = Stats_on_levels(epoch='79', layer=layer, condition='illit', verbose=False)
        unb_F_1_6, unb_p_1_6, unb_F_1_2, unb_p_1_2, unb_F_3_6, unb_p_3_6 = Stats_on_levels(epoch='79', layer=layer, condition='lit_no_bias', verbose=False)
        bia_F_1_6, bia_p_1_6, bia_F_1_2, bia_p_1_2, bia_F_3_6, bia_p_3_6 = Stats_on_levels(epoch='79', layer=layer, condition='lit_bias', verbose=False)
        print (pre_F_1_6+', '+pre_p_1_6+'\t'+ill_F_1_6+', '+ill_p_1_6+'\t'+unb_F_1_6+', '+unb_p_1_6+'\t'+bia_F_1_6+', '+bia_p_1_6)
        print (pre_F_1_2+', '+pre_p_1_2+'\t'+ill_F_1_2+', '+ill_p_1_2+'\t'+unb_F_1_2+', '+unb_p_1_2+'\t'+bia_F_1_2+', '+bia_p_1_2)
        print (pre_F_3_6+', '+pre_p_3_6+'\t'+ill_F_3_6+', '+ill_p_3_6+'\t'+unb_F_3_6+', '+unb_p_3_6+'\t'+bia_F_3_6+', '+bia_p_3_6)

def warpup(recompute = False):
    if recompute:
        warp_process_store(epoch=49, mode='pre', clip=False)
        warp_process_store(epoch=79, mode='illit', clip=False)
        warp_process_store(epoch=79, mode='lit_bias', clip=False)
        warp_process_store(epoch=79, mode='lit_no_bias', clip=False)
    plot_gradient2(show=True)
    warp_stats_and_print_tableS1()
    return 'done'
