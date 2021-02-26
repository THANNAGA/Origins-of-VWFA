import torch, torchvision, imageio, copy, gc, time
from torch.autograd import Variable
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFile
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
from tqdm import tqdm
from urllib.request import urlopen

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



"""
---------------------------- Look at weights from VWFA to word units
"""

def retrieve_net(epoch=79, mode='lit_bias', clip=False):
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
        #print ('model loaded')
        
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


def getAcc(codes=None, weights=None, bias=False, labels=list(range(1000)), verbose=False):
    out = np.dot(codes, weights.T) + bias
    out = np.argmax(out, axis=1)
    score = 100*np.mean(out==labels)
    if verbose:
        print ('score', score)
    return score    


"""
Lesion analysis. For each network (pre, illit, bias, no bias):
1. get the dense codes for 1000 images and 1000 word inputs.
2. get performance on images and on words as a function of the fraction of vwfa units removed.
3. plot as a figure of 4 subplots, each with mean image and mean word curves as a function of % vwfa lesion.
"""

# download 1 image for each class from the imagenet validation folder (with correct order)
def download_image(image_name='ILSVRC2012_val_00000001.JPEG', target_path='one_thousand_val_images/'):
    start = time.time()
    #download from website and save image in target_path
    url = 'http://169.44.201.108:7002/imagenet/val/'+image_name
    img = Image.open(urlopen(url))
    
    #save in folder
    img.save(target_path+image_name, format='jpeg')
    end = time.time()
    #print ('processing time:', end-start)

def select_images(source_list = 'randomized_val_list.txt'):
    file = open(source_list, 'r')
    image_list = []; indices = []
    for line in file:
        index = int(line[29:])
        if index not in indices:
            image_list += [line[:28]]
            indices += [index]
    print ('len(image_list)', len(image_list))
    print ('len(indices)', len(indices))
    s = np.argsort(indices)
    image_list = np.array(image_list)[s]
    indices = np.array(indices)[s]

    for i, image_name in enumerate(image_list):
        if i % 25 == 0:
            print ('iteration', i)
        download_image(image_name=image_name, target_path='one_thousand_val_images/')
    return image_list, indices

def save_images_as_numpy_in_order(image_path = 'one_thousand_val_images/'):
    #load images
    images = np.load(image_path+'image_list.npy')
    l = len(images)
    #go through all images in folder in the right order, numpyfy and save
    data = np.zeros((l, 3, 224, 224))
    for i, image in enumerate(images):

        if i % 25 == 0:
            print ('iteration', i)
        
        path = image_path+image
        f = open(path, 'rb')
        img = Image.open(f)
        if 'L' in img.getbands():
            img = img.convert('RGB')
        img = transform(img)
        arr = img.numpy()
        if arr.shape[0] == 1:
            arr = np.reshape(3*[arr], (3, arr.shape[1], arr.shape[2]))
        data[i] = arr[:, :224, :224]
        f.close()
        
    np.save(image_path+'image_data.npy', data)    
    return 'done'

def Read_process_and_store(model, data_path = '../save/all_words/data.npy', epoch=None, mode=None, stim=None):
    print ('reading, processing stimuli and storing codes')
    #load data
    data = np.load(data_path)

    #build labels
    if stim == 'images':
        labels = list(range(1000))
    if stim == 'words':
        labels = list(range(1000,2000))    

    #torchify data
    data = torch.Tensor(data)
    data = Variable(data)

    #process data by convnet
    v1, v2, v4, it, h, out = model(data, clip=False)

    #check perf
    out = out.detach().numpy()
    print ('np.shape(out)', out.shape)
    out = np.argmax(out, axis=1)
    print ('out[:10]', out[:10])
    score = 100*np.mean(out==labels)
    print ('score', score)
    
    #save codes
    h = h.detach().numpy()
    np.save('../save/' + stim[:-1] + '_codes_'+ mode + '_' + str(epoch) + '.npy', h)
    print ('codes have been saved as ', '../save/' + stim[:-1] + '_codes_'+ mode + '_' + str(epoch) + '.npy')
    return 'done'


def gen_dense_codes(epoch=79, mode='lit_bias', data_path = 'one_thousand_val_images/image_data.npy', stim='images'):
    # get model
    model = retrieve_net(epoch=epoch, mode=mode, clip=False)
    
    # generate and save dense codes for images for this model
    Read_process_and_store(model, data_path = data_path, epoch=epoch, mode=mode, stim=stim)

    return 'done'

def get_weights(net, mode=None, also_bias=False):
    
    if mode in ['pre', 'illit']:
        weights = net._modules['decoder']._modules['linear'].weight.data.cpu().numpy()
        bias = net._modules['decoder']._modules['linear'].bias.data.cpu().numpy()

    if mode == 'lit_bias':
        # to do
        wrd_weights = net._modules['bilinear'].lin_wrd.weight.data.cpu().numpy()
        wrd_bias = net._modules['bilinear'].lin_wrd.bias.data.cpu().numpy()
        img_weights = net._modules['bilinear'].lin_img.weight.data.cpu().numpy()
        img_bias = net._modules['bilinear'].lin_img.bias.data.cpu().numpy()
        out_img, inp_img = np.shape(img_weights)
        out_wrd, inp_wrd = np.shape(wrd_weights)
        print ('out_img, inp_img', out_img, inp_img)
        print ('out_wrd, inp_wrd', out_wrd, inp_wrd)
        zero_fill = np.zeros((1000, 512))
        weights = np.concatenate((img_weights, zero_fill), axis=0)
        weights[1000:, -49:] = wrd_weights
        bias = np.concatenate((img_bias, wrd_bias))
 
    if mode == 'lit_no_bias':
        weights = net._modules['linear'].weight.data.cpu().numpy()
        bias = net._modules['linear'].bias.data.cpu().numpy()


    out, inp = np.shape(weights)
    print ('out, inp', out, inp)

    if also_bias:
        return weights, bias
    else:
        return weights

def perf_under_lesion(mode='lit_bias', epoch=79, power=50):
    # get codes
    image_codes = np.load('../save/lesions/image_codes_'+ mode + '_' + str(epoch) + '.npy')
    word_codes = np.load('../save/lesions/word_codes_'+ mode + '_' + str(epoch) + '.npy')
    l,c = image_codes.shape

    # get weights
    net = retrieve_net(epoch=epoch, mode=mode, clip=False)
    weights, bias = get_weights(net, mode=mode, also_bias=True)

    vwfa_indices = list(range(510 - 49, 510));l_vwfa = len(vwfa_indices)
    img_scores = np.zeros((power,11))
    wrd_scores = np.zeros((power,11))
    for i, f in enumerate(np.linspace(0,100,11)):
        print ('lesion percent:', f)
        for n in range(power):
            # compute scores for all lesion fractions
            if f > 0:
                k = int(f*l_vwfa/100.0)
                #print ('k', k)
                indices_to_zero = random.sample(vwfa_indices, k)
                #print ('indices_to_zero', indices_to_zero)
                image_codes[:,indices_to_zero] = np.zeros((l, k))
                word_codes[:,indices_to_zero] = np.zeros((l, k))
                
            image_score = getAcc(codes = image_codes, weights = weights, bias = bias, labels = list(range(1000)))
            word_score = getAcc(codes = word_codes, weights = weights, bias = bias, labels = list(range(1000, 2000)))
            img_scores[n, i], wrd_scores[n, i] = image_score, word_score

    np.save('img_lesion_scores_'+ mode + '_' + str(epoch) + '.npy', img_scores)
    np.save('wrd_lesion_scores_'+ mode + '_' + str(epoch) + '.npy', wrd_scores)

    return img_scores, wrd_scores

def perf_under_lesion2(mode='lit_bias', target='all', epoch=79, power=50):
    # get codes
    image_codes = np.load('../save/lesions/image_codes_'+ mode + '_' + str(epoch) + '.npy')
    word_codes = np.load('../save/lesions/word_codes_'+ mode + '_' + str(epoch) + '.npy')
    l,c = image_codes.shape

    # get weights
    net = retrieve_net(epoch=epoch, mode=mode, clip=False)
    weights, bias = get_weights(net, mode=mode, also_bias=True)

    # select the pool of indices in which to carry out increasingly severe lesions
    if target == 'all':
        pool = list(range(510 - 49, 510))
    if target == 'ws':
        pool = np.load('../save/all_words/'+mode+'_'+str(epoch)+'_words_selective.npy')
    l_pool = len(pool)
    print ('length of pool:', l_pool)

    # do the job
    img_scores = np.zeros((power,11))
    wrd_scores = np.zeros((power,11))
    for i, f in enumerate(np.linspace(0,100,11)):
        print ('lesion percent:', f)
        for n in range(power):
            # compute scores for all lesion fractions
            if f > 0:
                k = int(f*l_pool/100.0)
                #print ('k', k)
                indices_to_zero = random.sample(list(pool), k)
                #print ('indices_to_zero', indices_to_zero)
                image_codes[:,indices_to_zero] = np.zeros((l, k))
                word_codes[:,indices_to_zero] = np.zeros((l, k))
                
            image_score = getAcc(codes = image_codes, weights = weights, bias = bias, labels = list(range(1000)))
            word_score = getAcc(codes = word_codes, weights = weights, bias = bias, labels = list(range(1000, 2000)))
            img_scores[n, i], wrd_scores[n, i] = image_score, word_score

    # save
    np.save('img_lesion_scores_'+ mode + '_' + target + '_' + str(epoch) + '.npy', img_scores)
    np.save('wrd_lesion_scores_'+ mode + '_' + target + '_' + str(epoch) + '.npy', wrd_scores)

    return img_scores, wrd_scores

def plot_lesion_analysis(epoch=79, save=True, show=True, load=False):

    #modes = ['pre', 'illit', 'lit_bias', 'lit_no_bias']
    #titles = ['pre', 'illit', 'bias', 'no bias']

    modes = ['lit_no_bias', 'lit_bias']
    titles = ['Unbiased literate', 'Biased literate']

    f = plt.figure(figsize=(8, 4))
    #plt.subplots_adjust(left = 0.05, bottom = 0.22, right = 0.95, top = 0.95, wspace = 0.2, hspace = 0.3)

    for i, (mode, title) in enumerate(zip(modes, titles)):
        img_scores = np.load('img_lesion_scores_'+ mode + '_' + str(epoch) + '.npy')
        wrd_scores = np.load('wrd_lesion_scores_'+ mode + '_' + str(epoch) + '.npy')

        plt.subplot(1, 2, i+1)
        x = np.linspace(0,100,11)
        mean_scores = np.mean(img_scores, axis=0)
        y_img = 100*mean_scores/mean_scores[0]
        error = scipy.stats.sem(img_scores, axis=0)
        plt.plot(x, y_img, label='images', color='#089FFF', linewidth=2)
        plt.fill_between(x, y_img-error, y_img+error, alpha=0.3, edgecolor='#089FFF', facecolor='#089FFF')

        mean_scores = np.mean(wrd_scores, axis=0)
        y_wrd = 100*mean_scores/mean_scores[0]
        error = scipy.stats.sem(wrd_scores, axis=0)
        plt.plot(x, y_wrd, label='words', color='#1B2ACC', linewidth=2)
        plt.fill_between(x, y_wrd-error, y_wrd+error, alpha=0.3, edgecolor='#1B2ACC', facecolor='#1B2ACC')

        plt.xticks(x, [str(int(j)) for j in np.linspace(0,100,11)], size = 8)
        plt.xlabel('Lesions in target units (%)', size=13)
        if i == 0:
            plt.ylabel('Top-1 Accuracy (normalized)', size=13)
        plt.ylim(0,105)
        plt.title(title, size=16)
        if i == 1:
            plt.legend(frameon=False, ncol=1, prop={'size': 10})
    
    if save:
        plt.savefig('lesions_'+str(epoch)+'.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.clf()

    return 'done'


def plot_lesion_analysis2(epoch=79, save=True, show=True, load=False):

    #modes = ['pre', 'illit', 'lit_bias', 'lit_no_bias']
    #titles = ['pre', 'illit', 'bias', 'no bias']

    modes = ['lit_no_bias_all', 'lit_bias_all', 'lit_no_bias_ws', 'lit_bias_ws']
    titles = ['Unbiased literate', 'Biased literate', 'Unbiased literate', 'Biased literate']

    f = plt.figure(figsize=(10, 6))
    plt.subplots_adjust(hspace = 0.4)
    ax = np.zeros(4).tolist()
    for i, (mode, title) in enumerate(zip(modes, titles)):
        print ('i, mode', i, mode)
        img_scores = np.load('img_lesion_scores_'+ mode + '_' + str(epoch) + '.npy')
        wrd_scores = np.load('wrd_lesion_scores_'+ mode + '_' + str(epoch) + '.npy')
        if i == 0:
            xlabel = 'Lesions in control vwfa units (%)'
        if i == 1:
            xlabel = 'Lesions in vwfa units (%)'
        if i >= 2:
            xlabel = 'Lesions in word-specific units (%)'

        ax[i] = f.add_subplot(2,2,i+1)
        ax[i].annotate('ABCD'[i], xy=(0., 0.), xycoords = "axes fraction", xytext=(-0.10, 1.05), size = 18,color='black',fontweight="bold")
        x = np.linspace(0,100,11)
        mean_scores = np.mean(img_scores, axis=0)
        y_img = 100*mean_scores/mean_scores[0]
        error = scipy.stats.sem(img_scores, axis=0)
        plt.plot(x, y_img, label='images', color='gray', lw=2, ls='--')
        plt.fill_between(x, y_img-error, y_img+error, alpha=0.3, edgecolor='#089FFF', facecolor='#089FFF')

        mean_scores = np.mean(wrd_scores, axis=0)
        y_wrd = 100*mean_scores/mean_scores[0]
        error = scipy.stats.sem(wrd_scores, axis=0)
        plt.plot(x, y_wrd, label='words', color='black', lw=2)
        plt.fill_between(x, y_wrd-error, y_wrd+error, alpha=0.3, edgecolor='#1B2ACC', facecolor='#1B2ACC')

        plt.xticks(x, [str(int(j)) for j in np.linspace(0,100,11)], size = 8)
        plt.xlabel(xlabel, size=13)
        if i in [0, 2]:
            plt.ylabel('Top-1 Accuracy\n(normalized)', size=13)
        plt.ylim(0,105)
        if i in [0, 1]:
            plt.title(title, size=16)
        if i == 1:
            plt.legend(frameon=False, ncol=1, prop={'size': 10})
    
    if save:
        plt.savefig('lesions_'+str(epoch)+'.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.clf()

    return 'done'





















