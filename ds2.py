#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 14:27:31 2018

@author: Thomas
"""

import torch, torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils import data
from PIL import Image, ImageDraw, ImageFont, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.warnings.simplefilter('ignore')
import subprocess, shlex, shutil, io, os, random, gc, time
from tqdm import tqdm
#from urllib.request import urlopen
import pickle

import argparse 
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--img_path', default='imagesets_nex',
                    help='path to ImageNet folder that contains train and val folders')
parser.add_argument('--wrd_path', default='wordsets_nex',
                    help='path to word folder that contains train and val folders')
parser.add_argument('--save_path', default='save/',
                    help='path for saving ')
parser.add_argument('--output_path', default='activations/',
                    help='path for storing activations')
parser.add_argument('--restore_file', default=None,
                    help='name of file from which to restore model (ought to be located in save path, e.g. as save/cornet_z_epoch25.pth.tar)')
parser.add_argument('--img_classes', default=1000,
                    help='number of image classes')
parser.add_argument('--wrd_classes', default=1000,
                    help='number of word classes')
parser.add_argument('--num_train_items', default=100,
                    help='number of training items in each category')
parser.add_argument('--num_val_items', default=50,
                    help='number of validation items in each category')
parser.add_argument('--mode', default='pre',
                    help='pre for pre-schooler mode, lit for literate mode')
parser.add_argument('--max_epochs_pre', default=30, type=int,
                    help='number of epochs to run as pre-schooler - training on images only')
parser.add_argument('--max_epochs_lit', default=30, type=int,
                    help='number of epochs to run as literate - training on images and words')
parser.add_argument('--batch_size', default=100, type=int,
                    help='mini-batch size')
parser.add_argument('--lr', '--learning_rate', default=.01, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_schedule', default='StepLR')
parser.add_argument('--step_size', default=10, type=int,
                    help='after how many epoch learning rate should be decreased 10x')
parser.add_argument('--momentum', default=.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay ')

FLAGS, _ = parser.parse_known_args()

img_classes = 100
img_classes2 = 100
wrd_classes = 100
batch_size = 100
num_train_items = 100
num_val_items = 50
num_workers = 0
max_epochs = 100

def debug_images():
    #identify all bugged images in ImageNet folder, put them in a list.
    bug_list = []
    c = 0
    path_in = 'imagesets/train'
    trainfolders =  [ item for item in os.listdir(path_in) if os.path.isdir(os.path.join(path_in, item)) ]
    print ('length of trainfolders:', len(trainfolders))     
    for folder in trainfolders:
        gc.collect()
        c +=1
        images = [item for item in os.listdir(path_in+'/'+folder)]
        #print ('number of images:', len(images))
        print ('folder number', c, 'length of bug_list:', len(bug_list))
        d = 0
        for image in images:
            d +=1
            path = path_in+'/'+folder+'/'+image
            try:
                f = open(path, 'rb')
                img = Image.open(f)
                img.close()
                f.close()
                del f, img
                if d %500 == 0:
                    gc.collect()
            except:
                bug_list += [path]
                #print ('length of bug_list:', len(bug_list))
                #download_and_replace_image(folder, image, path)
    
    print ('final length of bug_list:', len(bug_list))
    
    return bug_list
    
def debug_images3():
    bug_list = []
    c = 0
    path_in = 'imagesets/train'
    trainfolders =  [ item for item in os.listdir(path_in) if os.path.isdir(os.path.join(path_in, item)) ]
    print ('length of trainfolders:', len(trainfolders))
    for k in range(50):
        gc.collect()
        b_l, c = debug_by_batch(k)
        bug_list += b_l
        print ('length of bug_list:', len(bug_list))
        print ('bug_list:', bug_list)
    print ('final length of bug_list:', len(bug_list))
    return bug_list
    
def debug_by_batch(k):
    if k == 0:
        bug_list = []
    else:
        pickle_in = open("bug_list.pickle","rb")
        bug_list = pickle.load(pickle_in)
        print ('length of bug_list:',len(bug_list))
    path_in = 'imagesets/train'
    trainfolders =  [ item for item in os.listdir(path_in) if os.path.isdir(os.path.join(path_in, item)) ]
    batch = trainfolders[k*20 : min((999,(k+1)*20))]
    c = k*20
    for folder in batch:
        c += 1
        print ('folder number', c)
        b_l = debug_images_in_folder(path_in, folder)
        bug_list += b_l
    pickle_out = open("bug_list.pickle","wb")
    pickle.dump(bug_list, pickle_out)
    return bug_list
    
def debug_images2():
    #identify all bugged images in ImageNet folder, put them in a list.
    bug_list = []
    c = 0
    path_in = 'imagesets/train'
    trainfolders =  [ item for item in os.listdir(path_in) if os.path.isdir(os.path.join(path_in, item)) ]
    print ('length of trainfolders:', len(trainfolders))     
    for folder in trainfolders:
        c += 1
        b_l = debug_images_in_folder(path_in, folder)
        bug_list += b_l
        print ('folder number', c, 'length of bug_list:', len(bug_list))
    print ('final length of bug_list:', len(bug_list))
    
    return bug_list

def debug_images_in_folder(path_in, folder):
    #given n image foldern identify all bugged images, put them in a list.
    
    start = time.time()
    gc.collect()
    bug_list = []
    images = [item for item in os.listdir(path_in+'/'+folder)]
    print ('number of images:', len(images))
    d = 0
    for image in images:
        d +=1
        path = path_in+'/'+folder+'/'+image
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                img.close()
                f.close()
            
                del f, img
#                if d %500 == 0:
#                    gc.collect()
#            gc.collect()
        except:
            bug_list += [path]
            #print ('length of bug_list:', len(bug_list))
            #download_and_replace_image(folder, image, path)
    end = time.time()
    print ('processing time:', end-start)
    return bug_list

def fix_images():
    pickle_in = open("bug_list.pickle","rb")
    bug_list = pickle.load(pickle_in)
    for s in bug_list:
        print (s)
        folder = s[16:25]
        image = s[26:]
        download_and_replace_image(folder, image, s)
        
    return 'images replaced'
    
def download_and_replace_image(folder, image, target_path):
    start = time.time()
    #download from website and save image in target_path
    url = 'http://169.44.201.108:7002/imagenet/train/'+folder+'/'+image
    img = Image.open(urlopen(url))
    
    #save in folder
    img.save(target_path)
    end = time.time()
    print ('processing time:', end-start)


#def CreateMiniImageSet(path_in='/media/th257835/data crossc/ILSVRC2012_img_train', path_out='imagesets/train/',num_im=1000):
def CreateMiniImageSet(path_in='imagesets/val/', path_out='imagesets_nex/val/',num_im=50):
    # list all subdirectories in a directory
    #folders = [x[0] for x in os.walk(path_in)]
    dirlist = [ item for item in os.listdir(path_in) if os.path.isdir(os.path.join(path_in, item)) ]
    #dirlist = ['n02814533','n02825657','n02837789','n02859443','n04409515','n04487081','n04505470','n04507155','n04536866','n07930864']

    print ("len(dirlist)", len(dirlist))
    # create same (empty) subdirectories in a new directory
    l = len(path_in)
    count = 0
    for f in dirlist:
        print (count,)
        #print('file:',f)
        target_path = path_out+f
        os.mkdir(target_path)
        impath_list = os.listdir(path_in+'/'+f)
        print ("len(impath_list)", len(impath_list))
        #impath_list.remove('._.DS_Store')
        #print('impath_list[1:1+num_im]',impath_list[1:1+num_im])
        #for i in impath_list[1:1+num_im]:
        #for i in impath_list[:num_im]:
        for i in impath_list:
            # copy part of the original subdir into the new subdir
            #print('path in', path_in+f+'/'+i)
            #print('path out', target_path+'/'+i)
            if i[:2] != '._':
                shutil.copy(path_in+'/'+f+'/'+i, target_path+'/'+i)
                #print(i)
            
        count += 1
    
    return 'done'

def gen2(savepath='', text = 'text', index=1, mirror=False, invert=False, fontname='Arial', W = 500, H = 500, size=24, xshift=0, yshift=0, upper=0, show=None):
    if upper:
        text = text.upper()
    if invert:
        text = text[::-1]
    img = Image.new("RGB", (W,H), color = (255, 255, 255))
    #fnt = ImageFont.truetype('/Library/Fonts/'+fontname+'.ttf', size) #size in pixels
    fnt = ImageFont.truetype(fontname+'.ttf', size)
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
    if savepath == '':
##        print ('I was here')
##        img.show()
        return img
        
def CreateMiniWordSet(path_out='wordsets_100cat_100ex/',num_train=100, num_val=50):
    #define words, sizes, fonts
    words = words4yo()
    #random.shuffle(words)
    #wordlist =  ['dimanche', 'lundi', 'mots', 'samedi', 'semaine']
    wordlist = words[:100]
    sizes = [40, 50, 60, 70, 80]
    fonts = ['arial','tahoma']
    xshift = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
    yshift = [-30, -15, 0, 15, 30]
    
    #create train and val folders 
    for m in ['train', 'val']:
        for f in wordlist:
            target_path = path_out+m+'/'+f
            os.makedirs(target_path)
    
    #for each word, create num_train + num_val exemplars, then split randomly into train and val.
    for w in tqdm(wordlist):
        gc.collect()
        print (w,)
        for n in range(num_train + num_val):
            if n < num_train:
                path = path_out+'train/'+w+'/'
            else:
                path = path_out+'val/'+w+'/'
            
            f = random.choice(fonts)
            s = random.choice(sizes)
            u = random.choice([0,1])
            x = random.choice(xshift)
            y = random.choice(yshift)
            gen2(savepath=path, text=w, index=n, fontname=f, size=s, xshift=x, yshift=y, upper=u)

    return 'done'

def CreateMiniWordValSet(path_out='wordsets_100cat_100ex/',num_train=100, num_val=50):
    #define words, sizes, fonts
    words = words4yo()
    #wordlist =  ['dimanche', 'lundi', 'mots', 'samedi', 'semaine']
    wordlist = words[:100]
    sizes = [40, 50, 60, 70, 80]
    fonts = ['arial','tahoma']
    xshift = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
    yshift = [-30, -15, 0, 15, 30]
    
    #create train and val folders 
    #for m in ['train', 'val']:
    for m in ['val']:
        for f in wordlist:
            target_path = path_out+m+'/'+f
            os.makedirs(target_path)
    
    #for each word, create num_train + num_val exemplars, then split randomly into train and val.
    for w in tqdm(wordlist):
        gc.collect()
        print (w,)
        for n in range(num_val):
            path = path_out+'val/'+w+'/'
            
            f = random.choice(fonts)
            s = random.choice(sizes)
            u = random.choice([0,1])
            x = random.choice(xshift)
            y = random.choice(yshift)
            gen2(savepath=path, text=w, index=n, fontname=f, size=s, xshift=x, yshift=y, upper=u)

    return 'done'

def img_target_transform(target, classes=FLAGS.img_classes):
    onehot = torch.eye(classes)
    return onehot[target]

def ImageDataset(data_path='imagesets', folder='train', batch_size=2, workers=0):
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_path, folder),
        torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            #torchvision.transforms.Resize(size=(224, 224)),
            #torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),)
        #target_transform=img_target_transform,)
    return dataset

def phase2_target_transform_logit(target):
    onehot = torch.eye(FLAGS.img_classes + FLAGS.wrd_classes)
    return onehot[target + FLAGS.img_classes]

def phase2_target_transform(target):
    return target + FLAGS.img_classes

def WordDataset(data_path='wordsets', folder='train', batch_size=2, workers=0):
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_path, folder),
        torchvision.transforms.Compose([
            #torchvision.transforms.CenterCrop(224),
            torchvision.transforms.RandomResizedCrop(224, scale = (0.9,1), ratio= (1,1)),
            torchvision.transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        target_transform=phase2_target_transform,)
    return dataset

def img2_target_transform_logit(target):
    onehot = torch.eye(2*FLAGS.img_classes)
    return onehot[target]
    

def Image2Dataset(data_path='imagesets', folder='train', batch_size=2, workers=0):
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_path, folder),
        torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            #torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),)
        #target_transform=img2_target_transform,)
    return dataset

def Phase2Dataset(data_path='imagesets', folder='train', batch_size=2, workers=0):
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_path, folder),
        torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            #torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),)
        #target_transform=phase2_target_transform,)
    return dataset
