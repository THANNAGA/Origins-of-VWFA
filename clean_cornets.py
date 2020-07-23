from __future__ import (absolute_import, division,
                                                print_function, unicode_literals)
from builtins import (
                 bytes, dict, int, list, object, range, str,
                 ascii, chr, hex, input, next, oct, open,
                 pow, round, super,
                 filter, map, zip)

import numpy as np
import torch
from collections import OrderedDict
from torch import nn
import torch.nn.functional as F

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

class Flatten(nn.Module):

        def forward(self, x):
                return x.view(x.size(0), -1)


class Identity(nn.Module):

        def forward(self, x):
                return x
                
"""
class needed for the project
"""

class Bi_linear(nn.Module):
        def __init__(self, in_img=256, out_img=FLAGS.img_classes, in_wrd=49, out_wrd=FLAGS.wrd_classes):
                super().__init__()
                self.lin_img = nn.Linear(in_img, out_img)
                self.lin_wrd = nn.Linear(in_wrd, out_wrd)
                self.output = Identity()
                
        def forward(self, inp, in_wrd=49):
                x = torch.cat((self.lin_img(inp), self.lin_wrd(inp[:,-in_wrd:])), 1)
                #print('x.shape', x.shape)
                x = self.output(x)
                return x
                
"""
------------------------------------------------
------------------------------------------------
Cornet_Z models
------------------------------------------------
------------------------------------------------
"""
  
"""
Basic block for CORnet Z
"""
class CORblock_Z(nn.Module):

        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
                super().__init__()
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                                          stride=stride, padding=kernel_size // 2)
                self.nonlin = nn.ReLU()
                self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                self.output = Identity()

        def forward(self, inp):
                x = self.conv(inp)
                x = self.nonlin(x)
                x = self.pool(x)
                x = self.output(x)
                return x
                
                
def CORnet_Z():
        model = nn.Sequential(OrderedDict([
                ('V1', CORblock_Z(3, 64, kernel_size=7, stride=2)),
                ('V2', CORblock_Z(64, 128)),
                ('V4', CORblock_Z(128, 256)),
                ('IT', CORblock_Z(256, 512)),
                ('decoder', nn.Sequential(OrderedDict([
                        ('avgpool', nn.AdaptiveAvgPool2d(1)),
                        ('flatten', Flatten()),
                        ('linear', nn.Linear(512, 1000)),
                        ('output', Identity())
                ])))
        ]))

        # weight initialization
        for m in model.modules():
                if isinstance(m, nn.Conv2d):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                                nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                                nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()

        return model

# Easier model to use, straigthforwardly returns all the activations of each layer    
class CORnet_Z_tweak(nn.Module):
   
        def __init__(self, out_img=1000, out_wrd=0):
                super(CORnet_Z_tweak, self).__init__()

                self.V1 = CORblock_Z(3, 64, kernel_size=7, stride=2)
                self.V2 = CORblock_Z(64, 128)
                self.V4 = CORblock_Z(128, 256)
                self.IT = CORblock_Z(256, 512)

                # Decoder
                self.avgpool = nn.AdaptiveAvgPool2d(1)
                self.flatten = Flatten()
                self.softmax = nn.Softmax()
                self.sigmoid = nn.Sigmoid()
                self.output = Identity()
                self.decoder = nn.Sequential(OrderedDict([
                        ('linear', nn.Linear(512, out_img + out_wrd)),
                        ('output', Identity())
                        ]))

        def forward(self, image, clip=False):
                # Image
                v1 = self.V1(image)
                v2 = self.V2(v1)
                v4 = self.V4(v2)
                it = self.IT(v4)

                # decoder
                h = self.avgpool(it)
                h = self.flatten(h)
                out = self.decoder(h)
       
                if clip:
                        v1 = self.flatten(v1[:,:8,24:32,24:32].contiguous())
                        v2 = self.flatten(v2[:,:8,10:18,10:18].contiguous())
                        v4 = self.flatten(v4[:,:8,3:11,3:11].contiguous())
                        it = self.flatten(it[:,:11,:,:].contiguous())
                        return v1, v2, v4, it, h, out
                return self.flatten(v1), self.flatten(v2), self.flatten(v4), self.flatten(it), h, out


## Simplified biased word model with easily accessible activations
class CORNet_Z_biased_words(nn.Module):

        def __init__(self, init_model=None):
                super(CORNet_Z_biased_words, self).__init__()
                if init_model == None:
                    self.V1 = CORblock_Z(3, 64, kernel_size=7, stride=2)
                    self.V2 = CORblock_Z(64, 128)
                    self.V4 = CORblock_Z(128, 256)
                    self.IT = CORblock_Z(256, 512)
                if init_model != None:
                    self.V1 = init_model.V1
                    self.V2 = init_model.V2
                    self.V4 = init_model.V4
                    self.IT = init_model.IT

                # Decoder
                self.avgpool = nn.AdaptiveAvgPool2d(1)
                self.flatten = Flatten()
                self.bilinear = Bi_linear(in_img=512, out_img=1000, in_wrd=49, out_wrd=1000)
                #self.bilinear = Bi_linear(in_img=512, out_img=1000, in_wrd=49, out_wrd=2000)
                self.softmax = nn.Softmax()
                self.sigmoid = nn.Sigmoid()
                self.output = Identity()

                if init_model != None:
                    # Initializing the hidden-to-image linear network with weights taken from the previous model
                    self.bilinear.lin_img.weight.data = init_model._modules['decoder']._modules['linear'].weight.data
                    self.bilinear.lin_img.bias.data = init_model._modules['decoder']._modules['linear'].bias.data


        def forward(self, image, clip=False):
                # Image
                v1 = self.V1(image)
                v2 = self.V2(v1)
                v4 = self.V4(v2)
                it = self.IT(v4)

                # decoder
                h = self.avgpool(it)
                h = self.flatten(h)
                out = self.bilinear(h)
                if clip:
                        v1 = self.flatten(v1[:,:8,24:32,24:32].contiguous())
                        v2 = self.flatten(v2[:,:8,10:18,10:18].contiguous())
                        v4 = self.flatten(v4[:,:8,3:11,3:11].contiguous())
                        it = self.flatten(it[:,:11,:,:].contiguous())
                        return v1, v2, v4, it, h, out
                return self.flatten(v1), self.flatten(v2), self.flatten(v4), self.flatten(it), h, out


## Simplified non-biased word model with easily accessible activations
class CORNet_Z_nonbiased_words(nn.Module):

        def __init__(self, init_model=None, n_hid = 512, out_img=1000, out_wrd=1000):
                super(CORNet_Z_nonbiased_words, self).__init__()
                # Ventral
                if init_model == None:
                    self.V1 = CORblock_Z(3, 64, kernel_size=7, stride=2)
                    self.V2 = CORblock_Z(64, 128)
                    self.V4 = CORblock_Z(128, 256)
                    self.IT = CORblock_Z(256, 512)
                if init_model != None:
                        self.V1 = init_model.V1
                        self.V2 = init_model.V2
                        self.V4 = init_model.V4
                        self.IT = init_model.IT

                # Decoder
                self.avgpool = nn.AdaptiveAvgPool2d(1)
                self.flatten = Flatten()
                self.linear = nn.Linear(n_hid, out_img+out_wrd)
                self.softmax = nn.Softmax()
                self.sigmoid = nn.Sigmoid()
                self.output = Identity()

                if init_model != None:
                    # Initializing the hidden-to-image linear network with weights taken from the previous model
                    self.linear.weight.data[:out_img, :] = init_model._modules['decoder']._modules['linear'].weight.data
                    self.linear.bias.data[:out_img] = init_model._modules['decoder']._modules['linear'].bias.data


        def forward(self, image, clip=False):
                # Image
                v1 = self.V1(image)
                v2 = self.V2(v1)
                v4 = self.V4(v2)
                it = self.IT(v4)

                # decoder
                h = self.avgpool(it)
                h = self.flatten(h)
                out = self.linear(h)
                if clip:
                        v1 = self.flatten(v1[:,:8,24:32,24:32].contiguous())
                        v2 = self.flatten(v2[:,:8,10:18,10:18].contiguous())
                        v4 = self.flatten(v4[:,:8,3:11,3:11].contiguous())
                        it = self.flatten(it[:,:11,:,:].contiguous())
                        return v1, v2, v4, it, h, out
                return self.flatten(v1), self.flatten(v2), self.flatten(v4), self.flatten(it), h, out
  
#"""
#------------------------------------------------
#------------------------------------------------
#Cornet_S models
#------------------------------------------------
#------------------------------------------------
#"""
#  
#          
#"""
#Basic block for CORnet S
#"""
#
#class CORblock_S(nn.Module):
#
#    scale = 4
#
#    def __init__(self, in_channels, out_channels, times=1):
#        super().__init__()
#
#        self.times = times
#
#        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#        self.skip = nn.Conv2d(out_channels, out_channels,
#                              kernel_size=1, stride=2, bias=False)
#        self.norm_skip = nn.BatchNorm2d(out_channels)
#
#        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale,
#                               kernel_size=1, bias=False)
#        self.nonlin1 = nn.ReLU()
#
#        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,
#                               kernel_size=3, stride=2, padding=1, bias=False)
#        self.nonlin2 = nn.ReLU()
#
#        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels,
#                               kernel_size=1, bias=False)
#        self.nonlin3 = nn.ReLU()
#
#        self.output = Identity()
#
#        for t in range(self.times):
#            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
#            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
#            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))
#
#    def forward(self, inp):
#        x = self.conv_input(inp)
#
#        for t in range(self.times):
#            if t == 0:
#                skip = self.norm_skip(self.skip(x))
#                self.conv2.stride = (2, 2)
#            else:
#                skip = x
#                self.conv2.stride = (1, 1)
#
#            x = self.conv1(x)
#            x = getattr(self, f'norm1_{t}')(x)
#            x = self.nonlin1(x)
#
#            x = self.conv2(x)
#            x = getattr(self, f'norm2_{t}')(x)
#            x = self.nonlin2(x)
#
#            x = self.conv3(x)
#            x = getattr(self, f'norm3_{t}')(x)
#
#            x += skip
#            x = self.nonlin3(x)
#            output = self.output(x)
#
#        return output
#
#
#def CORnet_S():
#    model = nn.Sequential(OrderedDict([
#        ('V1', nn.Sequential(OrderedDict([
#            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                            bias=False)),
#            ('norm1', nn.BatchNorm2d(64)),
#            ('nonlin1', nn.ReLU()),
#            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
#            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
#                            bias=False)),
#            ('norm2', nn.BatchNorm2d(64)),
#            ('nonlin2', nn.ReLU()),
#            ('output', Identity())
#        ]))),
#        ('V2', CORblock_S(64, 128, times=2)),
#        ('V4', CORblock_S(128, 256, times=4)),
#        ('IT', CORblock_S(256, 512, times=2)),
#        ('decoder', nn.Sequential(OrderedDict([
#            ('avgpool', nn.AdaptiveAvgPool2d(1)),
#            ('flatten', Flatten()),
#            ('linear', nn.Linear(512, 1000)),
#            ('output', Identity())
#        ])))
#    ]))
#
#    # weight initialization
#    for m in model.modules():
#        if isinstance(m, nn.Conv2d):
#            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#            m.weight.data.normal_(0, math.sqrt(2. / n))
#        # nn.Linear is missing here for no good reason
#        elif isinstance(m, nn.BatchNorm2d):
#            m.weight.data.fill_(1)
#            m.bias.data.zero_()
#
#    return model
#
#
## Easier model to use, straigthforwardly returns all the activations of each layer    
#class CORnet_S_tweak(nn.Module):
#   
#    def __init__(self, out_img=1000, out_wrd=0):
#        super(CORnet_S_tweak, self).__init__()
#
#        self.V1 = nn.Sequential(OrderedDict([
#            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                            bias=False)),
#            ('norm1', nn.BatchNorm2d(64)),
#            ('nonlin1', nn.ReLU()),
#            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
#            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
#                            bias=False)),
#            ('norm2', nn.BatchNorm2d(64)),
#            ('nonlin2', nn.ReLU()),
#            ('output', Identity())
#        ])))
#        self.V2 = CORblock_S(64, 128, times=2)
#        self.V4 = CORblock_S(128, 256, times=4)
#        self.IT = CORblock_S(256, 512, times=2)
#
#        # Decoder
#        self.avgpool = nn.AdaptiveAvgPool2d(1)
#        self.flatten = Flatten()
#        self.softmax = nn.Softmax()
#        self.sigmoid = nn.Sigmoid()
#        self.output = Identity()
#        self.decoder = nn.Sequential(OrderedDict([
#            ('linear', nn.Linear(512, out_img + out_wrd)),
#            ('output', Identity())
#            ]))
#
#    def forward(self, image):
#        # Image
#        v1 = self.V1(image)
#        v2 = self.V2(v1)
#        v4 = self.V4(v2)
#        it = self.IT(v4)
#
#        # decoder
#        h = self.avgpool(it)
#        h = self.flatten(h)
#        out = self.decoder(h)
#               
#        return self.flatten(v1), self.flatten(v2), self.flatten(v4), self.flatten(it), h, out
#        
### Simplified biased word model with easily accessible activations
#class CORNet_S_biased_words(nn.Module):
#
#    def __init__(self, in_img=512, out_img=1000, in_wrd=49, out_wrd=1000):
#        super(CORnet_S_biased_words, self).__init__()
#        
#        # Ventral
#        self.V1 = nn.Sequential(OrderedDict([
#            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                            bias=False)),
#            ('norm1', nn.BatchNorm2d(64)),
#            ('nonlin1', nn.ReLU()),
#            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
#            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
#                            bias=False)),
#            ('norm2', nn.BatchNorm2d(64)),
#            ('nonlin2', nn.ReLU()),
#            ('output', Identity())
#        ])))
#        self.V2 = CORblock_S(64, 128, times=2)
#        self.V4 = CORblock_S(128, 256, times=4)
#        self.IT = CORblock_S(256, 512, times=2)
#
#       # Decoder
#       self.avgpool = nn.AdaptiveAvgPool2d(1)
#       self.flatten = Flatten()
#       self.bilinear = Bi_linear(in_img=512, out_img=1000, in_wrd=49, out_wrd=1000)
#       self.softmax = nn.Softmax()
#       self.sigmoid = nn.Sigmoid()
#       self.output = Identity()
#
#       # Initializing the hidden-to-image linear network with weights taken from the previous model
#       self.bilinear.lin_img.weight.data = init_model._modules['decoder']._modules['linear'].weight.data
#       self.bilinear.lin_img.bias.data = init_model._modules['decoder']._modules['linear'].bias.data
#
#
#   def forward(self, image):
#       # Image
#       v1 = self.V1(image)
#       v2 = self.V2(v1)
#       v4 = self.V4(v2)
#       it = self.IT(v4)
#
#       # decoder
#       h = self.avgpool(it)
#       h = self.flatten(h)
#       out = self.bilinear(h)
#                
#       return self.flatten(v1), self.flatten(v2), self.flatten(v4), self.flatten(it), h, out
#
### Simplified non-biased word model with easily accessible activations
#class CORNet_S_nonbiased_words(nn.Module):
#
#   def __init__(self, init_model, n_hid = 512, out_img=1000, out_wrd=1000):
#       super(CORNet_S_nonbiased_words, self).__init__()
#       
#        # Ventral
#        self.V1 = nn.Sequential(OrderedDict([
#            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                            bias=False)),
#            ('norm1', nn.BatchNorm2d(64)),
#            ('nonlin1', nn.ReLU()),
#            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
#            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
#                            bias=False)),
#            ('norm2', nn.BatchNorm2d(64)),
#            ('nonlin2', nn.ReLU()),
#            ('output', Identity())
#        ])))
#        self.V2 = CORblock_S(64, 128, times=2)
#        self.V4 = CORblock_S(128, 256, times=4)
#        self.IT = CORblock_S(256, 512, times=2)
#
#       # Decoder
#       self.avgpool = nn.AdaptiveAvgPool2d(1)
#       self.flatten = Flatten()
#       self.linear = nn.Linear(n_hid, out_img+out_wrd)
#       self.softmax = nn.Softmax()
#       self.sigmoid = nn.Sigmoid()
#       self.output = Identity()
#
#       # Initializing the hidden-to-image linear network with weights taken from the previous model
#       self.linear.weight.data[:out_img, :] = init_model._modules['decoder']._modules['linear'].weight.data
#       self.linear.bias.data[:out_img, :] = init_model._modules['decoder']._modules['linear'].bias.data
#
#
#   def forward(self, image):
#       # Image
#       v1 = self.V1(image)
#       v2 = self.V2(v1)
#       v4 = self.V4(v2)
#       it = self.IT(v4)
#
#       # decoder
#       h = self.avgpool(it)
#       h = self.flatten(h)
#       out = self.linear(h)
#                
#       return self.flatten(v1), self.flatten(v2), self.flatten(v4), self.flatten(it), h, out
# 
