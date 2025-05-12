from __future__ import print_function
from collections import namedtuple
from pathlib import Path
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module
from torchvision import transforms as trans
import numpy as np
import os
import torch
import time
import math

class Config:
    def __init__(self):
        self.net_depth = 50
        self.drop_ratio = 0.4
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.threshold = 82.0
        self.facebank_path = os.path.join(Path('data'), 'facebank')

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0 ,bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0 ,bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride ,bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3,3), (1,1),1 ,bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3,3), stride, 1 ,bias=False),
            BatchNorm2d(depth),
            SEModule(depth,16)
            )
    def forward(self,x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class Backbone(Module):
    def __init__(self, num_layers, drop_ratio):
        super(Backbone, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        blocks = get_blocks(num_layers)

        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer = Sequential(BatchNorm2d(512),
                                       Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * 7 * 7, 512),
                                       BatchNorm1d(512))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    bottleneck_IR_SE(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)

def l2_norm(input, axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''

def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]

def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units = 3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks

class FaceEngine(object):
    def __init__(self):
        self.conf = Config()
        self.model = Backbone(self.conf.net_depth, self.conf.drop_ratio).to(self.conf.device)
        self.threshold = self.conf.threshold

    def load_facebank(self):
        embeddings = torch.load(os.path.join(self.conf.facebank_path, "facebank.pth"))
        names = np.load(os.path.join(self.conf.facebank_path, "names.npy"))
        return embeddings, names
    
    def load_state(self, path_str):
        self.model.load_state_dict(torch.load(path_str, map_location=self.conf.device))
        self.model.eval()

    def infer(self, faces, target_embs, distance_method=0):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        '''      
        with torch.no_grad():
            if (distance_method == 1):
                embs = []
                for img in faces:
                    extract_start_time = time.time()
                    embs.append(self.model(self.conf.transform(img).to(self.conf.device).unsqueeze(0)))
                    print('Feature extraction time: {0} ms'.format(int((time.time() - extract_start_time) * 1000)))
                distance_start_time = time.time() 
                source_embs = torch.cat(embs)                
                diff = source_embs.unsqueeze(-1) - target_embs.to(self.conf.device).transpose(1, 0).unsqueeze(0)
                dist = torch.sum(torch.pow(diff, 2), dim=1)
                minimum, min_idx = torch.min(dist, dim=1)
                print('Distance calculation time (1xN): {0} ms'.format(int((time.time() - distance_start_time) * 1000)))   
            else:
                target_embs = target_embs.cpu().numpy()
                output_size = len(faces)
                minimum = np.zeros(shape=(output_size))
                min_idx = np.zeros(shape=(output_size), dtype= np.int32)

                for i, img in enumerate(faces):
                    extract_start_time = time.time()
                    emb = self.model(self.conf.transform(img).to(self.conf.device).unsqueeze(0))
                    print('Feature extraction time: {0} ms'.format(int((time.time() - extract_start_time) * 1000)))  
                    distance_start_time = time.time()
                    emb = emb.cpu().numpy()
                    dist = squared_euclidean_distances(emb, target_embs)
                    min_idx_temp = np.argmin(dist)            
                    minimum[i] = dist[min_idx_temp]         
                    min_idx[i] = min_idx_temp
                    print('Distance calculation time (1xN): {0} ms'.format(int((time.time() - distance_start_time) * 1000)))   
            minimum = (1.0 - (minimum / (2 * math.pi))) * 100  # convert to percentage
            min_idx[minimum < self.threshold] = -1  # if no match, set idx to -1
            
            return min_idx, minimum

    def extract(self, faces):
        '''
        faces : list of PIL Image
        '''      
        with torch.no_grad():
            embs = []
            for img in faces:
                emb = self.model(self.conf.transform(img).to(self.conf.device).unsqueeze(0))
                emb = emb.cpu().numpy()
                embs.append(emb[0])
        return embs

    def extract_single(self, face):
        '''
        face : single PIL Image
        '''      
        with torch.no_grad():
            emb =  self.model(self.conf.transform(face).to(self.conf.device).unsqueeze(0))
            return emb.cpu().numpy()
        
def squared_row_norms(X):
    # From http://stackoverflow.com/q/19094441/166749
    return np.einsum('ij,ij->i', X, X)

def squared_euclidean_distances(data, vec):
    data2 = squared_row_norms(data)
    vec2 = squared_row_norms(vec)
    d = np.dot(data, vec.T).ravel()
    d *= -2
    d += data2
    d += vec2
    return d