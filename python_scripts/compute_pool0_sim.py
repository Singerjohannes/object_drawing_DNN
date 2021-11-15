#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 17:05:59 2021

@author: johannessinger
"""

from PIL import Image
from collections import OrderedDict
import torch
import torchvision
import torchvision.models
from torch.utils import model_zoo as model_zoo
from torchvision import transforms
from os import listdir
import numpy as np
import os
import json
import scipy.io as sio
from scipy.stats import spearmanr, pearsonr, rankdata



def load_images(imgpath):
    return [Image.open(imgpath + '/' + fn) for fn in listdir(imgpath)]

def prep_batches(loaded_images):
    images_pre  = [val_transforms(img) for img in loaded_images]
    images_batch = [torch.unsqueeze(img, 0) for img in images_pre]  
    return images_batch


# specify path were activations should be saved to and name of network for saving
savepath = '/object_drawing_DNN/data'
        
# setup image-paths
photo_path = "/object_drawing_DNN/stimuli/photos"
drawing_path = "/object_drawing_DNN/stimuli/drawings"
sketch_path = "/object_drawing_DNN/stimuli/sketches"


# load the images 
loaded_photos   = load_images(photo_path)
loaded_drawings = load_images(drawing_path)
loaded_sketches = load_images(sketch_path)


# define the transforms 
val_transforms = transforms.Compose([
                                      transforms.Resize(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                    mean = [0.485, 0.456, 0.406], 
                                    std = [0.229, 0.224, 0.225])])     
               
# prepare batches and put all batches in one list for loop 
all_batches = [prep_batches(loaded_photos), prep_batches(loaded_drawings), prep_batches(loaded_sketches)]

all_rdvs = []
all_rdms = []

#loop over batches and compute RDMs
for batch in all_batches: 
    
    
    batch_array = torch.stack([img.flatten() for img in batch])
    rdm = 1 - np.corrcoef(batch_array.T, rowvar=False)
    np.fill_diagonal(rdm, 0) #accounts for minimal deviations from 0 
    rdv = rdm[np.triu_indices(rdm.shape[0], k=1)].reshape(-1,1)
    all_rdvs.append(rdv)
    all_rdms.append(rdm)
    
# calculate similarity based on raw pixel values
photo_drawing_sim_pool0 = spearmanr(all_rdvs[0], all_rdvs[1])
photo_sketch_sim_pool0 = spearmanr(all_rdvs[0], all_rdvs[2])
drawing_sketch_sim_pool0 = spearmanr(all_rdvs[1], all_rdvs[2])

# save RDMs based on raw pixel values 
input_RDMs = dict()
input_RDMs['photo_RDM'] = all_rdms[0]
input_RDMs['drawing_RDM'] = all_rdms[1]
input_RDMs['sketch_RDM'] = all_rdms[2]

sio.savemat(os.path.join(savepath, 'input_RDMs.mat'), input_RDMs)



    