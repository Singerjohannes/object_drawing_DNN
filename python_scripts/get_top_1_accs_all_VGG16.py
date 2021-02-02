# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:43:09 2020

get top-1 accuracies for pre-trained or finetuned models

@author: Johannes
"""

#import stuff 

from PIL import Image
from collections import OrderedDict
import torch
import torchvision
import torchvision.models
from torch.utils import model_zoo
from torchvision import transforms
from os import listdir
import numpy as np
import os
import json

# load model 

filepath = 'F:/final_analysis/VGG_finetune/geirhos_vgg16_imagenetsketches_ft_conv5-1-33.pt'
model = torchvision.models.vgg16(pretrained=True) # set True and comment out the following 2 lines to get plain VGG16
checkpoint = torch.load(filepath)
model.load_state_dict(checkpoint)

# path for files mapping imagenet labels to wordnet ids and their hyponyms/hypernyms
idx_path = 'C:/Users/Johannes/Documents/Leipzig/Masterarbeit/final_scripts/python/analysis/imnet_mapping'

# setup image-paths

photo_path = "C:/Users/Johannes/Documents/Leipzig/Masterarbeit/final_stimuli/photos"
drawing_path = "C:/Users/Johannes/Documents/Leipzig/Masterarbeit/final_stimuli/drawings"
sketch_path = "C:/Users/Johannes/Documents/Leipzig/Masterarbeit/final_stimuli/sketches"

# load the images 

loaded_photos = [Image.open(photo_path + '/'+ filename) for filename in listdir(photo_path)]
    
loaded_drawings = [Image.open(drawing_path + '/'+filename) for filename in listdir(drawing_path)]

loaded_sketches = [Image.open(sketch_path + '/'+filename) for filename in listdir(sketch_path)]

# define the transforms 

val_transforms = transforms.Compose([
                                      transforms.Resize(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                    mean = [0.485, 0.456, 0.406], 
                                    std = [0.229, 0.224, 0.225])])   
                                    
                                    
#%% prepare batches 
    
photos_pre = [val_transforms(img) for img in loaded_photos]
photo_batch = [torch.unsqueeze(img, 0) for img in photos_pre]

drawings_pre = [val_transforms(img) for img in loaded_drawings]
drawing_batch = [torch.unsqueeze(img, 0) for img in drawings_pre]

sketches_pre = [val_transforms(img) for img in loaded_sketches]
sketch_batch = [torch.unsqueeze(img, 0) for img in sketches_pre]

# put all batches in one list for loop 

all_batches = [photo_batch, drawing_batch, sketch_batch]

#%% classify 

# setup imnet label to category mapping 

with open(os.path.join(idx_path, 'imagenet_class_index.json')) as f:
    idxs = json.load(f)


# bring model in evaluation mode 
  
model.eval()

# setup accuracy array

all_accs = []

# setup category names 

categories = os.listdir(photo_path)
categories = [txt.split('.')[0] for txt in categories]

for batch in all_batches:
    
    this_acc = []
    
    for idx, this_batch in enumerate(batch):
    
        out = model(this_batch)
        
        _, index = torch.max(out, 1)
    
        # load correct indices for given image 
        
        with open(os.path.join(idx_path, categories[idx]+ '_idxs.txt')) as f:
            this_idxs = [line.strip() for line in f.readlines()]
            this_idxs = [txt.replace('-', '') for txt in this_idxs]
            
        # compare mapped category indices and the output 
    
        this_acc.append(idxs[str(index.item())][0] in this_idxs)
    
    all_accs.append(this_acc)
    
# get the mean 
    
all_accs = [np.mean(acc) for acc in all_accs]

#%% save accuracies 
    
import pickle as pkl    
    
results_path = 'C:/Users/Johannes/Documents/Leipzig/Masterarbeit/final_results/VGG16_with_finetuning'

pkl.dump(all_accs, open(os.path.join(results_path,'top_1_accs_geirhos_vgg16_imagenetsketches_ft_conv5-1-33.pkl'), 'wb'))

#%% save for matlab 

import scipy.io as sio

all_accs = [np.mean(acc) for acc in all_accs]

sio.savemat(os.path.join(results_path, 'top_1_acc_regular_vgg16_imagenetsketches_ft_conv5-1.mat'), {'VGG16_accs': all_accs})

#%% classifiy with readable labels - for sanity checking 
  

with open('imagenet1000_clsidx_to_labels.txt') as f:
  classes = [line.strip() for line in f.readlines()]
  
  
model.eval()

#sanity check for manual inspection of labels 

for idx, batch in enumerate(photo_batch):

    out = model(batch)
    
    _, index = torch.max(out, 1)

    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    print(listdir(photo_path)[idx], classes[index[0]], percentage[index[0]].item())

#%% top 5 
    _, indices = torch.sort(out, descending=True)

    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    print([(classes[idx], percentage[idx].item()) for idx in indices[0][:5]])
