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

# specify name of model 

net_name = 'VGG16' #either VGG16, VGG16_SIN or VGG16_FT

is_ft = 0 #  specify whether you want to work with the finetuned or plain VGG model
is_stylized = 0 #specify whether you want to work with the stylized or plain VGG model

# specify where results should be saved 

results_path = '/object_drawing_DNN/results'

# load model 
if is_ft: 
    filepath = '/object_drawing_DNN/models/'+ net_name+ '.pt' # specify path for the weights of the finetuned model
    model = torchvision.models.vgg16(pretrained=False)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint)
elif is_stylized: 
    assert("model" in locals()), "Please load the stylized VGG16 with the load_geirhos_model.py script"
elif not is_ft:
    model = torchvision.models.vgg16(pretrained=True)
    
print(model)

# path for files mapping imagenet labels to wordnet ids and their hyponyms/hypernyms
idx_path = '/object_drawing_DNN/python_scripts/imnet_mapping'

# setup image-paths

photo_path = "/object_drawing_DNN/stimuli/photos"
drawing_path = "/object_drawing_DNN/stimuli/drawings"
sketch_path = "/object_drawing_DNN/stimuli/sketches"

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

#%% save accuracies 
    
import pickle as pkl    

pkl.dump(all_accs, open(os.path.join(results_path,'top_1_accs_'+net_name+'.pkl'), 'wb'))

#%% save for matlab 

import scipy.io as sio

all_accs = [np.mean(acc) for acc in all_accs]

sio.savemat(os.path.join(results_path, 'top_1_accs_'+net_name+'.mat'), {'all_accs': all_accs})
