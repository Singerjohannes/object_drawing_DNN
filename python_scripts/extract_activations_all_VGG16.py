# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:24:55 2020

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
import scipy.io as sio

# specify path were activations should be saved to and name of network for saving

savepath = "F:/final_analysis/final"
net_name = 'VGG16_SIN'

# load model 

filepath = 'F:/final_analysis/VGG_finetune/'+ net_name+ '.pt'
model = torchvision.models.vgg16(pretrained=True) # set True and comment out the following 2 lines to get plain VGG16
checkpoint = torch.load(filepath)
model.load_state_dict(checkpoint)

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


#%% sancheck the preprocessed images 

import matplotlib.pyplot as plt

img = torch.squeeze(sketch_batch[38])
img = img.numpy().transpose(1,2,0)
plt.imshow(img)
plt.show()


#%% register the hooks for extracting features -VGG16   

def get_activation(name):
    def hook(model, input, output):
        this_activation = output.detach()
        this_activation = this_activation.flatten().cpu().numpy()
        activations[name] = this_activation
    return hook

# specify model.features.module for DataParallel features and model.features for Sequential features
    
for idx,layer in enumerate(model.features.module):
    layer.register_forward_hook(get_activation('Layer_'+ str(idx)))   

for idx,layer in enumerate(model.classifier):
    layer.register_forward_hook(get_activation('Classifier_'+ str(idx)))    


#%% extract the features for all batches one after another

cond = ['photo', 'drawing', 'sketch']    
relevant_layers = [4,9,16,23,30,0,3,6] 
   
    
for cond_idx, batch in enumerate(all_batches):

    these_activations = []
      
    for im_idx,img in enumerate(batch):
        activations = {}
        output = model(img)
        these_activations.append(activations)
        print('Image Nr. ' + str(im_idx))
       
    # and format 
    
    formatted_activations = {}
    
    for layer in relevant_layers:
        activation_array=[]
        for img_act in these_activations:
            if 'Layer_'+ str(layer) in these_activations[0].keys() and layer in [4,9,16,23,30]:
                activation_array.append(img_act['Layer_'+ str(layer)])
            elif 'Classifier_' + str(layer) in these_activations[0].keys():
                activation_array.append(img_act['Classifier_'+ str(layer)])
        formatted_activations[layer] = np.array(activation_array)
        
    # update keys 
        
    layer_names = ['Pool_1', 'Pool_2', 'Pool_3', 'Pool_4', 'Pool_5', 'Fc1', 'Fc2', 'Fc3']
    
    for idx, old_layer_name in enumerate(relevant_layers):
        formatted_activations[layer_names[idx]] = formatted_activations.pop(old_layer_name)
    
    # check activation sizes 
    
    for act in formatted_activations.values():
        
        print(act.shape)
    
    # save the formatted activations
        
    sio.savemat(os.path.join(savepath, 'all_'+ cond[cond_idx]+ '_activations_'+ net_name + '.mat'), formatted_activations)