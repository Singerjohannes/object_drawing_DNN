# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:24:55 2020

@author: Johannes
"""

#import stuff 

from PIL import Image
from torchvision import transforms
import torch 
from os import listdir
import numpy as np
import os
import json

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
                                      transforms.Resize(256),
                                      transforms.CenterCrop(224),
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


#%% register the hooks for extracting features -VGG16   

def get_activation(name):
    def hook(model, input, output):
        this_activation = output.detach()
        this_activation = this_activation.flatten().cpu().numpy()
        activations[name] = this_activation
    return hook

# specify model.features.module for DataParallel features and model.features for Sequential features
    
for idx,layer in enumerate(model.features):
    layer.register_forward_hook(get_activation('Layer_'+ str(idx)))   

for idx,layer in enumerate(model.classifier):
    layer.register_forward_hook(get_activation('Classifier_'+ str(idx)))    


#%% extract the features for photos

all_photo_activations = []
  
for im_idx,img in enumerate(photo_batch):
    activations = {}
    output = model(img)
    all_photo_activations.append(activations)
    print('Photo Nr. ' + str(im_idx))
   
# and format 

relevant_layers = [4,9,16,23,30,0,3,6]

formatted_photo_activations = {}

for layer in relevant_layers:
    photo_activation_array=[]
    for img_act in all_photo_activations:
        if 'Layer_'+ str(layer) in all_photo_activations[0].keys() and layer in [4,9,16,23,30]:
            photo_activation_array.append(img_act['Layer_'+ str(layer)])
        elif 'Classifier_' + str(layer) in all_photo_activations[0].keys():
            photo_activation_array.append(img_act['Classifier_'+ str(layer)])
    formatted_photo_activations[layer] = np.array(photo_activation_array)
    
# update keys 
    
layer_names = ['Pool_1', 'Pool_2', 'Pool_3', 'Pool_4', 'Pool_5', 'Fc1', 'Fc2', 'Fc3']

for idx, old_layer_name in enumerate(relevant_layers):
    formatted_photo_activations[layer_names[idx]] = formatted_photo_activations.pop(old_layer_name)
    
# clean up memory 
    
del all_photo_activations 

# check activation sizes 

for act in formatted_photo_activations.values():
    
    print(act.shape)

#%% extract features for drawings 

all_drawing_activations = []

for im_idx,img in enumerate(drawing_batch):
    activations = {}
    output = model(img)
    all_drawing_activations.append(activations)
    print('Drawing Nr. ' + str(im_idx))

# and format 
    
formatted_drawing_activations = {}

for layer in relevant_layers:
    drawing_activation_array=[]
    for img_act in all_drawing_activations:
        if 'Layer_'+ str(layer) in all_drawing_activations[0].keys() and layer in [4,9,16,23,30]:
            drawing_activation_array.append(img_act['Layer_'+ str(layer)])
        elif 'Classifier_' + str(layer) in all_drawing_activations[0].keys():
            drawing_activation_array.append(img_act['Classifier_'+ str(layer)])
    formatted_drawing_activations[layer] = np.array(drawing_activation_array)
    
# update keys 
    
layer_names = ['Pool_1', 'Pool_2', 'Pool_3', 'Pool_4', 'Pool_5', 'Fc1', 'Fc2', 'Fc3']

for idx, old_layer_name in enumerate(relevant_layers):
    formatted_drawing_activations[layer_names[idx]] = formatted_drawing_activations.pop(old_layer_name)
    
# clean up memory 
    
del all_drawing_activations 

# check activation sizes 

for act in formatted_drawing_activations.values():
    
    print(act.shape)
    

#%% extract features for sketches 
    
all_sketch_activations = []

for im_idx,img in enumerate(sketch_batch):
    activations = {}
    output = model(img)
    all_sketch_activations.append(activations)
    print('Sketch Nr. ' + str(im_idx))
    
formatted_sketch_activations = {}

for layer in relevant_layers:
    sketch_activation_array=[]
    for img_act in all_sketch_activations:
        if 'Layer_'+ str(layer) in all_sketch_activations[0].keys() and layer in [4,9,16,23,30]:
            sketch_activation_array.append(img_act['Layer_'+ str(layer)])
        elif 'Classifier_' + str(layer) in all_sketch_activations[0].keys():
            sketch_activation_array.append(img_act['Classifier_'+ str(layer)])
    formatted_sketch_activations[layer] = np.array(sketch_activation_array)
    
# update keys 
    
layer_names = ['Pool_1', 'Pool_2', 'Pool_3', 'Pool_4', 'Pool_5', 'Fc1', 'Fc2', 'Fc3']

for idx, old_layer_name in enumerate(relevant_layers):
    formatted_sketch_activations[layer_names[idx]] = formatted_sketch_activations.pop(old_layer_name)
    
# clean up memory 
    
del all_sketch_activations 

# check activation sizes 

for act in formatted_sketch_activations.values():
    
    print(act.shape)

#%% save as matfiles 

import scipy.io as sio

savepath = "F:/final_analysis/"

sio.savemat(os.path.join(savepath, 'all_drawing_activations_VGG16_ft_classL.mat'), formatted_drawing_activations)

sio.savemat(os.path.join(savepath, 'all_photo_activations_VGG16_ft_classL.mat'), formatted_photo_activations)

sio.savemat(os.path.join(savepath, 'all_sketch_activations_VGG16_ft_classL.mat'), formatted_sketch_activations)