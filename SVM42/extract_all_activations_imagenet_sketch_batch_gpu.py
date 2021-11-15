# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:43:47 2020

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

savepath = "/object_drawing_DNN/SVM_42/activations"

# load model 

model = torchvision.models.vgg16(pretrained=True) 

# setup image-paths

sketch_path = "/object_drawing_DNN/SVM_42/sketch" #this folder should contain all images from the ImageNet-Sketch dataset (https://github.com/HaohanWang/ImageNet-Sketch)

categories = sorted(listdir(sketch_path))

def set_device(): 
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
    print("GPU not enabled.")
  else:
    print("GPU is enabled.")

  return device

DEVICE = set_device()

# setup transforms

# define the transforms 
val_transforms = transforms.Compose([
                                      transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                    mean = [0.485, 0.456, 0.406], 
                                    std = [0.229, 0.224, 0.225])])

# register the hooks for extracting features -VGG16   
def get_activation(name):
    def hook(model, input, output):
        this_activation = output.detach()
        this_activation = this_activation.flatten(start_dim = 1).cpu().numpy()
        activations[name] = this_activation
    return hook

relevant_pool_layers = [4,9,16,23,30]
relevant_fc_layers = [0,3]

for idx,layer in enumerate(model.features):
    if idx in relevant_pool_layers:
        layer.register_forward_hook(get_activation('Layer_'+ str(idx)))   

for idx,layer in enumerate(model.classifier):
    if idx in relevant_fc_layers:
        layer.register_forward_hook(get_activation('Classifier_'+ str(idx))) 

# start with extracting activation

# set model to evaluation mode 
model.eval()
model.to(device=DEVICE)

for cat in categories: 

    # load the images 

    loaded_sketches = [Image.open(os.path.join(sketch_path, cat, filename)).convert('RGB') for filename in sorted(listdir(os.path.join(sketch_path, cat)))]

    sketches_pre = [val_transforms(img) for img in loaded_sketches]

    #sketch_batch = [torch.unsqueeze(img, 0) for img in sketches_pre]
    
    sketch_batch = torch.stack(sketches_pre)
      
    
    with torch.no_grad():
        activations = {}
        output = model(sketch_batch.to(device=DEVICE))
        print('Batch idx ' + cat)
               
    # update keys 
    layer_names = ['Pool_1', 'Pool_2', 'Pool_3', 'Pool_4', 'Pool_5', 'Fc1', 'Fc2']
    
    formatted_activations = dict()

    for idx, old_layer_name in enumerate(activations.keys()):
        formatted_activations[layer_names[idx]] = activations[old_layer_name]

    # check activation sizes 
    for act in activations.values():
        print(act.shape)
    
    # save the formatted activations
    sio.savemat(os.path.join(savepath, cat + '_activations.mat'), formatted_activations)

