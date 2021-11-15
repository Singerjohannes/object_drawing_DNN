"""
Created on Wed Jun 10 15:24:55 2020

@author: Johannes Singer
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


def load_images(imgpath):
    return [Image.open(imgpath + '/' + fn) for fn in listdir(imgpath)]

def prep_batches(loaded_images):
    images_pre  = [val_transforms(img) for img in loaded_images]
    images_batch = [torch.unsqueeze(img, 0) for img in images_pre]  
    return images_batch


# specify path were activations should be saved to and name of network for saving
savepath = '/object_drawing_DNN/'
net_name = 'AlexNet'  

# load model 
model = torchvision.models.alexnet(pretrained=True)
    
print(model)
        
idx_path = '/object_drawing_DNN/python_scripts/imnet_mapping'

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

# register the hooks for extracting features -VGG16   
def get_activation(name):
    def hook(model, input, output):
        this_activation = output.detach()
        this_activation = this_activation.flatten().cpu().numpy()
        activations[name] = this_activation
    return hook

for idx,layer in enumerate(model.features):
    print(layer)
    layer.register_forward_hook(get_activation('Layer_'+ str(idx)))   

for idx,layer in enumerate(model.classifier):
    print(layer)
    layer.register_forward_hook(get_activation('Classifier_'+ str(idx))) 

# extract the features for all batches one after another
cond = ['photo' ,'drawing', 'sketch']    
relevant_layers = [2,5,12,2,5] 


# set model to evaluation mode 
model.eval()
      
for cond_idx, batch in enumerate(all_batches):

    these_activations = []

    for im_idx,img in enumerate(batch):
        with torch.no_grad():
            activations = {}
            output = model(img)
            these_activations.append(activations)
            print('Image Nr. ' + str(im_idx))
       
    # and format 
    formatted_activations = {}

    for layer_idx,layer in enumerate(relevant_layers):
        activation_array = []
        for img_act in these_activations:
            if 'Layer_'+ str(layer) in these_activations[0].keys() and layer_idx<3:
                activation_array.append(img_act['Layer_' + str(layer)])
            elif 'Classifier_' + str(layer) in these_activations[0].keys():
                activation_array.append(img_act['Classifier_' + str(layer)])

        formatted_activations[layer_idx] = np.array(activation_array)
        
    # update keys 
    layer_names = ['Pool_1', 'Pool_2', 'Pool_3', 'Fc1', 'Fc2']

    for idx, old_layer_name in enumerate(relevant_layers):
        formatted_activations[layer_names[idx]] = formatted_activations.pop(idx)

    # check activation sizes 
    for act in formatted_activations.values():
        print(act.shape)
    
    # save the formatted activations
    sio.savemat(os.path.join(savepath, 'all_' + cond[cond_idx] + '_activations_' + net_name + '.mat'), formatted_activations)
