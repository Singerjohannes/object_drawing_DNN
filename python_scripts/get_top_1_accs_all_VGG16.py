"""
Created on Tue Aug 18 14:43:09 2020

get top-1 accuracies for pre-trained or finetuned models

@author: Johannes
"""

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
import pickle as pkl    



def load_images(imgpath):
    return [Image.open(imgpath + '/' + fn) for fn in listdir(imgpath)]

def prep_batches(loaded_images):
    images_pre  = [val_transforms(img) for img in loaded_images]
    images_batch = [torch.unsqueeze(img, 0) for img in images_pre]  
    return images_batch


# specify name of model 
net_name = 'VGG16' #either VGG16, VGG16_SIN or VGG16_FT

is_ft = False       # specify whether you want to work with the finetuned or plain VGG model
is_stylized = False # specify whether you want to work with the stylized or plain VGG model

# specify where results should be saved 
results_path = '/object_drawing_DNN/results'

# load model 
if is_ft: 
    filepath = '/object_drawing_DNN/models/'+ net_name+ '.pt' # specify path for the weights of the finetuned model
    assert os.path.exists(filepath), "Please download the finetuned VGG model yourself from the following link and save it locally: https://datashare.rzg.mpg.de/s/vUiWZ3oIV3QikZH"
    model = torchvision.models.vgg16(pretrained=False)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint)
elif is_stylized: 
    filepath = "/object_drawing_DNN/models/vgg16_train_60_epochs_lr0.01-6c6fcc9f.pth.tar" 
    assert os.path.exists(filepath), "Please download the VGG model yourself from the following link and save it locally: https://drive.google.com/drive/folders/1A0vUWyU6fTuc-xWgwQQeBvzbwi6geYQK (too large to be downloaded automatically like the other models)"
    model = torchvision.models.vgg16(pretrained=False)
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["state_dict"])

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


# classify
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

with torch.no_grad():
    for batch in all_batches:

        this_acc = []

        for idx, this_batch in enumerate(batch):

            out = model(this_batch)

            _, index = torch.max(out, 1)

            # load correct indices for given image 
            with open(os.path.join(idx_path, categories[idx] + '_idxs.txt')) as f:
                this_idxs = [line.strip() for line in f.readlines()]
                this_idxs = [txt.replace('-', '') for txt in this_idxs]

            # compare mapped category indices and the output
            this_acc.append(idxs[str(index.item())][0] in this_idxs)

        all_accs.append(this_acc)


# save accuracies 
pkl.dump(all_accs, open(os.path.join(results_path,'top_1_accs_'+net_name+'.pkl'), 'wb'))

# save for matlab 
all_accs = [np.mean(acc) for acc in all_accs]
sio.savemat(os.path.join(results_path, 'top_1_accs_'+net_name+'.mat'), {'all_accs': all_accs})


