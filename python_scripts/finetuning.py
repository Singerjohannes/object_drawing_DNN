import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import make_grid
import torchvision as tv
import torch.optim as optim
import shutil
from collections import OrderedDict
from PIL import Image

from torch.utils.data import random_split, Dataset, DataLoader

import string as s

import math, random
import matplotlib.pyplot as plt
import time
import os, sys
import copy
from tqdm import tqdm

from scipy.io import savemat


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    


def plot_train_and_val_loss(accuracies, fext):
    plt.figure()
    plt.plot( accuracies['val'], label='val' )
    plt.plot( accuracies['train'], label='train' )
    plt.xlabel('epochs') ; plt.ylabel('accuracy')
    plt.ylim([0.0,1.05])
    plt.legend()
    plt.savefig(fext + 'accuracies.png', bbox_inches='tight')


def train_model(model, dataloaders, criterion, optimizer, model_name, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    accuracies = { key:[] for key in dataloaders.keys() }

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in dataloaders.keys():
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]): 

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            accuracies[phase].append(epoch_acc)

            # store (deep copy) the model if improvement on validation set: 
            if (phase == 'val' or len(dataloaders.keys())==1) and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                save_name = model_name + '-' + str(epoch) + '.pt'
                torch.save(best_model_wts, os.path.join(save_name))

        if len(dataloaders.keys())>1: 
            # plot train and val loss
            plot_train_and_val_loss(accuracies, "current_")

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    if type(best_acc) != 'float':
        best_acc = best_acc.item()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, accuracies, best_acc



def get_vgg16_model(ft_start_layer):

    ft_start_layers = { 'vgg16_imagenetsketches_ft_3denseL':26, 
                        'vgg16_imagenetsketches_ft_classL':30, 
                        'vgg16_imagenetsketches_ft_conv5-1':20, 
                        'vgg16_imagenetsketches_ft_conv5-2':22, 
                        'vgg16_imagenetsketches_ft_conv5-3':24, 
                      }

    ft_from_param = ft_start_layers[ft_start_layer] 

    vgg16model = torchvision.models.vgg16(pretrained=True, progress=True)

    for pi, (name, param) in enumerate(vgg16model.named_parameters()): 
        #print(pi, name)
        if pi >= ft_from_param:
            param.requires_grad = True    # train layers >= ft_start_layer
        else: 
            param.requires_grad = False   # freeze layers < ft_start_layer

    vgg16model = vgg16model.to(device)

    print("The following list of parameters will be finetuned: ")
    params_to_update = []
    for name,param in vgg16model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)

    return vgg16model, params_to_update



if __name__=="__main__": 

    n_batch = 8

    debug = False

    n_epochs = 40 if not debug else 1
    n_folds = 1 if not debug else 1

    learning_rate = 0.001
    moment = 0.7
    
    fext = ''  # optional file name extension

    data_dir = './imagenetsketch'
    # assumes 1000 classes in directories saved in the same structure as in ILSVRC
    # download from: https://github.com/HaohanWang/ImageNet-Sketch

    ft_start_layer = 'vgg16_imagenetsketches_ft_conv5-1'

    data_transform = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ]) # Note: ImageNet-Stylized also used standard ImageNet preprocessing values

    criterion = nn.CrossEntropyLoss()

    print("Initializing Datasets and Dataloaders...")

    imagenetsketches_data = datasets.ImageFolder(data_dir, transform=data_transform)
    # len: 50889

    vgg16model, params_to_update = get_vgg16_model(ft_start_layer)

    # Train on all data with best params
    optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=moment )

    sketches_train_data, sketches_val_data = random_split(imagenetsketches_data, [48889, 2000])

    dataloaders = {}

    dataloaders['train'] = torch.utils.data.DataLoader( sketches_train_data, 
                                                        batch_size=n_batch, 
                                                        shuffle=True, num_workers=4)

    dataloaders['val']   = torch.utils.data.DataLoader( sketches_val_data, 
                                                        batch_size=1, 
                                                        shuffle=False, num_workers=4)

    # Train and evaluate
    vgg16model_final, accuracies_final, best_acc = train_model( vgg16model, dataloaders, criterion, 
                                                                optimizer, fext+ft_start_layer, num_epochs=n_epochs)

    # Save the model
    print("Saving model trained on all data")

    fext = fext + '_lr'+str(learning_rate) + '_mom'+str(moment) + '_'

    plot_train_and_val_loss(accuracies_final, fext)
    save_name = fext + ft_start_layer + ".pt"
    torch.save(vgg16model_final.state_dict(), save_name)

