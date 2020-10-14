# -*- coding: utf-8 -*-
"""
Created on Fri May 15 17:24:32 2020

get accuracies for the outputs of the networks B, BL, BLT only for the imagenet stimuli 

@author: Johannes
"""
# setup packages and paths 

import os 
import numpy as np   

data_path = 'F:/martin-hebart.com/data_from_tim_BLT/' # path for cluster ~/modelling/martin-hebart.com/data_from_tim_BLT/activations

# import ecoset_categories

ecoset_categories = open(os.path.join(data_path, 'ecoset_categories.txt'), 'r')

ecoset_categories = ecoset_categories.read().splitlines()

# split categories into name and number 

ecoset_names = []

for file in ecoset_categories:
    
    number, name = file.split('_') 
    
    ecoset_names.append(name)
    
# convert names to np array 
    
name_array = np.array(ecoset_names)

#%% get list of names that are in imagenet 

imagenet_files = 'C:/Users/Johannes/Documents/Leipzig/Masterarbeit/final_stimuli/photos'

imagenet_names = []

for filename in os.listdir(imagenet_files):
    
    name, end = filename.split('.')
    imagenet_names.append(name)

    
#%% check if category names are in ecoset names 
    

idxs = []
for filename in os.listdir(imagenet_files):
    
    name, end = filename.split('.')
    idxs.append(name in ecoset_names)
    
# change the not matching names in ecoset 
    
ecoset_names[ecoset_names.index('lobster')] = 'crayfish'
    

#%% get output for B - top 1 

data_path = 'F:/martin-hebart.com/data_from_tim_BLT/activations' # path for cluster ~/modelling/martin-hebart.com/data_from_tim_BLT/activations


network = ['B']

conds = ['photos', 'drawings', 'sketches']

accs_B = dict()

for cd in conds: 
    
    this_output = []
    
    for filename in os.listdir(os.path.join(data_path, 'B', cd)):
            if len(filename.split('_')) > 4:
                net ,layer, layernr, cond, obj = filename.split('_')
            else:
                output = np.load(os.path.join(data_path, 'B', cd, filename)).flatten()
                this_label_ind = int(np.where(output == max(output))[0])
                this_label = ecoset_names[this_label_ind]
                net ,_, cond, obj = filename.split('_')
                obj,_ = obj.split('.')
                if obj not in imagenet_names:
                    continue 
                this_acc = this_label == obj
                this_output.append(this_acc)
            accs_B[cd] = this_output

# get mean accuracies 
            
acc_photo_B = np.mean(accs_B['photos'])
acc_drawings_B = np.mean(accs_B['drawings'])
acc_sketches_B = np.mean(accs_B['sketches'])

#%% get output for BL - top 1
    
network = ['BL']

conds = ['photos', 'drawings', 'sketches']

accs_BL = dict()

for cd in conds: 
    
    this_output = []
    
    for filename in os.listdir(os.path.join(data_path, 'BL', cd)):
            if len(filename.split('_')) == 4:
                output = np.load(os.path.join(data_path, 'BL', cd, filename)).flatten()
                this_label_ind = int(np.where(output[-565:] == max(output[-565:]))[0])
                this_label = ecoset_names[this_label_ind]
                net ,_, cond, obj = filename.split('_')
                obj,_ = obj.split('.')
                if obj not in imagenet_names:
                    continue 
                this_acc = this_label == obj
                this_output.append(this_acc)
            accs_BL[cd] = this_output

# get mean accuracies 
            
acc_photo_BL = np.mean(accs_BL['photos'])
acc_drawings_BL = np.mean(accs_BL['drawings'])
acc_sketches_BL = np.mean(accs_BL['sketches'])


#%% get output for BLT - top 1 
    
network = ['BLT']

conds = ['photos', 'drawings', 'sketches']

accs_BLT = dict()

for cd in conds: 
    
    this_output = []
    
    for filename in os.listdir(os.path.join(data_path, 'BLT', cd)):
            if len(filename.split('_')) == 4:
                output = np.load(os.path.join(data_path, 'BLT', cd, filename)).flatten()
                this_label_ind = int(np.where(output[-565:] == max(output[-565:]))[0])
                this_label = ecoset_names[this_label_ind]
                net ,_, cond, obj = filename.split('_')
                obj,_ = obj.split('.')
                if obj not in imagenet_names:
                    continue 
                this_acc = this_label == obj
                this_output.append(this_acc)
            accs_BLT[cd] = this_output

# get mean accuracies 
            
acc_photo_BLT = np.mean(accs_BLT['photos'])
acc_drawings_BLT = np.mean(accs_BLT['drawings'])
acc_sketches_BLT = np.mean(accs_BLT['sketches'])


#%% convert accs to np array and save

import pickle

save_path = 'C:/Users/Johannes/Documents/Leipzig/Masterarbeit/final_results/recurrent_nets'

all_accs = np.array([acc_photo_B, acc_photo_BL, acc_photo_BLT,
                    acc_drawings_B, acc_drawings_BL, acc_drawings_BLT,
                    acc_sketches_B, acc_sketches_BL, acc_sketches_BLT])
#all_accs= np.reshape(all_accs, (3,3))

with open(os.path.join(save_path, 'top_1_accs_final_recurrent_nets.pkl'), 'wb') as output:
    pickle.dump(all_accs, output)