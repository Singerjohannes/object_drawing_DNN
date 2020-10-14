# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:10:22 2020

compute stats for the comparison of top_1 accuracies of different VGG16 networks 

@author: Johannes


"""

#%% import stuff 

import numpy as np 
from matplotlib import pyplot as plt 
import seaborn as sns
import pandas as pd 
import os 
import pickle 

# load data for the different networks 

results_path = 'C:/Users/Johannes/Documents/Leipzig/Masterarbeit/final_results/VGG16_with_without_SIN'

accs = pickle.load(open(os.path.join(results_path, 'top_1_accs_VGG16.pkl'),'rb'))

accs_SIN = pickle.load(open(os.path.join(results_path, 'top_1_accs_VGG16_SIN.pkl'),'rb'))

results_path = 'C:/Users/Johannes/Documents/Leipzig/Masterarbeit/final_results/VGG16_with_finetuning'

accs_ft = pickle.load(open(os.path.join(results_path, 'top_1_accs_regular_vgg16_imagenetsketches_ft_conv5-1.pkl'),'rb'))


#%% create contingency table for the networks 

from statsmodels.stats.contingency_tables import mcnemar

from statsmodels.stats.multitest import multipletests

# function for contingency table 

def run_mcnemar(acc_1, acc_2, n_stim):
    
    correct_1 = sum(acc_1)
    incorrect_1 = n_stim-sum(acc_1)
    correct_2 = sum(acc_2)
    incorrect_2 = n_stim-sum(acc_2)
    table = ([correct_1+correct_2, correct_1+incorrect_2],
             [incorrect_1 + correct_2, incorrect_1+incorrect_2])
    
    stat = mcnemar(table, exact=False)
    # if stat.pvalue <= 0.05/18:
    #     print('Its significant!')
    # else: print('Not today')
    return stat


#%% stats for VGG16 with finetuning 
    
stat_photo_drawing_ft = run_mcnemar(accs_ft[0], accs_ft[1], 42)
stat_photo_sketch_ft = run_mcnemar(accs_ft[0], accs_ft[2], 42)
stat_drawing_sketch_ft = run_mcnemar(accs_ft[1], accs_ft[2], 42)

print('Photo vs. Drawing VGG16 with finetuning ', format(stat_photo_drawing_ft.pvalue))
print('Photo vs. Sketch VGG16 with finetuning ', format(stat_photo_sketch_ft.pvalue))
print('Drawing vs. Sketch VGG16 with finetuning ', format(stat_drawing_sketch_ft.pvalue))

fdr_corrected_ft = multipletests([stat_photo_drawing_ft.pvalue, stat_photo_sketch_ft.pvalue, stat_drawing_sketch_ft.pvalue], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)


#%% stats for VGG16 

stat_photo_drawing = run_mcnemar(accs[0], accs[1], 42)
stat_photo_sketch = run_mcnemar(accs[0], accs[2], 42)
stat_drawing_sketch = run_mcnemar(accs[1], accs[2], 42)

print('Photo vs. Drawing VGG16 ', format(stat_photo_drawing.pvalue, '.8f'))
print('Photo vs. Sketch VGG16 ', format(stat_photo_sketch.pvalue, '.8f'))
print('Drawing vs. Sketch VGG16 ', format(stat_drawing_sketch.pvalue, '.8f'))

fdr_corrected_IN = multipletests([stat_photo_drawing.pvalue, stat_photo_sketch.pvalue, stat_drawing_sketch.pvalue], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
  

#%% stats for VGG16 SIN 

stat_photo_drawing_SIN = run_mcnemar(accs_SIN[0], accs_SIN[1], 42)
stat_photo_sketch_SIN = run_mcnemar(accs_SIN[0], accs_SIN[2], 42)
stat_drawing_sketch_SIN = run_mcnemar(accs_SIN[1], accs_SIN[2], 42)

print('Photo vs. Drawing VGG16 SIN ', format(stat_photo_drawing_SIN.pvalue, '.8f'))
print('Photo vs. Sketch VGG16 SIN ', format(stat_photo_sketch_SIN.pvalue, '.8f'))
print('Drawing vs. Sketch VGG16 SIN ', format(stat_drawing_sketch_SIN.pvalue, '.8f')) 

fdr_corrected_SIN = multipletests([stat_photo_drawing_SIN.pvalue, stat_photo_sketch_SIN.pvalue, stat_drawing_sketch_SIN.pvalue], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)

#%% stats for comparison between the VGG16 with and without finetuning 

stat_photo_VGG16_with_without_ft = run_mcnemar(accs[0], accs_ft[0], 42)
stat_photo_VGG16_with_without_SIN = run_mcnemar(accs[0], accs_SIN[0], 42)
stat_photo_VGG16_SIN_ft = run_mcnemar(accs_ft[0], accs_SIN[0], 42)


stat_drawing_VGG16_with_without_ft  = run_mcnemar(accs[1], accs_ft[1], 42)
stat_drawing_VGG16_with_without_SIN  = run_mcnemar(accs[1], accs_SIN[1], 42)
stat_drawing_VGG16_SIN_ft = run_mcnemar(accs_ft[1], accs_SIN[1], 42)

stat_sketch_VGG16_with_without_ft  = run_mcnemar(accs[2], accs_ft[2], 42)
stat_sketch_VGG16_with_without_SIN  = run_mcnemar(accs[2], accs_SIN[2], 42)
stat_sketch_VGG16_SIN_ft = run_mcnemar(accs_ft[2], accs_SIN[2], 42)


print('Photo - VGG16 before vs. after finetuning ', format(stat_photo_VGG16_with_without_ft.pvalue, '.8f'))
print('Drawing - VGG16 before vs. after finetuning ', format(stat_drawing_VGG16_with_without_ft.pvalue, '.8f'))
print('Sketch - VGG16 before vs. after finetuning ', format(stat_sketch_VGG16_with_without_ft.pvalue, '.8f'))

print('Photo - VGG16 with vs without SIN', format(stat_photo_VGG16_with_without_SIN.pvalue, '.8f'))
print('Drawing - VGG16 with vs without SIN ', format(stat_drawing_VGG16_with_without_SIN.pvalue, '.8f'))
print('Sketch - VGG16 with vs without SIN ', format(stat_sketch_VGG16_with_without_SIN.pvalue, '.8f'))

print('Photo - VGG16 with ft vs SIN', format(stat_photo_VGG16_SIN_ft.pvalue, '.8f'))
print('Drawing - VGG16 with ft vs SIN', format(stat_drawing_VGG16_SIN_ft.pvalue, '.8f'))
print('Sketch - VGG16 with ft vs SIN ', format(stat_sketch_VGG16_SIN_ft.pvalue, '.8f'))

#%% FDR correction 

# formula : adjusted p-value = p-value*(total number of hypotheses tested)/(rank of the p-value)

# correct p_values

fdr_corrected_with_without_ft = multipletests([stat_photo_VGG16_with_without_ft.pvalue, stat_drawing_VGG16_with_without_ft.pvalue, stat_sketch_VGG16_with_without_ft.pvalue],alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
fdr_corrected_IN_SIN = multipletests([stat_photo_VGG16_with_without_SIN.pvalue, stat_drawing_VGG16_with_without_SIN.pvalue, stat_sketch_VGG16_with_without_SIN.pvalue],alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
fdr_corrected_SIN_ft = multipletests([stat_photo_VGG16_SIN_ft.pvalue, stat_drawing_VGG16_SIN_ft.pvalue, stat_sketch_VGG16_SIN_ft.pvalue],alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)


