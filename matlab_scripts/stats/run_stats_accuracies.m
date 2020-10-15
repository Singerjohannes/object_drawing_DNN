% run statistics for all networks VGG16 

% run stats for behavior vs. DNN comparison for all nets VGG16 

clear all 
clc

% set up paths 

% path where CNN results are stored 

DNN_path = 'C:/Users/Johannes/Documents/Leipzig/Masterarbeit/final_results/VGG16_with_without_SIN';

% specify which network to use 

net_name = 'VGG16';

% path for the behavioral data 

behav_path = 'C:/Users/Johannes/Documents/Leipzig/Masterarbeit/final_results/behavior';

% load the behavioral data 

load(fullfile(behav_path, 'photo_accs_per_image'));

load(fullfile(behav_path, 'drawing_accs_per_image'));

load(fullfile(behav_path, 'sketch_accs_per_image'));

% load the DNN top-1 accuracies 

load(fullfile(DNN_path, ['top_1_acc_',net_name,'.mat']));

%% compute the signed-rank permutation tests and get the pvalues 

n_perm = 1000;

photo_human_cnn_pval = signed_perm_test(sel_photo_results, VGG16_accs(1),n_perm);
drawing_human_cnn_pval = signed_perm_test(sel_drawing_results, VGG16_accs(2),n_perm);
sketch_human_cnn_pval = signed_perm_test(sel_sketch_results, VGG16_accs(3),n_perm);

human_cnn_p = [photo_human_cnn_pval drawing_human_cnn_pval sketch_human_cnn_pval];
[human_cnn_decision,~,~,human_cnn_adj_p] = fdr_bh(human_cnn_p,0.05,'dep');

%% test human accuracies against each other 

[~,photo_drawing_human_acc_p,~, photo_drawing_human_stat]= ttest(sel_photo_results, sel_drawing_results);
[~, photo_sketch_human_acc_p, ~, photo_sketch_human_stat]= ttest(sel_photo_results, sel_sketch_results);
[~, drawing_sketch_human_acc_p, ~, drawing_sketch_human_stat]= ttest(sel_drawing_results, sel_sketch_results);

human_p = [photo_sketch_human_acc_p drawing_sketch_human_acc_p];
[human_acc_decision,~,~, human_acc_adj_p] = fdr_bh(human_p,0.05,'dep');

%% test for equality 

diff_photo_drawing = sel_photo_results - sel_drawing_results;
[photo_drawing_equ_p, stat] = tost('one_sample', [-1/length(sel_photo_results) 1/length(sel_photo_results)], diff_photo_drawing,0);
