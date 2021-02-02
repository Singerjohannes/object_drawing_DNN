%% run RDM sim stats between networks 

clear all
clc

% load RDMs for VGG16 - experiment 1 

savepath = 'C:/Users/Johannes/Documents/Leipzig/Masterarbeit/final_results/VGG16_with_without_SIN';

net_name = 'VGG16';

load(fullfile(savepath, ['photo_RDM_', net_name]))
load(fullfile(savepath, ['drawing_RDM_', net_name]))
load(fullfile(savepath, ['sketch_RDM_', net_name]))

photo_RDM_IN = photo_RDM;
drawing_RDM_IN = drawing_RDM;
sketch_RDM_IN = sketch_RDM;

% load RDMs for VGG16 SIN - experiment 2 

savepath = 'C:/Users/Johannes/Documents/Leipzig/Masterarbeit/final_results/VGG16_with_without_SIN';

net_name = 'VGG16_SIN';

load(fullfile(savepath, ['photo_RDM_', net_name]))
load(fullfile(savepath, ['drawing_RDM_', net_name]))
load(fullfile(savepath, ['sketch_RDM_', net_name]))

photo_RDM_SIN = photo_RDM;
drawing_RDM_SIN = drawing_RDM;
sketch_RDM_SIN = sketch_RDM;

% load RDMs for VGG16 with finetuning - experiment 3 

savepath = 'C:/Users/Johannes/Documents/Leipzig/Masterarbeit/final_results/VGG16_with_finetuning';

net_name = 'regular_vgg16_imagenetsketches_ft_conv5-1';

load(fullfile(savepath, ['photo_RDM_', net_name]))
load(fullfile(savepath, ['drawing_RDM_', net_name]))
load(fullfile(savepath, ['sketch_RDM_', net_name]))

photo_RDM_FT = photo_RDM;
drawing_RDM_FT = drawing_RDM;
sketch_RDM_FT = sketch_RDM;

% load behavioral data 

behav_path = 'C:\Users\Johannes\Documents\Leipzig\Behavior\data';

photo_RDM_behav = load(fullfile(behav_path, 'photos_mat.mat'), 'final_mat');
photo_RDM_behav = photo_RDM_behav.final_mat;

drawing_RDM_behav = load(fullfile(behav_path,'drawings_mat.mat'), 'final_mat');
drawing_RDM_behav = drawing_RDM_behav.final_mat;

sketch_RDM_behav = load(fullfile(behav_path,'sketches_mat.mat'), 'final_mat');
sketch_RDM_behav = sketch_RDM_behav.final_mat;

% select only imagenet objects from behavior RDMs 

% get ecoset filenames 

ecoset_path = 'C:\Users\Johannes\Documents\Leipzig\Modelling\Stimuli\ecoset\scaled\photos';

fp = ecoset_path;
fntmp = dir(fullfile(fp, '*.jpg'));
ecoset_fn = {fntmp.name}';

% get imagenet filenames 

imagenet_path = 'C:\Users\Johannes\Documents\Leipzig\Masterarbeit\final_stimuli\photos';

fp = imagenet_path;
fntmp = dir(fullfile(fp, '*.jpg'));
imagenet_fn = {fntmp.name}';

% get logical vector of overlay between imagenet and ecoset 

sel_vector = ismember(ecoset_fn, imagenet_fn);

% select rows and columns from the RDMs 

photo_RDM_behav = photo_RDM_behav(sel_vector, sel_vector);
drawing_RDM_behav = drawing_RDM_behav(sel_vector, sel_vector);
sketch_RDM_behav = sketch_RDM_behav(sel_vector, sel_vector);

%% compare RDM sims for VGG16 IN and VGG16 SIN 

for layer = 1:size(photo_RDM_IN,3)
    
    [photo_drawing_IN_vs_SIN_p(layer), photo_drawing_sim_IN(layer), photo_drawing_sim_SIN(layer)] = compute_RDM_bootstrap_correlation(photo_RDM_IN(:,:,layer), drawing_RDM_IN(:,:,layer), photo_RDM_SIN(:,:,layer), drawing_RDM_SIN(:,:,layer),1000);
    [photo_sketch_IN_vs_SIN_p(layer), photo_sketch_sim_IN(layer), photo_sketch_sim_SIN(layer)] = compute_RDM_bootstrap_correlation(photo_RDM_IN(:,:,layer), sketch_RDM_IN(:,:,layer), photo_RDM_SIN(:,:,layer), sketch_RDM_SIN(:,:,layer),1000);
    [drawing_sketch_IN_vs_SIN_p(layer), drawing_sketch_sim_IN(layer), drawing_sketch_sim_SIN(layer)] = compute_RDM_bootstrap_correlation(drawing_RDM_IN(:,:,layer), sketch_RDM_IN(:,:,layer), drawing_RDM_SIN(:,:,layer), sketch_RDM_SIN(:,:,layer),1000);

end 

[photo_drawing_IN_SIN_decision,~,~, photo_drawing_IN_SIN_adj_p] = fdr_bh(photo_drawing_IN_vs_SIN_p,0.05,'dep');
[photo_sketch_IN_SIN_decision,~,~, photo_sketch_IN_SIN_adj_p] = fdr_bh(photo_sketch_IN_vs_SIN_p,0.05,'dep');
[drawing_sketch_IN_SIN_decision,~, ~, drawing_sketch_IN_SIN_adj_p] = fdr_bh(drawing_sketch_IN_vs_SIN_p,0.05,'dep');

%% compare RDM sims for VGG16 IN and VGG16 FT 

for layer = 1:size(photo_RDM_IN,3)
    
    [photo_drawing_IN_vs_FT_p(layer), photo_drawing_sim_IN(layer), photo_drawing_sim_ft(layer)] = compute_RDM_bootstrap_correlation(photo_RDM_IN(:,:,layer), drawing_RDM_IN(:,:,layer), photo_RDM_FT(:,:,layer), drawing_RDM_FT(:,:,layer),1000);
    [photo_sketch_IN_vs_FT_p(layer), photo_sketch_sim_IN(layer), photo_sketch_sim_ft(layer)] = compute_RDM_bootstrap_correlation(photo_RDM_IN(:,:,layer), sketch_RDM_IN(:,:,layer), photo_RDM_FT(:,:,layer), sketch_RDM_FT(:,:,layer),1000);
    [drawing_sketch_IN_vs_FT_p(layer), drawing_sketch_sim_IN(layer), drawing_sketch_sim_ft(layer)] = compute_RDM_bootstrap_correlation(drawing_RDM_IN(:,:,layer), sketch_RDM_IN(:,:,layer), drawing_RDM_FT(:,:,layer), sketch_RDM_FT(:,:,layer),1000);

end 

[photo_drawing_IN_FT_decision,~,~,photo_drawing_IN_FT_adj_p] = fdr_bh(photo_drawing_IN_vs_FT_p(5:end),0.05,'dep');
[photo_sketch_IN_FT_decision,~,~,photo_sketch_IN_FT_adj_p] = fdr_bh(photo_sketch_IN_vs_FT_p(5:end),0.05,'dep');
[drawing_sketch_IN_FT_decision,~,~,drawing_sketch_IN_FT_adj_p] = fdr_bh(drawing_sketch_IN_vs_FT_p(5:end),0.05,'dep');

%% compare RDM sims for VGG16 SIN and VGG16 FT 

for layer = 1:size(photo_RDM_IN,3)
    
    [photo_drawing_SIN_vs_FT_p(layer), photo_drawing_sim_SIN(layer), photo_drawing_sim_ft(layer)] = compute_RDM_bootstrap_correlation(photo_RDM_SIN(:,:,layer), drawing_RDM_SIN(:,:,layer), photo_RDM_FT(:,:,layer), drawing_RDM_FT(:,:,layer),1000);
    [photo_sketch_SIN_vs_FT_p(layer), photo_sketch_sim_SIN(layer), photo_sketch_sim_ft(layer)] = compute_RDM_bootstrap_correlation(photo_RDM_SIN(:,:,layer), sketch_RDM_SIN(:,:,layer), photo_RDM_FT(:,:,layer), sketch_RDM_FT(:,:,layer),1000);
    [drawing_sketch_SIN_vs_FT_p(layer), drawing_sketch_sim_SIN(layer), drawing_sketch_sim_ft(layer)] = compute_RDM_bootstrap_correlation(drawing_RDM_SIN(:,:,layer), sketch_RDM_SIN(:,:,layer), drawing_RDM_FT(:,:,layer), sketch_RDM_FT(:,:,layer),1000);

end 

[photo_drawing_SIN_FT_decision,~,~,photo_drawing_SIN_FT_adj_p] = fdr_bh(photo_drawing_SIN_vs_FT_p(5:end),0.05,'dep');
[photo_sketch_SIN_FT_decision,~,~,photo_sketch_SIN_FT_adj_p] = fdr_bh(photo_sketch_SIN_vs_FT_p(5:end),0.05,'dep');
[drawing_sketch_SIN_FT_decision,~,~,drawing_sketch_SIN_FT_adj_p] = fdr_bh(drawing_sketch_SIN_vs_FT_p(5:end),0.05,'dep');

%% compare fit to human behavior for VGG16 IN and VGG16 SIN 

for layer = 1:size(photo_RDM_IN,3)
    
    [photo_DNN_human_IN_vs_SIN_p(layer), photo_DNN_human_sim_IN(layer),photo_DNN_human_sim_SIN(layer)]  = compute_RDM_bootstrap_correlation(photo_RDM_IN(:,:,layer),squareform(1-squareform(photo_RDM_behav)), photo_RDM_SIN(:,:,layer), squareform(1-squareform(photo_RDM_behav)),10000);
    [drawing_DNN_human_IN_vs_SIN_p(layer), drawing_DNN_human_sim_IN(layer),drawing_DNN_human_sim_SIN(layer)] = compute_RDM_bootstrap_correlation(drawing_RDM_IN(:,:,layer),squareform(1-squareform(drawing_RDM_behav)), drawing_RDM_SIN(:,:,layer), squareform(1-squareform(drawing_RDM_behav)),10000);
    [sketch_DNN_human_IN_vs_SIN_p(layer), sketch_DNN_human_sim_IN(layer),sketch_DNN_human_sim_SIN(layer)] = compute_RDM_bootstrap_correlation(sketch_RDM_IN(:,:,layer), squareform(1-squareform(sketch_RDM_behav)), sketch_RDM_SIN(:,:,layer), squareform(1-squareform(sketch_RDM_behav)),10000);

end 

[photo_DNN_human_IN_SIN_decision,~,~,photo_DNN_human_IN_SIN_adj_p] = fdr_bh(photo_DNN_human_IN_vs_SIN_p,0.05,'dep');
[drawing_DNN_human_IN_SIN_decision,~,~,drawing_DNN_human_IN_SIN_adj_p] = fdr_bh(drawing_DNN_human_IN_vs_SIN_p,0.05,'dep');
[sketch_DNN_human_IN_SIN_decision,~,~,sketch_DNN_human_IN_SIN_adj_p] = fdr_bh(sketch_DNN_human_IN_vs_SIN_p,0.05,'dep');

%% compare fit to human behavior for VGG16 IN and VGG16 FT 

for layer = 1:size(photo_RDM_IN,3)
    
    [photo_DNN_human_IN_vs_FT_p(layer), photo_DNN_human_sim_IN(layer),photo_DNN_human_sim_FT(layer)] = compute_RDM_bootstrap_correlation(photo_RDM_IN(:,:,layer), squareform(1-squareform(photo_RDM_behav)), photo_RDM_FT(:,:,layer), squareform(1-squareform(photo_RDM_behav)),1000);
    [drawing_DNN_human_IN_vs_FT_p(layer),  drawing_DNN_human_sim_IN(layer),drawing_DNN_human_sim_FT(layer)] = compute_RDM_bootstrap_correlation(drawing_RDM_IN(:,:,layer), squareform(1-squareform(drawing_RDM_behav)), drawing_RDM_FT(:,:,layer), squareform(1-squareform(drawing_RDM_behav)),1000);
    [sketch_DNN_human_IN_vs_FT_p(layer), sketch_DNN_human_sim_IN(layer),sketch_DNN_human_sim_FT(layer)] = compute_RDM_bootstrap_correlation(sketch_RDM_IN(:,:,layer), squareform(1-squareform(sketch_RDM_behav)), sketch_RDM_FT(:,:,layer), squareform(1-squareform(sketch_RDM_behav)),1000);

end 

[photo_DNN_human_IN_FT_decision,~,~,photo_DNN_human_IN_FT_adj_p] = fdr_bh(photo_DNN_human_IN_vs_FT_p(5:end),0.05,'dep');
[drawing_DNN_human_IN_FT_decision,~,~,drawing_DNN_human_IN_FT_adj_p] = fdr_bh(drawing_DNN_human_IN_vs_FT_p(5:end),0.05,'dep');
[sketch_DNN_human_IN_FT_decision,~,~,sketch_DNN_human_IN_FT_adj_p] = fdr_bh(sketch_DNN_human_IN_vs_FT_p(5:end),0.05,'dep');

%% compare fit to human behavior for VGG16 SIN and VGG16 FT 

for layer = 1:size(photo_RDM_IN,3)
    
    [photo_DNN_human_SIN_vs_FT_p(layer),photo_DNN_human_sim_IN(layer),photo_DNN_human_sim_FT(layer)] = compute_RDM_bootstrap_correlation(photo_RDM_IN(:,:,layer), photo_RDM_behav, photo_RDM_FT(:,:,layer), photo_RDM_behav,1000);
    [drawing_DNN_human_SIN_vs_FT_p(layer),  drawing_DNN_human_sim_IN(layer),drawing_DNN_human_sim_FT(layer)] = compute_RDM_bootstrap_correlation(drawing_RDM_IN(:,:,layer), drawing_RDM_behav, drawing_RDM_FT(:,:,layer), drawing_RDM_behav,1000);
    [sketch_DNN_human_IN_vs_FT_p(layer), sketch_DNN_human_sim_IN(layer),sketch_DNN_human_sim_FT(layer)] = compute_RDM_bootstrap_correlation(sketch_RDM_IN(:,:,layer), sketch_RDM_behav, sketch_RDM_FT(:,:,layer), sketch_RDM_behav,1000);

end 

[photo_DNN_human_IN_FT_decision,~] = fdr_bh(photo_DNN_human_IN_vs_FT_p(5:end),0.05,'dep');
[drawing_DNN_human_IN_FT_decision,~] = fdr_bh(drawing_DNN_human_IN_vs_FT_p(5:end),0.05,'dep');
[sketch_DNN_human_IN_FT_decision,~] = fdr_bh(sketch_DNN_human_IN_vs_FT_p(5:end),0.05,'dep');

