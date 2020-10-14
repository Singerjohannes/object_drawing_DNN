%% compute similarity between different depictions 

clear all
clc

% specify if statistics should be computed only for the finetuned layers 

is_ft = 1; % 1 for yes, 0 for no 

% load RDMs 

savepath = 'C:/Users/Johannes/Documents/Leipzig/Masterarbeit/final_results/VGG16_with_finetuning';

net_name = 'regular_vgg16_imagenetsketches_ft_conv5-1'%;'VGG16_SIN';

load(fullfile(savepath, ['photo_RDM_', net_name]))
load(fullfile(savepath, ['drawing_RDM_', net_name]))
load(fullfile(savepath, ['sketch_RDM_', net_name]))

if is_ft
    photo_RDM = photo_RDM(:,:,5:end);
    drawing_RDM = drawing_RDM(:,:,5:end);
    sketch_RDM = sketch_RDM(:,:,5:end);
end 
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


%% run stats on all similarities 

photo_drawing_sim = [];
photo_drawing_p = [];
photo_sketch_sim = [];
photo_sketch_p = []; 
drawing_sketch_sim = [];
drawing_sketch_p = []; 

for layer = 1: size(photo_RDM,3)
    
    [photo_DNN_human_sim(layer), photo_DNN_human_p(layer),~] = compute_RDM_perm_test(photo_RDM(:,:,layer), 1-photo_RDM_behav, 1000);
    [drawing_DNN_human_sim(layer), drawing_DNN_human_p(layer),~] = compute_RDM_perm_test(drawing_RDM(:,:,layer), 1-drawing_RDM_behav, 1000);
    [sketch_DNN_human_sim(layer), sketch_DNN_human_p(layer),~] = compute_RDM_perm_test(sketch_RDM(:,:,layer), 1-sketch_RDM_behav, 1000);

end 


[photo_DNN_human_RDM_sim_decision,~,~, photo_DNN_human_RDM_sim_adj_p] = fdr_bh(photo_DNN_human_p,0.05,'dep');
[drawing_DNN_human_RDM_sim_decision,~,~, drawing_DNN_human_RDM_sim_adj_p] = fdr_bh(drawing_DNN_human_p,0.05,'dep');
[sketch_DNN_human_RDM_sim_decision,~, ~, sketch_DNN_human_RDM_sim_adj_p] = fdr_bh(sketch_DNN_human_p,0.05,'dep');

%% test RDM correlation against each other

for this_layer = 1: size(photo_RDM,3)
        
        photo_drawing_comp_p(this_layer) = compute_RDM_bootstrap_correlation(photo_RDM(:,:,this_layer),squareform(1-squareform(photo_RDM_behav)), drawing_RDM(:,:,this_layer), squareform(1-squareform(drawing_RDM_behav)),1000);
        photo_sketch_comp_p(this_layer) = compute_RDM_bootstrap_correlation(photo_RDM(:,:,this_layer),squareform(1-squareform(photo_RDM_behav)), sketch_RDM(:,:,this_layer), squareform(1-squareform(sketch_RDM_behav)),1000);
        drawing_sketch_comp_p(this_layer) = compute_RDM_bootstrap_correlation(drawing_RDM(:,:,this_layer),squareform(1-squareform(drawing_RDM_behav)), sketch_RDM(:,:,this_layer),squareform(1-squareform(sketch_RDM_behav)),1000);

end 

[photo_drawing_comp_RDM_sim_decision,~,~, photo_drawing_comp_adj_p] = fdr_bh(photo_drawing_comp_p,0.05,'dep');
[photo_sketch_comp_RDM_sim_decision,~,~, photo_sketch_comp_adj_p] = fdr_bh(photo_sketch_comp_p,0.05,'dep');
[drawing_sketch_comp_RDM_sim_decision,~,~, drawing_sketch_comp_adj_p] = fdr_bh(drawing_sketch_comp_p,0.05,'dep');
