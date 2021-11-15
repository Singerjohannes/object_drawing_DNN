%% run stats for RDM similarities all VGG16

clear all 
clc

% load RDMs

% specify where RDMs are saved 

savepath = '/object_drawing_DNN/results';

% load input RDMs (based on raw pixel values after preprocessing) 

load(fullfile(savepath, 'input_RDMs.mat')) 

input_photo_RDM  = photo_RDM;
input_drawing_RDM = drawing_RDM; 
input_sketch_RDM = sketch_RDM; 

% specify which network to use

net_name = 'VGG16'; %'VGG16_SIN' or 'VGG16_FT

load(fullfile(savepath, ['photo_RDM_', net_name]))
load(fullfile(savepath, ['drawing_RDM_', net_name]))
load(fullfile(savepath, ['sketch_RDM_', net_name]))

% specify if statistics should be computed only for the finetuned layers 

is_ft = 0; % 1 for yes, 0 for no 

% select only the RDMs from the finetuned layers for statistical testing if is_ft is true 

if is_ft
    photo_RDM = photo_RDM(:,:,5:end);
    drawing_RDM = drawing_RDM(:,:,5:end);
    sketch_RDM = sketch_RDM(:,:,5:end);
end 

%% test for significant differences between RDM similarities at the input stage 

input_photo_drawing_photo_sketch_comp_p = compute_RDM_pairwise_randomization_test(input_photo_RDM,input_drawing_RDM, input_photo_RDM, input_sketch_RDM,1000);
input_photo_drawing_drawing_sketch_comp_p = compute_RDM_pairwise_randomization_test(input_photo_RDM,input_drawing_RDM, input_drawing_RDM, input_sketch_RDM,1000);
input_photo_sketch_drawing_sketch_comp_p = compute_RDM_pairwise_randomization_test(input_photo_RDM,input_sketch_RDM, input_drawing_RDM, input_sketch_RDM,1000);

[input_decision, ~,~, input_diff_adj_p] =  fdr_bh([input_photo_drawing_photo_sketch_comp_p input_photo_drawing_drawing_sketch_comp_p input_photo_sketch_drawing_sketch_comp_p],0.05,'dep');
%% test for significane of RDM similarities

photo_drawing_sim = [];
photo_sketch_sim = [];
drawing_sketch_sim = [];

for layer = 1: size(photo_RDM,3)
    
    [photo_drawing_sim(layer), photo_drawing_RDM_sim_p(layer)] = compute_RDM_randomization_test(photo_RDM(:,:,layer), drawing_RDM(:,:,layer), 1000);
    [photo_sketch_sim(layer), photo_sketch_RDM_sim_p(layer)] = compute_RDM_randomization_test(photo_RDM(:,:,layer), sketch_RDM(:,:,layer), 1000);
    [drawing_sketch_sim(layer), drawing_sketch_RDM_sim_p(layer)] = compute_RDM_randomization_test(drawing_RDM(:,:,layer), sketch_RDM(:,:,layer), 1000);

end 

[photo_drawing_RDM_sim_decision,~,~, photo_drawing_RDM_sim_adj_p] = fdr_bh(photo_drawing_RDM_sim_p,0.05,'dep');
[photo_sketch_RDM_sim_decision,~,~, photo_sketch_RDM_sim_adj_p] = fdr_bh(photo_sketch_RDM_sim_p,0.05,'dep');
[drawing_sketch_RDM_SIM_decision,~,~,drawing_sketch_RDM_sim_adj_p] = fdr_bh(drawing_sketch_RDM_sim_p,0.05,'dep');

%% test for overall signifance of differences in similarities between layers 

photo_drawing_all_p = compute_RDM_multiple_randomization_test(photo_RDM, drawing_RDM, 1000); 
photo_sketch_all_p = compute_RDM_multiple_randomization_test(photo_RDM, sketch_RDM, 1000); 
drawing_sketch_all_p = compute_RDM_multiple_randomization_test(photo_RDM, drawing_RDM, 1000); 

%% test RDM correlation against each other between layers 

% select layers which should be compared 

layer_sel = [1 4 7];

for this_comp = 1: length(layer_sel)-1
        photo_drawing_intra_comp_p(this_comp) = compute_RDM_pairwise_randomization_test(photo_RDM(:,:,layer_sel(this_comp)),drawing_RDM(:,:,layer_sel(this_comp)), photo_RDM(:,:,layer_sel(this_comp+1)), drawing_RDM(:,:,layer_sel(this_comp+1)),1000);
        photo_sketch_intra_comp_p(this_comp) = compute_RDM_pairwise_randomization_test(photo_RDM(:,:,layer_sel(this_comp)),sketch_RDM(:,:,layer_sel(this_comp)), photo_RDM(:,:,layer_sel(this_comp+1)), sketch_RDM(:,:,layer_sel(this_comp+1)),1000);
        drawing_sketch_intra_comp_p(this_comp) = compute_RDM_pairwise_randomization_test(drawing_RDM(:,:,layer_sel(this_comp)),sketch_RDM(:,:,layer_sel(this_comp)), drawing_RDM(:,:,layer_sel(this_comp+1)), sketch_RDM(:,:,layer_sel(this_comp+1)),1000);
end 

[photo_drawing_intra_comp_RDM_sim_decision,~,~,photo_drawing_intra_comp_RDM_sim_adj_p] = fdr_bh(photo_drawing_intra_comp_p,0.05,'dep');
[photo_sketch_intra_comp_RDM_sim_decision,~,~,photo_sketch_intra_comp_RDM_sim_adj_p] = fdr_bh(photo_sketch_intra_comp_p,0.05,'dep');
[drawing_sketch_intra_comp_RDM_sim_decision,~,~,drawing_sketch_intra_comp_RDM_sim_adj_p] = fdr_bh(drawing_sketch_intra_comp_p,0.05,'dep');

%% test RDM correlations against each other between depiction combinations 

for layer = 1: size(photo_RDM,3)
    
    photo_drawing_photo_sketch_comp_p(layer) = compute_RDM_pairwise_randomization_test(photo_RDM(:,:,layer),drawing_RDM(:,:,layer), photo_RDM(:,:,layer), sketch_RDM(:,:,layer),1000);
    photo_drawing_drawing_sketch_comp_p(layer) = compute_RDM_pairwise_randomization_test(photo_RDM(:,:,layer),drawing_RDM(:,:,layer), drawing_RDM(:,:,layer), sketch_RDM(:,:,layer),1000);
    photo_sketch_drawing_sketch_comp_p(layer) = compute_RDM_pairwise_randomization_test(photo_RDM(:,:,layer),sketch_RDM(:,:,layer), drawing_RDM(:,:,layer), sketch_RDM(:,:,layer),1000);

end 

[photo_drawing_photo_sketch_comp_decision,~,~, photo_drawing_photo_sketch_comp_adj_p] = fdr_bh(photo_drawing_photo_sketch_comp_p,0.05,'dep');
[photo_drawing_drawing_sketch_comp_decision,~,~, photo_drawing_drawing_sketch_comp_adj_p] = fdr_bh(photo_drawing_drawing_sketch_comp_p,0.05,'dep');
[photo_sketch_drawing_sketch_comp_decision,~,~, photo_sketch_drawing_sketch_comp_adj_p] = fdr_bh(photo_sketch_drawing_sketch_comp_p,0.05,'dep');


