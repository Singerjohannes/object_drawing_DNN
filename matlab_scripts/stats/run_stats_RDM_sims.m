%% run stats for RDM similarities all VGG16

clear all 
clc

% load RDMs

% specify where RDMs are saved 

savepath = 'C:/Users/Johannes/Documents/Leipzig/Masterarbeit/final_results/VGG16_with_without_SIN';

% specify which network to use

net_name = 'VGG16'; %'regular_vgg16_imagenetsketches_ft_conv5-1';

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

%% test for significane of RDM similarities

photo_drawing_sim = [];
photo_sketch_sim = [];
drawing_sketch_sim = [];

for layer = 1: size(photo_RDM,3)
    
    [photo_drawing_sim(layer), photo_drawing_RDM_sim_p(layer), dist] = compute_RDM_perm_test(photo_RDM(:,:,layer), drawing_RDM(:,:,layer), 1000);
    [photo_sketch_sim(layer), photo_sketch_RDM_sim_p(layer)] = compute_RDM_perm_test(photo_RDM(:,:,layer), sketch_RDM(:,:,layer), 1000);
    [drawing_sketch_sim(layer), drawing_sketch_RDM_sim_p(layer)] = compute_RDM_perm_test(drawing_RDM(:,:,layer), sketch_RDM(:,:,layer), 1000);

end 

[photo_drawing_RDM_sim_decision,~,~, photo_drawing_RDM_sim_adj_p] = fdr_bh(photo_drawing_RDM_sim_p,0.05,'dep');
[photo_sketch_RDM_sim_decision,~,~, photo_sketch_RDM_sim_adj_p] = fdr_bh(photo_sketch_RDM_sim_p,0.05,'dep');
[drawing_sketch_RDM_SIM_decision,~,~,drawing_sketch_RDM_sim_adj_p] = fdr_bh(drawing_sketch_RDM_sim_p,0.05,'dep');

%% test RDM correlations against each other between depiction combinations 

for layer = 1: size(photo_RDM,3)
photo_drawing_photo_sketch_comp_p(layer) = compute_RDM_bootstrap_correlation(photo_RDM(:,:,layer),drawing_RDM(:,:,layer), photo_RDM(:,:,layer), sketch_RDM(:,:,layer),1000);
photo_drawing_drawing_sketch_comp_p(layer) = compute_RDM_bootstrap_correlation(photo_RDM(:,:,layer),drawing_RDM(:,:,layer), drawing_RDM(:,:,layer), sketch_RDM(:,:,layer),1000);
photo_sketch_drawing_sketch_comp_p(layer) = compute_RDM_bootstrap_correlation(photo_RDM(:,:,layer),sketch_RDM(:,:,layer), drawing_RDM(:,:,layer), sketch_RDM(:,:,layer),1000);

end 

[photo_drawing_photo_sketch_comp_decision,~,~, photo_drawing_photo_sketch_comp_adj_p] = fdr_bh(photo_drawing_photo_sketch_comp_p,0.05,'dep');
[photo_drawing_drawing_sketch_comp_decision,~,~, photo_drawing_drawing_sketch_comp_adj_p] = fdr_bh(photo_drawing_drawing_sketch_comp_p,0.05,'dep');
[photo_sketch_drawing_sketch_comp_decision,~,~, photo_sketch_drawing_sketch_comp_adj_p] = fdr_bh(photo_sketch_drawing_sketch_comp_p,0.05,'dep');


%% test RDM correlation against each other between layers 

for this_layer = 1: size(photo_RDM,3)
    for that_layer = 1:size(photo_RDM,3)
        photo_drawing_intra_comp_p(this_layer,that_layer) = compute_RDM_bootstrap_correlation(photo_RDM(:,:,this_layer),drawing_RDM(:,:,this_layer), photo_RDM(:,:,that_layer), drawing_RDM(:,:,that_layer),1000);
        photo_sketch_intra_comp_p(this_layer,that_layer) = compute_RDM_bootstrap_correlation(photo_RDM(:,:,this_layer),sketch_RDM(:,:,this_layer), photo_RDM(:,:,that_layer), sketch_RDM(:,:,that_layer),1000);
        drawing_sketch_intra_comp_p(this_layer,that_layer) = compute_RDM_bootstrap_correlation(drawing_RDM(:,:,this_layer),sketch_RDM(:,:,this_layer), drawing_RDM(:,:,that_layer), sketch_RDM(:,:,that_layer),1000);
    end 
end 

[photo_drawing_intra_comp_RDM_sim_decision,~] = fdr_bh(squareform(photo_drawing_intra_comp_p),0.05,'dep');
photo_drawing_intra_comp_RDM_sim_decision = squareform(photo_drawing_intra_comp_RDM_sim_decision);
[photo_sketch_intra_comp_RDM_sim_decision,~] = fdr_bh(squareform(photo_sketch_intra_comp_p),0.05,'dep');
photo_sketch_intra_comp_RDM_sim_decision = squareform(photo_sketch_intra_comp_RDM_sim_decision);
[drawing_sketch_intra_comp_RDM_sim_decision,~] = fdr_bh(squareform(drawing_sketch_intra_comp_p),0.05,'dep');
drawing_sketch_intra_comp_RDM_sim_decision = squareform(drawing_sketch_intra_comp_RDM_sim_decision);

