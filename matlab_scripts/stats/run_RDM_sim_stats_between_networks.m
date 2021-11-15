%% run RDM sim stats between networks 

clear all
clc

% load RDMs for VGG16 - experiment 1 

savepath = '/object_drawing_DNN/results';

net_name = 'VGG16';

load(fullfile(savepath, ['photo_RDM_', net_name]))
load(fullfile(savepath, ['drawing_RDM_', net_name]))
load(fullfile(savepath, ['sketch_RDM_', net_name]))

photo_RDM_IN = photo_RDM;
drawing_RDM_IN = drawing_RDM;
sketch_RDM_IN = sketch_RDM;

% load RDMs for VGG16 with randomly initialized weights - experiment 1

net_name = 'VGG16_random';

load(fullfile(savepath, ['photo_RDM_', net_name]))
load(fullfile(savepath, ['drawing_RDM_', net_name]))
load(fullfile(savepath, ['sketch_RDM_', net_name]))

photo_RDM_random = photo_RDM;
drawing_RDM_random = drawing_RDM;
sketch_RDM_random = sketch_RDM;

% load RDMs for VGG16 SIN - experiment 2 

net_name = 'VGG16_SIN';

load(fullfile(savepath, ['photo_RDM_', net_name]))
load(fullfile(savepath, ['drawing_RDM_', net_name]))
load(fullfile(savepath, ['sketch_RDM_', net_name]))

photo_RDM_SIN = photo_RDM;
drawing_RDM_SIN = drawing_RDM;
sketch_RDM_SIN = sketch_RDM;

% load RDMs for VGG16 with finetuning - experiment 3 

net_name = 'VGG16_FT';

load(fullfile(savepath, ['photo_RDM_', net_name]))
load(fullfile(savepath, ['drawing_RDM_', net_name]))
load(fullfile(savepath, ['sketch_RDM_', net_name]))

photo_RDM_FT = photo_RDM;
drawing_RDM_FT = drawing_RDM;
sketch_RDM_FT = sketch_RDM;

% load behavioral data 

behav_path = '/Users/johannessinger/Documents/Leipzig/Modelling/object_drawing_DNN/data';

load(fullfile(behav_path, 'photo_behav_RDM.mat'));

load(fullfile(behav_path,'drawing_behav_RDM.mat'));

load(fullfile(behav_path,'sketch_behav_RDM.mat'));

%% compare RDM sims for VGG16 IN and VGG16 randomly initialized

for layer = 1:size(photo_RDM_IN,3)
    
    [photo_drawing_IN_vs_random_p(layer), photo_drawing_sim_IN(layer), photo_drawing_sim_random(layer)] = compute_RDM_pairwise_randomization_test(photo_RDM_IN(:,:,layer), drawing_RDM_IN(:,:,layer), photo_RDM_random(:,:,layer), drawing_RDM_random(:,:,layer),1000);
    [photo_sketch_IN_vs_random_p(layer), photo_sketch_sim_IN(layer), photo_sketch_sim_random(layer)] = compute_RDM_pairwise_randomization_test(photo_RDM_IN(:,:,layer), sketch_RDM_IN(:,:,layer), photo_RDM_random(:,:,layer), sketch_RDM_random(:,:,layer),1000);
    [drawing_sketch_IN_vs_random_p(layer), drawing_sketch_sim_IN(layer), drawing_sketch_sim_random(layer)] = compute_RDM_pairwise_randomization_test(drawing_RDM_IN(:,:,layer), sketch_RDM_IN(:,:,layer), drawing_RDM_random(:,:,layer), sketch_RDM_random(:,:,layer),1000);

end 

[photo_drawing_IN_random_decision,~,~, photo_drawing_IN_random_adj_p] = fdr_bh(photo_drawing_IN_vs_random_p,0.05,'dep');
[photo_sketch_IN_random_decision,~,~, photo_sketch_IN_random_adj_p] = fdr_bh(photo_sketch_IN_vs_random_p,0.05,'dep');
[drawing_sketch_IN_random_decision,~, ~, drawing_sketch_IN_random_adj_p] = fdr_bh(drawing_sketch_IN_vs_random_p,0.05,'dep');


%% compare RDM sims for VGG16 IN and VGG16 SIN 

for layer = 1:size(photo_RDM_IN,3)
    
    [photo_drawing_IN_vs_SIN_p(layer), photo_drawing_sim_IN(layer), photo_drawing_sim_SIN(layer)] = compute_RDM_pairwise_randomization_test(photo_RDM_IN(:,:,layer), drawing_RDM_IN(:,:,layer), photo_RDM_SIN(:,:,layer), drawing_RDM_SIN(:,:,layer),1000);
    [photo_sketch_IN_vs_SIN_p(layer), photo_sketch_sim_IN(layer), photo_sketch_sim_SIN(layer)] = compute_RDM_pairwise_randomization_test(photo_RDM_IN(:,:,layer), sketch_RDM_IN(:,:,layer), photo_RDM_SIN(:,:,layer), sketch_RDM_SIN(:,:,layer),1000);
    [drawing_sketch_IN_vs_SIN_p(layer), drawing_sketch_sim_IN(layer), drawing_sketch_sim_SIN(layer)] = compute_RDM_pairwise_randomization_test(drawing_RDM_IN(:,:,layer), sketch_RDM_IN(:,:,layer), drawing_RDM_SIN(:,:,layer), sketch_RDM_SIN(:,:,layer),1000);

end 

[photo_drawing_IN_SIN_decision,~,~, photo_drawing_IN_SIN_adj_p] = fdr_bh(photo_drawing_IN_vs_SIN_p,0.05,'dep');
[photo_sketch_IN_SIN_decision,~,~, photo_sketch_IN_SIN_adj_p] = fdr_bh(photo_sketch_IN_vs_SIN_p,0.05,'dep');
[drawing_sketch_IN_SIN_decision,~, ~, drawing_sketch_IN_SIN_adj_p] = fdr_bh(drawing_sketch_IN_vs_SIN_p,0.05,'dep');

%% compare RDM sims for VGG16 IN and VGG16 FT 

for layer = 1:size(photo_RDM_IN,3)
    
    [photo_drawing_IN_vs_FT_p(layer), photo_drawing_sim_IN(layer), photo_drawing_sim_ft(layer)] = compute_RDM_pairwise_randomization_test(photo_RDM_IN(:,:,layer), drawing_RDM_IN(:,:,layer), photo_RDM_FT(:,:,layer), drawing_RDM_FT(:,:,layer),10000);
    [photo_sketch_IN_vs_FT_p(layer), photo_sketch_sim_IN(layer), photo_sketch_sim_ft(layer)] = compute_RDM_pairwise_randomization_test(photo_RDM_IN(:,:,layer), sketch_RDM_IN(:,:,layer), photo_RDM_FT(:,:,layer), sketch_RDM_FT(:,:,layer),10000);
    [drawing_sketch_IN_vs_FT_p(layer), drawing_sketch_sim_IN(layer), drawing_sketch_sim_ft(layer)] = compute_RDM_pairwise_randomization_test(drawing_RDM_IN(:,:,layer), sketch_RDM_IN(:,:,layer), drawing_RDM_FT(:,:,layer), sketch_RDM_FT(:,:,layer),10000);

end 

[photo_drawing_IN_FT_decision,~,~,photo_drawing_IN_FT_adj_p] = fdr_bh(photo_drawing_IN_vs_FT_p(5:end),0.05,'dep');
[photo_sketch_IN_FT_decision,~,~,photo_sketch_IN_FT_adj_p] = fdr_bh(photo_sketch_IN_vs_FT_p(5:end),0.05,'dep');
[drawing_sketch_IN_FT_decision,~,~,drawing_sketch_IN_FT_adj_p] = fdr_bh(drawing_sketch_IN_vs_FT_p(5:end),0.05,'dep');

%% compare RDM sims for VGG16 SIN and VGG16 FT 

for layer = 1:size(photo_RDM_IN,3)
    
    [photo_drawing_SIN_vs_FT_p(layer), photo_drawing_sim_SIN(layer), photo_drawing_sim_ft(layer)] = compute_RDM_pairwise_randomization_test(photo_RDM_SIN(:,:,layer), drawing_RDM_SIN(:,:,layer), photo_RDM_FT(:,:,layer), drawing_RDM_FT(:,:,layer),1000);
    [photo_sketch_SIN_vs_FT_p(layer), photo_sketch_sim_SIN(layer), photo_sketch_sim_ft(layer)] = compute_RDM_pairwise_randomization_test(photo_RDM_SIN(:,:,layer), sketch_RDM_SIN(:,:,layer), photo_RDM_FT(:,:,layer), sketch_RDM_FT(:,:,layer),1000);
    [drawing_sketch_SIN_vs_FT_p(layer), drawing_sketch_sim_SIN(layer), drawing_sketch_sim_ft(layer)] = compute_RDM_pairwise_randomization_test(drawing_RDM_SIN(:,:,layer), sketch_RDM_SIN(:,:,layer), drawing_RDM_FT(:,:,layer), sketch_RDM_FT(:,:,layer),1000);

end 

[photo_drawing_SIN_FT_decision,~,~,photo_drawing_SIN_FT_adj_p] = fdr_bh(photo_drawing_SIN_vs_FT_p(5:end),0.05,'dep');
[photo_sketch_SIN_FT_decision,~,~,photo_sketch_SIN_FT_adj_p] = fdr_bh(photo_sketch_SIN_vs_FT_p(5:end),0.05,'dep');
[drawing_sketch_SIN_FT_decision,~,~,drawing_sketch_SIN_FT_adj_p] = fdr_bh(drawing_sketch_SIN_vs_FT_p(5:end),0.05,'dep');

%% compare fit to human behavior for VGG16 IN and VGG16 SIN 

for layer = 1:size(photo_RDM_IN,3)
    
    [photo_DNN_human_IN_vs_SIN_p(layer), photo_DNN_human_sim_IN(layer),photo_DNN_human_sim_SIN(layer)]  = compute_RDM_pairwise_randomization_test(photo_RDM_IN(:,:,layer),squareform(1-squareform(photo_RDM_behav)), photo_RDM_SIN(:,:,layer), squareform(1-squareform(photo_RDM_behav)),1000);
    [drawing_DNN_human_IN_vs_SIN_p(layer), drawing_DNN_human_sim_IN(layer),drawing_DNN_human_sim_SIN(layer)] = compute_RDM_pairwise_randomization_test(drawing_RDM_IN(:,:,layer),squareform(1-squareform(drawing_RDM_behav)), drawing_RDM_SIN(:,:,layer), squareform(1-squareform(drawing_RDM_behav)),1000);
    [sketch_DNN_human_IN_vs_SIN_p(layer), sketch_DNN_human_sim_IN(layer),sketch_DNN_human_sim_SIN(layer)] = compute_RDM_pairwise_randomization_test(sketch_RDM_IN(:,:,layer), squareform(1-squareform(sketch_RDM_behav)), sketch_RDM_SIN(:,:,layer), squareform(1-squareform(sketch_RDM_behav)),1000);

end 

[photo_DNN_human_IN_SIN_decision,~,~,photo_DNN_human_IN_SIN_adj_p] = fdr_bh(photo_DNN_human_IN_vs_SIN_p,0.05,'dep');
[drawing_DNN_human_IN_SIN_decision,~,~,drawing_DNN_human_IN_SIN_adj_p] = fdr_bh(drawing_DNN_human_IN_vs_SIN_p,0.05,'dep');
[sketch_DNN_human_IN_SIN_decision,~,~,sketch_DNN_human_IN_SIN_adj_p] = fdr_bh(sketch_DNN_human_IN_vs_SIN_p,0.05,'dep');

%% compare fit to human behavior for VGG16 IN and VGG16 FT 

for layer = 1:size(photo_RDM_IN,3)
    
    [photo_DNN_human_IN_vs_FT_p(layer), photo_DNN_human_sim_IN(layer),photo_DNN_human_sim_FT(layer)] = compute_RDM_pairwise_randomization_test(photo_RDM_IN(:,:,layer), squareform(1-squareform(photo_RDM_behav)), photo_RDM_FT(:,:,layer), squareform(1-squareform(photo_RDM_behav)),1000);
    [drawing_DNN_human_IN_vs_FT_p(layer),  drawing_DNN_human_sim_IN(layer),drawing_DNN_human_sim_FT(layer)] = compute_RDM_pairwise_randomization_test(drawing_RDM_IN(:,:,layer), squareform(1-squareform(drawing_RDM_behav)), drawing_RDM_FT(:,:,layer), squareform(1-squareform(drawing_RDM_behav)),1000);
    [sketch_DNN_human_IN_vs_FT_p(layer), sketch_DNN_human_sim_IN(layer),sketch_DNN_human_sim_FT(layer)] = compute_RDM_pairwise_randomization_test(sketch_RDM_IN(:,:,layer), squareform(1-squareform(sketch_RDM_behav)), sketch_RDM_FT(:,:,layer), squareform(1-squareform(sketch_RDM_behav)),1000);

end 

[photo_DNN_human_IN_FT_decision,~,~,photo_DNN_human_IN_FT_adj_p] = fdr_bh(photo_DNN_human_IN_vs_FT_p(5:end),0.05,'dep');
[drawing_DNN_human_IN_FT_decision,~,~,drawing_DNN_human_IN_FT_adj_p] = fdr_bh(drawing_DNN_human_IN_vs_FT_p(5:end),0.05,'dep');
[sketch_DNN_human_IN_FT_decision,~,~,sketch_DNN_human_IN_FT_adj_p] = fdr_bh(sketch_DNN_human_IN_vs_FT_p(5:end),0.05,'dep');

%% compare fit to human behavior for VGG16 SIN and VGG16 FT 

for layer = 1:size(photo_RDM_IN,3)
    
    [photo_DNN_human_SIN_vs_FT_p(layer),photo_DNN_human_sim_IN(layer),photo_DNN_human_sim_FT(layer)] = compute_RDM_pairwise_randomization_test(photo_RDM_IN(:,:,layer), photo_RDM_behav, photo_RDM_FT(:,:,layer), photo_RDM_behav,1000);
    [drawing_DNN_human_SIN_vs_FT_p(layer),  drawing_DNN_human_sim_IN(layer),drawing_DNN_human_sim_FT(layer)] = compute_RDM_pairwise_randomization_test(drawing_RDM_IN(:,:,layer), drawing_RDM_behav, drawing_RDM_FT(:,:,layer), drawing_RDM_behav,1000);
    [sketch_DNN_human_IN_vs_FT_p(layer), sketch_DNN_human_sim_IN(layer),sketch_DNN_human_sim_FT(layer)] = compute_RDM_pairwise_randomization_test(sketch_RDM_IN(:,:,layer), sketch_RDM_behav, sketch_RDM_FT(:,:,layer), sketch_RDM_behav,1000);

end 

[photo_DNN_human_IN_FT_decision,~] = fdr_bh(photo_DNN_human_IN_vs_FT_p(5:end),0.05,'dep');
[drawing_DNN_human_IN_FT_decision,~] = fdr_bh(drawing_DNN_human_IN_vs_FT_p(5:end),0.05,'dep');
[sketch_DNN_human_IN_FT_decision,~] = fdr_bh(sketch_DNN_human_IN_vs_FT_p(5:end),0.05,'dep');

