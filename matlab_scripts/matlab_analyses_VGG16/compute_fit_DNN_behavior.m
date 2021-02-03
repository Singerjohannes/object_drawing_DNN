function [photo_DNN_behav_sims, drawing_DNN_behav_sims, sketch_DNN_behav_sims] = compute_fit_DNN_behavior(photo_RDM, drawing_RDM, sketch_RDM, behav_path) 

% load behavioral data 

load(fullfile(behav_path, 'photo_behav_RDM.mat'));
load(fullfile(behav_path,'drawing_behav_RDM.mat'));
load(fullfile(behav_path,'sketch_behav_RDM.mat'));

% compute similarity between behavioral RDMs and DNN RDMs 

photo_DNN_behav_sims = [];
drawing_DNN_behav_sims = [];
sketch_DNN_behav_sims = [];

for layer = 1:size(photo_RDM,3)

[photo_DNN_behav_sims(layer), photo_DNN_behav_pval(layer)] = corr(1-squareform(photo_RDM_behav)', squareform(photo_RDM(:,:,layer))', 'Type', 'Spearman');

[drawing_DNN_behav_sims(layer), drawing_DNN_behav_pval(layer)] = corr(1-squareform(drawing_RDM_behav)', squareform(drawing_RDM(:,:,layer))', 'Type', 'Spearman');

[sketch_DNN_behav_sims(layer), sketch_DNN_behav_pval(layer)] = corr(1-squareform(sketch_RDM_behav)', squareform(sketch_RDM(:,:,layer))', 'Type', 'Spearman');

end
end 