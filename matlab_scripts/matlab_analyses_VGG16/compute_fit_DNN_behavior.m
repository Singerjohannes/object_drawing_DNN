function [photo_DNN_behav_sims, drawing_DNN_behav_sims, sketch_DNN_behav_sims] = compute_fit_DNN_behavior(photo_RDM, drawing_RDM, sketch_RDM) 

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