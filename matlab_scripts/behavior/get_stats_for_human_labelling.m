%% compute mean and SE for labelling data only for the imagenet images 

clc 
clear all

sketch_results = load('C:\Users\Johannes\Documents\Leipzig\Behavior\labelling\results/results_table_label_sketches.mat');
photo_results = load('C:\Users\Johannes\Documents\Leipzig\Behavior\labelling\results/results_table_label_photos.mat');
drawing_results = load('C:\Users\Johannes\Documents\Leipzig\Behavior\labelling\results/results_table_label_drawings.mat');


%% get the selection vector for the imagenet images 


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

%% save the selection 

save_path = 'C:\Users\Johannes\Documents\Leipzig\Masterarbeit\final_results\behavior'

sel_photo_results = photo_results.results_table.Accuracy(sel_vector)/40;
save(fullfile(save_path, 'photo_accs_per_image'), 'sel_photo_results');

sel_drawing_results = drawing_results.results_table.Accuracy(sel_vector)/40;
save(fullfile(save_path, 'drawing_accs_per_image'), 'sel_drawing_results');

sel_sketch_results = sketch_results.results_table.Accuracy(sel_vector)/40;
save(fullfile(save_path, 'sketch_accs_per_image'), 'sel_sketch_results');

%% compute mean and SE for the selection of imagenet images 

mean_accs(1) = mean(photo_results.results_table.Accuracy(sel_vector)/40);
stderrs(1) = (std(photo_results.results_table.Accuracy(sel_vector)/40))/sqrt(length(imagenet_fn));
stds(1) = std(photo_results.results_table.Accuracy(sel_vector)/40);

mean_accs(2) = mean(drawing_results.results_table.Accuracy(sel_vector)/40);
stderrs(2) = (std(drawing_results.results_table.Accuracy(sel_vector)/40))/sqrt(length(imagenet_fn));
stds(2) = std(drawing_results.results_table.Accuracy(sel_vector)/40);


mean_accs(3) = mean(sketch_results.results_table.Accuracy(sel_vector)/40);
stderrs(3) = (std(sketch_results.results_table.Accuracy(sel_vector)/40))/sqrt(length(imagenet_fn));
stds(3) = std(sketch_results.results_table.Accuracy(sel_vector)/40);
