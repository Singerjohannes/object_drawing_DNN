%% script to run all analyses for the features extracted from VGG16 networks 
%  with or without training on stylized imagenet, with or without finetuning
%  + get some simple plots right after 

clear all 
clc

% specify path where activations are stored 

path = 'F:/final_analysis/final';

% specify which activations to load 

net_name = 'regular_vgg16_imagenetsketches_ft_conv5-1';

% specify where results should be saved 

savepath = 'C:\Users\Johannes\Documents\Leipzig\Masterarbeit\final_results\VGG16_with_finetuning';

% load extracted activations from the network for each depiction seperately

photo_activations = load(fullfile(path, ['all_photo_activations_', net_name]));
drawing_activations = load(fullfile(path, ['all_drawing_activations_', net_name]));
sketch_activations = load(fullfile(path, ['all_sketch_activations_', net_name]));

%% compute RDMs 

[photo_RDM, drawing_RDM, sketch_RDM] = compute_RDMs(photo_activations, drawing_activations, sketch_activations);

%% save RDMs 

save(fullfile(savepath, ['photo_RDM_' , net_name]), 'photo_RDM')
save(fullfile(savepath, ['drawing_RDM_', net_name]), 'drawing_RDM')
save(fullfile(savepath, ['sketch_RDM_', net_name]), 'sketch_RDM')

%% compute betweeen depiction representational similarity across layers 

[photo_drawing_similarity, photo_sketch_similarity, drawing_sketch_similarity] = compute_RDM_sims(photo_RDM, drawing_RDM, sketch_RDM);

%% plot the RDM similarities

layer_names = {'pool_1'; 'pool_2'; 'pool_3'; 'pool_4'; 'pool_5'; 'fc_1'; 'fc_2'};


all_sims = cat(2, photo_drawing_similarity', drawing_sketch_similarity', photo_sketch_similarity')

bar(all_sims)
ylim([0 1])
xticklabels([layer_names])
xlabel('Layer in VGG16')
ylabel('RDM Correlation between Depictions')
legend({'Photo Drawing Similarity'; 'Drawing Sketch Similarity'; 'Photo Sketch Similarity'} ,'Location','northeast')
title('Representational Similarity between Depictions across Layers')

%% save the RDM similarities 

save(fullfile(savepath, ['RDM_sims_',net_name,'.mat']), 'photo_drawing_similarity', 'photo_sketch_similarity', 'drawing_sketch_similarity')

%% compute super RDM 

BIG_RDM = compute_BIG_RDM(photo_activations, drawing_activations, sketch_activations);

%% plot super RDM 

layer_names = {'pool_1'; 'pool_2'; 'pool_3'; 'pool_4'; 'pool_5'; 'fc_1'; 'fc_2'; 'fc_3'};

for layer=1:size(BIG_RDM,3)
subplot(2,4,layer)
imagesc(BIG_RDM(:,:,layer));
title(layer_names{layer})
%axis equal
end 
suptitle(['BIG RDMs across layers for ', net_name])

%% save super RDM

save(fullfile(savepath, ['BIG_RDM_', net_name]), 'BIG_RDM')

%% compute MDS and align the solution to the pooling layer 4 with procrustes alignment 

BIG_MDS_aligned = compute_MDS_procrustes(BIG_RDM);

%% plot the aligned MDS solutions across layers 

for layer=1:length(BIG_MDS_aligned)
subplot(2,4,layer)
hold on
title(['MDS for ' layer_names{layer}])
scatter(BIG_MDS_aligned{layer}(1:42,1),BIG_MDS_aligned{layer}(1:42,2), '*r');
scatter(BIG_MDS_aligned{layer}(43:84,1),BIG_MDS_aligned{layer}(43:84,2), '*b');
scatter(BIG_MDS_aligned{layer}(85:126,1),BIG_MDS_aligned{layer}(85:126,2), '*g');
end 
suptitle(['MDS for all depictions together across layers for ', net_name])
hL = legend('Photos', 'Drawings', 'Sketches')
newPosition = [0.85 0.4 0.2 0.2];
newUnits = 'normalized';
set(hL,'Position', newPosition,'Units', newUnits);

%% save the MDS solutions 

save(fullfile(savepath, ['BIG_MDS_aligned_' , net_name]), 'BIG_MDS_aligned')

%% compute fit to human behavior 

[photo_DNN_behav_sims, drawing_DNN_behav_sims, sketch_DNN_behav_sims] = compute_fit_DNN_behavior(photo_RDM, drawing_RDM, sketch_RDM) ;

%% plot the fit to human behavior 

layer_names = {'pool_1'; 'pool_2'; 'pool_3'; 'pool_4'; 'pool_5'; 'fc_1'; 'fc_2'};


all_sims = cat(2, photo_DNN_behav_sims', drawing_DNN_behav_sims', sketch_DNN_behav_sims');

bar(all_sims)
ylim([0 1])
xticklabels([layer_names])
xlabel('Layer in VGG16')
ylabel('RDM Correlation between DNN and Behavior')
legend({'Photo Similarity'; 'Drawing Similarity'; 'Sketch Similarity'} ,'Location','northeast')
title(['Representational Similarity between ', net_name, ' and Human Behavior across Layers'])

%% save the fit to human behavior 

save(fullfile(savepath,[net_name,'_behav_fit.mat']), 'photo_DNN_behav_sims', 'drawing_DNN_behav_sims', 'sketch_DNN_behav_sims')

%% compute the classification results for the manmade/natural distinctions based on the features extracted from the network across layers

% add libsvm 

addpath(genpath('C:\Users\Johannes\Documents\Leipzig\Modelling\Matlab Scripts\libsvm3.17'))

% do cross/decoding

fprintf(['Now running decoding for ', net_name, '\n'])
decoding_results = do_decoding_VGG16(photo_activations, drawing_activations, sketch_activations, 1000);
save(fullfile(savepath,['decoding_results_', net_name]),'-struct' ,'decoding_results')
fprintf(['Saved decoding results for ', net_name, '\n'])
fprintf(['Now running crossdecoding for ', net_name, '\n'])
crossdecoding_results = do_crossdecoding_VGG16(photo_activations, drawing_activations, sketch_activations, 1000);
save(fullfile(savepath, ['crossdecoding_results_', net_name]),'-struct', 'crossdecoding_results')
fprintf(['Saved crossdecoding results for ', net_name,'\n'])

%% plotting for decoding  

layer_names = {'pool_1'; 'pool_2'; 'pool_3'; 'pool_4'; 'pool_5'; 'fc_1'; 'fc_2'; 'fc_3'};

all_accs = cat(2, decoding_results.final_photo_acc', decoding_results.final_drawing_acc', decoding_results.final_sketch_acc');
all_se = cat(2, decoding_results.final_photo_se', decoding_results.final_drawing_se', decoding_results.final_sketch_se');

figure
bar(all_accs-0.5, 'grouped')
xticklabels([layer_names])
yticks([-0.1:0.1:0.5])
yticklabels([0.4:0.1:1])
xlabel('Layer in VGG16')
ylabel('Mean Classification Accuracy')
title('Mean Classification Accuracy across Layers For Different Depictions')

hold on
% Find the number of groups and the number of bars in each group

ngroups = size(all_accs, 1);
nbars = size(all_accs, 2);
% Calculate the width for each bar group

groupwidth = min(0.8, nbars/(nbars + 1.5));

% Set the position of each error bar in the centre of the main bar
% Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
for i = 1:nbars
    % Calculate center of each bar
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, all_accs(:,i)-0.5, all_se(:,i), 'k', 'linestyle', 'none');
end
legend({'Photos'; 'Drawings'; 'Sketches'} ,'Location','northwest')

%% plotting for crossdecoding

layer_names = {'pool_1'; 'pool_2'; 'pool_3'; 'pool_4'; 'pool_5'; 'fc_1'; 'fc_2'; 'fc_3'};

all_accs = cat(2, crossdecoding_results.final_photo_drawing_acc', crossdecoding_results.final_photo_sketch_acc',crossdecoding_results.final_drawing_sketch_acc');
all_se = cat(2, crossdecoding_results.final_photo_drawing_se', crossdecoding_results.final_photo_sketch_se', crossdecoding_results.final_drawing_sketch_se');

figure
bar(all_accs-0.5, 'grouped')
xticklabels([layer_names])
xlabel('Layer in VGG16')
yticks([-0.1:0.1:0.3])
yticklabels([0.4:0.1:0.8])
ylabel('Mean Classification Accuracy')
title(['Mean classification accuracy across layers for decoding across depictions for network ', net_name])

hold on
% Find the number of groups and the number of bars in each group

ngroups = size(all_accs, 1);
nbars = size(all_accs, 2);
% Calculate the width for each bar group

groupwidth = min(0.8, nbars/(nbars + 1.5));

% Set the position of each error bar in the centre of the main bar
% Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
for i = 1:nbars
    % Calculate center of each bar
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, all_accs(:,i)-0.5, all_se(:,i), 'k', 'linestyle', 'none');
end
legend({'Photo-Drawing'; 'Photo-Sketch'; 'Drawing-Sketch'} ,'Location','northwest')