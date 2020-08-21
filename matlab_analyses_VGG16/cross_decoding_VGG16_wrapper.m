%% decoding for VGG16 with and without SIN and save results (wrapper) 

clc 
clear all

% add libsvm 

addpath(genpath('C:\Users\Johannes\Documents\Leipzig\Modelling\Matlab Scripts\libsvm3.17'))

%load all activations 

loadpath = 'F:\final_analysis';
save_path = 'C:\Users\Johannes\Documents\Leipzig\Masterarbeit\final_results\VGG16_with_without_SIN';

%load activations for VGG16
photo_activations = load(fullfile(loadpath, 'all_photo_activations_VGG16'));
drawing_activations = load(fullfile(loadpath, 'all_drawing_activations_VGG16'));
sketch_activations = load(fullfile(loadpath, 'all_sketch_activations_VGG16'));

%load activations for VGG16 SIN
photo_activations_SIN = load(fullfile(loadpath, 'all_photo_activations_VGG16_SIN'));
drawing_activations_SIN = load(fullfile(loadpath, 'all_drawing_activations_VGG16_SIN'));
sketch_activations_SIN = load(fullfile(loadpath, 'all_sketch_activations_VGG16_SIN'));

% do the decoding and cross decoding and save results 

fprintf(['Now running decoding for VGG16 without SIN\n'])
decoding_results = do_decoding_VGG16(photo_activations, drawing_activations, sketch_activations, 100);
save(fullfile(save_path,'decoding_results_VGG16'),'-struct', 'decoding_results')
fprintf(['Saved decoding results for VGG16\n'])
fprintf(['Now running decoding for VGG16 with SIN\n'])
decoding_results_SIN = do_decoding_VGG16(photo_activations_SIN, drawing_activations_SIN, sketch_activations_SIN, 100);
save(fullfile(save_path,'decoding_results_VGG16_SIN'),'-struct', 'decoding_results_SIN')
fprintf(['Saved decoding results for VGG16 SIN\n'])
fprintf(['Now running crossdecoding for VGG16 without SIN\n'])
crossdecoding_results = do_crossdecoding_VGG16(photo_activations, drawing_activations, sketch_activations, 100);
save(fullfile(save_path,'crossdecoding_results_VGG16'),'-struct', 'crossdecoding_results')
fprintf(['Saved crossdecoding results for VGG16\n'])
fprintf(['Now running crossdecoding for VGG16 with SIN\n'])
crossdecoding_results_SIN = do_crossdecoding_VGG16(photo_activations_SIN, drawing_activations_SIN, sketch_activations_SIN, 100);
save(fullfile(save_path,'crossdecoding_results_VGG16_SIN'), '-struct', 'crossdecoding_results_SIN')
fprintf(['Saved crossdecoding results for VGG16 SIN\n'])

% %% plotting for decoding  
% 
% layer_names = {'pool_1'; 'pool_2'; 'pool_3'; 'pool_4'; 'pool_5'; 'fc_1'; 'fc_2'; 'fc_3'};
% 
% all_accs = cat(2, decoding_results.final_photo_acc', decoding_results.final_drawing_acc', decoding_results.final_sketch_acc');
% all_se = cat(2, decoding_results.final_photo_se', decoding_results.final_drawing_se', decoding_results.final_sketch_se');
% 
% figure
% bar(all_accs-0.5, 'grouped')
% xticklabels([layer_names])
% yticks([-0.1:0.1:0.5])
% yticklabels([0.4:0.1:1])
% xlabel('Layer in VGG16')
% ylabel('Mean Classification Accuracy')
% title('Mean Classification Accuracy across Layers For Different Depictions')
% 
% hold on
% % Find the number of groups and the number of bars in each group
% 
% ngroups = size(all_accs, 1);
% nbars = size(all_accs, 2);
% % Calculate the width for each bar group
% 
% groupwidth = min(0.8, nbars/(nbars + 1.5));
% 
% % Set the position of each error bar in the centre of the main bar
% % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
% for i = 1:nbars
%     % Calculate center of each bar
%     x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
%     errorbar(x, all_accs(:,i)-0.5, all_se(:,i), 'k', 'linestyle', 'none');
% end
% legend({'Photos'; 'Drawings'; 'Sketches'} ,'Location','northwest')
% 
% %% plotting for crossdecoding
% 
% layer_names = {'pool_1'; 'pool_2'; 'pool_3'; 'pool_4'; 'pool_5'; 'fc_1'; 'fc_2'; 'fc_3'};
% 
% all_accs = cat(2, crossdecoding_results.final_photo_drawing_acc', crossdecoding_results.final_photo_sketch_acc',crossdecoding_results.final_final_drawing_sketch_acc');
% all_se = cat(2, crossdecoding_results.final_photo_drawing_se', crossdecoding_results.final_photo_sketch_se', crossdecoding_results.final_drawing_sketch_se');
% 
% figure
% bar(all_accs-0.5, 'grouped')
% xticklabels([layer_names])
% xlabel('Layer in VGG16')
% yticks([-0.1:0.1:0.3])
% yticklabels([0.4:0.1:0.8])
% ylabel('Mean Classification Accuracy')
% title('Mean classification accuracy across layers for decoding across depictions for network BLT')
% 
% hold on
% % Find the number of groups and the number of bars in each group
% 
% ngroups = size(all_accs, 1);
% nbars = size(all_accs, 2);
% % Calculate the width for each bar group
% 
% groupwidth = min(0.8, nbars/(nbars + 1.5));
% 
% % Set the position of each error bar in the centre of the main bar
% % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
% for i = 1:nbars
%     % Calculate center of each bar
%     x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
%     errorbar(x, all_accs(:,i)-0.5, all_se(:,i), 'k', 'linestyle', 'none');
% end
% legend({'Photo-Drawing'; 'Photo-Sketch'; 'Drawing-Sketch'} ,'Location','northwest')
