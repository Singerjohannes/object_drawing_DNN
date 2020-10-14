%% run stats for classification analyses 

clear all 
clc

% specify path where activations are stored 

path = 'F:/final_analysis/final';

% specify which activations to load 

net_name = 'regular_vgg16_imagenetsketches_ft_conv5-1';%'VGG16_SIN'; %'VGG16_SIN'

% specify if stats should be computed only for the finetuned layers 

is_ft = 1; % 1 if yes, otherwise 0

% specify where results should be saved and loaded

if ~is_ft
savepath = 'C:\Users\Johannes\Documents\Leipzig\Masterarbeit\final_results\VGG16_with_without_SIN';
elseif is_ft
savepath = 'C:\Users\Johannes\Documents\Leipzig\Masterarbeit\final_results\VGG16_with_finetuning';
end 

% load extracted activations from the network for each depiction seperately
% 
% photo_activations = load(fullfile(path, ['all_photo_activations_', net_name]));
% drawing_activations = load(fullfile(path, ['all_drawing_activations_', net_name]));
% sketch_activations = load(fullfile(path, ['all_sketch_activations_', net_name]));

%% load empirical classfication results 

load(fullfile(savepath,['decoding_results_',net_name]))

photo_accs_emp = final_photo_acc;
drawing_accs_emp = final_drawing_acc;
sketch_accs_emp = final_sketch_acc;

load(fullfile(savepath,['crossdecoding_results_',net_name]))

photo_drawing_accs_emp = final_photo_drawing_acc;
photo_sketch_accs_emp = final_photo_sketch_acc;
drawing_sketch_accs_emp = final_drawing_sketch_acc;

%% run the classification with permuation 


% add libsvm 

addpath(genpath('C:\Users\Johannes\Documents\Leipzig\Modelling\Matlab Scripts\libsvm3.17'))

original_vector = [9 1 8 3 7 1 2 9 10 9 2 8 3 9 9 5 8, ...
               4 2 2 4 4 10 6 2 10 8 9 2 5 6 2 7 4, ...
               5 7 9 2 4 9 10 2]';

fn = fieldnames(sketch_activations);
clear xclass_kernel_photo_drawing xclass_kernel_photo_sketch xclass_kernel_drawing_sketch
for layer=length(fn):-1:1
        %initialize data matrix
        data_all = [];
        % assign data for classification
        data_all(:,:,1) = double(photo_activations.(fn{layer}));
        data_all(:,:,2) = double(drawing_activations.(fn{layer}));
        data_all(:,:,3) = double(sketch_activations.(fn{layer}));
        
        for i_kern = 1:3
            kernel_all(:,:,i_kern,layer) = data_all(:,:,i_kern)*data_all(:,:,i_kern)';
        end
        
        xclass_kernel_photo_drawing(:,:,layer) = [data_all(:,:,1); data_all(:,:,2)] * [data_all(:,:,1); data_all(:,:,2)]';
        xclass_kernel_photo_sketch(:,:,layer) = [data_all(:,:,1); data_all(:,:,3)] * [data_all(:,:,1); data_all(:,:,3)]';
        xclass_kernel_drawing_sketch(:,:,layer) = [data_all(:,:,2); data_all(:,:,3)] * [data_all(:,:,2); data_all(:,:,3)]';
end   
        
n_perm = 1000; 


for i=1:n_perm
    
    shuffle_vector = original_vector(randperm(length(original_vector))); 
    fprintf(['Now running decoding for ', net_name, ' Iteration ', num2str(i),'\n'])
    [photo_accs_shuffle(i,:), drawing_accs_shuffle(i,:), sketch_accs_shuffle(i,:)] = do_decoding_VGG16_for_stats(photo_activations, drawing_activations, sketch_activations, 1000, shuffle_vector, kernel_all);
    fprintf(['Now running crossdecoding for ', net_name, ' Iteration ', num2str(i), '\n'])
    [photo_drawing_accs_shuffle(i,:), photo_sketch_accs_shuffle(i,:), drawing_sketch_accs_shuffle(i,:)] = do_crossdecoding_VGG16_for_stats(xclass_kernel_photo_drawing, xclass_kernel_photo_sketch, xclass_kernel_drawing_sketch, 1000,shuffle_vector, kernel_all);
end 

save(fullfile(savepath,['permuted_classification_accuracies_',net_name]), 'photo_accs_shuffle', 'drawing_accs_shuffle', 'sketch_accs_shuffle');
save(fullfile(savepath,['permuted_crossclassification_accuracies_',net_name]), 'photo_drawing_accs_shuffle', 'photo_sketch_accs_shuffle', 'drawing_sketch_accs_shuffle');

%% load shuffled data 

load(fullfile(savepath,['permuted_classification_accuracies_',net_name]));
load(fullfile(savepath,['permuted_crossclassification_accuracies_',net_name]));

%% check for significance with permutation test

for layer = 1:size(photo_accs_emp,2)-1
    
   photo_p(layer)= mean([photo_accs_emp(layer); photo_accs_shuffle(:,layer)]>=photo_accs_emp(layer));
   drawing_p(layer)= mean([drawing_accs_emp(layer);drawing_accs_shuffle(:,layer)]>=drawing_accs_emp(layer));
   sketch_p(layer)= mean([sketch_accs_emp(layer); sketch_accs_shuffle(:,layer)]>=sketch_accs_emp(layer));
   
   photo_drawing_p(layer)= mean([photo_drawing_accs_emp(layer);photo_drawing_accs_shuffle(:,layer)]>=photo_drawing_accs_emp(layer));
   photo_sketch_p(layer)= mean([photo_sketch_accs_emp(layer);photo_sketch_accs_shuffle(:,layer)]>=photo_sketch_accs_emp(layer));
   drawing_sketch_p(layer)= mean([drawing_sketch_accs_emp(layer);drawing_sketch_accs_shuffle(:,layer)]>=drawing_sketch_accs_emp(layer));

end 

if ~is_ft
    [photo_chance_decision,~,~, photo_adj_p] = fdr_bh(photo_p,0.05,'dep');
    [drawing_chance_decision,~,~, drawing_adj_p] = fdr_bh(drawing_p,0.05,'dep');
    [sketch_chance_decision,~,~, sketch_adj_p] = fdr_bh(sketch_p,0.05,'dep');
    
    [photo_drawing_chance_decision,~, ~,photo_drawing_adj_p] = fdr_bh(photo_drawing_p,0.05,'dep');
    [photo_sketch_chance_decision,~,~, photo_sketch_adj_p] = fdr_bh(photo_sketch_p,0.05,'dep');
    [drawing_sketch_chance_decision,~,~, drawing_sketch_adj_p] = fdr_bh(drawing_sketch_p,0.05,'dep');
    
elseif is_ft
    [photo_chance_decision,~,~, photo_adj_p] = fdr_bh(photo_p(5:end),0.05,'dep');
    [drawing_chance_decision,~,~, drawing_adj_p] = fdr_bh(drawing_p(5:end),0.05,'dep');
    [sketch_chance_decision,~,~, sketch_adj_p] = fdr_bh(sketch_p(5:end),0.05,'dep');
    
    [photo_drawing_chance_decision,~, ~,photo_drawing_adj_p] = fdr_bh(photo_drawing_p(5:end),0.05,'dep');
    [photo_sketch_chance_decision,~,~, photo_sketch_adj_p] = fdr_bh(photo_sketch_p(5:end),0.05,'dep');
    [drawing_sketch_chance_decision,~,~, drawing_sketch_adj_p] = fdr_bh(drawing_sketch_p(5:end),0.05,'dep');
end

    
%% check for significance compared to each other 

photo_vs_drawing_emp = photo_accs_emp-drawing_accs_emp;
photo_vs_drawing_shuffle = photo_accs_shuffle-drawing_accs_shuffle;
photo_vs_drawing_shuffle = [photo_vs_drawing_emp; photo_vs_drawing_shuffle];
photo_vs_sketch_emp = photo_accs_emp-sketch_accs_emp;
photo_vs_sketch_shuffle = photo_accs_shuffle-sketch_accs_shuffle;
photo_vs_sketch_shuffle = [photo_vs_sketch_emp; photo_vs_sketch_shuffle];
drawing_vs_sketch_emp = drawing_accs_emp-sketch_accs_emp;
drawing_vs_sketch_shuffle = drawing_accs_shuffle-sketch_accs_shuffle;
drawing_vs_sketch_shuffle = [drawing_vs_sketch_emp; drawing_vs_sketch_shuffle];

for layer = 1:size(photo_accs_emp,2)-1

    % photo vs drawing 
    photo_vs_drawing_p(layer) = mean(photo_vs_drawing_shuffle(:,layer)>=photo_vs_drawing_emp(layer));
    photo_vs_sketch_p(layer) = mean(photo_vs_sketch_shuffle(:,layer)>=photo_vs_sketch_emp(layer));
    drawing_vs_sketch_p(layer) = mean(drawing_vs_sketch_shuffle(:,layer)>=drawing_vs_sketch_emp(layer));

end 

if ~is_ft
    [photo_vs_drawing_decision,~,~,photo_vs_drawing_adj_p] = fdr_bh(photo_vs_drawing_p,0.05,'dep');
    [photo_vs_sketch_decision,~,~,photo_vs_sketch_adj_p] = fdr_bh(photo_vs_sketch_p,0.05,'dep');
    [drawing_vs_sketch_decision,~,~,drawing_vs_sketch_adj_p] = fdr_bh(drawing_vs_sketch_p,0.05,'dep');
elseif is_ft
    [photo_vs_drawing_decision,~,~,photo_vs_drawing_adj_p] = fdr_bh(photo_vs_drawing_p(5:end),0.05,'dep');
    [photo_vs_sketch_decision,~,~,photo_vs_sketch_adj_p] = fdr_bh(photo_vs_sketch_p(5:end),0.05,'dep');
    [drawing_vs_sketch_decision,~,~,drawing_vs_sketch_adj_p] = fdr_bh(drawing_vs_sketch_p(5:end),0.05,'dep');
end
