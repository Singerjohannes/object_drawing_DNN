function SVM_42(cond) 
%
% This function can be used train Support Vector Machine classifiers 
% on the activations from the ImageNet-Sketch dataset separately 
% for every layer. The SVMs are trained on only the activations from the classes 
% corresponding to the 42 object categories in the experimental stimulus set 
% to classify the category of the image. In case there is more than one class 
% in ImageNet-Sketch corresponding to a given object category 
% (e.g., multiple dog classes for the object category dog) we randomly sample 
% 50 of these activations for that given object category. 
% Finally, the classifiers are evaluated on the activation patterns extracted 
% from VGG-16 for the drawing or sketch images from the experimental stimulus set
% separately for both types of depiction. This yields classification accuracies across layers for both drawings and sketches.
%
% Inputs: 
% cond = 2 if drawing and 3 if sketch activations should be used. 
%
% Outputs: 
% results struct containing accuracies for every layer and the trained
% model along with other information on the SVM procedure. 
%
% setup paths
addpath(genpath('object_drawing_DNN/libsvm3.17/'))
act_path = '/object_drawing_DNN/SVM_42/activations';
exp_act_path = '/object_drawing_DNN/SVM_42/VGG16_activations';
idx_path = '/object_drawing_DNN/python_scripts/imnet_mapping';

% get class names

classes = dir(fullfile(act_path, '*.mat'));
classes = {classes.name}';

% setup some parameters
n_obj = 42;
label_placeholder = [ones(21,1); 2*ones(21,1)]; % this is a placeholder
conds = {'photo'; 'drawing'; 'sketch'};

% load imagenet names and get the labels for the predicted inds
fname = fullfile(idx_path,'imagenet_class_index.json');
fid = fopen(fname);
raw = fread(fid,inf);
str = char(raw');
fclose(fid);
imagenet_ids = jsondecode(str);

% get all possible obj idxs
obj_idxs_files = dir(fullfile(idx_path,'*.txt'));
obj_idxs_files = {obj_idxs_files.name}';

% setup results structure 
results = struct();

%% find the class indices that correspond to the objects in experimental stimulus set 

cls_id_obj_id_mapping = [];

for obj = 1: n_obj
    
    % get the wordnet identifiers for a given object 
    possible_obj_ind = importdata(fullfile(obj_idxs_path,obj_idxs_files{obj}));
    % format the identifiers
    for idx = 2:length(possible_obj_ind)
        possible_obj_ind{idx} = possible_obj_ind{idx}(2:end);
    end
    % loop through to class idxs to find which cls_idxs fit with the
    % identifiers 
    for cls_idx = 1:length(classes)
        mapping_vec = [];
        if any(ismember(imagenet_ids.(['x' num2str(cls_idx-1)]){1},possible_obj_ind))
            
            cls_id_obj_id_mapping(cls_idx,:)= [mapping_vec,obj];
        end
        
    end
end 
    
%% load training data 

% load experimental activations
fprintf('Loading VGG16 activations for %s \n', conds{cond});
exp_act = load(fullfile(exp_act_path, ['all_' conds{cond} '_activations_VGG16.mat']));
layers = fieldnames(exp_act);

% loop through layers and initialize training data matrix for every layer
fprintf('Initilaizing training data matrices \n');
train_data = struct();

for layer = 1:length(layers)-1
    
    n_feat = size(exp_act.(layers{layer}),2);
    train_data(layer).X = zeros(51*n_obj,n_feat,'double');
    train_data(layer).labels = zeros(51*n_obj,1, 'double');
    
    mem = whos;
    fprintf('Memory in use %3f GB monitored by MATLAB; Layer %s initialized  \n' ,sum([mem.bytes])*1e-9,layers{layer});
    
end
cnt = 0; 

% loop over objects in the experimental stimulu (42) and for each object
% get a sample of CNN activations 
for obj = 1: n_obj
    
    im_net_idx = find(cls_id_obj_id_mapping==obj);
    if length(im_net_idx) > 1 % if there are more appropiate classes in imagenet for the object then subsample
        sampleind = cnt+1:cnt+50; 
        for layer = 1:length(layers)-1
            tmp = [];
            for i = 1:length(im_net_idx)
                this_act = load(fullfile(act_path,classes{im_net_idx(i)}));
                tmp = [tmp; this_act.(layers{layer})];
            end
            if layer ==1
                rand_sampleind = randsample(size(tmp,1),50);
            end
            subsample_tmp = tmp(rand_sampleind,:);
            %fprintf('Size subsample %i %i \n', size(subsample_tmp));
            train_data(layer).X(sampleind,:) = subsample_tmp;
            train_data(layer).labels(sampleind) = obj;
        end
        cnt = cnt+size(subsample_tmp,1); % need to check if dim1 or dim2
        clear tmp
    else % if there is just one corresponding class in imagenet then just take all images
    tmp = load(fullfile(act_path,classes{im_net_idx}));
    sampleind = cnt+1:cnt+size(tmp.(layers{1}),1); 
    % loop through layers to sample the activations
    for layer = 1:length(layers)-1
        tmp_layer = tmp.(layers{layer});
        train_data(layer).X(sampleind,:) = tmp_layer;
        train_data(layer).labels(sampleind) = obj;
    end
    cnt = cnt+size(tmp_layer,1);
    clear tmp
    end 
end

% calculate the SVM kernel to save memory
for layer = length(layers)-1:-1:1
    fprintf('Calculating Kernel for layer %s \n', layers{layer});
    train_data(layer).labels(cnt+1:end) = [];
    train_data(layer).X(cnt+1:end,:) = [];
    if layer ==1
    train_ind = [1:size(train_data(layer).X,1)];
    test_ind = [size(train_data(layer).X,1)+1:size(train_data(layer).X,1)+size(exp_act.(layers{layer}),1)];
    end 
    train_data(layer).X = [train_data(layer).X; exp_act.(layers{layer})];
    train_data(layer).X = train_data(layer).X*train_data(layer).X';
end

%% train and test the SVM

for layer = 1:length(layers)-1
    
    fprintf('Starting SVM training for layer %s \n', layers{layer});
    
    
    data_train_kernel = double(train_data(layer).X(train_ind,train_ind)); 
    data_test_kernel = double(train_data(layer).X(test_ind,train_ind)); 
        
    model = svmtrain(train_data(layer).labels, [(1:size(data_train_kernel,1))', data_train_kernel], '-s 0 -t 4 -c 1 -b 0 -q');
    
    %assign model to results structure
    results.model(layer) = model;
    
    fprintf('Starting SVM testing for condition %s and layer %s \n',conds{cond}, layers{layer});
    
    results.predicted_inds(:,layer) = svmpredict(label_placeholder, [(1:size(data_test_kernel,1))' data_test_kernel], model, '-q');
    
    results.all_accs(:,layer) = mean(results.predicted_inds(:,layer) == [1:42]');
    fprintf('Accuracy for condition %s and layer %s = %2f % \n', conds{cond},layers{layer},results.all_accs(:,layer)*100);
    fprintf('Done with layer %s \n',layers{layer});
end

fprintf('Saving results\n');
save(['SVM_42_results_' ,conds{cond},'.mat'], '-struct','results',  '-v7.3')
end 
