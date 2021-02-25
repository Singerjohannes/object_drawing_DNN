function [final_photo_acc, final_drawing_acc, final_sketch_acc] = do_decoding_VGG16_for_stats(photo_activations, drawing_activations, sketch_activations, n_iter, category_vector,passed_kernel)

% classify manmade/natural in each layer for different depictions -
% training and testing on the same depiction
% input: the activation cell arrays for each depiction seperately
% (*_activations) , the number of cross-validation iterations (n_iter), the
% category vector in case a shuffled category vector needs to be passed
% (category_vector), and the precomputed kernel for speeding up the
% analysis (passed_kernel) 
% output : decoding accuracies for training and testing on
% photos (final_photo_acc), for training and testing on
% drawings (final_drawing_acc), and for training and
% testing on sketches (final_sketch_acc) 


fn = fieldnames(sketch_activations);

% specify superordinate category number for every column in the activation matrix
if nargin<5
    category_vector = [9 1 8 3 7 1 2 9 10 9 2 8 3 9 9 5 8, ...
        4 2 2 4 4 10 6 2 10 8 9 2 5 6 2 7 4, ...
        5 7 9 2 4 9 10 2]';
else end;

% get the labels - 2 = natural , 1 = manmade
labels = 2*(category_vector < 6)-1;


% intialize accuracies for all layers
acc_all = zeros(n_iter,length(fn),3);

% classification loop

for layer=1:length(fn)
    
    if ~exist('passed_kernel','var')
        %initialize data matrix
        data_all = [];
        % assign data for classification
        data_all(:,:,1) = double(photo_activations.(fn{layer}));
        data_all(:,:,2) = double(drawing_activations.(fn{layer}));
        data_all(:,:,3) = double(sketch_activations.(fn{layer}));
        
        
        for i_kern = 1:3
            kernel_all(:,:,i_kern) = data_all(:,:,i_kern)*data_all(:,:,i_kern)';
        end
    else
        kernel_all = passed_kernel(:,:,:,layer);
    end
    
    n_cat = length(category_vector);
    
    TESTIND = false(n_cat,n_iter);
    [~,randind1] = sort(rand(n_cat/2,n_iter));
    [~,randind2] = sort(rand(n_cat/2,n_iter));
    randind1 = randind1(1:3,:) + [0:n_cat:(n_iter-1)*n_cat];
    randind2 = randind2(1:3,:) + [0:n_cat:(n_iter-1)*n_cat] + n_cat/2;
    TESTIND(randind1(:)) = true;
    TESTIND(randind2(:)) = true;
    TRAININD = ~TESTIND;
    
    for i = 1:n_iter
        
        testind = TESTIND(:,i);
        trainind = TRAININD(:,i);
        
        label_train = labels(trainind);
        
        % select training data for all depictions
        data_train_kernel = kernel_all(trainind,trainind,:);
        
        label_test = labels(testind);
        % select test data for all depictions
        data_test_kernel = kernel_all(testind,trainind,:);
        
        % train model for every depiction
        model_photo = svmtrain(label_train,[(1:size(data_train_kernel(:,:,1),1))' data_train_kernel(:,:,1)],'-s 0 -t 4 -c 1 -b 0 -q'); %#ok<SVMTRAIN>
        model_drawing = svmtrain(label_train,[(1:size(data_train_kernel(:,:,2),1))' data_train_kernel(:,:,2)],'-s 0 -t 4 -c 1 -b 0 -q'); %#ok<SVMTRAIN>
        model_sketch = svmtrain(label_train,[(1:size(data_train_kernel(:,:,3),1))' data_train_kernel(:,:,3)],'-s 0 -t 4 -c 1 -b 0 -q'); %#ok<SVMTRAIN>
        
        % get prediction for all depictions
        predicted_label_photo = svmpredict(label_test,[(1:size(data_test_kernel(:,:,1),1))'  data_test_kernel(:,:,1)],model_photo,'-q');
        predicted_label_drawing = svmpredict(label_test,[(1:size(data_test_kernel(:,:,2),1))'  data_test_kernel(:,:,2)],model_drawing,'-q');
        predicted_label_sketch = svmpredict(label_test,[(1:size(data_test_kernel(:,:,3),1))'  data_test_kernel(:,:,3)],model_sketch,'-q');
        
        % assign accuracies for this iteration
        acc_all(i,layer,1) = mean(predicted_label_photo == label_test);
        acc_all(i,layer,2) = mean(predicted_label_drawing == label_test);
        acc_all(i,layer,3) = mean(predicted_label_sketch == label_test);
    end
    
    % get the mean and std error for every layer of the iterations
    final_photo_acc(layer) = mean(acc_all(:,layer,1));
    
    final_drawing_acc(layer) = mean(acc_all(:,layer,2));
    
    final_sketch_acc(layer) = mean(acc_all(:,layer,3));
    
end
end



