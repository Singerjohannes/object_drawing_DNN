function [final_photo_drawing_acc, final_photo_sketch_acc,final_drawing_sketch_acc] = do_crossdecoding_VGG16(photo_activations, drawing_activations, sketch_activations, n_iter, category_vector, passed_kernel)

% classify manmade/natural in each layer for different depictions - but train
% on one depiction but test on the other
% input: the activation cell arrays for each depiction seperately
% (*_activations) , the number of cross-validation iterations (n_iter), the
% category vector in case a shuffled category vector needs to be passed
% (category_vector), and the precomputed kernel for speeding up the
% analysis (passed_kernel) 
% output : decoding accuracies for training on photos and testing on
% drawings (final_photo_drawing_acc), for training on photos and testing on
% sketches (final_photo_sketch_acc), and for training on drawings and
% testing on sketches (final_drawing_sketch_acc) 

if ~exist('passed_kernel','var')
    fn = fieldnames(sketch_activations);
    lfn = length(fn);
else
    lfn = size(sketch_activations,3);
end

if nargin<5
    category_vector = [9 1 8 3 7 1 2 9 10 9 2 8 3 9 9 5 8, ...
        4 2 2 4 4 10 6 2 10 8 9 2 5 6 2 7 4, ...
        5 7 9 2 4 9 10 2]';
end

labels = 2*(category_vector < 6)-1;

% intialize accuracies for all layers
acc_all = zeros(n_iter,lfn,3);

% classification loop

for layer=1:lfn
    
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
        
        xclass_kernel_photo_drawing = [data_all(:,:,1); data_all(:,:,2)] * [data_all(:,:,1); data_all(:,:,2)]';
        xclass_kernel_photo_sketch = [data_all(:,:,1); data_all(:,:,3)] * [data_all(:,:,1); data_all(:,:,3)]';
        xclass_kernel_drawing_sketch = [data_all(:,:,2); data_all(:,:,3)] * [data_all(:,:,2); data_all(:,:,3)]';
        
    else
        xclass_kernel_photo_drawing = photo_activations(:,:,layer);
        xclass_kernel_photo_sketch =  drawing_activations(:,:,layer);
        xclass_kernel_drawing_sketch = sketch_activations(:,:,layer); 
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
        
        testind = find(TESTIND(:,i));
        trainind = find(TRAININD(:,i));
     
        label_train = labels(trainind);
        
        % select training data for all depictions
        data_train_kernel = kernel_all(trainind,trainind,:);
        
        label_test = labels(testind);
        % select test data for all depictions
        xclass_test_photo_drawing = xclass_kernel_photo_drawing(testind+length(category_vector),trainind,:);
        xclass_test_photo_sketch = xclass_kernel_photo_sketch(testind+length(category_vector),trainind,:);
        xclass_test_drawing_sketch = xclass_kernel_drawing_sketch(testind+length(category_vector),trainind,:);
        
        % train model for every depiction
        model_photo = svmtrain(label_train,[(1:size(data_train_kernel(:,:,1),1))' data_train_kernel(:,:,1)],'-s 0 -t 4 -c 1 -b 0 -q'); %#ok<SVMTRAIN>
        model_drawing = svmtrain(label_train,[(1:size(data_train_kernel(:,:,2),1))' data_train_kernel(:,:,2)],'-s 0 -t 4 -c 1 -b 0 -q'); %#ok<SVMTRAIN>
        
        % get prediction across depictions
        predicted_label_photo_drawing = svmpredict(label_test,[(1:size(xclass_test_photo_drawing,1))'  xclass_test_photo_drawing],model_photo,'-q');
        predicted_label_photo_sketch = svmpredict(label_test,[(1:size(xclass_test_photo_sketch,1))'  xclass_test_photo_sketch],model_photo,'-q');
        predicted_label_drawing_sketch = svmpredict(label_test,[(1:size(xclass_test_drawing_sketch,1))'  xclass_test_drawing_sketch],model_drawing,'-q');
        
        % assign accuracies for this iteration
        acc_all(i,layer,1) = mean(predicted_label_photo_drawing == label_test);
        acc_all(i,layer,2) = mean(predicted_label_photo_sketch == label_test);
        acc_all(i,layer,3) = mean(predicted_label_drawing_sketch == label_test);
        
    end
    
    % get the mean and std error for every layer of the iterations
    final_photo_drawing_acc(layer) = mean(acc_all(:,layer,1));
    
    final_photo_sketch_acc(layer) = mean(acc_all(:,layer,2));
    
    final_drawing_sketch_acc(layer) = mean(acc_all(:,layer,3));
end
