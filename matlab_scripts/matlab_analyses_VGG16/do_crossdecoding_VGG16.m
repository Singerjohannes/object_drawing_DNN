function all_classification_results = do_crossdecoding_VGG16(photo_activations, drawing_activations, sketch_activations, n_iter)

% classify manmade/natural in each layer for different depictions - train
% on one depiction but test on the other (and vice versa)
% input: the activation cell arrays for each depiction seperately
% (*_activations) , the number of cross-validation iterations (n_iter)
% output : mean decoding accuracies and std. errors for a given train-test
% combination (e.g. photo-drawing) 

fn = fieldnames(sketch_activations);

% specify superordinate category number for every column in the activation matrix 
category_vector = [9 1 8 3 7 1 2 9 10 9 2 8 3 9 9 5 8, ...
               4 2 2 4 4 10 6 2 10 8 9 2 5 6 2 7 4, ...
               5 7 9 2 4 9 10 2]';

% get the labels - 2 = natural , 1 = manmade
labels = (category_vector < 6)+1;

% intialize accuracies for all layers 
acc_all = zeros(n_iter,length(fn),3);

% classification loop 

for layer=1:length(fn)

%initialize data matrix 
data_all = [];
% assign data for classification 
data_all(:,:,1) = double(photo_activations.(fn{layer}));
data_all(:,:,2) = double(drawing_activations.(fn{layer}));
data_all(:,:,3) = double(sketch_activations.(fn{layer}));

%precompute train kernel
for i_kern = 1:3
    kernel_all(:,:,i_kern) = data_all(:,:,i_kern)*data_all(:,:,i_kern)';
end

%precompute test kernels
xclass_kernel_photo_drawing= [data_all(:,:,1); data_all(:,:,2)] * [data_all(:,:,1); data_all(:,:,2)]';
xclass_kernel_photo_sketch = [data_all(:,:,1); data_all(:,:,3)] * [data_all(:,:,1); data_all(:,:,3)]';
xclass_kernel_drawing_sketch = [data_all(:,:,2); data_all(:,:,3)] * [data_all(:,:,2); data_all(:,:,3)]';

for i = 1:n_iter
    
    r_natural = randperm(length(category_vector)/2);
    r_manmade = randperm(length(category_vector)/2)+length(category_vector)/2;
    testind = [r_natural(1:3) r_manmade(1:3)]';
    trainind = setdiff(1:length(category_vector),testind)';
    
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
final_photo_drawing_se(layer) = std(acc_all(:,layer,1))/sqrt(n_iter);

final_photo_sketch_acc(layer) = mean(acc_all(:,layer,2));
final_photo_sketch_se(layer) = std(acc_all(:,layer,2))/sqrt(n_iter);

final_drawing_sketch_acc(layer) = mean(acc_all(:,layer,3));
final_drawing_sketch_se(layer) = std(acc_all(:,layer,3))/sqrt(n_iter);

end 
%% save results 

all_classification_results = struct();

all_classification_results.final_photo_drawing_acc = final_photo_drawing_acc;
all_classification_results.final_photo_drawing_se = final_photo_drawing_se;
all_classification_results.final_photo_sketch_acc = final_photo_sketch_acc;
all_classification_results.final_photo_sketch_se = final_photo_sketch_se;
all_classification_results.final_drawing_sketch_acc = final_drawing_sketch_acc;
all_classification_results.final_drawing_sketch_se = final_drawing_sketch_se;
end 