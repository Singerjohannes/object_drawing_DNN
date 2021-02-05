function [photo_RDM, drawing_RDM, sketch_RDM] = compute_RDMs(photo_activations, drawing_activations, sketch_activations)

% create RDMs for each layer and each condition 

% for photos

photo_RDM=[];
sketch_RDM=[];
drawing_RDM = [];

fn = fieldnames(sketch_activations);

for layer=1:length(fn)-1 % exclude softmax layer
            
photo_RDM(:,:,layer) = 1-corr(double(photo_activations.(fn{layer})'));

drawing_RDM(:,:,layer) = 1-corr(double(drawing_activations.(fn{layer})'));

sketch_RDM(:,:,layer) = 1-corr(double(sketch_activations.(fn{layer})'));

end 
end 