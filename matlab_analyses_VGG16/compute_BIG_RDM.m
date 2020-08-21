function BIG_RDM = compute_BIG_RDM(photo_activations, drawing_activations, sketch_activations)

% put all activations in one cell array 

all_activations = [];

fn = fieldnames(sketch_activations);

for layer=1:length(fn)-1 %exclude softmax layer
    
    all_activations{layer} = cat(2,photo_activations.(fn{layer})', drawing_activations.(fn{layer})', sketch_activations.(fn{layer})');
end 

% create BIG RDMs for all layers

BIG_RDM = []

for layer = 1: length(all_activations) 

    BIG_RDM(:,:,layer) = 1-corr(double(all_activations{layer}));
    
end 
end 