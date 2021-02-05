function BIG_MDS_aligned = compute_MDS_procrustes(BIG_RDM)

% do MDS


for layer=1:size(BIG_RDM,3)
    
BIG_MDS{layer} = mdscale(BIG_RDM(:,:,layer),2);

end 

% procrustes alignment to pooling layer 4

BIG_MDS_aligned = [];
BIG_MDS_aligned{4} = BIG_MDS{4};

for i=3:-1:1
    
    [~,BIG_MDS_aligned{i}] = procrustes(BIG_MDS_aligned{i+1}, BIG_MDS{i});
end 

for i=5:7

    [~,BIG_MDS_aligned{i}] = procrustes(BIG_MDS_aligned{i-1}, BIG_MDS{i});
end
end 