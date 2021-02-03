function [photo_drawing_similarity, photo_sketch_similarity, drawing_sketch_similarity] = compute_RDM_sims(photo_RDM, drawing_RDM, sketch_RDM)

% compute similarities

photo_sketch_similarity=[];
photo_drawing_similarity=[];
sketch_drawing_similarity=[];

for i=1:size(photo_RDM,3)
    
    photo_sketch_similarity(i) = corr(squareform(photo_RDM(:,:,i))', squareform(sketch_RDM(:,:,i))', 'Type', 'Spearman');
    
    photo_drawing_similarity(i) = corr(squareform(photo_RDM(:,:,i))', squareform(drawing_RDM(:,:,i))', 'Type', 'Spearman');
    
    drawing_sketch_similarity(i) = corr(squareform(sketch_RDM(:,:,i))', squareform(drawing_RDM(:,:,i))', 'Type', 'Spearman');
    
end
end 