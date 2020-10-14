function [photo_drawing_similarity, photo_sketch_similarity, drawing_sketch_similarity] = compute_RDM_sims(photo_RDM, drawing_RDM, sketch_RDM)

% compute similarities

photo_sketch_similarity=[];
photo_drawing_similarity=[];
sketch_drawing_similarity=[];

for i=1:size(photo_RDM,3)
    
    photo_sketch_similarity(i) = corr(reshape(squareform(photo_RDM(:,:,i)), 1,[])', reshape(squareform(sketch_RDM(:,:,i)),1,[])', 'Type', 'Spearman');
    
end 

for i=1:size(photo_RDM,3)
    
    photo_drawing_similarity(i) = corr(reshape(squareform(photo_RDM(:,:,i)), 1,[])', reshape(squareform(drawing_RDM(:,:,i)),1,[])', 'Type', 'Spearman');
    
end

for i=1:size(photo_RDM,3)
    
    drawing_sketch_similarity(i) = corr(reshape(squareform(sketch_RDM(:,:,i)), 1,[])', reshape(squareform(drawing_RDM(:,:,i)),1,[])', 'Type', 'Spearman');
    
end
end 