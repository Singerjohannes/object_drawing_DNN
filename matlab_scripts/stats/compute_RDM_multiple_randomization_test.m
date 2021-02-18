function [p_val, dist] = compute_RDM_multiple_randomization_test(RDM_vec1, RDM_vec2, n_iter) 

%%% function to perform a one-sided randomization test for a series of RDM correlations 
%%% inputs: two RDM-vectors each containing a series of RDMs, number of permutations
%%% outputs : p-value and the distribution of shuffled sum of squares

% setup the shuffle vector once for all layers 

n_items = size(RDM_vec1,1);
ind = tril(true(n_items),-1);
randmat = rand(n_items,n_iter);
[~,permmat] = sort(randmat);

for layer = 1: size(RDM_vec1,3)
    
    emp_corr(layer) = corr(squareform(RDM_vec1(:,:,layer))', squareform(RDM_vec2(:,:,layer))', 'type', 'spearman');

    shuffled_corr(:,layer)= zeros(n_iter,1);

    this_RDV = RDM_vec2(:,:,layer);
    this_RDV = this_RDV(ind); 

    for i=1:n_iter
   
        this_RDM_perm = RDM_vec1(permmat(:,i),permmat(:,i),layer); 
    
        shuffled_corr(i,layer) = corr(this_RDV, this_RDM_perm(ind), 'type', 'spearman');
        
    end 
end
    
    % compute sums of squared differences for every shuffle iteration
    
    shuffled_z = atanh(shuffled_corr);
    
    mean_z = mean(shuffled_z,2);
    
    shuffled_ss = tanh(sum((shuffled_z-mean_z).^2,2));
    
    % compute empirical sums of squared differences
    
    emp_z = atanh(emp_corr); 
    
    mean_emp_z = mean(emp_corr); 
    
    emp_ss = tanh(sum((emp_z-mean_emp_z).^2,2)); 
    
    shuffled_ss = [emp_ss ;shuffled_ss];
    
    % get the p-value by finding the percentile of your empirical value in the
    % permuted distribution
    p_val = mean(shuffled_ss>=emp_ss);
end 
