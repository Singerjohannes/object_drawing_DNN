function [p_val, emp_corr1, emp_corr2,emp_diff,diff_shuffled_corr] = compute_RDM_pairwise_randomization_test(this_RDM1, this_RDM2, that_RDM1,that_RDM2, n_iter) 

%%% function to perform a pairwise randomization test for RDM correlations 
%%% inputs: two RDM-pairs, number of permutations
%%% outputs : observed value of the empirical correlation and the
%%% distribution of shuffled correlation differences

emp_corr1 = corr(squareform(this_RDM1)', squareform(this_RDM2)', 'type', 'spearman');
emp_corr2 = corr(squareform(that_RDM1)', squareform(that_RDM2)', 'type', 'spearman');

shuffled_corr1= zeros(n_iter,1);
shuffled_corr2= zeros(n_iter,1);

n_items = size(this_RDM1,1);

ind = tril(true(n_items),-1);

this_RDV = this_RDM1(ind);
that_RDV= that_RDM1(ind);

randmat = rand(n_items,n_iter);
[~,permmat] = sort(randmat);

for i=1:n_iter
   
    this_RDM_perm = this_RDM2(permmat(:,i),permmat(:,i)); 
    that_RDM_perm = that_RDM2(permmat(:,i),permmat(:,i));
    
    shuffled_corr1(i) = corr(this_RDV, this_RDM_perm(ind), 'type', 'spearman');
   
    shuffled_corr2(i) = corr(that_RDV, that_RDM_perm(ind), 'type', 'spearman');
end 

% apply z-transform
emp_z1 = atanh(emp_corr1);
emp_z2 = atanh(emp_corr2);
shuffled_z1 = atanh(shuffled_corr1);
shuffled_z2 = atanh(shuffled_corr2);

% get the difference and transform back to r-values
if emp_corr1 >emp_corr2
emp_diff = tanh(emp_z1-emp_z2);
diff_shuffled_corr = tanh(shuffled_z1-shuffled_z2);
else 
emp_diff = tanh(emp_z2-emp_z1);
diff_shuffled_corr = tanh(shuffled_z2-shuffled_z1);
end

diff_shuffled_corr = [emp_diff ;diff_shuffled_corr];

% get the p-value by finding the percentile of your empirical value in the
% permutated distribution
p_val = mean(diff_shuffled_corr>=emp_diff); 

end