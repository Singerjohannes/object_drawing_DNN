function [emp_corr, pval, shuffled_corr] = compute_RDM_perm_test(this_RDM, that_RDM, n_iter) 

%%% function to perform a permutation test for RDM correlations 
%%% inputs: two RDMs, number of permutations 
%%% outputs : p_value of the empirical correlation
n_items = size(that_RDM,1);

ind = tril(true(n_items),-1);

shuffled_corr= zeros(n_iter,1);

this_RDV = this_RDM(ind);

emp_corr = corr(this_RDV, that_RDM(ind), 'type', 'spearman');

randmat = rand(n_items,n_iter);
[~,permmat] = sort(randmat);

for i=1:n_iter
    that_RDM_perm = that_RDM(permmat(:,i),permmat(:,i)); 
    shuffled_corr(i) = corr(this_RDV, that_RDM_perm(ind), 'type','spearman');
end 

shuffled_corr= [emp_corr; shuffled_corr];

% get the p-value by finding the percentile of your empirical value in the
% permutated distribution

pval = mean(abs(shuffled_corr) >= abs(emp_corr));

end