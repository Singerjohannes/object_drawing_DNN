function [p_val, emp_corr1, emp_corr2,emp_diff,diff_shuffled_corr] = compute_RDM_bootstrap_correlation(this_RDM1, this_RDM2, that_RDM1,that_RDM2, n_iter, n_sel) 

%%% function to perform a bootstrap for RDM correlations 
%%% inputs: two RDMs, number of permutations, number of rows and columns
%%% that should be selected from the RDM 
%%% outputs : observed value of the empirical correlation and the lower and
%%% upper bounds for the confidence interval of the correlation

emp_corr1 = corr(squareform(this_RDM1)', squareform(this_RDM2)', 'type', 'spearman');
emp_corr2 = corr(squareform(that_RDM1)', squareform(that_RDM2)', 'type', 'spearman');

shuffled_corr1= zeros(n_iter,1);
shuffled_corr2= zeros(n_iter,1);

this_selmat = randi(length(squareform(this_RDM1)),length(squareform(this_RDM1)),n_iter);

sel_this_RDM1 = squareform(this_RDM1);
sel_this_RDM2 = squareform(this_RDM2);
sel_that_RDM1 = squareform(that_RDM1);
sel_that_RDM2 = squareform(that_RDM2);

for i=1:n_iter
   
    this_sel = this_selmat(:,i);

    shuffled_corr1(i) = corr(sel_this_RDM1(this_sel)', sel_this_RDM2(this_sel)', 'type', 'spearman');
   
    shuffled_corr2(i) = corr(sel_that_RDM1(this_sel)', sel_that_RDM2(this_sel)', 'type', 'spearman');
end 

% make Z-transform
emp_z1 = atanh(emp_corr1);
emp_z2 = atanh(emp_corr2);
shuffled_z1 = atanh(shuffled_corr1);
shuffled_z2 = atanh(shuffled_corr2);

% get the difference
if emp_corr1 >emp_corr2
emp_diff = tanh(emp_z1-emp_z2);
diff_shuffled_corr = tanh(shuffled_z1-shuffled_z2);
else 
emp_diff = tanh(emp_z2-emp_z1);
diff_shuffled_corr = tanh(shuffled_z2-shuffled_z1);
end

diff_shuffled_corr = [0 ;diff_shuffled_corr];
% get the p-value by finding the percentile of your empirical value in the
% permutated distribution
if emp_diff == 0; p_val=0; else 
p_val = mean(diff_shuffled_corr<=0); end 

end