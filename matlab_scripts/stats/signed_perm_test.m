function [p_val, emp_value, distribution] = signed_perm_test(obs_val, fix_val, n_perm) 

%%% function for computing a signed permutation test 
%%% inputs: 
%%% the values per observation (val_per_obs), 
%%% the value to compare your observations with (fix_val)
%%% and the number of permutations you want to perform for your test 
%%% outputs: 
%%% the significance value for the test (p_val)
%%% the empirical value which the permutated distribution is tested against (emp_value)
%%% and the distribution which is obtained using random permutation of your
%%% data 

% compute the empirical value by substracting the fixed value from every observation and sum the results  

emp_value = sum(obs_val - fix_val)/length(obs_val); 

% do the permutation and obtain the distribution 

distribution = ((obs_val - fix_val)' * (2*round(rand(length(obs_val),n_perm))-1))/length(obs_val);

distribution = [emp_value distribution];

% get the p-value by finding the percentile of your empirical value in the
% permutated distribution

p_val = mean(distribution>=emp_value);

end 