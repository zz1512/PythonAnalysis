function my_result = kernel_measure_z(ds,argus)
   
    my_result = struct();
    
    F_yy_q = ds.samples(1:35,:);
    F_yy_h = ds.samples(36:70,:);
    S_yy_q = ds.samples(71:105,:);
    S_yy_h = ds.samples(106:140,:);
    
    F_kj_q = ds.samples(141:175,:);
    F_kj_h = ds.samples(176:210,:);
    S_kj_q = ds.samples(211:245,:);
    S_kj_h = ds.samples(246:280,:);
    
    
     r_F_yy = 0;
     for x = 1:35
          r = cosmo_corr(F_yy_q(x,:)',F_yy_h(x,:)');
          r_F_yy = r_F_yy + atanh(r);
     end
     mean_r_F_yy = r_F_yy/35;
     
     r_S_yy = 0;
     for x = 1:35
          r = cosmo_corr(S_yy_q(x,:)',S_yy_h(x,:)');
          r_S_yy = r_S_yy + atanh(r);
     end
     mean_r_S_yy = r_S_yy/35;
     
     r_F_kj = 0;
     for x = 1:35
          r = cosmo_corr(F_kj_q(x,:)',F_kj_h(x,:)');
          r_F_kj = r_F_kj + atanh(r);
     end
     mean_r_F_kj = r_F_kj/35;
     
     r_S_kj = 0;
     for x = 1:35
          r = cosmo_corr(S_kj_q(x,:)',S_kj_h(x,:)');
          r_S_kj = r_S_kj + atanh(r);
     end
     mean_r_S_kj = r_S_kj/35;
     
     observed = mean_r_S_yy-mean_r_F_yy + mean_r_S_kj-mean_r_F_kj;
     
     %%
     num_permutations = 1000;
     permutation_val = zeros(num_permutations, 1);
for i = 1:num_permutations
    % 对原始数据进行重新排列
    permuted_F_yy_q = F_yy_q(randperm(size(F_yy_q,1)),:);
    % 计算每次排列后的zhi
     r_F_yy = 0;
     for x = 1:35
          r = cosmo_corr(permuted_F_yy_q(x,:)',F_yy_h(x,:)');
          r_F_yy = r_F_yy + atanh(r);
     end
     mean_r_F_yy = r_F_yy/35;
     %%
    permuted_S_yy_q = S_yy_q(randperm(size(S_yy_q,1)),:);
    r_S_yy = 0;
     for x = 1:35
          r = cosmo_corr(permuted_S_yy_q(x,:)',S_yy_h(x,:)');
          r_S_yy = r_S_yy + atanh(r);
     end
     mean_r_S_yy = r_S_yy/35;
     %%
     permuted_F_kj_q = F_kj_q(randperm(size(F_kj_q,1)),:);
     r_F_kj = 0;
     for x = 1:35
          r = cosmo_corr(permuted_F_kj_q(x,:)',F_kj_h(x,:)');
          r_F_kj = r_F_kj + atanh(r);
     end
     mean_r_F_kj = r_F_kj/35;
     %%
     permuted_S_kj_q = S_kj_q(randperm(size(S_kj_q,1)),:);
     r_S_kj = 0;
     for x = 1:35
          r = cosmo_corr(permuted_S_kj_q(x,:)',S_kj_h(x,:)');
          r_S_kj = r_S_kj + atanh(r);
     end
     mean_r_S_kj = r_S_kj/35;
     
     %%
    permutation_val(i) = mean_r_S_yy-mean_r_F_yy + mean_r_S_kj-mean_r_F_kj;
end
     
p = sum(permutation_val >= observed) / num_permutations;
my_result.samples = norminv(1 - p);
end

