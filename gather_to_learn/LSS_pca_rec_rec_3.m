%% recontruct with first leading pc
clear;clc;
subjects = {'sub-01';'sub-02';'sub-03';'sub-04';'sub-05';'sub-06';'sub-07';'sub-08';'sub-09';'sub-10';'sub-11';'sub-12';'sub-13';'sub-14';'sub-15';'sub-16';'sub-17';'sub-18';'sub-19';'sub-20';'sub-21';'sub-22';'sub-23';'sub-24';'sub-25';'sub-26';'sub-27';'sub-28'};
masks = {'sphere_8--6_-60_32.nii'}; % H:\metaphor\LSS\words_with_juzi\run12_with_run3\result\sphere_8--6_-60_32.nii
study_path='H:\metaphor\LSS\rsa\pattern_similarity_change\mvpa_pattern';
roi_path = 'H:\metaphor\mask'; 
n_subjects = numel(subjects);
msk = masks{1};
pcpc = 3;

counter = 0;
for s = 1:length(subjects)
    sub = subjects{s};
    sub_path=fullfile(study_path,sub);
    mask_fn=fullfile(roi_path,msk);
    %%  run7 kj
    S_kj_s = fullfile(sub_path,'F_kj_start_word.nii');
    S_kj_s_data = cosmo_fmri_dataset(S_kj_s,'mask',mask_fn);
    
    S_kj_e = fullfile(sub_path,'F_kj_end_word.nii');
    S_kj_e_data = cosmo_fmri_dataset(S_kj_e,'mask',mask_fn);
    
    run7=(S_kj_s_data.samples + S_kj_e_data.samples)*0.5;
   
    input = run7'; 
    %% juzi1 kj
    syyjuzi = fullfile(sub_path,'F_kj_juzi.nii');
    syyjuzi = cosmo_fmri_dataset(syyjuzi,'mask',mask_fn);
    juzi_kj = syyjuzi.samples'; 
    %%
    [coeff,score,~,~,explained,mu]=pca(input);
     
    total_explained=0;
    
%     for m = 1:length(explained)
%         total_explained = total_explained + explained(m);
%         if total_explained > EV
%             break
%         end
%     end
    
   %num_pcs = m;
   num_pcs = pcpc;
   top_pcs = coeff(:,1:num_pcs);
   %full_rec = score * coeff'+ mu;
   reconstructed_data = score(:, 1:num_pcs) * top_pcs'+mu;

   predict = juzi_kj;
   
%    predict_dsm = cosmo_pdist(predict', 'correlation');
%    rec_dsm = cosmo_pdist(reconstructed_data', 'correlation');
%    r_hsc = cosmo_corr(rec_dsm(:), predict_dsm(:), 'Spearman');
   %%
   rkj = 0;
   
   for x = 1:size(input,2)
         %r = 1 - pdist2(predict(x,:), reconstructed_data(x,:), 'cosine');
         r = corr(predict(:,x),reconstructed_data(:,x));
         rkj= rkj + r;
   end

   %% yy   
   %%  run7 yy
    S_kj_s = fullfile(sub_path,'F_yy_start_word.nii');
    S_kj_s_data = cosmo_fmri_dataset(S_kj_s,'mask',mask_fn);
    
    S_kj_e = fullfile(sub_path,'F_yy_end_word.nii');
    S_kj_e_data = cosmo_fmri_dataset(S_kj_e,'mask',mask_fn);
    
    run7=(S_kj_s_data.samples + S_kj_e_data.samples)*0.5;
   
    input = run7'; 
    %% juzi1 kj
    syyjuzi = fullfile(sub_path,'F_yy_juzi.nii');
    syyjuzi = cosmo_fmri_dataset(syyjuzi,'mask',mask_fn);
    juzi_yy = syyjuzi.samples'; 
    
    
    
    [coeff,score,~,~,explained,mu]=pca(input);
     
    total_explained=0;
    
%     for m = 1:length(explained)
%         total_explained = total_explained + explained(m);
%         if total_explained > EV
%             break
%         end
%     end
    
   %num_pcs = m;
   num_pcs = pcpc;
   
   top_pcs = coeff(:,1:num_pcs);
   %full_rec = score * coeff'+ mu;
   reconstructed_data = score(:, 1:num_pcs) * top_pcs'+mu;

   predict = juzi_yy;
   %[A,B,rr,U,V] = canoncorr(predict,reconstructed_data);
  
  
   %%
   ryy = 0;
   for x = 1:size(input,2)
         %r = 1 - pdist2(predict(x,:), reconstructed_data(x,:), 'cosine');
       %  r = corr(predict(:,x),reconstructed_data(:,x));
         r = corr(predict(:,x),reconstructed_data(:,x));
         ryy= ryy + r;
   end
   
   counter = counter + 1;

    kj(counter,:) = rkj;
    yy(counter,:) = ryy;
    
%     H_rr(counter,:) = rhh;
%     L_rr(counter,:) = rll;
    
end

[h,p,ci,stats] = ttest(kj,yy)
% [h3,p3,ci3,stats3] = ttest(H_rr,L_rr)




