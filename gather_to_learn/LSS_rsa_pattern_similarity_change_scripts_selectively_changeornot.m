% pattern similarity change
clear;clc;
subjects = {'sub-01';'sub-02';'sub-03';'sub-04';'sub-05';'sub-06';'sub-07';'sub-08';'sub-09';'sub-10';'sub-11';'sub-12';'sub-13';'sub-14';'sub-15';'sub-16';'sub-17';'sub-18';'sub-19';'sub-20';'sub-21';'sub-22';'sub-23';'sub-24';'sub-25';'sub-26';'sub-27';'sub-28'};
masks = {'Hippocampus_L.nii'};
s_path = 'H:\metaphor\LSS\rsa\pattern_similarity_change\mvpa_pattern';
roi_path = 'H:\metaphor\mask'; 
msk1 = masks{1};
counter = 0;

% 第一次看yy 首詞 和 基綫 之間的距離
for s = 1:length(subjects)
    sub = subjects{s};
    sub_path=fullfile(s_path,sub);
    mask_fn=fullfile(roi_path,msk1);

    % 
    data_shouci=fullfile(sub_path,'F_yy_start_word.nii');     %%%%%%%
    ds_shouci=cosmo_fmri_dataset(data_shouci,'mask',mask_fn);
    [number,~]=size(ds_shouci.samples);
    % 
    data_weici=fullfile(sub_path,'F_yy_end_word.nii');     %%%%%%
    ds_weici=cosmo_fmri_dataset(data_weici,'mask',mask_fn);
    
    data_jixian = fullfile(sub_path,'F_jx_word.nii');     %%%%%%
    data_jixian=cosmo_fmri_dataset(data_jixian,'mask',mask_fn);
    jixian = mean(data_jixian.samples);
    %
     allr_shou = 0;
     for x = 1:number
          r = corr(ds_shouci.samples(x,:)',jixian');
          allr_shou = allr_shou + r;
     end
    
     allr_wei = 0;
     for x = 1:number
          r = corr(ds_weici.samples(x,:)',jixian');
          allr_wei = allr_wei + r;
     end
     
     
    counter = counter+1;
    first_yy_lIFG_shou_corr_with_jixian(counter,:) = allr_shou/number;  
    first_yy_lIFG_wei_corr_with_jixian(counter,:) = allr_wei/number; 
end

% 第2次看yy 首詞 和 基綫 之間的距離
counter = 0;
for s = 1:length(subjects)
    sub = subjects{s};
    sub_path=fullfile(s_path,sub);
    mask_fn=fullfile(roi_path,msk1);

    % 
    data_shouci=fullfile(sub_path,'S_yy_start_word.nii');     %%%%%%%
    ds_shouci=cosmo_fmri_dataset(data_shouci,'mask',mask_fn);
    [number,~]=size(ds_shouci.samples);
    % 
    data_weici=fullfile(sub_path,'S_yy_end_word.nii');     %%%%%%
    ds_weici=cosmo_fmri_dataset(data_weici,'mask',mask_fn);
    
    data_jixian = fullfile(sub_path,'S_jx_word.nii');     %%%%%%
    data_jixian=cosmo_fmri_dataset(data_jixian,'mask',mask_fn);
    jixian = mean(data_jixian.samples);
    
    allr_shou = 0;
     for x = 1:number
          r = corr(ds_shouci.samples(x,:)',jixian');
          allr_shou = allr_shou + r;
     end
    
     allr_wei = 0;
     for x = 1:number
          r = corr(ds_weici.samples(x,:)',jixian');
          allr_wei = allr_wei + r;
     end
     
     
    counter = counter+1;
    
    second_yy_lIFG_shou_corr_with_jixian(counter,:) = allr_shou/number;  
    second_yy_lIFG_wei_corr_with_jixian(counter,:) = allr_wei/number; 
end
%
[h,p,ci,stats] = ttest(first_yy_lIFG_wei_corr_with_jixian,second_yy_lIFG_wei_corr_with_jixian)
% mean(my_r_F) 
% mean(my_r_S)
[h1,p1,ci1,stats1] = ttest(first_yy_lIFG_shou_corr_with_jixian,second_yy_lIFG_shou_corr_with_jixian)






