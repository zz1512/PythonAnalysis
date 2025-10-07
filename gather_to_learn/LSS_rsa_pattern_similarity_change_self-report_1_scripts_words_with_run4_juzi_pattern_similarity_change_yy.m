%% pattern similarity change
clear;clc;
subjects = {'sub-01';'sub-02';'sub-03';'sub-04';'sub-05';'sub-06';'sub-07';'sub-08';'sub-09';'sub-10';'sub-11';'sub-13';'sub-14';'sub-15';'sub-16';'sub-17';'sub-18';'sub-19';'sub-20';'sub-21';'sub-22';'sub-24';'sub-25';'sub-26';'sub-27';'sub-28'};
masks = {'IFG_L.nii'};
s_path = 'H:\metaphor\LSS\rsa\pattern_similarity_change_self-report_1\mvpa_pattern';
roi_path = 'H:\metaphor\mask'; 
msk1 = masks{1};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
counter = 0;
for s = 1:length(subjects)
    sub = subjects{s};
    sub_path=fullfile(s_path,sub);
    mask_fn=fullfile(roi_path,msk1);
    %% 
    data_shouci=fullfile(sub_path,'F_yy_start_word.nii');     %%%%%%%
    ds_shouci=cosmo_fmri_dataset(data_shouci,'mask',mask_fn);
    [number,~]=size(ds_shouci.samples);
    
    data_weici=fullfile(sub_path,'F_yy_end_word.nii');     %%%%%%%
    ds_weici=cosmo_fmri_dataset(data_weici,'mask',mask_fn);
   
    %% 
    data_juzi=fullfile(sub_path,'S_yy_juzi.nii');     %%%%%%
    ds_juzi=cosmo_fmri_dataset(data_juzi,'mask',mask_fn);
    
    %% hscµÄr
     allr_s = 0;
     for x = 1:number
          r = corr(ds_shouci.samples(x,:)',ds_juzi.samples(x,:)');
          allr_s = allr_s + r;
     end
    
     allr_w = 0;
     for x = 1:number
          r = corr(ds_weici.samples(x,:)',ds_juzi.samples(x,:)');
          allr_w = allr_w + r;
     end
     
     allr = (allr_s + allr_w)/2;
     
    counter = counter+1;
    F_YY_word_juzi(counter,:) = allr/number;  
    
end

counter = 0;
for s = 1:length(subjects)
    sub = subjects{s};
    sub_path=fullfile(s_path,sub);
    mask_fn=fullfile(roi_path,msk1);
    %% 
    data_shouci=fullfile(sub_path,'S_yy_start_word.nii');     %%%%%%%
    ds_shouci=cosmo_fmri_dataset(data_shouci,'mask',mask_fn);
    [number,~]=size(ds_shouci.samples);
    
    data_weici=fullfile(sub_path,'S_yy_end_word.nii');     %%%%%%%
    ds_weici=cosmo_fmri_dataset(data_weici,'mask',mask_fn);
   
    %% 
    data_juzi=fullfile(sub_path,'S_yy_juzi.nii');     %%%%%%
    ds_juzi=cosmo_fmri_dataset(data_juzi,'mask',mask_fn);
    
    %% hscµÄr
     allr_s = 0;
     for x = 1:number
          r = corr(ds_shouci.samples(x,:)',ds_juzi.samples(x,:)');
          allr_s = allr_s + r;
     end
    
     allr_w = 0;
     for x = 1:number
          r = corr(ds_weici.samples(x,:)',ds_juzi.samples(x,:)');
          allr_w = allr_w + r;
     end
     
     allr = (allr_s + allr_w)/2;
     
    counter = counter+1;
    S_YY_word_juzi(counter,:) = allr/number;  
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[h,p,ci,stats] = ttest(F_YY_word_juzi,S_YY_word_juzi)

mean(F_YY_word_juzi)
mean(S_YY_word_juzi)

save sme_lIFG_yy_run12_run4_run56.mat F_YY_word_juzi S_YY_word_juzi
