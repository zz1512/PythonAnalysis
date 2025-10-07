%% pattern similarity change
clear;clc;
subjects = {'sub-01';'sub-02';'sub-03';'sub-04';'sub-05';'sub-06';'sub-07';'sub-08';'sub-09';'sub-10';'sub-11';'sub-12';'sub-13';'sub-14';'sub-15';'sub-16';'sub-17';'sub-18';'sub-19';'sub-20';'sub-21';'sub-22';'sub-23';'sub-24';'sub-25';'sub-26';'sub-27';'sub-28'};
masks = {'IFG_L.nii'};
s_path = 'H:\metaphor\First_level\single_trial_v1\pattern';
roi_path = 'H:\metaphor\mask'; 
msk1 = masks{1};
counter = 0;

%% first yy 
for s = 1:length(subjects)
    sub = subjects{s};
    sub_path=fullfile(s_path,sub);
    mask_fn=fullfile(roi_path,msk1);

    %% 
    data_shouci=fullfile(sub_path,'F_yy_sword.nii');    
    ds_shouci=cosmo_fmri_dataset(data_shouci,'mask',mask_fn);
    beta =mean(mean(ds_shouci.samples,2));
    counter = counter+1;
    F_yy_sword(counter,:) = beta;  
    
end

%% second yy
counter = 0;
for s = 1:length(subjects)
    sub = subjects{s};
    sub_path=fullfile(s_path,sub);
    mask_fn=fullfile(roi_path,msk1);

    %% 
    data_shouci=fullfile(sub_path,'F_yy_eword.nii');    
    ds_shouci=cosmo_fmri_dataset(data_shouci,'mask',mask_fn);
    beta =mean(mean(ds_shouci.samples,2));
    counter = counter+1;
    F_yy_eword(counter,:) = beta;  
    
end

counter = 0;
for s = 1:length(subjects)
    sub = subjects{s};
    sub_path=fullfile(s_path,sub);
    mask_fn=fullfile(roi_path,msk1);

    %% 
    data_shouci=fullfile(sub_path,'S_yy_sword.nii');    
    ds_shouci=cosmo_fmri_dataset(data_shouci,'mask',mask_fn);
    beta =mean(mean(ds_shouci.samples,2));
    counter = counter+1;
    S_yy_sword(counter,:) = beta;  
    
end

counter = 0;
for s = 1:length(subjects)
    sub = subjects{s};
    sub_path=fullfile(s_path,sub);
    mask_fn=fullfile(roi_path,msk1);

    %% 
    data_shouci=fullfile(sub_path,'S_yy_eword.nii');    
    ds_shouci=cosmo_fmri_dataset(data_shouci,'mask',mask_fn);
    beta =mean(mean(ds_shouci.samples,2));
    counter = counter+1;
    S_yy_eword(counter,:) = beta;  
    
end

mean(F_yy_sword)
mean(F_yy_eword)
mean(S_yy_sword)
mean(S_yy_eword)
F = F_yy_sword+F_yy_eword;
S = S_yy_sword+S_yy_eword;

[h,p,ci,stats] = ttest(F,S)
