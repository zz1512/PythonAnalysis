%% pattern similarity change
clear;clc;
subjects = {'sub-01';'sub-02';'sub-03';'sub-04';'sub-05';'sub-06';'sub-07';'sub-08';'sub-09';'sub-10';'sub-11';'sub-12';'sub-13';'sub-14';'sub-15';'sub-16';'sub-17';'sub-18';'sub-19';'sub-20';'sub-21';'sub-22';'sub-23';'sub-24';'sub-25';'sub-26';'sub-27';'sub-28'};
masks = {'parahippocampal_gyrus_L.nii'};
s_path = 'J:\metaphor\First_level\single_trial_v1\pattern';
roi_path = 'J:\metaphor\mask'; 
msk1 = masks{1};
counter = 0;

%% first yy 
for s = 1:length(subjects)
    sub = subjects{s};
    sub_path=fullfile(s_path,sub);
    mask_fn=fullfile(roi_path,msk1);

    %% 
    data_shouci=fullfile(sub_path,'F_yy_eword.nii');     %%%%%%%
    ds_shouci=cosmo_fmri_dataset(data_shouci,'mask',mask_fn);
    [number,~]=size(ds_shouci.samples);
    %% 
    data_weici=fullfile(sub_path,'F_yy_sword.nii');     %%%%%%
    ds_weici=cosmo_fmri_dataset(data_weici,'mask',mask_fn);
    
    %% hscµÄr
     allr = 0;
     for x = 1:number
          r = corr(ds_shouci.samples(x,:)',ds_weici.samples(x,:)');
          allr = allr + r;
     end
    
    counter = counter+1;
    my_r_shou(counter,:) = allr/number;  
    
end


%% second yy
counter = 0;
for s = 1:length(subjects)
    sub = subjects{s};
    sub_path=fullfile(s_path,sub);
    mask_fn=fullfile(roi_path,msk1);

    %% 
    data_shouci=fullfile(sub_path,'S_yy_eword.nii');     %%%%%%%
    ds_shouci=cosmo_fmri_dataset(data_shouci,'mask',mask_fn);
    [number,~]=size(ds_shouci.samples);
    %% 
    data_weici=fullfile(sub_path,'S_yy_sword.nii');     %%%%%%
    ds_weici=cosmo_fmri_dataset(data_weici,'mask',mask_fn);
    
    %% hscµÄr
     allr = 0;
     for x = 1:number
          r = corr(ds_shouci.samples(x,:)',ds_weici.samples(x,:)');
          allr = allr + r;
     end
      
    counter = counter+1;
    my_r_wei(counter,:) = allr/number;  
    
end

%%
counter = 0;
for s = 1:length(subjects)
    sub = subjects{s};
    sub_path=fullfile(s_path,sub);
    mask_fn=fullfile(roi_path,msk1);

    %% 
    data_shouci=fullfile(sub_path,'F_jx_word.nii');     %%%%%%%
    ds_shouci=cosmo_fmri_dataset(data_shouci,'mask',mask_fn);
    [number,~]=size(ds_shouci.samples);
    %% 
    data_weici=fullfile(sub_path,'F_jc_word.nii');     %%%%%%
    ds_weici=cosmo_fmri_dataset(data_weici,'mask',mask_fn);
    
    %% hscµÄr
     allr = 0;
     for x = 1:20
          r = corr(ds_shouci.samples(x,:)',ds_weici.samples(x,:)');
          allr = allr + r;
     end
      
    counter = counter+1;
    my_r_jx(counter,:) = allr/number;  
    
end

%%
[h,p,ci,stats] = ttest(my_r_shou,my_r_jx)
mean(my_r_shou) 
mean(my_r_wei)
mean(my_r_jx)









