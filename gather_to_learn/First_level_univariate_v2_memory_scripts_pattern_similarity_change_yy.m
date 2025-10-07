%% pattern similarity change
clear;clc;
subjects = {'sub-02';'sub-03';'sub-04';'sub-05';'sub-06';'sub-07';'sub-08';'sub-09';'sub-10';'sub-13';'sub-14';'sub-15';'sub-16';'sub-17';'sub-18';'sub-19';'sub-20';'sub-21';'sub-22';'sub-24';'sub-25';'sub-26';'sub-27';'sub-28'};
masks = {'Hippocampus_L.nii'}; %
s_path = 'J:\metaphor\First_level\univariate_v2';
roi_path = 'J:\metaphor\mask'; 
msk1 = masks{1};
counter = 0;
%% first yy 
for s = 1:length(subjects)
    sub = subjects{s};
    sub_path=fullfile(s_path,sub);
    mask_fn=fullfile(roi_path,msk1);
    %% 
    data_shouci=fullfile(sub_path,'con_0002.nii');     %%%%%%%
    ds_shouci=cosmo_fmri_dataset(data_shouci,'mask',mask_fn);
    [ds_shouci,msk]=cosmo_remove_useless_data(ds_shouci,1, 'finite');
    %% 
    data_weici=fullfile(sub_path,'con_0004.nii');     %%%%%%
    ds_weici=cosmo_fmri_dataset(data_weici,'mask',mask_fn);
    [ds_weici,msk]=cosmo_remove_useless_data(ds_weici,1, 'finite');
    
    r1 = corr(ds_shouci.samples',ds_weici.samples');
   
    %%
    counter = counter+1;
    my_r_F(counter,:) = r1;  
    
end


%% second yy
counter = 0;
for s = 1:length(subjects)
    sub = subjects{s};
    sub_path=fullfile(s_path,sub);
    mask_fn=fullfile(roi_path,msk1);

    %% 
    data_shouci=fullfile(sub_path,'con_0006.nii');     %%%%%%%
    ds_shouci=cosmo_fmri_dataset(data_shouci,'mask',mask_fn);
  
     [ds_shouci,msk]=cosmo_remove_useless_data(ds_shouci,1, 'finite');
    %% 
    data_weici=fullfile(sub_path,'con_0008.nii');     %%%%%%
    ds_weici=cosmo_fmri_dataset(data_weici,'mask',mask_fn);
  [ds_weici,msk]=cosmo_remove_useless_data(ds_weici,1, 'finite');
  %%
    r1 = corr(ds_shouci.samples',ds_weici.samples');
    %%
    %%
    counter = counter+1;
    my_r_S(counter,:) = r1;  

end

%%
[h,p,ci,stats] = ttest(my_r_F,my_r_S)
mean(my_r_F) 
mean(my_r_S)
