 %%
clear
clc
subjects = {'sub-01';'sub-02';'sub-03';'sub-04';'sub-05';'sub-06';'sub-07';'sub-08';'sub-09';'sub-10';'sub-11';'sub-12';'sub-13';'sub-14';'sub-15';'sub-16';'sub-17';'sub-18';'sub-19';'sub-20';'sub-21';'sub-22';'sub-23';'sub-24';'sub-25';'sub-26';'sub-27';'sub-28'};
data_path = 'H:\metaphor\First_level\single_trial_v1\pattern';
mask_path = 'H:\metaphor\First_level\single_trial_v1'; 
out_path = 'H:\metaphor\First_level\single_trial_v1_searchlight';

%% first yy 
for s = 1:length(subjects) 
    sub = subjects{s};
    sub_mask_path = fullfile(mask_path,sub);
    output_path = fullfile(out_path,sub);
    mask_fn = fullfile(sub_mask_path,'mask.nii');
    
    sub_path = fullfile(data_path,sub);
    
     if ~exist(output_path)
        mkdir(output_path);
     end
    %% 
    data_q=fullfile(sub_path,'F_kj_sword.nii');  
    ds_q=cosmo_fmri_dataset(data_q,'mask',mask_fn);
    data_h=fullfile(sub_path,'F_kj_eword.nii');  
    ds_h=cosmo_fmri_dataset(data_h,'mask',mask_fn);
    %%
    all_ds = cosmo_stack({ds_q,ds_h});
    measure = @kernel_measure;
    voxel_count = 100;
    nbrhood = cosmo_spherical_neighborhood(all_ds,'count',voxel_count);
    result = cosmo_searchlight(all_ds,nbrhood,measure);
    cosmo_map2fmri(result, ...
        fullfile(output_path,'kj_First.nii'));
end

%% second yy 
for s = 1:length(subjects) 
    sub = subjects{s};
    sub_mask_path = fullfile(mask_path,sub);
    output_path = fullfile(out_path,sub);
    mask_fn = fullfile(sub_mask_path,'mask.nii');
    
    sub_path = fullfile(data_path,sub);
    
     if ~exist(output_path)
        mkdir(output_path);
     end
    %% 
    data_q=fullfile(sub_path,'S_kj_sword.nii');  
    ds_q=cosmo_fmri_dataset(data_q,'mask',mask_fn);
    data_h=fullfile(sub_path,'S_kj_eword.nii');  
    ds_h=cosmo_fmri_dataset(data_h,'mask',mask_fn);
    %%
    all_ds = cosmo_stack({ds_q,ds_h});
    measure = @kernel_measure;
    voxel_count = 100;
    nbrhood = cosmo_spherical_neighborhood(all_ds,'count',voxel_count);
    result = cosmo_searchlight(all_ds,nbrhood,measure);
    cosmo_map2fmri(result, ...
        fullfile(output_path,'kj_Second.nii'));
end
