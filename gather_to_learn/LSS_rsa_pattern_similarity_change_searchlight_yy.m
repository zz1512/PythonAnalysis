 %%
clear
clc
subjects = {'sub-01';'sub-02';'sub-03';'sub-04';'sub-05';'sub-06';'sub-07';'sub-08';'sub-09';'sub-10';'sub-11';'sub-12';'sub-13';'sub-14';'sub-15';'sub-16';'sub-17';'sub-18';'sub-19';'sub-20';'sub-21';'sub-22';'sub-23';'sub-24';'sub-25';'sub-26';'sub-27';'sub-28'};
data_path = 'H:\metaphor\LSS\rsa\pattern_similarity_change\mvpa_pattern';
mask_path = 'H:\metaphor\LSS'; 
out_path = 'H:\metaphor\LSS\rsa\pattern_similarity_change\searchlight_compare_first_with_second';

%% first yy 
for s = 1:length(subjects) 
    sub = subjects{s};
   
    output_path = fullfile(out_path,sub);
    mask_fn = fullfile(mask_path,'gm.nii');
    
    sub_path = fullfile(data_path,sub);
    
     if ~exist(output_path)
        mkdir(output_path);
     end
    %% 
    data_q=fullfile(sub_path,'F_yy_start_word.nii');  
    ds_q=cosmo_fmri_dataset(data_q,'mask',mask_fn);
    data_h=fullfile(sub_path,'F_yy_end_word.nii');  
    ds_h=cosmo_fmri_dataset(data_h,'mask',mask_fn);
    %%
    all_ds = cosmo_stack({ds_q,ds_h});
    measure = @kernel_measure;
    voxel_count = 100;
    nbrhood = cosmo_spherical_neighborhood(all_ds,'count',voxel_count);
    result = cosmo_searchlight(all_ds,nbrhood,measure);
    cosmo_map2fmri(result, ...
        fullfile(output_path,'yy_First.nii'));
end

%% second yy 
for s = 1:length(subjects) 
    sub = subjects{s};
    
    output_path = fullfile(out_path,sub);
    mask_fn = fullfile(mask_path,'gm.nii');
    
    sub_path = fullfile(data_path,sub);
    
     if ~exist(output_path)
        mkdir(output_path);
     end
    %% 
    data_q=fullfile(sub_path,'S_yy_start_word.nii');  
    ds_q=cosmo_fmri_dataset(data_q,'mask',mask_fn);
    data_h=fullfile(sub_path,'S_yy_end_word.nii');  
    ds_h=cosmo_fmri_dataset(data_h,'mask',mask_fn);
    %%
    all_ds = cosmo_stack({ds_q,ds_h});
    measure = @kernel_measure;
    voxel_count = 100;
    nbrhood = cosmo_spherical_neighborhood(all_ds,'count',voxel_count);
    result = cosmo_searchlight(all_ds,nbrhood,measure);
    cosmo_map2fmri(result, ...
        fullfile(output_path,'yy_Second.nii'));
end
