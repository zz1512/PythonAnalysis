clear
clc

subjects = {'sub-01';'sub-02';'sub-03';'sub-04';'sub-05';'sub-06';'sub-07';'sub-08';'sub-09';'sub-10';'sub-11';'sub-12';'sub-13';'sub-14';'sub-15';'sub-16';'sub-17';'sub-18';'sub-19';'sub-20';'sub-21';'sub-22';'sub-23';'sub-24';'sub-25';'sub-26';'sub-27';'sub-28'};
data_path = 'H:\metaphor\LSS\rsa\pattern_similarity_change\mvpa_pattern';
mask_path = 'H:\metaphor\mask'; 
out_path = 'H:\metaphor\LSS\rsa\pattern_similarity_change\searchlight_permu\model1_lrHPC';


for s = 1:length(subjects) 
    sub = subjects{s};
  
    output_path = fullfile(out_path,sub);
    mask_fn = fullfile(mask_path,'Hippocampus_L_R.nii');
    
    sub_path = fullfile(data_path,sub);
    
     if ~exist(output_path)
        mkdir(output_path);
     end
    %% 
    yy_data_q=fullfile(sub_path,'F_yy_start_word.nii');  
    F_yy_ds_q=cosmo_fmri_dataset(yy_data_q,'mask',mask_fn);
    yy_data_h=fullfile(sub_path,'F_yy_end_word.nii');  
    F_yy_ds_h=cosmo_fmri_dataset(yy_data_h,'mask',mask_fn);
    
    yy_data_q=fullfile(sub_path,'S_yy_start_word.nii');  
    S_yy_ds_q=cosmo_fmri_dataset(yy_data_q,'mask',mask_fn);
    yy_data_h=fullfile(sub_path,'S_yy_end_word.nii');  
    S_yy_ds_h=cosmo_fmri_dataset(yy_data_h,'mask',mask_fn);
    %%
    kj_data_q=fullfile(sub_path,'F_kj_start_word.nii');  
    F_kj_ds_q=cosmo_fmri_dataset(kj_data_q,'mask',mask_fn);
    kj_data_h=fullfile(sub_path,'F_kj_end_word.nii');  
    F_kj_ds_h=cosmo_fmri_dataset(kj_data_h,'mask',mask_fn);
    
    kj_data_q=fullfile(sub_path,'S_kj_start_word.nii');  
    S_kj_ds_q=cosmo_fmri_dataset(kj_data_q,'mask',mask_fn);
    kj_data_h=fullfile(sub_path,'S_kj_end_word.nii');  
    S_kj_ds_h=cosmo_fmri_dataset(kj_data_h,'mask',mask_fn);
    %%
    all_ds = cosmo_stack({F_yy_ds_q,F_yy_ds_h,S_yy_ds_q,S_yy_ds_h,F_kj_ds_q,F_kj_ds_h,S_kj_ds_q,S_kj_ds_h});
    measure = @kernel_measure;
    voxel_count = 125;
    nbrhood = cosmo_spherical_neighborhood(all_ds,'count',voxel_count);
    result = cosmo_searchlight(all_ds,nbrhood,measure);
    cosmo_map2fmri(result, ...
        fullfile(output_path,'model1.nii'));
end