
clear
clc
subject_ids={'sub-01';'sub-02';'sub-03';'sub-04';'sub-05';'sub-06';'sub-07';'sub-08';'sub-09';'sub-10';'sub-11';'sub-12';'sub-13';'sub-14';'sub-15';'sub-16';'sub-17';'sub-18';'sub-19';'sub-20';'sub-21';'sub-22';'sub-23';'sub-24';'sub-25';'sub-26';'sub-27';'sub-28'};
nsubjects=numel(subject_ids);

run_path='H:\metaphor\LSS\rsa\pattern_similarity_change\mvpa_pattern';  % F_yy_juzi.nii

mask_path='H:\metaphor\LSS';  % grey matter
out_path='H:\metaphor\LSS\ps_increase\result';

for i_subj=1:nsubjects
    %%
    subject_id=subject_ids{i_subj};
    mask_fn=fullfile(mask_path,'gm.nii');
    %%
    sub_path=fullfile(run_path,subject_id);
  
    %%
      output_path=fullfile(out_path,subject_id);
    
     if ~exist(output_path)
        mkdir(output_path);
    end
    %% yy
    
    Syy=fullfile(sub_path,'S_yy_juzi.nii');
    ds_Syy=cosmo_fmri_dataset(Syy,'mask',mask_fn);
    %%%
    Syystart=fullfile(sub_path,'S_yy_start_word.nii');
    ds_syystart=cosmo_fmri_dataset(Syystart,'mask',mask_fn);
 
    Syyend=fullfile(sub_path,'S_yy_end_word.nii');
    ds_syyend=cosmo_fmri_dataset(Syyend,'mask',mask_fn);
    %%%
    Fyystart=fullfile(sub_path,'F_yy_start_word.nii');
    ds_fyystart=cosmo_fmri_dataset(Fyystart,'mask',mask_fn);
 
    Fyyend=fullfile(sub_path,'F_yy_end_word.nii');
    ds_fyyend=cosmo_fmri_dataset(Fyyend,'mask',mask_fn);
    %%
    
    all_ds_yy=cosmo_stack({ds_Syy,ds_syystart});
    all_ds_yy=cosmo_stack({all_ds_yy,ds_syyend});
    all_ds_yy=cosmo_stack({all_ds_yy,ds_fyystart});
    all_ds_yy=cosmo_stack({all_ds_yy,ds_fyyend});
    
    measure = @compare_ps_v1;
    voxel_count = 100;
    nbrhood=cosmo_spherical_neighborhood(all_ds_yy,'count',voxel_count);
    result_yy = cosmo_searchlight(all_ds_yy,nbrhood,measure);
    
    cosmo_map2fmri(result_yy, ...
        fullfile(output_path,'searchlight_yy_v1.nii'));
    
   %%
   
end
