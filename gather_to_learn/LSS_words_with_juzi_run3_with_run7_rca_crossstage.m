clear
clc
subject_ids={'sub-01';'sub-02';'sub-03';'sub-04';'sub-05';'sub-06';'sub-07';'sub-08';'sub-09';'sub-10';'sub-11';'sub-13';'sub-14';'sub-15';'sub-16';'sub-17';'sub-18';'sub-19';'sub-20';'sub-21';'sub-22';'sub-24';'sub-25';'sub-26';'sub-27';'sub-28'};
nsubjects=numel(subject_ids);

run3_path='H:\metaphor\LSS\rsa\pattern_similarity_change\mvpa_pattern';  % F_yy_juzi.nii
run7_path='H:\metaphor\LSS\rsa\pattern_similarity_change\mvpa_pattern';   % S_yy_juzi.nii

mask_path='H:\metaphor\LSS'; 
out_path='H:\metaphor\LSS\words_with_juzi\run3_with_run7';

for i_subj=1:nsubjects
    %%
    subject_id=subject_ids{i_subj};
    mask_fn=fullfile(mask_path,'gm.nii');
    %%
    sub_3_path=fullfile(run3_path,subject_id);
    sub_7_path=fullfile(run7_path,subject_id);
    %%
    output_path=fullfile(out_path,subject_id);
    
     if ~exist(output_path)
        mkdir(output_path);
    end
    %% yy
    
    Fyy=fullfile(sub_3_path,'F_yy_juzi.nii');
    ds_fyy=cosmo_fmri_dataset(Fyy,'mask',mask_fn);
    %%%
    Syystart=fullfile(sub_7_path,'run7_yy_start_word.nii');
    ds_syystart=cosmo_fmri_dataset(Syystart,'mask',mask_fn);
 
    Syyend=fullfile(sub_7_path,'run7_yy_end_word.nii');
    ds_syyend=cosmo_fmri_dataset(Syyend,'mask',mask_fn);
    
    all_ds_yy=cosmo_stack({ds_fyy,ds_syystart});
    all_ds_yy=cosmo_stack({all_ds_yy,ds_syyend});
    
    measure = @new_measure_dsm_corr;
    voxel_count = 100;
    nbrhood=cosmo_spherical_neighborhood(all_ds_yy,'count',voxel_count);
    result_yy = cosmo_searchlight(all_ds_yy,nbrhood,measure);
    
    cosmo_map2fmri(result_yy, ...
        fullfile(output_path,'searchlight_yy.nii'));
    
    %% 
   Fkj=fullfile(sub_3_path,'F_kj_juzi.nii');
   ds_Fkj=cosmo_fmri_dataset(Fkj,'mask',mask_fn);
    
   Skjstart=fullfile(sub_7_path,'run7_kj_start_word.nii');
   ds_Skjstart=cosmo_fmri_dataset(Skjstart,'mask',mask_fn);

   Skjend=fullfile(sub_7_path,'run7_kj_end_word.nii');
   ds_Skjend=cosmo_fmri_dataset(Skjend,'mask',mask_fn);
   
   all_ds_kj=cosmo_stack({ds_Fkj,ds_Skjstart});
   all_ds_kj=cosmo_stack({all_ds_kj,ds_Skjend});
    
    measure = @new_measure_dsm_corr;
    voxel_count=100;
    nbrhood1=cosmo_spherical_neighborhood(all_ds_kj,'count',voxel_count);
    r_kj = cosmo_searchlight(all_ds_kj,nbrhood1,measure);
    
    cosmo_map2fmri(r_kj, ...
        fullfile(output_path,'searchlight_kj.nii'));
    
end
