%% yy
clear
clc
subject_ids = {'sub-01';'sub-02';'sub-03';'sub-04';'sub-05';'sub-06';'sub-07';'sub-08';'sub-09';'sub-10';'sub-11';'sub-13';'sub-14';'sub-15';'sub-16';'sub-17';'sub-18';'sub-19';'sub-20';'sub-21';'sub-22';'sub-24';'sub-25';'sub-26';'sub-27';'sub-28'};
nsubjects=numel(subject_ids);
study_path='H:\metaphor\LSS\rsa\pattern_similarity_change\mvpa_pattern';
out_path='H:\metaphor\LSS\pca\searchlight\pc3';

for i_subj=1:nsubjects
    subject_id=subject_ids{i_subj};
    
    sub_path=fullfile(study_path,subject_id);
    
    mask_fn ='H:\metaphor\LSS\gm.nii';
    
    output_path=fullfile(out_path,subject_id);
    
     if ~exist(output_path)
        mkdir(output_path);
     end
    
    %% yy
    S__s = fullfile(sub_path,'run7_yy_start_word.nii');
    S_s_data = cosmo_fmri_dataset(S__s,'mask',mask_fn);
    
    S__e = fullfile(sub_path,'run7_yy_end_word.nii');
    S_e_data = cosmo_fmri_dataset(S__e,'mask',mask_fn);
    
    all = cosmo_stack({S_s_data,S_e_data});

    %%
    measure = @my_var_measure;
    measure_args = struct();

    voxel_count = 100;
    nbrhood = cosmo_spherical_neighborhood(all,'count',voxel_count);
    %%
    r = cosmo_searchlight(all,nbrhood,measure,measure_args);
    cosmo_map2fmri(r, ...
        fullfile(output_path,'searchlight_yy_run7_pc3_ev.nii'));
end
