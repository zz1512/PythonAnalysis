
clear; clc;
spm_jobman('initcfg');
spm('defaults', 'fMRI');

data_dir = 'H:\metaphor\Pro_proc_data';
sublist = dir([data_dir,'\sub*']);

%%
for nsub = 1:length(sublist)   
    sub_dir = [data_dir,filesep,sublist(nsub).name];
    %-----------------------------------------------------------------------
    out_dir = ['H:\metaphor\gppi\univariate_ppi_run3\',sublist(nsub).name]; 
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    %% 
   
    run3 = spm_select('ExtFPList', fullfile(sub_dir,'run3'),'smooth.*\.nii$');
    
   
    %% condition
    run_con_all = dir([sub_dir,filesep,'condition\sub*.tsv']);

    run3_con=struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(3).name]));
    
   
    %% rp
   multi_reg_path=[sub_dir,filesep,'multi_reg'];
    %% our dir
    matlabbatch{1}.spm.stats.fmri_spec.dir = {out_dir};
    matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 2;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 30;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 15;
    %%
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).scans =  cellstr(run3);
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(1).name = 'yy';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(1).onset = run3_con.onset(string(run3_con.trial_type) =='yy');
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(1).duration = 5;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).name = 'kj';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).onset = run3_con.onset(string(run3_con.trial_type) =='kj');
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).duration = 5;
  
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).multi_reg ={[multi_reg_path,filesep,'multi_reg_run3.txt']};
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).hpf = 192;
   
    %%
    matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
    matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
    matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
    matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
    matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.8;
    matlabbatch{1}.spm.stats.fmri_spec.mask = {''};
    matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';
    
    %%
    matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('fMRI model specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
    matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
    matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;
    
    %%
   
    
    spm_jobman('run', matlabbatch);
    clear matlabbatch;
    disp(['Stats 1st-level done: ' sub_dir]);
end

