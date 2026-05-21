clear; clc;
spm_jobman('initcfg');
spm('defaults', 'fMRI');

data_dir = 'H:\metaphor\Pro_proc_data';
sublist = dir([data_dir,'\sub*']);
%%
for nsub = 1:length(sublist)   
    sub_dir = [data_dir,filesep,sublist(nsub).name];
    %-----------------------------------------------------------------------
    out_dir = ['H:\metaphor\gppi\univariate_ppi_run1_2\',sublist(nsub).name]; 
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    %% 
    run1 = spm_select('ExtFPList', fullfile(sub_dir,'run1'),'smooth.*\.nii$');
    run2 = spm_select('ExtFPList', fullfile(sub_dir,'run2'),'smooth.*\.nii$');
   
    %% condition
    run_con_all = dir([sub_dir,filesep,'condition\sub*.tsv']);

    run1_con=struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(1).name]));
    run2_con=struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(2).name]));
   
    %% rp
   multi_reg_path=[sub_dir,filesep,'multi_reg'];
    %% our dir
    matlabbatch{1}.spm.stats.fmri_spec.dir = {out_dir};
    matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 2;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 30;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 15;
    %%
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).scans =  cellstr(run1);
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(1).name = 'yyw';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(1).onset = run1_con.onset(string(run1_con.trial_type) =='yyw ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(1).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).name = 'yyew';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).onset = run1_con.onset(string(run1_con.trial_type) =='yyew');
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).name = 'kjw';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).onset = run1_con.onset(string(run1_con.trial_type) =='kjw ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).name = 'kjew';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).onset = run1_con.onset(string(run1_con.trial_type) =='kjew');
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).duration = 2;  
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(5).name = 'jc';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(5).onset = run1_con.onset(string(run1_con.trial_type) =='jc  ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(5).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(6).name = 'jx';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(6).onset = run1_con.onset(string(run1_con.trial_type) =='jx  ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(6).duration = 2;
  
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).multi_reg ={[multi_reg_path,filesep,'multi_reg_run1.txt']};
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).hpf = 192;
    
    %%
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).scans =  cellstr(run2);
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).cond(1).name = 'yyw';
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).cond(1).onset = run2_con.onset(string(run2_con.trial_type) =='yyw ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).cond(1).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).cond(2).name = 'yyew';
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).cond(2).onset = run2_con.onset(string(run2_con.trial_type) =='yyew');
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).cond(2).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).cond(3).name = 'kjw';
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).cond(3).onset = run2_con.onset(string(run2_con.trial_type) =='kjw ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).cond(3).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).cond(4).name = 'kjew';
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).cond(4).onset = run2_con.onset(string(run2_con.trial_type) =='kjew');
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).cond(4).duration = 2;  
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).cond(5).name = 'jc';
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).cond(5).onset = run2_con.onset(string(run2_con.trial_type) =='jc  ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).cond(5).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).cond(6).name = 'jx';
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).cond(6).onset = run2_con.onset(string(run2_con.trial_type) =='jx  ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).cond(6).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).multi_reg ={[multi_reg_path,filesep,'multi_reg_run2.txt']};
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).hpf = 192;
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

