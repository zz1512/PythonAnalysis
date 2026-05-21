
clear; clc;
spm_jobman('initcfg');
spm('defaults', 'fMRI');

data_dir = 'H:\metaphor\Pro_proc_data';
sublist = dir([data_dir,'\sub*']);

%%
for nsub = 1:length(sublist)   %
    if nsub==12 || nsub==23
        continue;
    end
    sub_dir = [data_dir,filesep,sublist(nsub).name];
    %-----------------------------------------------------------------------
    out_dir = ['H:\metaphor\First_level\univariate_v1\',sublist(nsub).name]; 
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    %% 
    run1 = spm_select('ExtFPList', fullfile(sub_dir,'run1'),'smooth.*\.nii$');
    run2 = spm_select('ExtFPList', fullfile(sub_dir,'run2'),'smooth.*\.nii$');
    run3 = spm_select('ExtFPList', fullfile(sub_dir,'run3'),'smooth.*\.nii$');
    run4 = spm_select('ExtFPList', fullfile(sub_dir,'run4'),'smooth.*\.nii$');
    run5 = spm_select('ExtFPList', fullfile(sub_dir,'run5'),'smooth.*\.nii$');
    run6 = spm_select('ExtFPList', fullfile(sub_dir,'run6'),'smooth.*\.nii$');
    run7 = spm_select('ExtFPList', fullfile(sub_dir,'run7'),'smooth.*\.nii$');
    
    %% condition
    run_con_all = dir([sub_dir,filesep,'condition\sub*.tsv']);
    run1_con= struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(1).name]));
    run2_con= struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(2).name]));
    run3_con=struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(3).name]));
    run4_con=struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(4).name]));
    run5_con=struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(5).name]));
    run6_con=struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(6).name])); 
    run7_con=struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(7).name])); 
    %% rp
    multi_reg=dir([sub_dir,filesep,'multi_reg\sub*.tsv']);
    multi_reg_path=[sub_dir,filesep,'multi_reg'];
    
    rp_run1=tdfread([sub_dir,filesep,'multi_reg\',multi_reg(1).name]);
    multi_reg_run1=[rp_run1.rot_x,rp_run1.rot_y,rp_run1.rot_z,rp_run1.trans_x,rp_run1.trans_y,rp_run1.trans_z];
    
    rp_run2=tdfread([sub_dir,filesep,'multi_reg\',multi_reg(2).name]);
    multi_reg_run2=[rp_run2.rot_x,rp_run2.rot_y,rp_run2.rot_z,rp_run2.trans_x,rp_run2.trans_y,rp_run2.trans_z];
    
    rp_run3=tdfread([sub_dir,filesep,'multi_reg\',multi_reg(3).name]);
    multi_reg_run3=[rp_run3.rot_x,rp_run3.rot_y,rp_run3.rot_z,rp_run3.trans_x,rp_run3.trans_y,rp_run3.trans_z];
    
    rp_run4=tdfread([sub_dir,filesep,'multi_reg\',multi_reg(4).name]);
    multi_reg_run4=[rp_run4.rot_x,rp_run4.rot_y,rp_run4.rot_z,rp_run4.trans_x,rp_run4.trans_y,rp_run4.trans_z];
    
    rp_run5=tdfread([sub_dir,filesep,'multi_reg\',multi_reg(5).name]);
    multi_reg_run5=[rp_run5.rot_x,rp_run5.rot_y,rp_run5.rot_z,rp_run5.trans_x,rp_run5.trans_y,rp_run5.trans_z];
    
    rp_run6=tdfread([sub_dir,filesep,'multi_reg\',multi_reg(6).name]);
    multi_reg_run6=[rp_run6.rot_x,rp_run6.rot_y,rp_run6.rot_z,rp_run6.trans_x,rp_run6.trans_y,rp_run6.trans_z];
    
    rp_run7=tdfread([sub_dir,filesep,'multi_reg\',multi_reg(7).name]);
    multi_reg_run7=[rp_run7.rot_x,rp_run7.rot_y,rp_run7.rot_z,rp_run7.trans_x,rp_run7.trans_y,rp_run7.trans_z];
    
    cd (multi_reg_path)
    save multi_reg_run1.txt multi_reg_run1 -ascii
    save multi_reg_run2.txt multi_reg_run2 -ascii
    save multi_reg_run3.txt multi_reg_run3 -ascii
    save multi_reg_run4.txt multi_reg_run4 -ascii  
    save multi_reg_run5.txt multi_reg_run5 -ascii
    save multi_reg_run6.txt multi_reg_run6 -ascii  
    save multi_reg_run7.txt multi_reg_run7 -ascii  
    %% our dir
    matlabbatch{1}.spm.stats.fmri_spec.dir = {out_dir};
    matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 2;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 30;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 15;
    %% run1   kjw kjew  jc jx yyw yyew
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).scans = cellstr(run1);
    
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
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).hpf = 128;
    %% run2
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
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).hpf = 128;
    
    %%
    matlabbatch{1}.spm.stats.fmri_spec.sess(3).scans =  cellstr(run3);
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(3).cond(1).name = 'yy';
    matlabbatch{1}.spm.stats.fmri_spec.sess(3).cond(1).onset = run3_con.onset(string(run3_con.trial_type) =='yy');
    matlabbatch{1}.spm.stats.fmri_spec.sess(3).cond(1).duration = 5;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(3).cond(2).name = 'kj';
    matlabbatch{1}.spm.stats.fmri_spec.sess(3).cond(2).onset = run3_con.onset(string(run3_con.trial_type) =='kj');
    matlabbatch{1}.spm.stats.fmri_spec.sess(3).cond(2).duration = 5;
  
    matlabbatch{1}.spm.stats.fmri_spec.sess(3).multi_reg ={[multi_reg_path,filesep,'multi_reg_run3.txt']};
    matlabbatch{1}.spm.stats.fmri_spec.sess(3).hpf = 128;
    
    %%
    matlabbatch{1}.spm.stats.fmri_spec.sess(4).scans =  cellstr(run4);

    matlabbatch{1}.spm.stats.fmri_spec.sess(4).cond(1).name = 'yy';
    matlabbatch{1}.spm.stats.fmri_spec.sess(4).cond(1).onset = run4_con.onset(string(run4_con.trial_type) =='yy');
    matlabbatch{1}.spm.stats.fmri_spec.sess(4).cond(1).duration = 3;
   
    matlabbatch{1}.spm.stats.fmri_spec.sess(4).cond(2).name = 'kj';
    matlabbatch{1}.spm.stats.fmri_spec.sess(4).cond(2).onset = run4_con.onset(string(run4_con.trial_type) =='kj');
    matlabbatch{1}.spm.stats.fmri_spec.sess(4).cond(2).duration = 3;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(4).multi_reg ={[multi_reg_path,filesep,'multi_reg_run4.txt']};
    matlabbatch{1}.spm.stats.fmri_spec.sess(4).hpf = 128;
    %% run5
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).scans =  cellstr(run5);

    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(1).name = 'yyw';
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(1).onset = run5_con.onset(string(run5_con.trial_type) =='yyw ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(1).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(2).name = 'yyew';
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(2).onset = run5_con.onset(string(run5_con.trial_type) =='yyew');
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(2).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(3).name = 'kjw';
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(3).onset = run5_con.onset(string(run5_con.trial_type) =='kjw ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(3).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(4).name = 'kjew';
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(4).onset = run5_con.onset(string(run5_con.trial_type) =='kjew');
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(4).duration = 2;  
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(5).name = 'jc';
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(5).onset = run5_con.onset(string(run5_con.trial_type) =='jc  ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(5).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(6).name = 'jx';
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(6).onset = run5_con.onset(string(run5_con.trial_type) =='jx  ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(6).duration = 2;
   
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).multi_reg ={[multi_reg_path,filesep,'multi_reg_run5.txt']};
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).hpf = 128;
    
    
    %% run6
     matlabbatch{1}.spm.stats.fmri_spec.sess(6).scans =  cellstr(run6);

      matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(1).name = 'yyw';
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(1).onset = run6_con.onset(string(run6_con.trial_type) =='yyw ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(1).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(2).name = 'yyew';
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(2).onset = run6_con.onset(string(run6_con.trial_type) =='yyew');
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(2).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(3).name = 'kjw';
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(3).onset = run6_con.onset(string(run6_con.trial_type) =='kjw ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(3).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(4).name = 'kjew';
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(4).onset = run6_con.onset(string(run6_con.trial_type) =='kjew');
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(4).duration = 2;  
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(5).name = 'jc';
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(5).onset = run6_con.onset(string(run6_con.trial_type) =='jc  ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(5).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(6).name = 'jx';
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(6).onset = run6_con.onset(string(run6_con.trial_type) =='jx  ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(6).duration = 2;
   
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).multi_reg ={[multi_reg_path,filesep,'multi_reg_run6.txt']};
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).hpf = 128;
    %%
     matlabbatch{1}.spm.stats.fmri_spec.sess(7).scans =  cellstr(run7);

    matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(1).name = 'yyw';
    matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(1).onset = run7_con.onset(string(run7_con.trial_type) =='yyw ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(1).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(2).name = 'yyew';
    matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(2).onset = run7_con.onset(string(run7_con.trial_type) =='yyew');
    matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(2).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(3).name = 'kjw';
    matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(3).onset = run7_con.onset(string(run7_con.trial_type) =='kjw ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(3).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(4).name = 'kjew';
    matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(4).onset = run7_con.onset(string(run7_con.trial_type) =='kjew');
    matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(4).duration = 2;  
   
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(7).multi_reg ={[multi_reg_path,filesep,'multi_reg_run7.txt']};
    matlabbatch{1}.spm.stats.fmri_spec.sess(7).hpf = 128;
    
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
%     matlabbatch{3}.spm.stats.con.spmmat(1) = cfg_dep('Model estimation: SPM.mat File', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
%     
%     matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'first_time_yy_word';
%     matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = [1,1,zeros(1,10),1,1,zeros(1,10),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12),zeros(1,10)];
%     matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
%     
%     matlabbatch{3}.spm.stats.con.consess{2}.tcon.name = 'first_time_kj_word';
%     matlabbatch{3}.spm.stats.con.consess{2}.tcon.weights = [0,0,1,1,zeros(1,8),0,0,1,1,zeros(1,8),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12),zeros(1,10)];
%     matlabbatch{3}.spm.stats.con.consess{2}.tcon.sessrep = 'none';
%     
%     %%jx
%     
%     %%jc
%     
%     %%
%     matlabbatch{3}.spm.stats.con.consess{3}.tcon.name = 'yy_juzi';
%     matlabbatch{3}.spm.stats.con.consess{3}.tcon.weights = [zeros(1,12),zeros(1,12),1,0,zeros(1,6),1,0,zeros(1,6),zeros(1,12),zeros(1,12),zeros(1,10)];
%     matlabbatch{3}.spm.stats.con.consess{3}.tcon.sessrep = 'none';
%     
%     matlabbatch{3}.spm.stats.con.consess{4}.tcon.name = 'kj_juzi';
%     matlabbatch{3}.spm.stats.con.consess{4}.tcon.weights = [zeros(1,12),zeros(1,12),0,1,zeros(1,6),0,1,zeros(1,6),zeros(1,12),zeros(1,12),zeros(1,10)];
%     matlabbatch{3}.spm.stats.con.consess{4}.tcon.sessrep = 'none';
%     
%    %%
%     matlabbatch{3}.spm.stats.con.consess{5}.tcon.name = 'second_time_yy';
%     matlabbatch{3}.spm.stats.con.consess{5}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),1,1,zeros(1,10),1,1,zeros(1,10),zeros(1,10)];
%     matlabbatch{3}.spm.stats.con.consess{5}.tcon.sessrep = 'none';
%     
%     matlabbatch{3}.spm.stats.con.consess{6}.tcon.name = 'second_time_jk';
%     matlabbatch{3}.spm.stats.con.consess{6}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),0,0,1,1,zeros(1,8),0,0,1,1,zeros(1,8),zeros(1,10)];
%     matlabbatch{3}.spm.stats.con.consess{6}.tcon.sessrep = 'none';
%     %%jx
%     
%     
%     
%     %%jc
%     
%     
%     %%
%     matlabbatch{3}.spm.stats.con.consess{7}.tcon.name = 'recall_yy';
%     matlabbatch{3}.spm.stats.con.consess{7}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12),1,1,zeros(1,8)];
%     matlabbatch{3}.spm.stats.con.consess{7}.tcon.sessrep = 'none';
%     
%     matlabbatch{3}.spm.stats.con.consess{8}.tcon.name = 'recall_kj';
%     matlabbatch{3}.spm.stats.con.consess{8}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12),0,0,1,1,zeros(1,6)];
%     matlabbatch{3}.spm.stats.con.consess{8}.tcon.sessrep = 'none';
%    
%     matlabbatch{3}.spm.stats.con.delete = 0;
    
    spm_jobman('run', matlabbatch);
    clear matlabbatch;
    disp(['Stats 1st-level done: ' sub_dir]);
end

%%
for nsub = [12,23]  %
    sub_dir = [data_dir,filesep,sublist(nsub).name];
    %-----------------------------------------------------------------------
    out_dir = ['H:\metaphor\First_level\univariate_v1\',sublist(nsub).name]; 
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    %% 
    run1 = spm_select('ExtFPList', fullfile(sub_dir,'run1'),'smooth.*\.nii$');
    run2 = spm_select('ExtFPList', fullfile(sub_dir,'run2'),'smooth.*\.nii$');
    run3 = spm_select('ExtFPList', fullfile(sub_dir,'run3'),'smooth.*\.nii$');
    run4 = spm_select('ExtFPList', fullfile(sub_dir,'run4'),'smooth.*\.nii$');
    run5 = spm_select('ExtFPList', fullfile(sub_dir,'run5'),'smooth.*\.nii$');
    run6 = spm_select('ExtFPList', fullfile(sub_dir,'run6'),'smooth.*\.nii$');
    %run7 = spm_select('ExtFPList', fullfile(sub_dir,'run7'),'smooth.*\.nii$');
    %% condition
    run_con_all = dir([sub_dir,filesep,'condition\sub*.tsv']);
    run1_con= struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(1).name]));
    run2_con= struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(2).name]));
    run3_con=struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(3).name]));
    run4_con=struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(4).name]));
    run5_con=struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(5).name]));
    run6_con=struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(6).name])); 
    %run7_con=struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(7).name])); 
    %% rp
    multi_reg=dir([sub_dir,filesep,'multi_reg\sub*.tsv']);
    multi_reg_path=[sub_dir,filesep,'multi_reg'];
    
    rp_run1=tdfread([sub_dir,filesep,'multi_reg\',multi_reg(1).name]);
    multi_reg_run1=[rp_run1.rot_x,rp_run1.rot_y,rp_run1.rot_z,rp_run1.trans_x,rp_run1.trans_y,rp_run1.trans_z];
    
    rp_run2=tdfread([sub_dir,filesep,'multi_reg\',multi_reg(2).name]);
    multi_reg_run2=[rp_run2.rot_x,rp_run2.rot_y,rp_run2.rot_z,rp_run2.trans_x,rp_run2.trans_y,rp_run2.trans_z];
    
    rp_run3=tdfread([sub_dir,filesep,'multi_reg\',multi_reg(3).name]);
    multi_reg_run3=[rp_run3.rot_x,rp_run3.rot_y,rp_run3.rot_z,rp_run3.trans_x,rp_run3.trans_y,rp_run3.trans_z];
    
    rp_run4=tdfread([sub_dir,filesep,'multi_reg\',multi_reg(4).name]);
    multi_reg_run4=[rp_run4.rot_x,rp_run4.rot_y,rp_run4.rot_z,rp_run4.trans_x,rp_run4.trans_y,rp_run4.trans_z];
    
    rp_run5=tdfread([sub_dir,filesep,'multi_reg\',multi_reg(5).name]);
    multi_reg_run5=[rp_run5.rot_x,rp_run5.rot_y,rp_run5.rot_z,rp_run5.trans_x,rp_run5.trans_y,rp_run5.trans_z];
    
    rp_run6=tdfread([sub_dir,filesep,'multi_reg\',multi_reg(6).name]);
    multi_reg_run6=[rp_run6.rot_x,rp_run6.rot_y,rp_run6.rot_z,rp_run6.trans_x,rp_run6.trans_y,rp_run6.trans_z];
    
    %rp_run7=tdfread([sub_dir,filesep,'multi_reg\',multi_reg(7).name]);
    %multi_reg_run7=[rp_run7.rot_x,rp_run7.rot_y,rp_run7.rot_z,rp_run7.trans_x,rp_run7.trans_y,rp_run7.trans_z];
    
    cd (multi_reg_path)
    save multi_reg_run1.txt multi_reg_run1 -ascii
    save multi_reg_run2.txt multi_reg_run2 -ascii
    save multi_reg_run3.txt multi_reg_run3 -ascii
    save multi_reg_run4.txt multi_reg_run4 -ascii  
    save multi_reg_run5.txt multi_reg_run5 -ascii
    save multi_reg_run6.txt multi_reg_run6 -ascii  
    %save multi_reg_run7.txt multi_reg_run7 -ascii  
    %% our dir
    matlabbatch{1}.spm.stats.fmri_spec.dir = {out_dir};
    matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 2;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 30;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 15;
    %% run1   kjw kjew  jc jx yyw yyew
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).scans = cellstr(run1);
    
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
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).hpf = 128;
    %% run2
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
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).hpf = 128;
    
    %%
    matlabbatch{1}.spm.stats.fmri_spec.sess(3).scans =  cellstr(run3);
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(3).cond(1).name = 'yy';
    matlabbatch{1}.spm.stats.fmri_spec.sess(3).cond(1).onset = run3_con.onset(string(run3_con.trial_type) =='yy');
    matlabbatch{1}.spm.stats.fmri_spec.sess(3).cond(1).duration = 5;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(3).cond(2).name = 'kj';
    matlabbatch{1}.spm.stats.fmri_spec.sess(3).cond(2).onset = run3_con.onset(string(run3_con.trial_type) =='kj');
    matlabbatch{1}.spm.stats.fmri_spec.sess(3).cond(2).duration = 5;
  
    matlabbatch{1}.spm.stats.fmri_spec.sess(3).multi_reg ={[multi_reg_path,filesep,'multi_reg_run3.txt']};
    matlabbatch{1}.spm.stats.fmri_spec.sess(3).hpf = 128;
    
    %%
    matlabbatch{1}.spm.stats.fmri_spec.sess(4).scans =  cellstr(run4);

    matlabbatch{1}.spm.stats.fmri_spec.sess(4).cond(1).name = 'yy';
    matlabbatch{1}.spm.stats.fmri_spec.sess(4).cond(1).onset = run4_con.onset(string(run4_con.trial_type) =='yy');
    matlabbatch{1}.spm.stats.fmri_spec.sess(4).cond(1).duration = 3;
   
    matlabbatch{1}.spm.stats.fmri_spec.sess(4).cond(2).name = 'kj';
    matlabbatch{1}.spm.stats.fmri_spec.sess(4).cond(2).onset = run4_con.onset(string(run4_con.trial_type) =='kj');
    matlabbatch{1}.spm.stats.fmri_spec.sess(4).cond(2).duration = 3;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(4).multi_reg ={[multi_reg_path,filesep,'multi_reg_run4.txt']};
    matlabbatch{1}.spm.stats.fmri_spec.sess(4).hpf = 128;
    %% run5
    
     matlabbatch{1}.spm.stats.fmri_spec.sess(5).scans =  cellstr(run5);

     matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(1).name = 'yyw';
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(1).onset = run5_con.onset(string(run5_con.trial_type) =='yyw ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(1).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(2).name = 'yyew';
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(2).onset = run5_con.onset(string(run5_con.trial_type) =='yyew');
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(2).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(3).name = 'kjw';
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(3).onset = run5_con.onset(string(run5_con.trial_type) =='kjw ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(3).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(4).name = 'kjew';
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(4).onset = run5_con.onset(string(run5_con.trial_type) =='kjew');
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(4).duration = 2;  
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(5).name = 'jc';
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(5).onset = run5_con.onset(string(run5_con.trial_type) =='jc  ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(5).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(6).name = 'jx';
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(6).onset = run5_con.onset(string(run5_con.trial_type) =='jx  ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(6).duration = 2;
   
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).multi_reg ={[multi_reg_path,filesep,'multi_reg_run5.txt']};
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).hpf = 128;
    
    
    %% run6
     matlabbatch{1}.spm.stats.fmri_spec.sess(6).scans =  cellstr(run6);

      matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(1).name = 'yyw';
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(1).onset = run6_con.onset(string(run6_con.trial_type) =='yyw ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(1).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(2).name = 'yyew';
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(2).onset = run6_con.onset(string(run6_con.trial_type) =='yyew');
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(2).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(3).name = 'kjw';
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(3).onset = run6_con.onset(string(run6_con.trial_type) =='kjw ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(3).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(4).name = 'kjew';
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(4).onset = run6_con.onset(string(run6_con.trial_type) =='kjew');
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(4).duration = 2;  
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(5).name = 'jc';
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(5).onset = run6_con.onset(string(run6_con.trial_type) =='jc  ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(5).duration = 2;
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(6).name = 'jx';
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(6).onset = run6_con.onset(string(run6_con.trial_type) =='jx  ');
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(6).duration = 2;
   
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).multi_reg ={[multi_reg_path,filesep,'multi_reg_run6.txt']};
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).hpf = 128;
    %%
%      matlabbatch{1}.spm.stats.fmri_spec.sess(7).scans =  cellstr(run7);
% 
%     matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(1).name = 'yyw';
%     matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(1).onset = run7_con.onset(string(run7_con.trial_type) =='yyw ');
%     matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(1).duration = 2;
%     
%     matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(2).name = 'yyew';
%     matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(2).onset = run7_con.onset(string(run7_con.trial_type) =='yyew');
%     matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(2).duration = 2;
%     
%     matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(3).name = 'kjw';
%     matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(3).onset = run7_con.onset(string(run7_con.trial_type) =='kjw ');
%     matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(3).duration = 2;
%     
%     matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(4).name = 'kjew';
%     matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(4).onset = run7_con.onset(string(run7_con.trial_type) =='kjew');
%     matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(4).duration = 2;  
%    
%     
%     matlabbatch{1}.spm.stats.fmri_spec.sess(7).multi_reg ={[multi_reg_path,filesep,'multi_reg_run7.txt']};
%     matlabbatch{1}.spm.stats.fmri_spec.sess(7).hpf = 128;
    
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
%     matlabbatch{3}.spm.stats.con.spmmat(1) = cfg_dep('Model estimation: SPM.mat File', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
%     
%     matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'first_time_yy_word';
%     matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = [1,1,zeros(1,10),1,1,zeros(1,10),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12)];
%     matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
%     
%     matlabbatch{3}.spm.stats.con.consess{2}.tcon.name = 'first_time_kj_word';
%     matlabbatch{3}.spm.stats.con.consess{2}.tcon.weights = [0,0,1,1,zeros(1,8),0,0,1,1,zeros(1,8),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12)];
%     matlabbatch{3}.spm.stats.con.consess{2}.tcon.sessrep = 'none';
%     
%     matlabbatch{3}.spm.stats.con.consess{3}.tcon.name = 'yy_juzi';
%     matlabbatch{3}.spm.stats.con.consess{3}.tcon.weights = [zeros(1,12),zeros(1,12),1,0,zeros(1,6),1,0,zeros(1,6),zeros(1,12),zeros(1,12)];
%     matlabbatch{3}.spm.stats.con.consess{3}.tcon.sessrep = 'none';
%     
%     matlabbatch{3}.spm.stats.con.consess{4}.tcon.name = 'kj_juzi';
%     matlabbatch{3}.spm.stats.con.consess{4}.tcon.weights = [zeros(1,12),zeros(1,12),0,1,zeros(1,6),0,1,zeros(1,6),zeros(1,12),zeros(1,12)];
%     matlabbatch{3}.spm.stats.con.consess{4}.tcon.sessrep = 'none';
%    
%     matlabbatch{3}.spm.stats.con.consess{5}.tcon.name = 'second_time_yy';
%     matlabbatch{3}.spm.stats.con.consess{5}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),1,1,zeros(1,10),1,1,zeros(1,10)];
%     matlabbatch{3}.spm.stats.con.consess{5}.tcon.sessrep = 'none';
%     
%     matlabbatch{3}.spm.stats.con.consess{6}.tcon.name = 'second_time_jk';
%     matlabbatch{3}.spm.stats.con.consess{6}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),0,0,1,1,zeros(1,8),0,0,1,1,zeros(1,8)];
%     matlabbatch{3}.spm.stats.con.consess{6}.tcon.sessrep = 'none';
%     
%     
%    
%     matlabbatch{3}.spm.stats.con.delete = 0;
    
    spm_jobman('run', matlabbatch);
    clear matlabbatch;
    disp(['Stats 1st-level done: ' sub_dir]);
end
