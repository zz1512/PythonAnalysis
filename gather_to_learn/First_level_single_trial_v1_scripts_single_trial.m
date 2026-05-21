clear; clc;
spm_jobman('initcfg');
spm('defaults', 'fMRI');
data_dir = 'H:\metaphor\Pro_proc_data';
sublist = dir([data_dir,'\sub*']);
%%
for nsub = [12, 23]
    sub_dir = [data_dir,filesep,sublist(nsub).name];
    %-----------------------------------------------------------------------
    out_dir = ['H:\metaphor\First_level\single_trial_v1\',sublist(nsub).name]; 
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    run1 = spm_select('ExtFPList', fullfile(sub_dir,'run1'),'^sub-.*\.nii$');%
    run2 = spm_select('ExtFPList', fullfile(sub_dir,'run2'),'^sub-.*\.nii$');
    run3 = spm_select('ExtFPList', fullfile(sub_dir,'run3'),'^sub-.*\.nii$');
    run4 = spm_select('ExtFPList', fullfile(sub_dir,'run4'),'^sub-.*\.nii$');
    run5 = spm_select('ExtFPList', fullfile(sub_dir,'run5'),'^sub-.*\.nii$');
    run6 = spm_select('ExtFPList', fullfile(sub_dir,'run6'),'^sub-.*\.nii$');
    
    run_con_all = dir([sub_dir,filesep,'condition\sub*.tsv']);
    
    run1_con= struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(1).name]));
    run2_con= struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(2).name]));
    run3_con=struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(3).name]));
    run4_con=struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(4).name]));
    run5_con=struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(5).name]));
    run6_con=struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(6).name])); 
   
    multi_reg_path=[sub_dir,filesep,'multi_reg'];
     
    %%   
    matlabbatch{1}.spm.stats.fmri_spec.dir = {out_dir};
    matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 2;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 30;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 15;
    %%
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).scans = cellstr(run1);
    
    yyw = run1_con.onset(string(run1_con.trial_type) =='yyw ');
    yyw_pic = run1_con.pic_num(string(run1_con.trial_type) =='yyw ');
    yyew = run1_con.onset(string(run1_con.trial_type) =='yyew');
    yyew_pic = run1_con.pic_num(string(run1_con.trial_type) =='yyew');
    kjw = run1_con.onset(string(run1_con.trial_type) =='kjw ');
    kjw_pic = run1_con.pic_num(string(run1_con.trial_type) =='kjw ');
    kjew = run1_con.onset(string(run1_con.trial_type) =='kjew');
    kjew_pic = run1_con.pic_num(string(run1_con.trial_type) =='kjew');
    jc = run1_con.onset(string(run1_con.trial_type) =='jc  ');
    jc_pic = run1_con.pic_num(string(run1_con.trial_type) =='jc  ');
    jx = run1_con.onset(string(run1_con.trial_type) =='jx  ');
    jx_pic = run1_con.pic_num(string(run1_con.trial_type) =='jx  ');
    
    onset_all = [yyw;yyew;kjw;kjew;jc;jx];
    allpic = num2str([yyw_pic;yyew_pic;kjw_pic;kjew_pic;jc_pic;jx_pic]);
    allpic = cellstr([allpic;'r1';'r2';'r3';'r4';'r5';'r6']);
    
    condition_label={}; %x
    condition_label(1:length(yyw),1)={'F_yyw'};
    condition_label(length(condition_label)+1:length(condition_label)+length(yyew),1)={'F_yyew'};
    condition_label(length(condition_label)+1:length(condition_label)+length(kjw),1)={'F_kjw'};
    condition_label(length(condition_label)+1:length(condition_label)+length(kjew),1)={'F_kjew'};
    condition_label(length(condition_label)+1:length(condition_label)+length(jc),1)={'F_jc'};
    condition_label(length(condition_label)+1:length(condition_label)+length(jx),1)={'F_jx'};
    condition_label(length(condition_label)+1:length(condition_label)+ 6,1)={'regressor'};
    
    run1_label = join([condition_label,allpic],"_");
    run1_label = strrep(run1_label,' ','');
    
    for n_condi=1:length(onset_all) %30个试次
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(n_condi).name = ['trial_' num2str(n_condi)];
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(n_condi).onset = onset_all(n_condi); 
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(n_condi).duration = 2; 
    end
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).multi_reg ={[multi_reg_path,filesep,'multi_reg_run1.txt']};
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).hpf = 192;
    %% run2
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).scans =  cellstr(run2);
    
    yyw = run2_con.onset(string(run2_con.trial_type) =='yyw ');
    yyw_pic = run2_con.pic_num(string(run2_con.trial_type) =='yyw ');
    yyew = run2_con.onset(string(run2_con.trial_type) =='yyew');
    yyew_pic = run2_con.pic_num(string(run2_con.trial_type) =='yyew');
    kjw = run2_con.onset(string(run2_con.trial_type) =='kjw ');
    kjw_pic = run2_con.pic_num(string(run2_con.trial_type) =='kjw ');
    kjew = run2_con.onset(string(run2_con.trial_type) =='kjew');
    kjew_pic = run2_con.pic_num(string(run2_con.trial_type) =='kjew');
    jc = run2_con.onset(string(run2_con.trial_type) =='jc  ');
    jc_pic = run2_con.pic_num(string(run2_con.trial_type) =='jc  ');
    jx = run2_con.onset(string(run2_con.trial_type) =='jx  ');
    jx_pic = run2_con.pic_num(string(run2_con.trial_type) =='jx  ');
    
    onset_all = [yyw;yyew;kjw;kjew;jc;jx];
    allpic = num2str([yyw_pic;yyew_pic;kjw_pic;kjew_pic;jc_pic;jx_pic]);
    allpic = cellstr([allpic;'r1';'r2';'r3';'r4';'r5';'r6']);
    
    condition_label={}; %x
    condition_label(1:length(yyw),1)={'F_yyw'};
    condition_label(length(condition_label)+1:length(condition_label)+length(yyew),1)={'F_yyew'};
    condition_label(length(condition_label)+1:length(condition_label)+length(kjw),1)={'F_kjw'};
    condition_label(length(condition_label)+1:length(condition_label)+length(kjew),1)={'F_kjew'};
    condition_label(length(condition_label)+1:length(condition_label)+length(jc),1)={'F_jc'};
    condition_label(length(condition_label)+1:length(condition_label)+length(jx),1)={'F_jx'};
    condition_label(length(condition_label)+1:length(condition_label)+ 6,1)={'regressor'};
    
    run2_label = join([condition_label,allpic],"_");
    run2_label = strrep(run2_label,' ','');
    
    for n_condi=1:length(onset_all) %30个试次
        matlabbatch{1}.spm.stats.fmri_spec.sess(2).cond(n_condi).name = ['trial_' num2str(n_condi)];
        matlabbatch{1}.spm.stats.fmri_spec.sess(2).cond(n_condi).onset = onset_all(n_condi); 
        matlabbatch{1}.spm.stats.fmri_spec.sess(2).cond(n_condi).duration = 2; 
    end
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).multi_reg ={[multi_reg_path,filesep,'multi_reg_run2.txt']};
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).hpf = 192;
    %% run3
    matlabbatch{1}.spm.stats.fmri_spec.sess(3).scans =  cellstr(run3);
    
    yy = run3_con.onset(string(run3_con.trial_type) =='yy');
    yy_pic = run3_con.pic_num(string(run3_con.trial_type) =='yy');
    kj = run3_con.onset(string(run3_con.trial_type) =='kj');
    kj_pic = run3_con.pic_num(string(run3_con.trial_type) =='kj');
  
    onset_all = [yy;kj];
    allpic = num2str([yy_pic;kj_pic]);
    allpic = cellstr([allpic;'r1';'r2';'r3';'r4';'r5';'r6']);
    
    condition_label={}; %x
    condition_label(1:length(yy),1)={'F_yy'};
    condition_label(length(condition_label)+1:length(condition_label)+length(kj),1)={'F_kj'};
    condition_label(length(condition_label)+1:length(condition_label)+ 6,1)={'regressor'};
    
    run3_label = join([condition_label,allpic],"_");
    run3_label = strrep(run3_label,' ','');
    
    for n_condi=1:length(onset_all) %30个试次
        matlabbatch{1}.spm.stats.fmri_spec.sess(3).cond(n_condi).name = ['trial_' num2str(n_condi)];
        matlabbatch{1}.spm.stats.fmri_spec.sess(3).cond(n_condi).onset = onset_all(n_condi); 
        matlabbatch{1}.spm.stats.fmri_spec.sess(3).cond(n_condi).duration = 5; 
    end
   
    matlabbatch{1}.spm.stats.fmri_spec.sess(3).multi_reg ={[multi_reg_path,filesep,'multi_reg_run3.txt']};
    matlabbatch{1}.spm.stats.fmri_spec.sess(3).hpf = 192;
    %%
    matlabbatch{1}.spm.stats.fmri_spec.sess(4).scans =  cellstr(run4);
   
    yy = run4_con.onset(string(run4_con.trial_type) =='yy');
    yy_pic = run4_con.pic_num(string(run4_con.trial_type) =='yy');
    kj = run4_con.onset(string(run4_con.trial_type) =='kj');
    kj_pic = run4_con.pic_num(string(run4_con.trial_type) =='kj');
  
    onset_all = [yy;kj];
    allpic = num2str([yy_pic;kj_pic]);
    allpic = cellstr([allpic;'r1';'r2';'r3';'r4';'r5';'r6']);
    
    condition_label={}; %x
    condition_label(1:length(yy),1)={'S_yy'};
    condition_label(length(condition_label)+1:length(condition_label)+length(kj),1)={'S_kj'};
    condition_label(length(condition_label)+1:length(condition_label)+ 6,1)={'regressor'};
    
    run4_label = join([condition_label,allpic],"_");
    run4_label = strrep(run4_label,' ','');
    
    for n_condi=1:length(onset_all) %30个试次
        matlabbatch{1}.spm.stats.fmri_spec.sess(4).cond(n_condi).name = ['trial_' num2str(n_condi)];
        matlabbatch{1}.spm.stats.fmri_spec.sess(4).cond(n_condi).onset = onset_all(n_condi); 
        matlabbatch{1}.spm.stats.fmri_spec.sess(4).cond(n_condi).duration = 3; 
    end
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(4).multi_reg ={[multi_reg_path,filesep,'multi_reg_run4.txt']};
    matlabbatch{1}.spm.stats.fmri_spec.sess(4).hpf = 192;
    %% run5
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).scans =  cellstr(run5);
     
    yyw = run5_con.onset(string(run5_con.trial_type) =='yyw ');
    yyw_pic = run5_con.pic_num(string(run5_con.trial_type) =='yyw ');
    yyew = run5_con.onset(string(run5_con.trial_type) =='yyew');
    yyew_pic = run5_con.pic_num(string(run5_con.trial_type) =='yyew');
    kjw = run5_con.onset(string(run5_con.trial_type) =='kjw ');
    kjw_pic = run5_con.pic_num(string(run5_con.trial_type) =='kjw ');
    kjew = run5_con.onset(string(run5_con.trial_type) =='kjew');
    kjew_pic = run5_con.pic_num(string(run5_con.trial_type) =='kjew');
    jc = run5_con.onset(string(run5_con.trial_type) =='jc  ');
    jc_pic = run5_con.pic_num(string(run5_con.trial_type) =='jc  ');
    jx = run5_con.onset(string(run5_con.trial_type) =='jx  ');
    jx_pic = run5_con.pic_num(string(run5_con.trial_type) =='jx  ');
    
    onset_all = [yyw;yyew;kjw;kjew;jc;jx];
    allpic = num2str([yyw_pic;yyew_pic;kjw_pic;kjew_pic;jc_pic;jx_pic]);
    allpic = cellstr([allpic;'r1';'r2';'r3';'r4';'r5';'r6']);
    
    condition_label={}; %x
    condition_label(1:length(yyw),1)={'S_yyw'};
    condition_label(length(condition_label)+1:length(condition_label)+length(yyew),1)={'S_yyew'};
    condition_label(length(condition_label)+1:length(condition_label)+length(kjw),1)={'S_kjw'};
    condition_label(length(condition_label)+1:length(condition_label)+length(kjew),1)={'S_kjew'};
    condition_label(length(condition_label)+1:length(condition_label)+length(jc),1)={'S_jc'};
    condition_label(length(condition_label)+1:length(condition_label)+length(jx),1)={'S_jx'};
    condition_label(length(condition_label)+1:length(condition_label)+ 6,1)={'regressor'};
    
    run5_label = join([condition_label,allpic],"_");
    run5_label = strrep(run5_label,' ','');
    
    for n_condi=1:length(onset_all) %30个试次
        matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(n_condi).name = ['trial_' num2str(n_condi)];
        matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(n_condi).onset = onset_all(n_condi); 
        matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(n_condi).duration = 2; 
    end
    
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).multi_reg ={[multi_reg_path,filesep,'multi_reg_run5.txt']};
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).hpf = 192;
    %% run6
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).scans =  cellstr(run6);
    
    yyw = run6_con.onset(string(run6_con.trial_type) =='yyw ');
    yyw_pic = run6_con.pic_num(string(run6_con.trial_type) =='yyw ');
    yyew = run6_con.onset(string(run6_con.trial_type) =='yyew');
    yyew_pic = run6_con.pic_num(string(run6_con.trial_type) =='yyew');
    kjw = run6_con.onset(string(run6_con.trial_type) =='kjw ');
    kjw_pic = run6_con.pic_num(string(run6_con.trial_type) =='kjw ');
    kjew = run6_con.onset(string(run6_con.trial_type) =='kjew');
    kjew_pic = run6_con.pic_num(string(run6_con.trial_type) =='kjew');
    jc = run6_con.onset(string(run6_con.trial_type) =='jc  ');
    jc_pic = run6_con.pic_num(string(run6_con.trial_type) =='jc  ');
    jx = run6_con.onset(string(run6_con.trial_type) =='jx  ');
    jx_pic = run6_con.pic_num(string(run6_con.trial_type) =='jx  ');
    
    onset_all = [yyw;yyew;kjw;kjew;jc;jx];
    allpic = num2str([yyw_pic;yyew_pic;kjw_pic;kjew_pic;jc_pic;jx_pic]);
    allpic = cellstr([allpic;'r1';'r2';'r3';'r4';'r5';'r6']);
    
    condition_label={}; %x
    condition_label(1:length(yyw),1)={'S_yyw'};
    condition_label(length(condition_label)+1:length(condition_label)+length(yyew),1)={'S_yyew'};
    condition_label(length(condition_label)+1:length(condition_label)+length(kjw),1)={'S_kjw'};
    condition_label(length(condition_label)+1:length(condition_label)+length(kjew),1)={'S_kjew'};
    condition_label(length(condition_label)+1:length(condition_label)+length(jc),1)={'S_jc'};
    condition_label(length(condition_label)+1:length(condition_label)+length(jx),1)={'S_jx'};
    condition_label(length(condition_label)+1:length(condition_label)+ 6,1)={'regressor'};
    
    run6_label = join([condition_label,allpic],"_");
    run6_label = strrep(run6_label,' ','');
    
    for n_condi=1:length(onset_all) %30个试次
        matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(n_condi).name = ['trial_' num2str(n_condi)];
        matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(n_condi).onset = onset_all(n_condi); 
        matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(n_condi).duration = 2; 
    end
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).multi_reg ={[multi_reg_path,filesep,'multi_reg_run6.txt']};
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).hpf = 192;
    %%
    all_label = [run1_label;run2_label;run3_label;run4_label;run5_label;run6_label];
    %%
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
    
    matlabbatch{3}.spm.stats.con.spmmat(1) = cfg_dep('Model estimation: SPM.mat File', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));    
    null_contra = zeros(1,length(all_label)); 
    
   for n_contrast=1:length(all_label)
        matlabbatch{3}.spm.stats.con.consess{n_contrast}.tcon.name = ['contra_' num2str(n_contrast)];
        real_contra = null_contra;
        real_contra(1,n_contrast) = 1;
        matlabbatch{3}.spm.stats.con.consess{n_contrast}.tcon.weights = real_contra;
        matlabbatch{3}.spm.stats.con.consess{n_contrast}.tcon.sessrep = 'none';
    end
    matlabbatch{3}.spm.stats.con.delete = 0;
    spm_jobman('run', matlabbatch);
    clear matlabbatch;
    
    
    cd(out_dir); 
    for n_spmt=1:length(all_label)
        if n_spmt<10
            orignal_name = ['con_000' num2str(n_spmt) '.nii'];   
            movefile(orignal_name,[all_label{n_spmt} '_con_000' num2str(n_spmt) '.nii'])
        elseif n_spmt<100
            orignal_name = ['con_00' num2str(n_spmt) '.nii'];
            movefile(orignal_name,[all_label{n_spmt} '_con_00' num2str(n_spmt) '.nii'])
        else 
            orignal_name = ['con_0' num2str(n_spmt) '.nii'];
            movefile(orignal_name,[all_label{n_spmt} '_con_0' num2str(n_spmt) '.nii'] )
        end

    end
    
    disp(['Stats 1st-level done: ' sub_dir]);
end

for nsub = 1:length(sublist)
    if nsub == 12 || nsub == 23
        continue;
    end
    sub_dir = [data_dir,filesep,sublist(nsub).name];
    %-----------------------------------------------------------------------
    out_dir = ['H:\metaphor\First_level\single_trial_v1\',sublist(nsub).name]; 
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    run1 = spm_select('ExtFPList', fullfile(sub_dir,'run1'),'^sub-.*\.nii$');%
    run2 = spm_select('ExtFPList', fullfile(sub_dir,'run2'),'^sub-.*\.nii$');
    run3 = spm_select('ExtFPList', fullfile(sub_dir,'run3'),'^sub-.*\.nii$');
    run4 = spm_select('ExtFPList', fullfile(sub_dir,'run4'),'^sub-.*\.nii$');
    run5 = spm_select('ExtFPList', fullfile(sub_dir,'run5'),'^sub-.*\.nii$');
    run6 = spm_select('ExtFPList', fullfile(sub_dir,'run6'),'^sub-.*\.nii$');
    run7 = spm_select('ExtFPList', fullfile(sub_dir,'run7'),'^sub-.*\.nii$');
    
    
    run_con_all = dir([sub_dir,filesep,'condition\sub*.tsv']);
    
    run1_con= struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(1).name]));
    run2_con= struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(2).name]));
    run3_con=struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(3).name]));
    run4_con=struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(4).name]));
    run5_con=struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(5).name]));
    run6_con=struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(6).name])); 
    run7_con=struct2table(tdfread([sub_dir,filesep,'condition\',run_con_all(7).name])); 
   
    multi_reg_path=[sub_dir,filesep,'multi_reg'];
     
    %%   
    matlabbatch{1}.spm.stats.fmri_spec.dir = {out_dir};
    matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 2;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 30;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 15;
    %%
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).scans = cellstr(run1);
    
    yyw = run1_con.onset(string(run1_con.trial_type) == 'yyw ');
    yyw_pic = run1_con.pic_num(string(run1_con.trial_type) == 'yyw ');
    yyew = run1_con.onset(string(run1_con.trial_type) =='yyew');
    yyew_pic = run1_con.pic_num(string(run1_con.trial_type) == 'yyew');
    kjw = run1_con.onset(string(run1_con.trial_type) =='kjw ');
    kjw_pic = run1_con.pic_num(string(run1_con.trial_type) == 'kjw ');
    kjew = run1_con.onset(string(run1_con.trial_type) == 'kjew');
    kjew_pic = run1_con.pic_num(string(run1_con.trial_type) == 'kjew');
    jc = run1_con.onset(string(run1_con.trial_type) =='jc  ');
    jc_pic = run1_con.pic_num(string(run1_con.trial_type) == 'jc  ');
    jx = run1_con.onset(string(run1_con.trial_type) == 'jx  ');
    jx_pic = run1_con.pic_num(string(run1_con.trial_type) == 'jx  ');
    
    onset_all = [yyw;yyew;kjw;kjew;jc;jx];
    allpic = num2str([yyw_pic;yyew_pic;kjw_pic;kjew_pic;jc_pic;jx_pic]);
    allpic = cellstr([allpic;'r1';'r2';'r3';'r4';'r5';'r6']);
    
    condition_label={}; 
    condition_label(1:length(yyw),1)={'F_yyw'};
    condition_label(length(condition_label)+1:length(condition_label)+length(yyew),1)={'F_yyew'};
    condition_label(length(condition_label)+1:length(condition_label)+length(kjw),1)={'F_kjw'};
    condition_label(length(condition_label)+1:length(condition_label)+length(kjew),1)={'F_kjew'};
    condition_label(length(condition_label)+1:length(condition_label)+length(jc),1)={'F_jc'};
    condition_label(length(condition_label)+1:length(condition_label)+length(jx),1)={'F_jx'};
    condition_label(length(condition_label)+1:length(condition_label)+ 6,1)={'regressor'};
    
    run1_label = join([condition_label,allpic],"_");
    run1_label = strrep(run1_label,' ','');
    
    for n_condi=1:length(onset_all) 
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(n_condi).name = ['trial_' num2str(n_condi)];
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(n_condi).onset = onset_all(n_condi); 
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(n_condi).duration = 2; 
    end
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).multi_reg ={[multi_reg_path,filesep,'multi_reg_run1.txt']};
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).hpf = 192;
    %% run2
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).scans =  cellstr(run2);
    
    yyw = run2_con.onset(string(run2_con.trial_type) =='yyw ');
    yyw_pic = run2_con.pic_num(string(run2_con.trial_type) =='yyw ');
    yyew = run2_con.onset(string(run2_con.trial_type) =='yyew');
    yyew_pic = run2_con.pic_num(string(run2_con.trial_type) =='yyew');
    kjw = run2_con.onset(string(run2_con.trial_type) =='kjw ');
    kjw_pic = run2_con.pic_num(string(run2_con.trial_type) =='kjw ');
    kjew = run2_con.onset(string(run2_con.trial_type) =='kjew');
    kjew_pic = run2_con.pic_num(string(run2_con.trial_type) =='kjew');
    jc = run2_con.onset(string(run2_con.trial_type) =='jc  ');
    jc_pic = run2_con.pic_num(string(run2_con.trial_type) =='jc  ');
    jx = run2_con.onset(string(run2_con.trial_type) =='jx  ');
    jx_pic = run2_con.pic_num(string(run2_con.trial_type) =='jx  ');
    
    onset_all = [yyw;yyew;kjw;kjew;jc;jx];
    allpic = num2str([yyw_pic;yyew_pic;kjw_pic;kjew_pic;jc_pic;jx_pic]);
    allpic = cellstr([allpic;'r1';'r2';'r3';'r4';'r5';'r6']);
    
    condition_label={}; %x
    condition_label(1:length(yyw),1)={'F_yyw'};
    condition_label(length(condition_label)+1:length(condition_label)+length(yyew),1)={'F_yyew'};
    condition_label(length(condition_label)+1:length(condition_label)+length(kjw),1)={'F_kjw'};
    condition_label(length(condition_label)+1:length(condition_label)+length(kjew),1)={'F_kjew'};
    condition_label(length(condition_label)+1:length(condition_label)+length(jc),1)={'F_jc'};
    condition_label(length(condition_label)+1:length(condition_label)+length(jx),1)={'F_jx'};
    condition_label(length(condition_label)+1:length(condition_label)+ 6,1)={'regressor'};
    
    run2_label = join([condition_label,allpic],"_");
    run2_label = strrep(run2_label,' ','');
    
    for n_condi=1:length(onset_all) %30个试次
        matlabbatch{1}.spm.stats.fmri_spec.sess(2).cond(n_condi).name = ['trial_' num2str(n_condi)];
        matlabbatch{1}.spm.stats.fmri_spec.sess(2).cond(n_condi).onset = onset_all(n_condi); 
        matlabbatch{1}.spm.stats.fmri_spec.sess(2).cond(n_condi).duration = 2; 
    end
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).multi_reg ={[multi_reg_path,filesep,'multi_reg_run2.txt']};
    matlabbatch{1}.spm.stats.fmri_spec.sess(2).hpf = 192;
    %% run3
    matlabbatch{1}.spm.stats.fmri_spec.sess(3).scans =  cellstr(run3);
    
    yy = run3_con.onset(string(run3_con.trial_type) =='yy');
    yy_pic = run3_con.pic_num(string(run3_con.trial_type) =='yy');
    kj = run3_con.onset(string(run3_con.trial_type) =='kj');
    kj_pic = run3_con.pic_num(string(run3_con.trial_type) =='kj');
  
    onset_all = [yy;kj];
    allpic = num2str([yy_pic;kj_pic]);
    allpic = cellstr([allpic;'r1';'r2';'r3';'r4';'r5';'r6']);
    
    condition_label={}; %x
    condition_label(1:length(yy),1)={'F_yy'};
    condition_label(length(condition_label)+1:length(condition_label)+length(kj),1)={'F_kj'};
    condition_label(length(condition_label)+1:length(condition_label)+ 6,1)={'regressor'};
    
    run3_label = join([condition_label,allpic],"_");
    run3_label = strrep(run3_label,' ','');
    
    for n_condi=1:length(onset_all) %30个试次
        matlabbatch{1}.spm.stats.fmri_spec.sess(3).cond(n_condi).name = ['trial_' num2str(n_condi)];
        matlabbatch{1}.spm.stats.fmri_spec.sess(3).cond(n_condi).onset = onset_all(n_condi); 
        matlabbatch{1}.spm.stats.fmri_spec.sess(3).cond(n_condi).duration = 5; 
    end
   
    matlabbatch{1}.spm.stats.fmri_spec.sess(3).multi_reg ={[multi_reg_path,filesep,'multi_reg_run3.txt']};
    matlabbatch{1}.spm.stats.fmri_spec.sess(3).hpf = 192;
    %%
    matlabbatch{1}.spm.stats.fmri_spec.sess(4).scans =  cellstr(run4);
   
    yy = run4_con.onset(string(run4_con.trial_type) =='yy');
    yy_pic = run4_con.pic_num(string(run4_con.trial_type) =='yy');
    kj = run4_con.onset(string(run4_con.trial_type) =='kj');
    kj_pic = run4_con.pic_num(string(run4_con.trial_type) =='kj');
  
    onset_all = [yy;kj];
    allpic = num2str([yy_pic;kj_pic]);
    allpic = cellstr([allpic;'r1';'r2';'r3';'r4';'r5';'r6']);
    
    condition_label={}; %x
    condition_label(1:length(yy),1)={'S_yy'};
    condition_label(length(condition_label)+1:length(condition_label)+length(kj),1)={'S_kj'};
    condition_label(length(condition_label)+1:length(condition_label)+ 6,1)={'regressor'};
    
    run4_label = join([condition_label,allpic],"_");
    run4_label = strrep(run4_label,' ','');
    
    for n_condi=1:length(onset_all) %30个试次
        matlabbatch{1}.spm.stats.fmri_spec.sess(4).cond(n_condi).name = ['trial_' num2str(n_condi)];
        matlabbatch{1}.spm.stats.fmri_spec.sess(4).cond(n_condi).onset = onset_all(n_condi); 
        matlabbatch{1}.spm.stats.fmri_spec.sess(4).cond(n_condi).duration = 3; 
    end
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(4).multi_reg ={[multi_reg_path,filesep,'multi_reg_run4.txt']};
    matlabbatch{1}.spm.stats.fmri_spec.sess(4).hpf = 192;
    %% run5
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).scans =  cellstr(run5);
    yyw = run5_con.onset(string(run5_con.trial_type) =='yyw ');
    yyw_pic = run5_con.pic_num(string(run5_con.trial_type) =='yyw ');
    yyew = run5_con.onset(string(run5_con.trial_type) =='yyew');
    yyew_pic = run5_con.pic_num(string(run5_con.trial_type) =='yyew');
    kjw = run5_con.onset(string(run5_con.trial_type) =='kjw ');
    kjw_pic = run5_con.pic_num(string(run5_con.trial_type) =='kjw ');
    kjew = run5_con.onset(string(run5_con.trial_type) =='kjew');
    kjew_pic = run5_con.pic_num(string(run5_con.trial_type) =='kjew');
    jc = run5_con.onset(string(run5_con.trial_type) =='jc  ');
    jc_pic = run5_con.pic_num(string(run5_con.trial_type) =='jc  ');
    jx = run5_con.onset(string(run5_con.trial_type) =='jx  ');
    jx_pic = run5_con.pic_num(string(run5_con.trial_type) =='jx  ');
    
    onset_all = [yyw;yyew;kjw;kjew;jc;jx];
    allpic = num2str([yyw_pic;yyew_pic;kjw_pic;kjew_pic;jc_pic;jx_pic]);
    allpic = cellstr([allpic;'r1';'r2';'r3';'r4';'r5';'r6']);
    
    condition_label={}; %x
    condition_label(1:length(yyw),1)={'S_yyw'};
    condition_label(length(condition_label)+1:length(condition_label)+length(yyew),1)={'S_yyew'};
    condition_label(length(condition_label)+1:length(condition_label)+length(kjw),1)={'S_kjw'};
    condition_label(length(condition_label)+1:length(condition_label)+length(kjew),1)={'S_kjew'};
    condition_label(length(condition_label)+1:length(condition_label)+length(jc),1)={'S_jc'};
    condition_label(length(condition_label)+1:length(condition_label)+length(jx),1)={'S_jx'};
    condition_label(length(condition_label)+1:length(condition_label)+ 6,1)={'regressor'};
    
    run5_label = join([condition_label,allpic],"_");
    run5_label = strrep(run5_label,' ','');
    
    for n_condi=1:length(onset_all) %30个试次
        matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(n_condi).name = ['trial_' num2str(n_condi)];
        matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(n_condi).onset = onset_all(n_condi); 
        matlabbatch{1}.spm.stats.fmri_spec.sess(5).cond(n_condi).duration = 2; 
    end
    
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).multi_reg ={[multi_reg_path,filesep,'multi_reg_run5.txt']};
    matlabbatch{1}.spm.stats.fmri_spec.sess(5).hpf = 192;
    %% run6
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).scans =  cellstr(run6);
    
    yyw = run6_con.onset(string(run6_con.trial_type) =='yyw ');
    yyw_pic = run6_con.pic_num(string(run6_con.trial_type) =='yyw ');
    yyew = run6_con.onset(string(run6_con.trial_type) =='yyew');
    yyew_pic = run6_con.pic_num(string(run6_con.trial_type) =='yyew');
    kjw = run6_con.onset(string(run6_con.trial_type) =='kjw ');
    kjw_pic = run6_con.pic_num(string(run6_con.trial_type) =='kjw ');
    kjew = run6_con.onset(string(run6_con.trial_type) =='kjew');
    kjew_pic = run6_con.pic_num(string(run6_con.trial_type) =='kjew');
    jc = run6_con.onset(string(run6_con.trial_type) =='jc  ');
    jc_pic = run6_con.pic_num(string(run6_con.trial_type) =='jc  ');
    jx = run6_con.onset(string(run6_con.trial_type) =='jx  ');
    jx_pic = run6_con.pic_num(string(run6_con.trial_type) =='jx  ');
    
    onset_all = [yyw;yyew;kjw;kjew;jc;jx];
    allpic = num2str([yyw_pic;yyew_pic;kjw_pic;kjew_pic;jc_pic;jx_pic]);
    allpic = cellstr([allpic;'r1';'r2';'r3';'r4';'r5';'r6']);
    
    condition_label={}; %x
    condition_label(1:length(yyw),1)={'S_yyw'};
    condition_label(length(condition_label)+1:length(condition_label)+length(yyew),1)={'S_yyew'};
    condition_label(length(condition_label)+1:length(condition_label)+length(kjw),1)={'S_kjw'};
    condition_label(length(condition_label)+1:length(condition_label)+length(kjew),1)={'S_kjew'};
    condition_label(length(condition_label)+1:length(condition_label)+length(jc),1)={'S_jc'};
    condition_label(length(condition_label)+1:length(condition_label)+length(jx),1)={'S_jx'};
    condition_label(length(condition_label)+1:length(condition_label)+ 6,1)={'regressor'};
    
    run6_label = join([condition_label,allpic],"_");
    run6_label = strrep(run6_label,' ','');
    
    for n_condi=1:length(onset_all) %30个试次
        matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(n_condi).name = ['trial_' num2str(n_condi)];
        matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(n_condi).onset = onset_all(n_condi); 
        matlabbatch{1}.spm.stats.fmri_spec.sess(6).cond(n_condi).duration = 2; 
    end
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).multi_reg ={[multi_reg_path,filesep,'multi_reg_run6.txt']};
    matlabbatch{1}.spm.stats.fmri_spec.sess(6).hpf = 192;
    %%
    matlabbatch{1}.spm.stats.fmri_spec.sess(7).scans =  cellstr(run7);
    
    yyw = run7_con.onset(string(run7_con.trial_type) =='yyw ');
    yyw_pic = run7_con.pic_num(string(run7_con.trial_type) =='yyw ');
    yyew = run7_con.onset(string(run7_con.trial_type) =='yyew');
    yyew_pic = run7_con.pic_num(string(run7_con.trial_type) =='yyew');
    kjw = run7_con.onset(string(run7_con.trial_type) =='kjw ');
    kjw_pic = run7_con.pic_num(string(run7_con.trial_type) =='kjw ');
    kjew = run7_con.onset(string(run7_con.trial_type) =='kjew');
    kjew_pic = run7_con.pic_num(string(run7_con.trial_type) =='kjew');
    
    onset_all = [yyw;yyew;kjw;kjew];
    allpic = num2str([yyw_pic;yyew_pic;kjw_pic;kjew_pic]);
    allpic = cellstr([allpic;'r1';'r2';'r3';'r4';'r5';'r6']);
    
    condition_label={}; %x
    condition_label(1:length(yyw),1)={'M_yyw'};
    condition_label(length(condition_label)+1:length(condition_label)+length(yyew),1)={'M_yyew'};
    condition_label(length(condition_label)+1:length(condition_label)+length(kjw),1)={'M_kjw'};
    condition_label(length(condition_label)+1:length(condition_label)+length(kjew),1)={'M_kjew'};
    condition_label(length(condition_label)+1:length(condition_label)+ 6,1)={'regressor'};
    
    run7_label = join([condition_label,allpic],"_");
    run7_label = strrep(run7_label,' ','');
    
    for n_condi=1:length(onset_all) %30个试次
        matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(n_condi).name = ['trial_' num2str(n_condi)];
        matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(n_condi).onset = onset_all(n_condi); 
        matlabbatch{1}.spm.stats.fmri_spec.sess(7).cond(n_condi).duration = 2; 
    end
  
    matlabbatch{1}.spm.stats.fmri_spec.sess(7).multi_reg ={[multi_reg_path,filesep,'multi_reg_run7.txt']};
    matlabbatch{1}.spm.stats.fmri_spec.sess(7).hpf = 192;
    %%
    all_label = [run1_label;run2_label;run3_label;run4_label;run5_label;run6_label;run7_label];
    %%
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
    matlabbatch{3}.spm.stats.con.spmmat(1) = cfg_dep('Model estimation: SPM.mat File', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));    
    null_contra = zeros(1,length(all_label)); 
    
   for n_contrast=1:length(all_label)
        matlabbatch{3}.spm.stats.con.consess{n_contrast}.tcon.name = ['contra_' num2str(n_contrast)];
        real_contra = null_contra;
        real_contra(1,n_contrast) = 1;
        matlabbatch{3}.spm.stats.con.consess{n_contrast}.tcon.weights = real_contra;
        matlabbatch{3}.spm.stats.con.consess{n_contrast}.tcon.sessrep = 'none';
    end
    matlabbatch{3}.spm.stats.con.delete = 0;
    spm_jobman('run', matlabbatch);
    clear matlabbatch;
    
    
    cd(out_dir); 
    for n_spmt=1:length(all_label)
        if n_spmt<10
            orignal_name = ['con_000' num2str(n_spmt) '.nii'];   %spmT  con
            movefile(orignal_name,[all_label{n_spmt} '_con_000' num2str(n_spmt) '.nii'])
        elseif n_spmt<100
            orignal_name = ['con_00' num2str(n_spmt) '.nii'];
            movefile(orignal_name,[all_label{n_spmt} '_con_00' num2str(n_spmt) '.nii'])
        else 
            orignal_name = ['con_0' num2str(n_spmt) '.nii'];
            movefile(orignal_name,[all_label{n_spmt} '_con_0' num2str(n_spmt) '.nii'] )
        end

    end
    disp(['Stats 1st-level done: ' sub_dir]);
end



