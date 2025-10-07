clear; clc;
spm_jobman('initcfg');
spm('defaults', 'fMRI');
%%
data_dir = 'H:\metaphor\LSS\words_with_juzi\self-report-mem\run3_with_run7';   % con map
matlabbatch{1}.spm.stats.factorial_design.dir = {'H:\metaphor\LSS\words_with_juzi\self-report-mem\second\yy_1_minus_0'};  %%  output path
%%
n = 1;
s = {'01','02','03','04','05','07','08','09','10','13','14','15','16','18','19','21','22','24','25','26','28'};
sublist = dir([data_dir,'\sub*']);

for subj= 1:numel(s)
    
    i = s(subj);
    dir1 = strcat(data_dir,filesep,strcat('sub-',string(i)),filesep,'searchlight_yy_1.nii');
    dir2 = strcat(data_dir,filesep,strcat('sub-',string(i)),filesep,'searchlight_yy_0.nii');
    
    con = [dir1;dir2];
    matlabbatch{1}.spm.stats.factorial_design.des.pt.pair(n).scans = cellstr(con);
    n=n+1;
end

matlabbatch{1}.spm.stats.factorial_design.des.pt.gmsca = 0;
matlabbatch{1}.spm.stats.factorial_design.des.pt.ancova = 0;
matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;
matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('Factorial design specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;
matlabbatch{3}.spm.stats.con.spmmat(1) = cfg_dep('Model estimation: SPM.mat File', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = '1-0';
matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = [1 -1];
matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
matlabbatch{3}.spm.stats.con.consess{2}.tcon.name = '0-1';
matlabbatch{3}.spm.stats.con.consess{2}.tcon.weights = [-1 1];
matlabbatch{3}.spm.stats.con.consess{2}.tcon.sessrep = 'none';
matlabbatch{3}.spm.stats.con.delete = 0;
spm_jobman('run', matlabbatch);