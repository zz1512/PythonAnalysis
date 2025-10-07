clear; clc;
spm_jobman('initcfg');
spm('defaults', 'fMRI');
data_dir = 'J:\metaphor\Pro_proc_data';
sublist = dir([data_dir,'\sub*']);
%%
for nsub=28
    sub_dir = [data_dir,filesep,sublist(nsub).name];
    %-----------------------------------------------------------------------
%     run1 = spm_select('ExtFPList', fullfile(sub_dir,'run1'),'sub-.*\.nii$');
%     run2 = spm_select('ExtFPList', fullfile(sub_dir,'run2'),'sub-.*\.nii$');
%     run3 = spm_select('ExtFPList', fullfile(sub_dir,'run3'),'sub-.*\.nii$');
%     run4 = spm_select('ExtFPList', fullfile(sub_dir,'run4'),'sub-.*\.nii$');
%     run5 = spm_select('ExtFPList', fullfile(sub_dir,'run5'),'sub-.*\.nii$');
%     run6 = spm_select('ExtFPList', fullfile(sub_dir,'run6'),'sub-.*\.nii$');
    run7 = spm_select('ExtFPList', fullfile(sub_dir,'run7'),'sub-.*\.nii$');
    %%
    matlabbatch{1}.spm.spatial.smooth.data = cellstr(run7);
    matlabbatch{1}.spm.spatial.smooth.fwhm = [6 6 6];
    matlabbatch{1}.spm.spatial.smooth.dtype = 0;
    matlabbatch{1}.spm.spatial.smooth.im = 0;
    matlabbatch{1}.spm.spatial.smooth.prefix = 'smooth';
    
%     matlabbatch{2}.spm.spatial.smooth.data = cellstr(run2);
%     matlabbatch{2}.spm.spatial.smooth.fwhm = [6 6 6];
%     matlabbatch{2}.spm.spatial.smooth.dtype = 0;
%     matlabbatch{2}.spm.spatial.smooth.im = 0;
%     matlabbatch{2}.spm.spatial.smooth.prefix = 'smooth';
%     
%     matlabbatch{3}.spm.spatial.smooth.data = cellstr(run3);
%     matlabbatch{3}.spm.spatial.smooth.fwhm = [6 6 6];
%     matlabbatch{3}.spm.spatial.smooth.dtype = 0;
%     matlabbatch{3}.spm.spatial.smooth.im = 0;
%     matlabbatch{3}.spm.spatial.smooth.prefix = 'smooth';
%     
%     matlabbatch{4}.spm.spatial.smooth.data = cellstr(run4);
%     matlabbatch{4}.spm.spatial.smooth.fwhm = [6 6 6];
%     matlabbatch{4}.spm.spatial.smooth.dtype = 0;
%     matlabbatch{4}.spm.spatial.smooth.im = 0;
%     matlabbatch{4}.spm.spatial.smooth.prefix = 'smooth';
%     
%     matlabbatch{5}.spm.spatial.smooth.data = cellstr(run5);
%     matlabbatch{5}.spm.spatial.smooth.fwhm = [6 6 6];
%     matlabbatch{5}.spm.spatial.smooth.dtype = 0;
%     matlabbatch{5}.spm.spatial.smooth.im = 0;
%     matlabbatch{5}.spm.spatial.smooth.prefix = 'smooth';
%     
%     matlabbatch{6}.spm.spatial.smooth.data = cellstr(run6);
%     matlabbatch{6}.spm.spatial.smooth.fwhm = [6 6 6];
%     matlabbatch{6}.spm.spatial.smooth.dtype = 0;
%     matlabbatch{6}.spm.spatial.smooth.im = 0;
%     matlabbatch{6}.spm.spatial.smooth.prefix = 'smooth';
%     
%     matlabbatch{7}.spm.spatial.smooth.data = cellstr(run7);
%     matlabbatch{7}.spm.spatial.smooth.fwhm = [6 6 6];
%     matlabbatch{7}.spm.spatial.smooth.dtype = 0;
%     matlabbatch{7}.spm.spatial.smooth.im = 0;
%     matlabbatch{7}.spm.spatial.smooth.prefix = 'smooth';
    
    spm_jobman('run', matlabbatch);
    clear matlabbatch;
end
