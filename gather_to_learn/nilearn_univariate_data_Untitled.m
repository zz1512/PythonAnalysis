
source_root = 'H:\metaphor\nilearn_univariate\data';
target_root = 'H:\metaphor\nilearn_univariate\data';

% 获取所有子目录的列表
sub_folders = dir(fullfile(source_root, 'sub-*'));
% no 12 23
for sub_idx = 24:length(sub_folders)
    sub_folder = sub_folders(sub_idx);
    sub_path = fullfile(source_root, sub_folder.name);
    
    run_path = fullfile(sub_path,  'run-7');
        
    files  = filenames(fullfile(run_path, '*sub*.nii'));
    
spm_jobman('initcfg');
spm('defaults', 'fMRI');
 
    matlabbatch{1}.spm.util.imcalc.input = cellstr(files);
    matlabbatch{1}.spm.util.imcalc.output = 'run7_yy-kj';
    matlabbatch{1}.spm.util.imcalc.outdir = {run_path};
    matlabbatch{1}.spm.util.imcalc.expression = 'i2-i1';
    matlabbatch{1}.spm.util.imcalc.var = struct('name', {}, 'value', {});
    matlabbatch{1}.spm.util.imcalc.options.dmtx = 0;
    matlabbatch{1}.spm.util.imcalc.options.mask = 0;
    matlabbatch{1}.spm.util.imcalc.options.interp = 1;
    matlabbatch{1}.spm.util.imcalc.options.dtype = 4;
    spm_jobman('run', matlabbatch);
    clear matlabbatch;
     
end
    
