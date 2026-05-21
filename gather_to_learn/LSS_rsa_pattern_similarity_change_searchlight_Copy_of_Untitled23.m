
source_root = 'H:\metaphor\LSS\rsa\pattern_similarity_change\searchlight_compare_first_with_second';
target_root = 'H:\metaphor\LSS\rsa\pattern_similarity_change\searchlight_compare_first_with_second';

% 获取所有子目录的列表
sub_folders = dir(fullfile(source_root, 'sub-*'));
% no 
for sub_idx = 1:28
    sub_folder = sub_folders(sub_idx);
    sub_path = fullfile(source_root, sub_folder.name);
    
    run_path = fullfile(sub_path);
        
    yy2  = filenames(fullfile(run_path, 'kj_Second.nii'));
    yy1  = filenames(fullfile(run_path, 'kj_First.nii'));
    
    files=[yy2;yy1];
    
    spm_jobman('initcfg');
    spm('defaults', 'fMRI');
 
    matlabbatch{1}.spm.util.imcalc.input = cellstr(files);
    matlabbatch{1}.spm.util.imcalc.output = 'KJ_second_minus_first';
    matlabbatch{1}.spm.util.imcalc.outdir = {run_path};
    matlabbatch{1}.spm.util.imcalc.expression = 'i1-i2';
    matlabbatch{1}.spm.util.imcalc.var = struct('name', {}, 'value', {});
    matlabbatch{1}.spm.util.imcalc.options.dmtx = 0;
    matlabbatch{1}.spm.util.imcalc.options.mask = 0;
    matlabbatch{1}.spm.util.imcalc.options.interp = 1;
    matlabbatch{1}.spm.util.imcalc.options.dtype = 4;
    spm_jobman('run', matlabbatch);
    clear matlabbatch;
     
end
    
