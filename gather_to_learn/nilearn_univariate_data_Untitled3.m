
source_root = 'H:\metaphor\nilearn_univariate\data';
target_root = 'H:\metaphor\nilearn_univariate\data';

% 获取所有子目录的列表
sub_folders = dir(fullfile(source_root, 'sub-*'));
% 遍历每个子目录
for sub_idx = 1:length(sub_folders)
    sub_folder = sub_folders(sub_idx);
    sub_path = fullfile(source_root, sub_folder.name);
    
    run_path = fullfile(sub_path,  'first_words_view');
        
    files  = filenames(fullfile(run_path, '*yy*.nii'));
    
spm_jobman('initcfg');
spm('defaults', 'fMRI');
 
    matlabbatch{1}.spm.util.imcalc.input = cellstr(files);
    matlabbatch{1}.spm.util.imcalc.output = 'yy';
    matlabbatch{1}.spm.util.imcalc.outdir = {run_path};
    matlabbatch{1}.spm.util.imcalc.expression = '(i1+i2)/2';
    matlabbatch{1}.spm.util.imcalc.var = struct('name', {}, 'value', {});
    matlabbatch{1}.spm.util.imcalc.options.dmtx = 0;
    matlabbatch{1}.spm.util.imcalc.options.mask = 0;
    matlabbatch{1}.spm.util.imcalc.options.interp = 1;
    matlabbatch{1}.spm.util.imcalc.options.dtype = 4;
    spm_jobman('run', matlabbatch);
    clear matlabbatch;
     
end
    
 for sub_idx = 1:length(sub_folders)
    sub_folder = sub_folders(sub_idx);
    sub_path = fullfile(source_root, sub_folder.name);
    
    run_path = fullfile(sub_path,  'first_words_view');
        
    files  = filenames(fullfile(run_path, '*kj*.nii'));
    
spm_jobman('initcfg');
spm('defaults', 'fMRI');
 
    matlabbatch{1}.spm.util.imcalc.input = cellstr(files);
    matlabbatch{1}.spm.util.imcalc.output = 'kj';
    matlabbatch{1}.spm.util.imcalc.outdir = {run_path};
    matlabbatch{1}.spm.util.imcalc.expression = '(i1+i2)/2';
    matlabbatch{1}.spm.util.imcalc.var = struct('name', {}, 'value', {});
    matlabbatch{1}.spm.util.imcalc.options.dmtx = 0;
    matlabbatch{1}.spm.util.imcalc.options.mask = 0;
    matlabbatch{1}.spm.util.imcalc.options.interp = 1;
    matlabbatch{1}.spm.util.imcalc.options.dtype = 4;
    spm_jobman('run', matlabbatch);
    clear matlabbatch;
     
 end

 for sub_idx = 1:length(sub_folders)
    sub_folder = sub_folders(sub_idx);
    sub_path = fullfile(source_root, sub_folder.name);
    
    run_path = fullfile(sub_path,  'first_words_view');
        
    files  = filenames(fullfile(run_path, '*jx*.nii'));
    
spm_jobman('initcfg');
spm('defaults', 'fMRI');
 
    matlabbatch{1}.spm.util.imcalc.input = cellstr(files);
    matlabbatch{1}.spm.util.imcalc.output = 'jx';
    matlabbatch{1}.spm.util.imcalc.outdir = {run_path};
    matlabbatch{1}.spm.util.imcalc.expression = '(i1+i2)/2';
    matlabbatch{1}.spm.util.imcalc.var = struct('name', {}, 'value', {});
    matlabbatch{1}.spm.util.imcalc.options.dmtx = 0;
    matlabbatch{1}.spm.util.imcalc.options.mask = 0;
    matlabbatch{1}.spm.util.imcalc.options.interp = 1;
    matlabbatch{1}.spm.util.imcalc.options.dtype = 4;
    spm_jobman('run', matlabbatch);
    clear matlabbatch;
     
 end

for sub_idx = 1:length(sub_folders)
    sub_folder = sub_folders(sub_idx);
    sub_path = fullfile(source_root, sub_folder.name);
    
    run_path = fullfile(sub_path,  'first_words_view');
        
    files  = filenames(fullfile(run_path, '*jc*.nii'));
    
spm_jobman('initcfg');
spm('defaults', 'fMRI');
 
    matlabbatch{1}.spm.util.imcalc.input = cellstr(files);
    matlabbatch{1}.spm.util.imcalc.output = 'jc';
    matlabbatch{1}.spm.util.imcalc.outdir = {run_path};
    matlabbatch{1}.spm.util.imcalc.expression = '(i1+i2)/2';
    matlabbatch{1}.spm.util.imcalc.var = struct('name', {}, 'value', {});
    matlabbatch{1}.spm.util.imcalc.options.dmtx = 0;
    matlabbatch{1}.spm.util.imcalc.options.mask = 0;
    matlabbatch{1}.spm.util.imcalc.options.interp = 1;
    matlabbatch{1}.spm.util.imcalc.options.dtype = 4;
    spm_jobman('run', matlabbatch);
    clear matlabbatch;
     
end