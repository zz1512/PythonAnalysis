clear;clc;
spm('Defaults','fMRI');
spm_jobman('initcfg');
S_dir = 'H:\metaphor\First_level\single_trial_v2_memory';  

data_name = dir([S_dir filesep 'sub*']);

data_out_dir = 'H:\metaphor\First_level\single_trial_v2\pattern';

if ~exist(data_out_dir)
    mkdir(data_out_dir);
end

%% first time yy word
for nsub = 1:size(data_name,1)
    stage_dir = [S_dir filesep data_name(nsub).name];
    %%
    pattern1 = filenames(fullfile(stage_dir,'*F_yyw_*nii'));
    
if string(pattern1{1}) == '找不到文件'
     
        continue;
    
end
    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern1);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'First_F_yy_start_word.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
end

for nsub = 1:size(data_name,1)
    stage_dir = [S_dir filesep data_name(nsub).name];
    %%
    pattern2 = filenames(fullfile(stage_dir,'*F_yyew_*nii'));
    
if string(pattern2{1}) == '找不到文件'
     
        continue;
    
end
    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern2);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'First_F_yy_end_word.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
end


%%   yy word

for nsub = 1:size(data_name,1)
    stage_dir = [S_dir filesep data_name(nsub).name];
    %%
    pattern8 = filenames(fullfile(stage_dir,'*Second_F_yyw_*nii'));

    if string(pattern8{1}) == '找不到文件'
     
        continue;
    
end
    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern8);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'Second_F_yy_start_word.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end

for nsub = 1:size(data_name,1)
    stage_dir = [S_dir filesep data_name(nsub).name];
    %%
    pattern6 = filenames(fullfile(stage_dir,'*Second_F_yyew_*nii'));
if string(pattern6{1}) == '找不到文件'
     
        continue;
    
end
    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern6);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'Second_F_yy_end_word.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end

