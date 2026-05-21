clear;clc;
spm('Defaults','fMRI');
spm_jobman('initcfg');
S_dir = 'H:\metaphor\First_level\single_trial_v1';  
data_name = dir([S_dir filesep 'sub*']);
data_out_dir = 'H:\metaphor\First_level\single_trial_v1\pattern';

if ~exist(data_out_dir)
    mkdir(data_out_dir);
end

%% 
for nsub = 1:size(data_name,1)
    stage_dir = [S_dir filesep data_name(nsub).name];
    %%
    pattern = filenames(fullfile(stage_dir,'*F_yy*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'F_yy_juzi.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end


for nsub = 1:size(data_name,1)
    stage_dir = [S_dir filesep data_name(nsub).name];
    %%
    pattern = filenames(fullfile(stage_dir,'*S_yy*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'S_yy_juzi.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end


for nsub = 1:size(data_name,1)
    stage_dir = [S_dir filesep data_name(nsub).name];
    %%
    pattern = filenames(fullfile(stage_dir,'*F_kj*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'F_kj_juzi.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end


for nsub = 1:size(data_name,1)
    stage_dir = [S_dir filesep data_name(nsub).name];
    %%
    pattern = filenames(fullfile(stage_dir,'*S_kj*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'S_kj_juzi.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end
