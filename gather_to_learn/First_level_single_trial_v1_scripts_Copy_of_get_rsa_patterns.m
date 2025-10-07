clear;clc;
spm('Defaults','fMRI');
spm_jobman('initcfg');
S_dir = 'H:\metaphor\First_level\single_trial_v1';  

data_name = dir([S_dir filesep 'sub*']);

data_out_dir = 'H:\metaphor\First_level\single_trial_v1\pattern';

if ~exist(data_out_dir)
    mkdir(data_out_dir);
end

%% first time yy word
for nsub = 1:size(data_name,1)
    if nsub == 12 || nsub == 23
        continue;
    end
    stage_dir = [S_dir filesep data_name(nsub).name];
    %
    pattern1 = filenames(fullfile(stage_dir,'*M_yyw_*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern1);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'M_yy_sword.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
end

for nsub  = 1:size(data_name,1)
    if nsub == 12 || nsub == 23
        continue;
    end
    stage_dir = [S_dir filesep data_name(nsub).name];
    %%
    pattern2 = filenames(fullfile(stage_dir,'*M_yyew_*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern2);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'M_yy_eword.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
end

%% first time kj word
for nsub = 1:size(data_name,1)
     if nsub == 12 || nsub == 23
        continue;
    end
    stage_dir = [S_dir filesep data_name(nsub).name];
    %%
    pattern1 = filenames(fullfile(stage_dir,'*M_kjw_*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern1);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'M_kj_sword.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end

for nsub = 1:size(data_name,1)
     if nsub == 12 || nsub == 23
        continue;
    end
    stage_dir = [S_dir filesep data_name(nsub).name];
    %%
    pattern2 = filenames(fullfile(stage_dir,'*M_kjew_*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern2);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'M_kj_eword.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end

