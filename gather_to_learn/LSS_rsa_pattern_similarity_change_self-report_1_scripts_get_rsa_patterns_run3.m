clear;clc;
spm('Defaults','fMRI');
spm_jobman('initcfg');
S_dir = 'H:\metaphor\LSS\data_self_report_memory';  

data_name = dir([S_dir filesep 'sub*']);

data_out_dir = 'H:\metaphor\LSS\rsa\pattern_similarity_change_self-report_1\mvpa_pattern';

if ~exist(data_out_dir)
    mkdir(data_out_dir);
end

%% no 25
for nsub = 1:size(data_name,1)
    stage_dir = [S_dir filesep data_name(nsub).name '\run-3'];
    %%
    pattern1 = filenames(fullfile(stage_dir,'*yy_*1.0*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern1);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'F_yy_juzi.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
end
% no 25
for nsub = 1:size(data_name,1)
    stage_dir = [S_dir filesep data_name(nsub).name '\run-3'];
    %%
    pattern1 = filenames(fullfile(stage_dir,'*kj_*1.0*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern1);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'F_kj_juzi.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
end