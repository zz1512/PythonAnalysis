clear;clc;
spm('Defaults','fMRI');
spm_jobman('initcfg');
S_dir = 'H:\metaphor\LSS\data';  

data_name = dir([S_dir filesep 'sub*']);

data_out_dir = 'H:\metaphor\LSS\rsa\pattern_similarity_change\mvpa_pattern';

if ~exist(data_out_dir)
    mkdir(data_out_dir);
end


for nsub = 1:size(data_name,1)
    if nsub == 12
        continue;
    end
    
     if nsub == 23
         continue;
     end
    stage_dir = [S_dir filesep data_name(nsub).name '\run-7'];
    %%
    pattern = filenames(fullfile(stage_dir,'*kjw_*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'run7_kj_start_word.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end

for nsub = 1:size(data_name,1)
    if nsub == 12
        continue;
    end
    
     if nsub == 23
         continue;
     end
    
    stage_dir = [S_dir filesep data_name(nsub).name '\run-7'];
    %%
    pattern = filenames(fullfile(stage_dir,'*kjew_*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'run7_kj_end_word.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end


for nsub = 1:size(data_name,1)
    
    if nsub == 12
        continue;
    end
    
     if nsub == 23
         continue;
     end
    stage_dir = [S_dir filesep data_name(nsub).name '\run-7'];
    %
    pattern = filenames(fullfile(stage_dir,'*yyw_*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'run7_yy_start_word.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end

for nsub = 1:size(data_name,1)
    
    if nsub == 12
        continue;
    end
    
     if nsub == 23
         continue;
     end
     
    stage_dir = [S_dir filesep data_name(nsub).name '\run-7'];
    %
    pattern = filenames(fullfile(stage_dir,'*yyew_*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'run7_yy_end_word.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end
