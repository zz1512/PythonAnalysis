clear;clc;
spm('Defaults','fMRI');
spm_jobman('initcfg');
S_dir = 'H:\metaphor\LSS\data_self_report_memory';  

data_name = dir([S_dir filesep 'sub*']);

data_out_dir = 'H:\metaphor\LSS\rsa\pattern_similarity_change_self-report_no1\mvpa_pattern';

if ~exist(data_out_dir)
    mkdir(data_out_dir);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% first time kj word
for nsub = 1:size(data_name,1)
    
    stage_dir = [S_dir filesep data_name(nsub).name '\first_words_view'];
    %%
    
    pattern05 = filenames(fullfile(stage_dir,'*kjw_*0.5*nii'));
    pattern00 = filenames(fullfile(stage_dir,'*kjw_*0.0*nii'));
    pattern = [pattern05;pattern00];
    
    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'F_kj_start_word.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end

for nsub = 1:size(data_name,1)
    
    stage_dir = [S_dir filesep data_name(nsub).name '\first_words_view'];
    %%
    
    pattern05 = filenames(fullfile(stage_dir,'*kjew_*0.5*nii'));
    pattern00 = filenames(fullfile(stage_dir,'*kjew_*0.0*nii'));
    pattern = [pattern05;pattern00];
    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'F_kj_end_word.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end

%% first time yy 
for nsub = 1:size(data_name,1)
    
      if nsub == 11
        continue;
    end
    stage_dir = [S_dir filesep data_name(nsub).name '\first_words_view'];
    %%
  %  pattern = filenames(fullfile(stage_dir,'*yyw_*1.0*nii'));
    pattern05 = filenames(fullfile(stage_dir,'*yyw_*0.5*nii'));
    pattern00 = filenames(fullfile(stage_dir,'*yyw_*0.0*nii'));
    pattern = [pattern05;pattern00];
    
    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'F_yy_start_word.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end


for nsub = 1:size(data_name,1)
    
      if nsub == 11
        continue;
    end
    stage_dir = [S_dir filesep data_name(nsub).name '\first_words_view'];
    %%
     pattern05 = filenames(fullfile(stage_dir,'*yyew_*0.5*nii'));
    pattern00 = filenames(fullfile(stage_dir,'*yyew_*0.0*nii'));
    pattern = [pattern05;pattern00];

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'F_yy_end_word.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end




%%  second yy 
for nsub = 1:size(data_name,1)
    
      if nsub == 11
        continue;
    end
    stage_dir = [S_dir filesep data_name(nsub).name '\second_words_view'];
    %%
    pattern05 = filenames(fullfile(stage_dir,'*yyw_*0.5*nii'));
    pattern00 = filenames(fullfile(stage_dir,'*yyw_*0.0*nii'));
    pattern = [pattern05;pattern00];

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'S_yy_start_word.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end


for nsub = 1:size(data_name,1)
    
      if nsub == 11
        continue;
    end
    stage_dir = [S_dir filesep data_name(nsub).name '\second_words_view'];
    %%
     pattern05 = filenames(fullfile(stage_dir,'*yyew_*0.5*nii'));
    pattern00 = filenames(fullfile(stage_dir,'*yyew_*0.0*nii'));
    pattern = [pattern05;pattern00];
    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'S_yy_end_word.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end










% %% first time kj word
for nsub = 1:size(data_name,1)
    stage_dir = [S_dir filesep data_name(nsub).name '\second_words_view'];
    %%
   % pattern = filenames(fullfile(stage_dir,'*kjw_*1.0*nii'));
    pattern05 = filenames(fullfile(stage_dir,'*kjw_*0.5*nii'));
    pattern00 = filenames(fullfile(stage_dir,'*kjw_*0.0*nii'));
    pattern = [pattern05;pattern00];
    
    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'S_kj_start_word.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end

for nsub = 1:size(data_name,1)
    stage_dir = [S_dir filesep data_name(nsub).name '\second_words_view'];
    %%
   % pattern = filenames(fullfile(stage_dir,'*kjew_*1.0*nii'));
    pattern05 = filenames(fullfile(stage_dir,'*kjew_*0.5*nii'));
    pattern00 = filenames(fullfile(stage_dir,'*kjew_*0.0*nii'));
    pattern = [pattern05;pattern00];
    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'S_kj_end_word.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end
