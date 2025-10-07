clear;clc;
spm('Defaults','fMRI');
spm_jobman('initcfg');
S_dir = 'H:\metaphor\First_level\single_trial_v1';  

data_name = dir([S_dir filesep 'sub*']);

data_out_dir = 'H:\metaphor\First_level\single_trial_v1\pattern';

if ~exist(data_out_dir)
    mkdir(data_out_dir);
end

% %% first time yy word
% for nsub = 1:size(data_name,1)
%     stage_dir = [S_dir filesep data_name(nsub).name];
%     %%
%     pattern1 = filenames(fullfile(stage_dir,'*F_yy_*nii'));
% 
%     out_dir = [data_out_dir filesep data_name(nsub).name];
%     
%     if ~exist(out_dir)
%         mkdir(out_dir);
%     end
%     
%     matlabbatch{1}.spm.util.cat.vols = cellstr(pattern1);
%     matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'F_yy_juzi.nii'];
%     matlabbatch{1}.spm.util.cat.dtype = 4;
%     matlabbatch{1}.spm.util.cat.RT = NaN;
%     spm_jobman('run',matlabbatch);
%     clear matlabbatch
% end
% 
% for nsub = 1:size(data_name,1)
%     stage_dir = [S_dir filesep data_name(nsub).name];
%     %%
%     pattern2 = filenames(fullfile(stage_dir,'*S_yy_*nii'));
% 
%     out_dir = [data_out_dir filesep data_name(nsub).name];
%     
%     if ~exist(out_dir)
%         mkdir(out_dir);
%     end
%     
%     matlabbatch{1}.spm.util.cat.vols = cellstr(pattern2);
%     matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'S_yy_juzi.nii'];
%     matlabbatch{1}.spm.util.cat.dtype = 4;
%     matlabbatch{1}.spm.util.cat.RT = NaN;
%     spm_jobman('run',matlabbatch);
%     clear matlabbatch
% end
% 
% %% first time yy word
% for nsub = 1:size(data_name,1)
%     stage_dir = [S_dir filesep data_name(nsub).name];
%     %%
%     pattern1 = filenames(fullfile(stage_dir,'*F_kj_*nii'));
% 
%     out_dir = [data_out_dir filesep data_name(nsub).name];
%     
%     if ~exist(out_dir)
%         mkdir(out_dir);
%     end
%     
%     matlabbatch{1}.spm.util.cat.vols = cellstr(pattern1);
%     matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'F_kj_juzi.nii'];
%     matlabbatch{1}.spm.util.cat.dtype = 4;
%     matlabbatch{1}.spm.util.cat.RT = NaN;
%     spm_jobman('run',matlabbatch);
%     clear matlabbatch
% end
% 
% for nsub = 1:size(data_name,1)
%     stage_dir = [S_dir filesep data_name(nsub).name];
%     %%
%     pattern2 = filenames(fullfile(stage_dir,'*S_kj_*nii'));
% 
%     out_dir = [data_out_dir filesep data_name(nsub).name];
%     
%     if ~exist(out_dir)
%         mkdir(out_dir);
%     end
%     
%     matlabbatch{1}.spm.util.cat.vols = cellstr(pattern2);
%     matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'S_kj_juzi.nii'];
%     matlabbatch{1}.spm.util.cat.dtype = 4;
%     matlabbatch{1}.spm.util.cat.RT = NaN;
%     spm_jobman('run',matlabbatch);
%     clear matlabbatch
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% first time kj word
for nsub = 1:size(data_name,1)
    stage_dir = [S_dir filesep data_name(nsub).name];
    %%
    pattern = filenames(fullfile(stage_dir,'*F_kjw_*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'F_kj_sword.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end

for nsub = 1:size(data_name,1)
    stage_dir = [S_dir filesep data_name(nsub).name];
    %%
    pattern = filenames(fullfile(stage_dir,'*F_kjew_*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'F_kj_eword.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end

%% first time yy 
for nsub = 1:size(data_name,1)
    stage_dir = [S_dir filesep data_name(nsub).name];
    %%
    pattern = filenames(fullfile(stage_dir,'*F_yyw_*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'F_yy_sword.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end

for nsub = 1:size(data_name,1)
    stage_dir = [S_dir filesep data_name(nsub).name];
    %%
    pattern = filenames(fullfile(stage_dir,'*F_yyew_*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'F_yy_eword.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end

%% first time jc word
for nsub = 1:size(data_name,1)
    stage_dir = [S_dir filesep data_name(nsub).name];
    %%
    pattern = filenames(fullfile(stage_dir,'*F_jc*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'F_jc_word.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end



%% first time jx word
for nsub = 1:size(data_name,1)
    stage_dir = [S_dir filesep data_name(nsub).name];
    %%
    pattern = filenames(fullfile(stage_dir,'*F_jx*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'F_jx_word.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end

%
%
%
%
%
%

%%  second yy 
for nsub = 1:size(data_name,1)
    stage_dir = [S_dir filesep data_name(nsub).name];
    %%
    pattern = filenames(fullfile(stage_dir,'*S_yyw_*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'S_yy_sword.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end


for nsub = 1:size(data_name,1)
    stage_dir = [S_dir filesep data_name(nsub).name];
    %%
    pattern = filenames(fullfile(stage_dir,'*S_yyew_*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'S_yy_eword.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end

% %% first time kj word
for nsub = 1:size(data_name,1)
    stage_dir = [S_dir filesep data_name(nsub).name];
    %%
    pattern = filenames(fullfile(stage_dir,'*S_kjw_*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'S_kj_sword.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end

for nsub = 1:size(data_name,1)
    stage_dir = [S_dir filesep data_name(nsub).name];
    %%
    pattern = filenames(fullfile(stage_dir,'*S_kjew_*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'S_kj_eword.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end


%% first time jc word
for nsub = 1:size(data_name,1)
    stage_dir = [S_dir filesep data_name(nsub).name];
    %%
    pattern = filenames(fullfile(stage_dir,'*S_jc*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'S_jc_word.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end
%% first time jx word
for nsub = 1:size(data_name,1)
    stage_dir = [S_dir filesep data_name(nsub).name];
    %%
    pattern = filenames(fullfile(stage_dir,'*S_jx*nii'));

    out_dir = [data_out_dir filesep data_name(nsub).name];
    
    if ~exist(out_dir)
        mkdir(out_dir);
    end
    
    matlabbatch{1}.spm.util.cat.vols = cellstr(pattern);
    matlabbatch{1}.spm.util.cat.name = [out_dir filesep 'S_jx_word.nii'];
    matlabbatch{1}.spm.util.cat.dtype = 4;
    matlabbatch{1}.spm.util.cat.RT = NaN;
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
end




