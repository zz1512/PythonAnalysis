clear; clc;
spm_jobman('initcfg');
spm('defaults', 'fMRI');

data_dir = 'J:\metaphor\Pro_proc_data';
sublist = dir([data_dir,'\sub*']);

for nsub = 1:length(sublist)
    if nsub == 12 || nsub == 23
        continue;
    end
    mat = ['J:\metaphor\First_level\univariate_v1\',sublist(nsub).name,'\SPM.mat'];
    %%
    matlabbatch{1}.spm.stats.con.spmmat(1) = {mat};
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = 'first_time_yy_sword_run1';
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = [1,0,zeros(1,10),0,0,zeros(1,10),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12),zeros(1,0)];
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
    %%
     matlabbatch{1}.spm.stats.con.consess{2}.tcon.name = 'first_time_yy_sword_run2';
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.weights = [0,0,zeros(1,10),1,0,zeros(1,10),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12),zeros(1,0)];
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.sessrep = 'none';
    %%
     matlabbatch{1}.spm.stats.con.consess{3}.tcon.name = 'first_time_yy_eword_run1';
    matlabbatch{1}.spm.stats.con.consess{3}.tcon.weights = [0,1,zeros(1,10),0,0,zeros(1,10),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12),zeros(1,0)];
    matlabbatch{1}.spm.stats.con.consess{3}.tcon.sessrep = 'none';
    %%
    matlabbatch{1}.spm.stats.con.consess{4}.tcon.name = 'first_time_yy_eword_run2';
    matlabbatch{1}.spm.stats.con.consess{4}.tcon.weights = [0,0,zeros(1,10),0,1,zeros(1,10),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12),zeros(1,0)];
    matlabbatch{1}.spm.stats.con.consess{4}.tcon.sessrep = 'none';
    %%
    matlabbatch{1}.spm.stats.con.consess{5}.tcon.name = 'second_time_yy_sword_run1';
    matlabbatch{1}.spm.stats.con.consess{5}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),1,0,zeros(1,10),0,0,zeros(1,10),zeros(1,0)];
    matlabbatch{1}.spm.stats.con.consess{5}.tcon.sessrep = 'none';
    %%
     matlabbatch{1}.spm.stats.con.consess{6}.tcon.name = 'second_time_yy_sword_run2';
    matlabbatch{1}.spm.stats.con.consess{6}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),0,0,zeros(1,10),1,0,zeros(1,10),zeros(1,0)];
    matlabbatch{1}.spm.stats.con.consess{6}.tcon.sessrep = 'none';
    %%
    matlabbatch{1}.spm.stats.con.consess{7}.tcon.name = 'second_time_yy_eword_run1';
    matlabbatch{1}.spm.stats.con.consess{7}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),0,1,zeros(1,10),0,0,zeros(1,10),zeros(1,0)];
    matlabbatch{1}.spm.stats.con.consess{7}.tcon.sessrep = 'none';
    %%
   matlabbatch{1}.spm.stats.con.consess{8}.tcon.name = 'second_time_yy_eword_run2';
    matlabbatch{1}.spm.stats.con.consess{8}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),0,0,zeros(1,10),0,1,zeros(1,10),zeros(1,0)];
    matlabbatch{1}.spm.stats.con.consess{8}.tcon.sessrep = 'none';
    
    matlabbatch{1}.spm.stats.con.delete = 1;
    
    spm_jobman('run', matlabbatch);
    clear matlabbatch;
    
    
end

