% clear; clc;
% spm_jobman('initcfg');
% spm('defaults', 'fMRI');
% 
% data_dir = 'H:\metaphor\Pro_proc_data';
% sublist = dir([data_dir,'\sub*']);
% %%
% for nsub = [12,23]  %
%     
%     mat = ['H:\metaphor\First_level\univariate_v1\',sublist(nsub).name,'\SPM.mat'];
%     
%     %%
%     matlabbatch{1}.spm.stats.con.spmmat(1) = {mat};
%     
%     %%
%     matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = 'first_time_yy_sword';
%     matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = [1,0,zeros(1,10),1,0,zeros(1,10),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12)];
%     matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
%     %%
%     matlabbatch{1}.spm.stats.con.consess{2}.tcon.name = 'first_time_yy_eword';
%     matlabbatch{1}.spm.stats.con.consess{2}.tcon.weights = [0,1,zeros(1,10),0,1,zeros(1,10),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12)];
%     matlabbatch{1}.spm.stats.con.consess{2}.tcon.sessrep = 'none';
%     %%
%     matlabbatch{1}.spm.stats.con.consess{3}.tcon.name = 'first_time_yy_word';
%     matlabbatch{1}.spm.stats.con.consess{3}.tcon.weights = [1,1,zeros(1,10),1,1,zeros(1,10),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12)];
%     matlabbatch{1}.spm.stats.con.consess{3}.tcon.sessrep = 'none';
%     %% kj
%     matlabbatch{1}.spm.stats.con.consess{4}.tcon.name = 'first_time_kj_sword';
%     matlabbatch{1}.spm.stats.con.consess{4}.tcon.weights = [0,0,1,0,zeros(1,8),0,0,1,0,zeros(1,8),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12)];
%     matlabbatch{1}.spm.stats.con.consess{4}.tcon.sessrep = 'none';
%     %%
%     matlabbatch{1}.spm.stats.con.consess{5}.tcon.name = 'first_time_kj_eword';
%     matlabbatch{1}.spm.stats.con.consess{5}.tcon.weights = [0,0,0,1,zeros(1,8),0,0,0,1,zeros(1,8),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12)];
%     matlabbatch{1}.spm.stats.con.consess{5}.tcon.sessrep = 'none';
%     %%
%     matlabbatch{1}.spm.stats.con.consess{6}.tcon.name = 'first_time_kj_word';
%     matlabbatch{1}.spm.stats.con.consess{6}.tcon.weights = [0,0,1,1,zeros(1,8),0,0,1,1,zeros(1,8),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12)];
%     matlabbatch{1}.spm.stats.con.consess{6}.tcon.sessrep = 'none';
%     %% jc
%     matlabbatch{1}.spm.stats.con.consess{7}.tcon.name = 'first_time_jc';
%     matlabbatch{1}.spm.stats.con.consess{7}.tcon.weights = [0,0,0,0,1,0,zeros(1,6),0,0,0,0,1,0,zeros(1,6),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12)];
%     matlabbatch{1}.spm.stats.con.consess{7}.tcon.sessrep = 'none';
%     %% jx
%     matlabbatch{1}.spm.stats.con.consess{8}.tcon.name = 'first_time_jx';
%     matlabbatch{1}.spm.stats.con.consess{8}.tcon.weights = [0,0,0,0,0,1,zeros(1,6),0,0,0,0,0,1,zeros(1,6),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12)];
%     matlabbatch{1}.spm.stats.con.consess{8}.tcon.sessrep = 'none';
%     %% first yy
%     matlabbatch{1}.spm.stats.con.consess{9}.tcon.name = 'first_time_yy_juzi';
%     matlabbatch{1}.spm.stats.con.consess{9}.tcon.weights = [zeros(1,12),zeros(1,12),1,0,zeros(1,6),zeros(1,8),zeros(1,12),zeros(1,12)];
%     matlabbatch{1}.spm.stats.con.consess{9}.tcon.sessrep = 'none';
%     %% first kj
%     matlabbatch{1}.spm.stats.con.consess{10}.tcon.name = 'first_time_kj_juzi';
%     matlabbatch{1}.spm.stats.con.consess{10}.tcon.weights = [zeros(1,12),zeros(1,12),0,1,zeros(1,6),zeros(1,8),zeros(1,12),zeros(1,12)];
%     matlabbatch{1}.spm.stats.con.consess{10}.tcon.sessrep = 'none';
%     %% second yy
%     matlabbatch{1}.spm.stats.con.consess{11}.tcon.name = 'second_time_yy_juzi';
%     matlabbatch{1}.spm.stats.con.consess{11}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),1,0,zeros(1,6),zeros(1,12),zeros(1,12)];
%     matlabbatch{1}.spm.stats.con.consess{11}.tcon.sessrep = 'none';
%     %% second kj
%     matlabbatch{1}.spm.stats.con.consess{12}.tcon.name = 'second_time_kj_juzi';
%     matlabbatch{1}.spm.stats.con.consess{12}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),0,1,zeros(1,6),zeros(1,12),zeros(1,12)];
%     matlabbatch{1}.spm.stats.con.consess{12}.tcon.sessrep = 'none';
%     %%
%     matlabbatch{1}.spm.stats.con.consess{13}.tcon.name = 'second_time_yy_sword';
%     matlabbatch{1}.spm.stats.con.consess{13}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),1,0,zeros(1,10),1,0,zeros(1,10)];
%     matlabbatch{1}.spm.stats.con.consess{13}.tcon.sessrep = 'none';
%     %%
%     matlabbatch{1}.spm.stats.con.consess{14}.tcon.name = 'second_time_yy_eword';
%     matlabbatch{1}.spm.stats.con.consess{14}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),0,1,zeros(1,10),0,1,zeros(1,10)];
%     matlabbatch{1}.spm.stats.con.consess{14}.tcon.sessrep = 'none';
%     
%     matlabbatch{1}.spm.stats.con.consess{15}.tcon.name = 'second_time_yy_word';
%     matlabbatch{1}.spm.stats.con.consess{15}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),1,1,zeros(1,10),1,1,zeros(1,10)];
%     matlabbatch{1}.spm.stats.con.consess{15}.tcon.sessrep = 'none';
%     %% kj
%     matlabbatch{1}.spm.stats.con.consess{16}.tcon.name = 'second_time_kj_sword';
%     matlabbatch{1}.spm.stats.con.consess{16}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),0,0,1,0,zeros(1,8),0,0,1,0,zeros(1,8)];
%     matlabbatch{1}.spm.stats.con.consess{16}.tcon.sessrep = 'none';
%     %%
%     matlabbatch{1}.spm.stats.con.consess{17}.tcon.name = 'second_time_kj_eword';
%     matlabbatch{1}.spm.stats.con.consess{17}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),0,0,0,1,zeros(1,8),0,0,0,1,zeros(1,8)];
%     matlabbatch{1}.spm.stats.con.consess{17}.tcon.sessrep = 'none';
%     
%     %%
%     matlabbatch{1}.spm.stats.con.consess{18}.tcon.name = 'second_time_kj_word';
%     matlabbatch{1}.spm.stats.con.consess{18}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),0,0,1,1,zeros(1,8),0,0,1,1,zeros(1,8)];
%     matlabbatch{1}.spm.stats.con.consess{18}.tcon.sessrep = 'none';
%     %% jc
%     matlabbatch{1}.spm.stats.con.consess{19}.tcon.name = 'second_time_jc';
%     matlabbatch{1}.spm.stats.con.consess{19}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),0,0,0,0,1,0,zeros(1,6),0,0,0,0,1,0,zeros(1,6)];
%     matlabbatch{1}.spm.stats.con.consess{19}.tcon.sessrep = 'none';
%     %% jx
%     matlabbatch{1}.spm.stats.con.consess{20}.tcon.name = 'second_time_jx';
%     matlabbatch{1}.spm.stats.con.consess{20}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),0,0,0,0,0,1,zeros(1,6),0,0,0,0,0,1,zeros(1,6)];
%     matlabbatch{1}.spm.stats.con.consess{20}.tcon.sessrep = 'none';
%     
%     
%     matlabbatch{1}.spm.stats.con.delete = 1;  % 刪除之前的舊文件
%     
%     spm_jobman('run', matlabbatch);
%     clear matlabbatch;
%     
% end

clear; clc;
spm_jobman('initcfg');
spm('defaults', 'fMRI');

data_dir = 'H:\metaphor\Pro_proc_data';
sublist = dir([data_dir,'\sub*']);


for nsub = 1:length(sublist)
    if nsub == 12 || nsub == 23
        continue;
    end
    
    mat = ['H:\metaphor\First_level\univariate_v1\',sublist(nsub).name,'\SPM.mat'];
    %%
    matlabbatch{1}.spm.stats.con.spmmat(1) = {mat};
    
    %%
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = 'first_time_yy_sword';
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = [1,0,zeros(1,10),1,0,zeros(1,10),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12),zeros(1,10)];
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
    %%
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.name = 'first_time_yy_eword';
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.weights = [0,1,zeros(1,10),0,1,zeros(1,10),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12),zeros(1,10)];
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.sessrep = 'none';
    %%
    matlabbatch{1}.spm.stats.con.consess{3}.tcon.name = 'first_time_yy_word';
    matlabbatch{1}.spm.stats.con.consess{3}.tcon.weights = [1,1,zeros(1,10),1,1,zeros(1,10),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12),zeros(1,10)];
    matlabbatch{1}.spm.stats.con.consess{3}.tcon.sessrep = 'none';
    %% kj
    matlabbatch{1}.spm.stats.con.consess{4}.tcon.name = 'first_time_kj_sword';
    matlabbatch{1}.spm.stats.con.consess{4}.tcon.weights = [0,0,1,0,zeros(1,8),0,0,1,0,zeros(1,8),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12),zeros(1,10)];
    matlabbatch{1}.spm.stats.con.consess{4}.tcon.sessrep = 'none';
    %%
    matlabbatch{1}.spm.stats.con.consess{5}.tcon.name = 'first_time_kj_eword';
    matlabbatch{1}.spm.stats.con.consess{5}.tcon.weights = [0,0,0,1,zeros(1,8),0,0,0,1,zeros(1,8),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12),zeros(1,10)];
    matlabbatch{1}.spm.stats.con.consess{5}.tcon.sessrep = 'none';
    %%
    matlabbatch{1}.spm.stats.con.consess{6}.tcon.name = 'first_time_kj_word';
    matlabbatch{1}.spm.stats.con.consess{6}.tcon.weights = [0,0,1,1,zeros(1,8),0,0,1,1,zeros(1,8),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12),zeros(1,10)];
    matlabbatch{1}.spm.stats.con.consess{6}.tcon.sessrep = 'none';
    %% jc
    matlabbatch{1}.spm.stats.con.consess{7}.tcon.name = 'first_time_jc';
    matlabbatch{1}.spm.stats.con.consess{7}.tcon.weights = [0,0,0,0,1,0,zeros(1,6),0,0,0,0,1,0,zeros(1,6),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12),zeros(1,10)];
    matlabbatch{1}.spm.stats.con.consess{7}.tcon.sessrep = 'none';
    %% jx
    matlabbatch{1}.spm.stats.con.consess{8}.tcon.name = 'first_time_jx';
    matlabbatch{1}.spm.stats.con.consess{8}.tcon.weights = [0,0,0,0,0,1,zeros(1,6),0,0,0,0,0,1,zeros(1,6),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12),zeros(1,10)];
    matlabbatch{1}.spm.stats.con.consess{8}.tcon.sessrep = 'none';
    %% first yy
    matlabbatch{1}.spm.stats.con.consess{9}.tcon.name = 'first_time_yy_juzi';
    matlabbatch{1}.spm.stats.con.consess{9}.tcon.weights = [zeros(1,12),zeros(1,12),1,0,zeros(1,6),zeros(1,8),zeros(1,12),zeros(1,12),zeros(1,10)];
    matlabbatch{1}.spm.stats.con.consess{9}.tcon.sessrep = 'none';
    %% first kj
    matlabbatch{1}.spm.stats.con.consess{10}.tcon.name = 'first_time_kj_juzi';
    matlabbatch{1}.spm.stats.con.consess{10}.tcon.weights = [zeros(1,12),zeros(1,12),0,1,zeros(1,6),zeros(1,8),zeros(1,12),zeros(1,12),zeros(1,10)];
    matlabbatch{1}.spm.stats.con.consess{10}.tcon.sessrep = 'none';
    %% second yy
    matlabbatch{1}.spm.stats.con.consess{11}.tcon.name = 'second_time_yy_juzi';
    matlabbatch{1}.spm.stats.con.consess{11}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),1,0,zeros(1,6),zeros(1,12),zeros(1,12),zeros(1,10)];
    matlabbatch{1}.spm.stats.con.consess{11}.tcon.sessrep = 'none';
    %% second kj
    matlabbatch{1}.spm.stats.con.consess{12}.tcon.name = 'second_time_kj_juzi';
    matlabbatch{1}.spm.stats.con.consess{12}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),0,1,zeros(1,6),zeros(1,12),zeros(1,12),zeros(1,10)];
    matlabbatch{1}.spm.stats.con.consess{12}.tcon.sessrep = 'none';
    %%
    matlabbatch{1}.spm.stats.con.consess{13}.tcon.name = 'second_time_yy_sword';
    matlabbatch{1}.spm.stats.con.consess{13}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),1,0,zeros(1,10),1,0,zeros(1,10),zeros(1,10)];
    matlabbatch{1}.spm.stats.con.consess{13}.tcon.sessrep = 'none';
    %%
    matlabbatch{1}.spm.stats.con.consess{14}.tcon.name = 'second_time_yy_eword';
    matlabbatch{1}.spm.stats.con.consess{14}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),0,1,zeros(1,10),0,1,zeros(1,10),zeros(1,10)];
    matlabbatch{1}.spm.stats.con.consess{14}.tcon.sessrep = 'none';
    
    matlabbatch{1}.spm.stats.con.consess{15}.tcon.name = 'second_time_yy_word';
    matlabbatch{1}.spm.stats.con.consess{15}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),1,1,zeros(1,10),1,1,zeros(1,10),zeros(1,10)];
    matlabbatch{1}.spm.stats.con.consess{15}.tcon.sessrep = 'none';
    %% kj
    matlabbatch{1}.spm.stats.con.consess{16}.tcon.name = 'second_time_kj_sword';
    matlabbatch{1}.spm.stats.con.consess{16}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),0,0,1,0,zeros(1,8),0,0,1,0,zeros(1,8),zeros(1,10)];
    matlabbatch{1}.spm.stats.con.consess{16}.tcon.sessrep = 'none';
    %%
    matlabbatch{1}.spm.stats.con.consess{17}.tcon.name = 'second_time_kj_eword';
    matlabbatch{1}.spm.stats.con.consess{17}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),0,0,0,1,zeros(1,8),0,0,0,1,zeros(1,8),zeros(1,10)];
    matlabbatch{1}.spm.stats.con.consess{17}.tcon.sessrep = 'none';
    
    %%
    matlabbatch{1}.spm.stats.con.consess{18}.tcon.name = 'second_time_kj_word';
    matlabbatch{1}.spm.stats.con.consess{18}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),0,0,1,1,zeros(1,8),0,0,1,1,zeros(1,8),zeros(1,10)];
    matlabbatch{1}.spm.stats.con.consess{18}.tcon.sessrep = 'none';
    %% jc
    matlabbatch{1}.spm.stats.con.consess{19}.tcon.name = 'second_time_jc';
    matlabbatch{1}.spm.stats.con.consess{19}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),0,0,0,0,1,0,zeros(1,6),0,0,0,0,1,0,zeros(1,6),zeros(1,10)];
    matlabbatch{1}.spm.stats.con.consess{19}.tcon.sessrep = 'none';
    %% jx
    matlabbatch{1}.spm.stats.con.consess{20}.tcon.name = 'second_time_jx';
    matlabbatch{1}.spm.stats.con.consess{20}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),0,0,0,0,0,1,zeros(1,6),0,0,0,0,0,1,zeros(1,6),zeros(1,10)];
    matlabbatch{1}.spm.stats.con.consess{20}.tcon.sessrep = 'none';
    
    %% memory yyw
    matlabbatch{1}.spm.stats.con.consess{21}.tcon.name = 'memory_yy_sword';
    matlabbatch{1}.spm.stats.con.consess{21}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12),1,0,0,0,zeros(1,6)];
    matlabbatch{1}.spm.stats.con.consess{21}.tcon.sessrep = 'none';
    
    %% memory yyew
    matlabbatch{1}.spm.stats.con.consess{22}.tcon.name = 'memory_yy_eword';
    matlabbatch{1}.spm.stats.con.consess{22}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12),0,1,0,0,zeros(1,6)];
    matlabbatch{1}.spm.stats.con.consess{22}.tcon.sessrep = 'none';
    
    %% memory kjw
    matlabbatch{1}.spm.stats.con.consess{23}.tcon.name = 'memory_kj_sword';
    matlabbatch{1}.spm.stats.con.consess{23}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12),0,0,1,0,zeros(1,6)];
    matlabbatch{1}.spm.stats.con.consess{23}.tcon.sessrep = 'none';
    
    %% memory kjew
    matlabbatch{1}.spm.stats.con.consess{24}.tcon.name = 'memory_kj_eword';
    matlabbatch{1}.spm.stats.con.consess{24}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12),0,0,0,1,zeros(1,6)];
    matlabbatch{1}.spm.stats.con.consess{24}.tcon.sessrep = 'none';
    %%
    matlabbatch{1}.spm.stats.con.consess{25}.tcon.name = 'memory_yy_word';
    matlabbatch{1}.spm.stats.con.consess{25}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12),1,1,0,0,zeros(1,6)];
    matlabbatch{1}.spm.stats.con.consess{25}.tcon.sessrep = 'none';
    matlabbatch{1}.spm.stats.con.consess{26}.tcon.name = 'memory_kj_word';
    matlabbatch{1}.spm.stats.con.consess{26}.tcon.weights = [zeros(1,12),zeros(1,12),zeros(1,8),zeros(1,8),zeros(1,12),zeros(1,12),0,0,1,1,zeros(1,6)];
    matlabbatch{1}.spm.stats.con.consess{26}.tcon.sessrep = 'none';
    
    
    
    matlabbatch{1}.spm.stats.con.delete = 1;
    
    spm_jobman('run', matlabbatch);
    clear matlabbatch;
    
    
end

