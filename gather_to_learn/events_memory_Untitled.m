clear;
clc;
data_dir = 'H:\metaphor\events_memory'; 
sublist = dir([data_dir,'\sub*']);
%%
counter=0;
%% 1back
for nsub=1:length(sublist)
    if nsub == 12
        continue;
    end
    
    if nsub ==23
        continue;
    end
    sub_dir = [data_dir,filesep,sublist(nsub).name];
    %-----------------------------------------------------------------------
    run_con_all = dir([sub_dir,filesep,'sub*.txt']);
    run7_con=readtable([sub_dir,filesep,run_con_all(7).name]);
 
    yy_3 = sum(contains(run7_con.trial_type,'yyew') & run7_con.recall == 3);
    yy_4 = sum(contains(run7_con.trial_type,'yyew') & run7_con.recall == 4);
    kj_3 = sum(contains(run7_con.trial_type,'kjew') & run7_con.recall == 3);
    kj_4 = sum(contains(run7_con.trial_type,'kjew') & run7_con.recall == 4);
    counter=counter+1;
    
    memory_yy(counter,:)=yy_3;
    memory_kj(counter,:)=kj_3;
end

[h,p,ci,stats] = ttest(memory_yy,memory_kj)

mean(memory_yy)
mean(memory_kj)
