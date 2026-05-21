clear;
clc;
data_dir = 'H:\metaphor\events_true_memory'; 
sublist = dir([data_dir,'\sub*']);
%%
counter=0;
%% 1back
for nsub=1:length(sublist)
   
    sub_dir = [data_dir,filesep,sublist(nsub).name];
    %-----------------------------------------------------------------------
    run_con_all = dir([sub_dir,filesep,'sub*.txt']);
    run7_con=readtable([sub_dir,filesep,run_con_all(7).name]);
 
    yy_1 = sum(contains(run7_con.trial_type,'yyew') & run7_con.memory3 == 1);
    yy_05 = 0.5*sum(contains(run7_con.trial_type,'yyew') & run7_con.memory3 == 0.5);
    kj_1 = sum(contains(run7_con.trial_type,'kjew') & run7_con.memory3 == 1);
    kj_05 = 0.5*sum(contains(run7_con.trial_type,'kjew') & run7_con.memory3 == 0.5);
    counter=counter+1;
    
    memory_yy(counter,:)=yy_1+yy_05;
    memory_kj(counter,:)=kj_1+kj_05;
end

[h,p,ci,stats] = ttest(memory_yy,memory_kj);

mean(memory_yy)
mean(memory_kj)

save memory1050_no_sub11.mat memory_yy memory_kj


