clear;clc
data_path = 'H:\metaphor\events_true_memory';%数据路径,存放每个被试的文件夹
subjects_ids={'sub-01','sub-02','sub-03','sub-04','sub-05','sub-06','sub-07','sub-08','sub-09','sub-10','sub-12','sub-13','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20','sub-21','sub-22','sub-23','sub-24','sub-25','sub-26','sub-27','sub-28'};% 被试编号
% 被试编号
nsubjects=numel(subjects_ids);% 被试个数
output_path = 'H:\metaphor\events_true_memory';

for i_subj=1:nsubjects
    subjects_id=subjects_ids{i_subj};
    sub_path=fullfile(data_path,subjects_id);
    sub_fn=dir([sub_path,'\*.txt']);
    output = fullfile(output_path,subjects_id);

if ~exist(output)
    mkdir(output);
end

%% first, we should get run1 run2 data
   k = 1;
   runfile=sub_fn(k).name;
   runfilename=[sub_path,filesep,runfile];
   rundata_1=readtable(runfilename);

   k = 2;
   runfile=sub_fn(k).name;
   runfilename=[sub_path,filesep,runfile];
   rundata_2=readtable(runfilename);
      
   alldata = [rundata_1;rundata_2];
   
   k = 7;
   runfile=sub_fn(k).name;
   runfilename=[sub_path,filesep,runfile];
   rundata_7=readtable(runfilename);
   
for i = 1:height(rundata_7)
    % 提取当前行的 trial_type 和 pic_num
    current_trial_type = rundata_7.trial_type{i};
    current_pic_num = rundata_7.pic_num(i);
    
    % 在 alldata 中查找匹配的行
    matching_row_indices = find(strncmp(current_trial_type, alldata.trial_type, 2) & alldata.pic_num == current_pic_num);
    
    % 如果找到匹配的行，则将 new_variable 的值赋给 rundata_3
    if ~isempty(matching_row_indices)
        rundata_7.memory3(i) = alldata.memory3(matching_row_indices(1));
    end
end
   
%%
        run_number=num2str(k);

        T_name=strcat(subjects_id,'_run-',run_number,'_events.txt'); 
        
        save_fn=[output,filesep,T_name];
        
        writetable(rundata_7,save_fn,'FileType','text','Delimiter','\t');


%%
  
end
