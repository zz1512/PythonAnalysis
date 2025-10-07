clear;clc;
data_origin_p = 'C:\Metaphor\Pre_proc\dcm\';
data_all = dir([data_origin_p,'*']);
data_all(1:2)=[];
nifti_path = 'D:\Metaphor\Pre_proc\nii\';
for n=1:numel(data_all)
    %% convert all dicom to nifti
    data_in = [data_origin_p data_all(n).name];
    data_out = [nifti_path data_all(n).name];
    if ~exist(data_out)
        mkdir(data_out);
        dcm2niix = 'C:\guowei_dcm2bids\MRIcron\Resources\dcm2niix.exe';
        system([dcm2niix ' -z y -f %p_%t_%s -o ' data_out ' ' data_in]);
        %% exclude the useless nii files
        file_out_all = dir([data_out '\*.json']);
        file_out_all = {file_out_all.name};
        exclude_type = {'localizer';'localizer2';'ADC';'FA';'TRACEW';'ColFA'};
        for file = 1:size(file_out_all,2)
            data = loadjson([data_out filesep file_out_all{file}]);
            SeriesDescription = data.SeriesDescription;
            SeriesDescription = strsplit(SeriesDescription,'_');
            [path name ext] = fileparts([data_out filesep file_out_all{file}]);
            disp(SeriesDescription{end});
            if cellfind(exclude_type,SeriesDescription{end})
               delete([path filesep name '*']);
            end
        end
    else
        continue
    end
end
%% convert name to bids style
out_bids_path = 'D:\Metaphor\Pre_proc\bids\';
for sub_num = 1:numel(data_all)
    data_out = [nifti_path data_all(sub_num).name];
    data_all(sub_num).name
    if sub_num <10
        subid = ['0' num2str(sub_num)];
    else
        subid = [num2str(sub_num)];
    end
    file_out_all = dir([data_out '\*.json']);
    file_out_all = {file_out_all.name};
    r_run=1;
    
    for file = 1:size(file_out_all,2)
        data = loadjson([data_out filesep file_out_all{file}]);
        SeriesDescription = data.SeriesDescription;
        SeriesDescription = strsplit(SeriesDescription,'_');
        [path name ext] = fileparts([data_out filesep file_out_all{file}]);
        
        serises = 1;
        serises_type = SeriesDescription{serises};
            if contains(serises_type,'YY')
                   fileout = [out_bids_path filesep 'sub-' subid '\func'];
                   mkdir(fileout); 
                   copyfile([path filesep name '.nii.gz'],[fileout filesep 'sub-',subid,'_task-yy_run-' num2str(r_run) '_bold.nii.gz']);
                   copyfile([path filesep name '.json'],[fileout filesep 'sub-',subid,'_task-yy_run-' num2str(r_run) '_bold.json']);
                   r_run = r_run+1;
            elseif contains(serises_type,'T1')
               fileout = [out_bids_path filesep 'sub-' subid '\anat'];
               mkdir(fileout); 
               copyfile([path filesep name '.nii.gz'],[fileout filesep 'sub-',subid,'_T1w.nii.gz']);
               copyfile([path filesep name '.json'],[fileout filesep 'sub-',subid,'_T1w.json']); 
            end
        
    end
    
    
    participants(sub_num).participant_id =['sub-' subid];
    participants(sub_num).name = data_all(n).name;
    participants(sub_num).gender = 'M';
    participants(sub_num).age = 20;
end

participants_tsv =struct2table(participants);
writetable(participants_tsv,[out_bids_path '\participants.csv'],'Delimiter','\t');
movefile([out_bids_path '\participants.csv'],[out_bids_path '\participants.tsv']);

%% create the bids validator files
jsonmesh=struct('Name','yy',...
         'BIDSVersion', '1.0.0rc1',...
         'License','CC0');
descrip_name = [out_bids_path '\dataset_description.json'];
savejson('',jsonmesh,descrip_name);
% 
% copyfile([out_bids_path filesep 'sub-' subid '\func' filesep 'sub-',subid,'_task-rest_bold.json'],...
%          [out_bids_path,'\task-rest_bold.json']);
% task_json =  loadjson([out_bids_path,'\task-rest_bold.json']);
% task_json.TaskName = 'rest';
% savejson('',task_json,[out_bids_path,'\task-rest_bold.json']);


