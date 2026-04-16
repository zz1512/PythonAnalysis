clear all; clc; close all;

%% 指定路径，建立初始文件
fMRI_dir = 'H:\shengyu\BIDS\Dicom';
 if ~exist (fMRI_dir,'file')
    mkdir(fMRI_dir);
 end
fMRI_data = dir(fMRI_dir);
fMRI_out = fullfile(fileparts(fMRI_dir), 'Nifti');
 if ~exist (fMRI_out,'file')
    mkdir(fMRI_out);
 end
%% 解压
% 逻辑：将文件解压后，将被试的Dicom文件转移到刚才建立的Dicom文件夹路径下，并将名称改为001,002...
rawPath = 'H:\shengyu\BIDS\RAW';
destPath = 'H:\shengyu\BIDS\Dicom';

% 获取 RAW 路径下的所有 zip 文件
zipFiles = dir(fullfile(rawPath, '*.zip'));

% 按修改时间排序（从早到晚）
[~, idx] = sort([zipFiles.datenum]);
zipFiles = zipFiles(idx);

% 初始化计数器，用于命名 DICOM 文件夹
folderCounter = 1;

% 循环处理每个 zip 文件
for i = 1:length(zipFiles)
    zipFilePath = fullfile(rawPath, zipFiles(i).name);
    
    % 提取 zip 文件名（不带扩展名）
    [~, zipName, ~] = fileparts(zipFiles(i).name);
    
    % 解压到同名文件夹
    unzipFolder = fullfile(rawPath, zipName);
    if ~exist(unzipFolder, 'dir')
        unzip(zipFilePath, unzipFolder);
    end
    
    % 查找解压后文件夹内的子文件夹（假设只有一个，即 DICOM 文件夹所在）
    subFolders = dir(unzipFolder);
    subFolders = subFolders([subFolders.isdir] & ~ismember({subFolders.name}, {'.', '..'}));
    
    if ~isempty(subFolders)
        % 找到 DICOM 文件夹（假设在子文件夹内）
        dicomParentFolder = fullfile(unzipFolder, subFolders(1).name);
        dicomFolder = fullfile(dicomParentFolder, 'DICOM'); % DICOM 文件夹
        
        if exist(dicomFolder, 'dir')
            % 新名字按顺序编号，如 001, 002, 003...
            newDicomFolderName = sprintf('%03d', folderCounter);
            newDicomFolder = fullfile(destPath, newDicomFolderName);
            
            % 如果目标已经存在，先删除
            if exist(newDicomFolder, 'dir')
                rmdir(newDicomFolder, 's');
            end
            
            % 移动 DICOM 文件夹到目标路径并重命名
            movefile(dicomFolder, newDicomFolder);
            
            % 计数器加 1
            folderCounter = folderCounter + 1;
        else
            warning('未找到 DICOM 文件夹: %s', dicomFolder);
        end
    else
        warning('解压文件夹为空: %s', unzipFolder);
    end
end

disp('所有文件处理完成！');
%%
%【根据需求改】创建一个README.txt文件
% README.txt 文件内容
readmeContent = 'Data description: Building on the model of our initial NKI-RS effort';
readmeFilename = 'README.txt';% 指定 README.txt 文件名
readmeFullFilePath = fullfile(fMRI_out, readmeFilename);% 与路径字符串结合
readmeFileID = fopen(readmeFullFilePath, 'w');% 打开文件进行写入
fprintf(readmeFileID, '%s', readmeContent);% 将 README.txt 内容写入文件
fclose(readmeFileID);% 关闭文件
disp(['README.txt 文件已成功创建：' readmeFullFilePath]);

% 【根据需求改】创建一个dataset_description.json文件
data.Name = 'NKI-Rockland Sample - Multiband Imaging Test-RetestPilot Dataset';
data.BIDSVersion = '1.0.2';
data.Authors = {'Dawn Thomsen','Marissa Jones Issa','Nancy Duan'};
jsonStr = jsonencode(data);% 将结构体转换为JSON格式的字符串
formattedJsonStr = sprintf('{\n  "Name":"%s",\n  "BIDSVersion":"%s",\n  "Authors": [\n    "%s"\n  ]\n}', data.Name, data.BIDSVersion, strjoin(data.Authors, '",\n    "'));
filename = 'dataset_description.json';% 指定要写入的文件名
fullFilePath = fullfile(fMRI_out, filename);% 与路径字符串结合
fileID = fopen(fullFilePath, 'w');% 打开文件进行写入
fprintf(fileID, '%s', formattedJsonStr);% 将JSON字符串写入文件
fclose(fileID);% 关闭文件
disp(['JSON文件已成功创建：' filename]);

% 【根据需求改】创建一个participants.json文件【注意：participantData之后也要改】
participantData.age.Description = 'age of the participant';
participantData.age.Units = 'year';
participantData.sex.Description = 'sex of the participant as reported by the participant';
participantData.sex.Levels.M = 'male';
participantData.sex.Levels.F = 'female';
participantData.handedness.Description = 'handedness of the participant as reported by the participant';
participantData.handedness.Levels.left = 'left';
participantData.handedness.Levels.right = 'right';
participantData.group.Description = 'experimental group the participant belonged to';
participantData.group.Levels.read = 'participants who read an inspirational text before the experiment';
participantData.group.Levels.write = 'participants who wrote an inspirational text before the experiment';
jsonStr = jsonencode(participantData);% 将结构体转换为JSON格式的字符串
% 格式化JSON字符串
formattedJsonStr = sprintf('{\n    "age": {\n        "Description": "%s",\n        "Units": "%s"\n    },\n    "sex": {\n        "Description": "%s",\n        "Levels": {\n            "M": "%s",\n            "F": "%s"\n        }\n    },\n    "handedness": {\n        "Description": "%s",\n        "Levels": {\n            "left": "%s",\n            "right": "%s"\n        }\n    },\n    "group": {\n        "Description": "%s",\n        "Levels": {\n            "read": "%s",\n            "write": "%s"\n        }\n    }\n}', ...
    participantData.age.Description, participantData.age.Units, ...
    participantData.sex.Description, participantData.sex.Levels.M, participantData.sex.Levels.F, ...
    participantData.handedness.Description, participantData.handedness.Levels.left, participantData.handedness.Levels.right, ...
    participantData.group.Description, participantData.group.Levels.read, participantData.group.Levels.write);
filename = 'participants.json';% 指定要写入的文件名
fullFilePath = fullfile(fMRI_out, filename);% 与路径字符串结合
fileID = fopen(fullFilePath, 'w');% 打开文件进行写入
fprintf(fileID, '%s', formattedJsonStr);% 将JSON字符串写入文件
fclose(fileID);
disp(['JSON文件已成功创建：' filename]);

%% 建立anat文件
for i =3:length(fMRI_data)
  participantNumber = fMRI_data(i).name;%【注意】这里需要所有的被试名称以数字进行标记
% 建立sub文件夹
  sub_folders = fullfile(fMRI_out, ['sub-' participantNumber]);%fileparts()是向上一级读取路径
  if ~exist (sub_folders,'file')
     mkdir(sub_folders);
  end
% 采用MRIcroGL进行数据转换
  sub_in = fullfile(fMRI_dir, participantNumber);
  dcm2niix = 'H:\MRIcroGL\Resources\dcm2niix.exe';
  system([dcm2niix ' -z y -f %p_%t_%s -o ' sub_folders ' ' sub_in]);

  file_out_all = dir([sub_folders '\*.json']);
        file_out_all = {file_out_all.name};
        exclude_type = {'localizer';'localizer2';'ADC';'FA';'TRACEW';'ColFA'};
%         for file = 1:size(file_out_all,2)
%             data = loadjson([sub_folders filesep file_out_all{file}]);
%             SeriesDescription = data.SeriesDescription;
%             SeriesDescription = strsplit(SeriesDescription,'_');
%             [path, name ext] = fileparts([sub_folders filesep file_out_all{file}]);
%             disp(SeriesDescription{end});
%             if cellfind(exclude_type,SeriesDescription{end})
%                delete([path filesep name '*']);
%             end
%         end

  
  % 对文件进行重命名
  % 获取生成的.nii.gz文件
  cd(sub_folders);
  niigz_files = dir(fullfile(sub_folders,'*.nii.gz'));
      for j = 1:numel(niigz_files)
            old_name = fullfile(sub_folders, niigz_files(j).name);
            % 解析文件名和扩展名
            [pathstr, name, ext] = fileparts(old_name);
            % 解析文件名中的信息
            parts = strsplit(name, '_');
            % 定义新文件名
            if any(strcmp(parts, 'WIP')) && any(strcmp(parts, 'T1W'))
                new_name = fullfile(pathstr, sprintf('sub-%s_T1w.nii%s', participantNumber, ext));
                %重命名
                movefile(old_name, new_name);
            elseif any(strcmp(parts, 'WIP')) && any(strcmp(parts, 'JQ')) && any(strcmp(parts, '301.nii'))
                new_name = fullfile(pathstr, sprintf('sub-%s_run-1_bold.nii%s', participantNumber, ext));
                movefile(old_name, new_name);
            elseif any(strcmp(parts, 'WIP')) && any(strcmp(parts, 'JQ')) && any(strcmp(parts, '401.nii'))
                new_name = fullfile(pathstr, sprintf('sub-%s_run-2_bold.nii%s', participantNumber, ext));
                movefile(old_name, new_name);
                
            elseif any(strcmp(parts, 'VWIP')) && any(strcmp(parts, 'T1W'))
                file_to_delete = fullfile(old_name);
                    if exist(file_to_delete, 'file')  % 检查文件是否存在
                        delete(file_to_delete);  % 删除文件
                        fprintf('Deleted: %s\n', file_to_delete);
                    end
            elseif any(strcmp(parts, 'WIP')) && any(strcmp(parts, 'Survey')) && any(strcmp(parts, '32ch'))
                file_to_delete = fullfile(old_name);
                    if exist(file_to_delete, 'file')  % 检查文件是否存在
                        delete(file_to_delete);  % 删除文件
                        fprintf('Deleted: %s\n', file_to_delete);
                    end        
            else
                continue; % 跳过不符合命名规则的文件
            end
      end
  % 获取生成的.json文件
    json_files = dir(fullfile(sub_folders, '*.json'));
      for j = 1:numel(json_files)
            old_name = fullfile(sub_folders, json_files(j).name);
            % 解析文件名和扩展名
            [pathstr, name, ext] = fileparts(old_name);
            % 解析文件名中的信息
            parts = strsplit(name, '_');
            % 定义新文件名
            if any(strcmp(parts, 'WIP')) && any(strcmp(parts, 'T1W'))
                new_name = fullfile(pathstr, sprintf('sub-%s_T1w.nii%s', participantNumber, ext));
                %重命名
                movefile(old_name, new_name);
            elseif any(strcmp(parts, 'WIP')) && any(strcmp(parts, 'JQ')) && any(strcmp(parts, '301'))
                new_name = fullfile(pathstr, sprintf('sub-%s_run-1_bold.nii%s', participantNumber, ext));
                movefile(old_name, new_name);
            elseif any(strcmp(parts, 'WIP')) && any(strcmp(parts, 'JQ')) && any(strcmp(parts, '401'))
                new_name = fullfile(pathstr, sprintf('sub-%s_run-2_bold.nii%s', participantNumber, ext));
                movefile(old_name, new_name);
                
            elseif any(strcmp(parts, 'VWIP')) && any(strcmp(parts, 'T1W'))
                file_to_delete = fullfile(old_name);
                    if exist(file_to_delete, 'file')  % 检查文件是否存在
                        delete(file_to_delete);  % 删除文件
                        fprintf('Deleted: %s\n', file_to_delete);
                    end
            elseif any(strcmp(parts, 'WIP')) && any(strcmp(parts, 'Survey')) && any(strcmp(parts, '32ch'))
                file_to_delete = fullfile(old_name);
                    if exist(file_to_delete, 'file')  % 检查文件是否存在
                        delete(file_to_delete);  % 删除文件
                        fprintf('Deleted: %s\n', file_to_delete);
                    end        
            else
                continue; % 跳过不符合命名规则的文件
            end
      end   
end
      
%% 储存到anat和func文件

% 遍历所有被试文件夹
for i = 1:length(fMRI_data)-2
    subID = sprintf('sub-%03d', i);
    subDir = fullfile(fMRI_out, subID);
    
    % 确保anat和func文件夹存在
    anatDir = fullfile(subDir, 'anat');
    funcDir = fullfile(subDir, 'func');
    if ~exist(anatDir, 'dir'); mkdir(anatDir); end
    if ~exist(funcDir, 'dir'); mkdir(funcDir); end
    
    
    % 获取所有.gz和.json文件
    files = dir(fullfile(subDir, '*.*'));
    
    for file = files'
        fileName = file.name;
        filePath = fullfile(subDir, fileName);
        
        % 移动 T1w 相关文件到 anat 目录
        if contains(fileName, 'T1w')
            movefile(filePath, anatDir);
        elseif contains(fileName, 'run')
            movefile(filePath, funcDir);
        end
    end
end

disp('文件分类和重命名完成！');

%% 手动建立任务介绍文件
%在每一个func中对应任务建立一个.tsv文件。按照onset duration trial_type……描述数据，具体见bids手册



