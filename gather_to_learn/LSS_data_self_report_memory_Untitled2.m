% 定义源目录和目标目录的根路径
source_root = 'H:\metaphor\LSS\data_self_report_memory';
target_root = 'H:\metaphor\LSS\data_self_report_memory';

% 获取所有子目录的列表
sub_folders = dir(fullfile(source_root, 'sub-*'));

% 遍历每个子目录
for sub_idx = 1:length(sub_folders)
    sub_folder = sub_folders(sub_idx);
    sub_path = fullfile(source_root, sub_folder.name);
    
    % 获取当前子目录下的所有 run 子目录
    run_folders = dir(fullfile(sub_path, 'run-*'));
    
    % run 子目录
    for run_idx = 1:2
        run_folder = run_folders(run_idx);
        run_path = fullfile(sub_path, run_folder.name);
        outpath =fullfile(sub_path, 'first_words_view');
         if ~exist(outpath)
                mkdir(outpath);
            end
        % 获取当前 run 子目录下的所有nii文件
        files = dir(fullfile(run_path, '*.nii'));
        
        % 遍历每个文件，将其移动到相应的子目录下
        for file_idx = 1:length(files)
            file = files(file_idx);
            file_path = fullfile(run_path, file.name);
            target_path = fullfile(outpath, file.name);
            
            % 移动文件
           
            movefile(file_path, target_path);
        end
        
        % 删除空的 run 子目录
        rmdir(run_path);
    end
    
    for run_idx = 5:6
        run_folder = run_folders(run_idx);
        run_path = fullfile(sub_path, run_folder.name);
        outpath =fullfile(sub_path, 'second_words_view');
         if ~exist(outpath)
                mkdir(outpath);
            end
        % 获取当前 run 子目录下的所有nii文件
        files = dir(fullfile(run_path, '*.nii'));
        
        % 遍历每个文件，将其移动到相应的子目录下
        for file_idx = 1:length(files)
            file = files(file_idx);
            file_path = fullfile(run_path, file.name);
            target_path = fullfile(outpath, file.name);
           
            % 移动文件
            movefile(file_path, target_path);
        end
        
        % 删除空的 run 子目录
        rmdir(run_path);
    end
    
    
    
end
