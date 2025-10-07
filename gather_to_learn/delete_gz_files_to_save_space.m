% 获取当前目录及全部各级别的子目录下的所有.gz文件
gzFiles = dir('**/*.gz');

% 循环处理所有找到的.gz文件
for i = 1:length(gzFiles)
    filePath = fullfile(gzFiles(i).folder, gzFiles(i).name);
    
    % 删除.gz文件
    delete(filePath);
    disp(['Deleted: ' filePath]);
end
