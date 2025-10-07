clear; clc;
%%
data_dir = 'H:\metaphor\LSS\words_with_juzi\run3_with_run7';   % con map
s={'01','02','03','04','05','06','07','08','09','10','11','13','14','15','16','17','18','19','20','21','22','24','25','26','27','28'};

for subj= 1:numel(s)
spm_jobman('initcfg');
spm('defaults', 'fMRI');

i = s(subj);
dir1 = strcat(data_dir,filesep,strcat('sub-',string(i)),filesep,'searchlight_yy.nii');
dir2 = strcat(data_dir,filesep,strcat('sub-',string(i)),filesep,'searchlight_kj.nii');
dir = [dir1;dir2];

matlabbatch{1}.spm.util.imcalc.input = cellstr(dir);
matlabbatch{1}.spm.util.imcalc.output = 'yy-kj';
matlabbatch{1}.spm.util.imcalc.outdir = cellstr(strcat(data_dir,filesep,strcat('sub-',string(i))));
matlabbatch{1}.spm.util.imcalc.expression = 'i1-i2';
matlabbatch{1}.spm.util.imcalc.var = struct('name', {}, 'value', {});
matlabbatch{1}.spm.util.imcalc.options.dmtx = 0;
matlabbatch{1}.spm.util.imcalc.options.mask = 0;
matlabbatch{1}.spm.util.imcalc.options.interp = 1;
matlabbatch{1}.spm.util.imcalc.options.dtype = 4;

spm_jobman('run', matlabbatch);
end
