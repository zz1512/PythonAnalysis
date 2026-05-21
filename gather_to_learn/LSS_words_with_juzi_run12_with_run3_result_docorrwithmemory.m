clear;
clc;
masks = {'sphere_8-2_-42_48.nii'};
roi_path='H:\metaphor\LSS\words_with_juzi\run3_with_run7\result';
%%
msk = masks{1};
mask = fmri_mask_image(fullfile(roi_path,msk));
mydir = 'H:\metaphor\LSS\words_with_juzi\run3_with_run7';
%%
image_names_yy = filenames(fullfile(mydir, 'searchlight_yy*nii'), 'absolute'); % NA组con图  包括全部的例子     %剔除sub06  sub19
image_obj_yy = fmri_data(image_names_yy); 
beta_yy= extract_roi_averages(image_obj_yy, mask);
yy = beta_yy.dat;
%%
image_names_kj = filenames(fullfile(mydir, 'searchlight_kj*nii'), 'absolute'); % NA组con图  包括全部的例子     %剔除sub06  sub19
image_obj_kj = fmri_data(image_names_kj); 
beta_kj= extract_roi_averages(image_obj_kj, mask);
kj = beta_kj.dat;
%%

load H:\metaphor\events_true_memory\memory10_no_sub11.mat
memory_yy([11,22]) = [];
memory_kj([11,22]) = [];
yy([11]) = [];
kj([11]) = [];
brain = yy-kj;
beh = memory_yy-memory_kj;
[r,p]=corr(brain,beh)
[r,p]=corr(yy,memory_yy)
[r,p]=corr(kj,memory_kj)


