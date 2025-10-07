clear;
clc;
masks = {'Reslice_Lrostral_Hippo.nii'};
%roi_path='H:\metaphor\LSS\words_with_juzi\run4_with_run7';
roi_path='H:\metaphor\mask';
%%

msk = masks{1};

mask = fmri_mask_image(fullfile(roi_path,msk));

mydir = 'H:\metaphor\LSS\words_with_juzi\run4_with_run7';
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

load H:\metaphor\events_memory\selfreportmemory1050.mat

[a,b]=corr(memory_yy,yy)
[a,b]=corr(memory_kj,kj)







