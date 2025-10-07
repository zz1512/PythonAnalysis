clear;
clc;
masks = {'Hippocampus_L.nii'; 'Hippocampus_R.nii';'Precuneus_L.nii';'Precuneus_R.nii';'IFG_L.nii';'IFG_R.nii'};
roi_path='H:\metaphor\mask';
% right pcg right pcc left ifg
allp = [];
allh = [];
%%
for n = 1:length(masks)

msk = masks{n};
mask = fmri_mask_image(fullfile(roi_path,msk));
mydir = 'H:\metaphor\LSS\words_with_juzi\self-report-mem\run3_with_run7';
%%
image_names = filenames(fullfile(mydir, 'searchlight_yy_1*nii'), 'absolute');  % NA组con图  包括全部的例子     %剔除sub06  sub19

image_obj = fmri_data(image_names); 
beta = extract_roi_averages(image_obj, mask);
reme = beta.dat;
%%
image_names = filenames(fullfile(mydir, 'searchlight_yy_0*nii'), 'absolute');  % NA组con图  包括全部的例子     %剔除sub06  sub19

image_obj = fmri_data(image_names); 
beta = extract_roi_averages(image_obj, mask);
forg = beta.dat;
%%
[h,p] = ttest(reme,forg)
allp = [allp;p];
allh = [allh;h];

end










