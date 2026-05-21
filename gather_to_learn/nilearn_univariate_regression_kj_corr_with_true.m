clear;
clc;
masks = {'Hippocampus_L.nii'; 'Hippocampus_R.nii';'Precuneus_L.nii';'Precuneus_R.nii';'PCC_L.nii';'PCC_R.nii';'IFG_L.nii';'IFG_R.nii'};
roi_path='H:\metaphor\mask';
counter = 1;
% right pcg right pcc left ifg
%%
load H:\metaphor\events_true_memory\true_memory_kj_no_sub11_12_23.mat
beh = memory_kj;
allp = [];
%%:
for n = 1:length(masks)

msk = masks{n};

mask = fmri_mask_image(fullfile(roi_path,msk));

mydir = 'H:\metaphor\nilearn_univariate\data';
%%
image_names = filenames(fullfile(mydir, '*run-7_kj_recall_zmap*nii'), 'absolute');  % NA组con图  包括全部的例子     %剔除sub06  sub19

indices_to_remove = 11;
% Remove elements at specified indices
image_names(indices_to_remove) = [];
image_obj = fmri_data(image_names); 

beta = extract_roi_averages(image_obj, mask);

data = beta.dat;

[h,p] = corr (data,beh);

allp = [allp;p];

alldata(:,counter) = data;
counter = counter +1;
end











