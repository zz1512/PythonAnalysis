clear;
clc;
masks = {'Hippocampus_R.nii';'PCC_R.nii';'sphere_20--50_16_12.nii'};
roi_path='H:\metaphor\mask';
counter = 1;
% right pcg right pcc left ifg
%%
load H:\metaphor\events_memory\selfreportmemory34.mat
beh = memory_yy-memory_kj;
allp = [];
allr = [];
%%:
for n = 1:length(masks)

msk = masks{n};

mask = fmri_mask_image(fullfile(roi_path,msk));

mydir = 'H:\metaphor\nilearn_univariate\data';
%%
image_names = filenames(fullfile(mydir, '*run7_yy-kj*nii'), 'absolute');  % NA组con图  包括全部的例子     %剔除sub06  sub19

image_obj = fmri_data(image_names); 

beta = extract_roi_averages(image_obj, mask);

data = beta.dat;

[r,p] = corr(data,beh);

allp = [allp;p];
allr = [allr;r];
alldata(:,counter) = data;
counter = counter +1;
end



%save yy-kj-selfreport.mat alldata beh






