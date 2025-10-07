clear;
clc;
masks = {'Hippocampus_L.nii'; 'Hippocampus_R.nii';'Precuneus_L.nii';'Precuneus_R.nii';'IFG_L.nii';'IFG_R.nii'};
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

mydir = 'H:\metaphor\LSS\rsa\pattern_similarity_change\searchlight_compare_first_with_second';

%%
image_names = filenames(fullfile(mydir, 'YY_second_minus_first*nii'), 'absolute'); % NA组con图  包括全部的例子     %剔除sub06  sub19
indices_to_remove = [12,23];
% Remove elements at specified indices
image_names(indices_to_remove) = [];
image_obj = fmri_data(image_names); 

beta = extract_roi_averages(image_obj, mask);
data = beta.dat;
[r,p] = corr (data,beh,'type','Spearman');

allp = [allp;p];
allr = [allr;r];
alldata(:,counter) = data;
counter = counter +1;
end



%save yy-kj-selfreport.mat alldata beh






