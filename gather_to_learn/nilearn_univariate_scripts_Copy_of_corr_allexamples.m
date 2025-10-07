
%%get 
clear
clc
mask = fmri_mask_image('H:\metaphor\nilearn_univariate\scripts\sphere_8--44_8_30.nii');
%mask = fmri_mask_image('H:\metaphor\mask\IFG_L.nii');
mydir = 'H:\metaphor\nilearn_univariate\data';
%%
image_names = filenames(fullfile(mydir, '*run-4_yy_juzi_zmap*nii'), 'absolute'); % 
image_obj = fmri_data(image_names); 
brain = extract_roi_averages(image_obj, mask);
brain_dat=brain.dat;
%%
image_names1 = filenames(fullfile(mydir, '*run-4_kj_juzi_zmap*nii'), 'absolute'); % 
image_obj1 = fmri_data(image_names1); 
brain1 = extract_roi_averages(image_obj1, mask);
brain_dat1=brain1.dat;
%%
b = brain_dat-brain_dat1;

indices_to_remove = [12, 23];
% Remove elements at specified indices
b(indices_to_remove) = [];

load H:\metaphor\events_memory\selfreportmemory1050.mat

beh = memory_yy-memory_kj;
%beh = memory_yy;
[h,p]= corr(b,beh)



