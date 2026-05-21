
%%get 
clear
clc
mask = fmri_mask_image('H:\metaphor\gppi\lHPC_seed\output_run3\second\yy_kj\sphere_10-6_-38_6.nii');   % sphere_10-2_-50_52good
%mask = fmri_mask_image('H:\metaphor\mask\MFG_R.nii');

mydir = 'H:\metaphor\gppi\lHPC_seed\output_run3';

image_names = filenames(fullfile(mydir, 'con_PPI_yy-kj_sub-*nii'), 'absolute'); % 

image_obj = fmri_data(image_names); 

brain = extract_roi_averages(image_obj, mask);

brain_dat=brain.dat;
indices_to_remove = [12, 23];
% Remove elements at specified indices
brain_dat(indices_to_remove) = [];

load H:\metaphor\events_memory\selfreportmemory1050.mat

beh = memory_yy-memory_kj;

[h,p]= corr(brain_dat,beh)



