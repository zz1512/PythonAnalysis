
%%get 
clear
clc
%mask = fmri_mask_image('H:\metaphor\gppi\lHPC_seed\output_run3\second\yy_kj\sphere_8--46_38_-10.nii');
mask = fmri_mask_image('H:\metaphor\mask\Hippocampus_L.nii');

mydir = 'H:\metaphor\gppi\LIFG_mask\output_run3';

image_names = filenames(fullfile(mydir, 'con_PPI_yy-kj_sub-*nii'), 'absolute'); % 

image_obj = fmri_data(image_names); 

brain = extract_roi_averages(image_obj, mask);

brain_dat=brain.dat;
indices_to_remove = [12, 23];
% Remove elements at specified indices
brain_dat(indices_to_remove) = [];

load H:\metaphor\events_memory\selfreportmemory34.mat

beh = memory_yy-memory_kj;

[h,p]= corr(brain_dat,beh)



