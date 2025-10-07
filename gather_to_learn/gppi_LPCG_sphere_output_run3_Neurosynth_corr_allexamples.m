
%%get 
clear
clc
mask = fmri_mask_image('H:\metaphor\mask\Hippocampus_L.nii');    % good

%mask = fmri_mask_image('H:\metaphor\gppi\LIFG_mask\sphere_10--46_38_-10.nii');     %ﬂ@ÇÄø…“‘sig      %

mydir = 'H:\metaphor\gppi\LPCG_sphere\output_run3_Neurosynth';

image_names = filenames(fullfile(mydir, 'con_PPI_yy-kj_sub-*nii'), 'absolute'); % 

image_obj = fmri_data(image_names); 

brain = extract_roi_averages(image_obj, mask);

brain_dat=brain.dat;

indices_to_remove = [12, 23];

brain_dat(indices_to_remove) = [];

load H:\metaphor\events_memory\selfreportmemory34.mat

beh = memory_yy-memory_kj;

[h,p]= corr(brain_dat,beh)



