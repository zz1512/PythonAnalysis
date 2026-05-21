%%
clear;
clc;
mask = fmri_mask_image('H:\metaphor\mask\Lrostral_Hippo.nii');

mydir = 'H:\metaphor\First_level\univariate_v1';

image_names = filenames(fullfile(mydir, 'con_0013*nii'), 'absolute'); 

image_obj = fmri_data(image_names); 

beta_yyw1 = extract_roi_averages(image_obj, mask);

beta_yyw1=beta_yyw1.dat;
%%
image_names = filenames(fullfile(mydir, 'con_0014*nii'), 'absolute'); 

image_obj = fmri_data(image_names); 

beta_yyw5 = extract_roi_averages(image_obj, mask);

beta_yyw5=beta_yyw5.dat;

%%
mean(beta_yyw1)
mean(beta_yyw5)

[h,p,ci,stats] = ttest(beta_yyw1,beta_yyw5)
