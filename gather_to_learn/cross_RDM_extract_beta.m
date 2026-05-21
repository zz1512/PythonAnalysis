
clear
clc
mask = fmri_mask_image('H:\metaphor\mask\angular_R.nii');
mydir = 'H:\metaphor\cross_RDM\run3_and_4\yy_and_kj';

yyimage_names = filenames(fullfile(mydir, 'searchlight_yy*nii'), 'absolute'); % NA组con图  包括全部的例子     %剔除sub06  sub19

image_objyy = fmri_data(yyimage_names); 

beta_yy = extract_roi_averages(image_objyy, mask);

yy=beta_yy.dat;
%%
kjimage_names = filenames(fullfile(mydir, 'searchlight_kj*nii'), 'absolute'); % NA组con图  包括全部的例子     %剔除sub06  sub19

image_objkj = fmri_data(kjimage_names); 

beta_kj = extract_roi_averages(image_objkj, mask);

kj=beta_kj.dat;
%%
[h,p,ci,stats] = ttest(yy,kj)
mean(yy) 
mean(kj)





