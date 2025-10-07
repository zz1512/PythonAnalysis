
%%get 
clear
clc

%mask = fmri_mask_image('H:\GJXX_2_reanalysis\NA_regression\scripts\sphere_4-32_-12_-16.nii'); %parahippocampal_gyrus_L
mask = fmri_mask_image('H:\GJXX_2_reanalysis\NA_regression\scripts\sphere_6-32_-12_-16.nii');

mydir = 'H:\GJXX_2_reanalysis\NA_regression\rd\rsa';

image_names = filenames(fullfile(mydir, 'rdvar*nii'), 'absolute'); % NA组con图  包括全部的例子     %剔除sub06  sub19

image_obj = fmri_data(image_names); 

beta_hsc = extract_roi_averages(image_obj, mask);

X=beta_hsc.dat;
%X=rankdata(X);

beh = importdata('M.txt');% NA组novelty分数
Y = beh(:, 1);  
%Y=rankdata(Y);
[rho,pval] = corr(X,Y)
thedata=[X,Y];




