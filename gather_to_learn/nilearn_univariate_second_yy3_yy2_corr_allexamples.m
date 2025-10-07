
%%get 
clear
clc
%mask = fmri_mask_image('H:\metaphor\mask\parahippocampal_gyrus_L.nii'); %parahippocampal_gyrus_L
mask = fmri_mask_image('H:\metaphor\nilearn_univariate\second\yy2_yy1\sphere_8--32_-42_-10.nii'); 
data_dir = 'H:\metaphor\nilearn_univariate\data';
%%
s={'01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28'};
%%
yy1 = [];
for subj= 1:numel(s)
    i = s(subj);
    dir1 = strcat(data_dir,filesep,strcat('sub-',string(i)),filesep,'first_words_view\yy.nii');  
    yy1 = [yy1;dir1];
end
%%

yy2 = [];
for subj= 1:numel(s)
    i = s(subj);
    dir2 = strcat(data_dir,filesep,strcat('sub-',string(i)),filesep,'second_words_view\yy.nii');  
    yy2 = [yy2;dir2];
end
%%
image_obj1 = fmri_data(cellstr(yy1)); 
brain1 = extract_roi_averages(image_obj1, mask);
%%
image_obj2 = fmri_data(cellstr(yy2)); 
brain2 = extract_roi_averages(image_obj2, mask);

%%
brain_dat2=brain2.dat;
brain_dat1=brain1.dat;
%%
[a,b,c,d]=ttest(brain_dat1,brain_dat2)



