%%yy PS between first time shouci and second time shouci
%%yy PS between first time weici and second time weici
%%jx ps 

clear;clc;
subjects = {'sub-01';'sub-02';'sub-03';'sub-04';'sub-05';'sub-06';'sub-07';'sub-08';'sub-09';'sub-10';'sub-11';'sub-12';'sub-13';'sub-14';'sub-15';'sub-16';'sub-17';'sub-18';'sub-19';'sub-20';'sub-21';'sub-22';'sub-23';'sub-24';'sub-25';'sub-26';'sub-27';'sub-28'};
masks = {'IFG_L.nii'};
s_path = 'H:\metaphor\First_level\single_trial_v1\pattern';
roi_path = 'H:\metaphor\mask'; 
msk1 = masks{1};
counter = 0;

for s = 1:length(subjects)
    sub = subjects{s};
    sub_path=fullfile(s_path,sub);
    mask_fn=fullfile(roi_path,msk1);

    %% 
    data_f=fullfile(sub_path,'F_yy_sword.nii');     %%%%%%%
    ds_f=cosmo_fmri_dataset(data_f,'mask',mask_fn);
    [number,~]=size(ds_f.samples);
    %% 
    data_s=fullfile(sub_path,'S_yy_sword.nii');     %%%%%%
    ds_s=cosmo_fmri_dataset(data_s,'mask',mask_fn);
    
    %% hscµÄr
     allr = 0;
     for x = 1:number
          r = corr(ds_f.samples(x,:)',ds_s.samples(x,:)');
          allr = allr + r;
     end
    
    counter = counter+1;
    my_r_yy_S(counter,:) = allr/number;  
    
end

counter = 0;

for s = 1:length(subjects)
    sub = subjects{s};
    sub_path=fullfile(s_path,sub);
    mask_fn=fullfile(roi_path,msk1);

    %% 
    data_f=fullfile(sub_path,'F_yy_eword.nii');     %%%%%%%
    ds_f=cosmo_fmri_dataset(data_f,'mask',mask_fn);
    [number,~]=size(ds_f.samples);
    %% 
    data_s=fullfile(sub_path,'S_yy_eword.nii');     %%%%%%
    ds_s=cosmo_fmri_dataset(data_s,'mask',mask_fn);
    
    %% hscµÄr
     allr = 0;
     for x = 1:number
          r = corr(ds_f.samples(x,:)',ds_s.samples(x,:)');
          allr = allr + r;
     end
    
    counter = counter+1;
    my_r_yy_E(counter,:) = allr/number;  
    
end


counter = 0;

for s = 1:length(subjects)
    sub = subjects{s};
    sub_path=fullfile(s_path,sub);
    mask_fn=fullfile(roi_path,msk1);

    %% 
    data_f=fullfile(sub_path,'F_jx_word.nii');     %%%%%%%
    ds_f=cosmo_fmri_dataset(data_f,'mask',mask_fn);
    [number,~]=size(ds_f.samples);
    %% 
    data_s=fullfile(sub_path,'S_jx_word.nii');     %%%%%%
    ds_s=cosmo_fmri_dataset(data_s,'mask',mask_fn);
    
    %% hscµÄr
     allr = 0;
     for x = 1:number
          r = corr(ds_f.samples(x,:)',ds_s.samples(x,:)');
          allr = allr + r;
     end
    
    counter = counter+1;
    my_r_jx(counter,:) = allr/number;  
    
end


mean(my_r_yy_S)
mean(my_r_yy_E)
mean(my_r_jx)

[h,p,ci,stats] = ttest(my_r_yy_E,my_r_yy_S)

% £É£Æ£Ç¡¡



