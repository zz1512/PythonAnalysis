%% 
clear;clc;
subjects = {'sub-01';'sub-02';'sub-03';'sub-04';'sub-05';'sub-06';'sub-07';'sub-08';'sub-09';'sub-10';'sub-11';'sub-12';'sub-13';'sub-14';'sub-15';'sub-16';'sub-17';'sub-18';'sub-19';'sub-20';'sub-21';'sub-22';'sub-23';'sub-24';'sub-25';'sub-26';'sub-27';'sub-28'};
masks = {'Precuneus_R.nii'}; %Precuneus_R   %Hippocampus_R
s_path='H:\metaphor\LSS\rsa\pattern_similarity_change\mvpa_pattern';
roi_path='H:\metaphor\mask'; 
msk1 = masks{1};
counter = 0;

%% first yy 
for s = 1:length(subjects)
    sub = subjects{s};
    sub_path=fullfile(s_path,sub);
    mask_fn=fullfile(roi_path,msk1);
    %% 
    data_shouci=fullfile(sub_path,'F_kj_start_word.nii');     %%%%%%%
    ds_shouci=cosmo_fmri_dataset(data_shouci,'mask',mask_fn);
    [number,~]=size(ds_shouci.samples);
    %% 
    data_weici=fullfile(sub_path,'F_kj_end_word.nii');     %%%%%%
    ds_weici=cosmo_fmri_dataset(data_weici,'mask',mask_fn);
    
    %% hscµÄr
     allr = 0;
     for x = 1:number
          r = corr(ds_shouci.samples(x,:)',ds_weici.samples(x,:)');
          allr = allr + r;
     end
      
    counter = counter+1;
    my_r_first_kj(counter,:) = allr/number;  
    
end


%% second yy
counter = 0;
for s = 1:length(subjects)
    sub = subjects{s};
    sub_path=fullfile(s_path,sub);
    mask_fn=fullfile(roi_path,msk1);

    %% 
    data_shouci=fullfile(sub_path,'S_kj_start_word.nii');     %%%%%%%
    ds_shouci=cosmo_fmri_dataset(data_shouci,'mask',mask_fn);
    [number,~]=size(ds_shouci.samples);
    
    %% 
    data_weici=fullfile(sub_path,'S_kj_end_word.nii');     %%%%%%
    ds_weici=cosmo_fmri_dataset(data_weici,'mask',mask_fn);
    
    %% hscµÄr
     allr = 0;
     for x = 1:number
          r = corr(ds_shouci.samples(x,:)',ds_weici.samples(x,:)');
          allr = allr + r;
     end
      
    counter = counter+1;
    my_r_second_kj(counter,:) = allr/number;  
    
end

%%
%[h,p,ci,stats] = ttest(my_r_F,my_r_S)
% mean(my_r_F)
% mean(my_r_S)

save my_r_first_kj_rPCG.mat  my_r_first_kj
save my_r_second_kj_rPCG.mat my_r_second_kj
