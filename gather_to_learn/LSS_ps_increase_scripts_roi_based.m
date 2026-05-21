
clear
clc
subjects = {'sub-01';'sub-02';'sub-03';'sub-04';'sub-05';'sub-06';'sub-07';'sub-08';'sub-09';'sub-10';'sub-11';'sub-12';'sub-13';'sub-14';'sub-15';'sub-16';'sub-17';'sub-18';'sub-19';'sub-20';'sub-21';'sub-22';'sub-23';'sub-24';'sub-25';'sub-26';'sub-27';'sub-28'};
masks = {'Hippocampus_L.nii'};
s_path = 'H:\metaphor\LSS\rsa\pattern_similarity_change\mvpa_pattern';
roi_path = 'H:\metaphor\mask'; 
msk1 = masks{1};
counter = 0;

for s=1:length(subjects)
    %%
    sub = subjects{s};
    sub_path=fullfile(s_path,sub);
    mask_fn=fullfile(roi_path,msk1);
    %
    %% yy
    
    Syy=fullfile(sub_path,'S_yy_juzi.nii');
    ds_Syy=cosmo_fmri_dataset(Syy,'mask',mask_fn);
    %%%
    Syystart=fullfile(sub_path,'S_yy_start_word.nii');
    ds_syystart=cosmo_fmri_dataset(Syystart,'mask',mask_fn);
 
    Syyend=fullfile(sub_path,'S_yy_end_word.nii');
    ds_syyend=cosmo_fmri_dataset(Syyend,'mask',mask_fn);
    %%%
    Fyystart=fullfile(sub_path,'F_yy_start_word.nii');
    ds_fyystart=cosmo_fmri_dataset(Fyystart,'mask',mask_fn);
 
    Fyyend=fullfile(sub_path,'F_yy_end_word.nii');
    ds_fyyend=cosmo_fmri_dataset(Fyyend,'mask',mask_fn);
    %%
    
    my_ds=cosmo_stack({ds_Syy,ds_syystart});
    my_ds=cosmo_stack({my_ds,ds_syyend});
    my_ds=cosmo_stack({my_ds,ds_fyystart});
    my_ds=cosmo_stack({my_ds,ds_fyyend});
    
    run4 = my_ds.samples(1:35,:);
    run5 = my_ds.samples(36:70,:);
    run6 = my_ds.samples(71:105,:);
    run1 = my_ds.samples(106:140,:);
    run2 = my_ds.samples(141:175,:);
    
    run56 = (run5+run6)/2;
    run12 = (run1+run2)/2;
    
    
    allr1 = 0;
     for x = 1:35
          rr = corr(run4(x,:)',run12(x,:)');
          allr1 = allr1+rr;
     end
    
      allr2 = 0;
     for x = 1:35
          rr = corr(run4(x,:)',run56(x,:)');
          allr2 = allr2+rr;
      end
   %%
   counter = counter+1;
   mydata(counter,:) =  (allr2-allr1)/35; 

end
[a,b,c,d] = ttest(mydata,0)
