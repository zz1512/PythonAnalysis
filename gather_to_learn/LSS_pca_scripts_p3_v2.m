clear;clc;
subjects = {'sub-01';'sub-02';'sub-03';'sub-04';'sub-05';'sub-06';'sub-07';'sub-08';'sub-09';'sub-10';'sub-11';'sub-13';'sub-14';'sub-15';'sub-16';'sub-17';'sub-18';'sub-19';'sub-20';'sub-21';'sub-22';'sub-24';'sub-25';'sub-26';'sub-27';'sub-28'};
masks = {'Hippocampus_L.nii'};% mask    前幾個pc kj可以解釋更多主成分，説明kj維度低
study_path='H:\metaphor\LSS\rsa\pattern_similarity_change\mvpa_pattern';
roi_path = 'H:\metaphor\mask'; 
n_subjects=numel(subjects);
n_masks=numel(masks);
msk = masks{1};
alldata = [];
counterp = 0;

counter=0;
counter1=0;
pcpc=5;
for s = 1:length(subjects) 
    %% 指标的初始值
   % RD_eig_lsc=0;
    total_explained_yy=0;
    RD_var_yy=0;
 %   myexplained_lsc=0;
    %%
  %  RD_eig_hsc=0;
    total_explained_kj=0;
    RD_var_kj=0;
   % myexplained_hsc=0;
    %%
    sub = subjects{s};
    sub_path=fullfile(study_path,sub);
    mask_fn=fullfile(roi_path,msk);
    %%  HSC组，计算3个指标
    S_kj_s = fullfile(sub_path,'run7_kj_start_word.nii');
    S_kj_s_data = cosmo_fmri_dataset(S_kj_s,'mask',mask_fn);
    
    S_kj_e = fullfile(sub_path,'run7_kj_end_word.nii');
    S_kj_e_data = cosmo_fmri_dataset(S_kj_e,'mask',mask_fn);
    
    kjs=(S_kj_s_data.samples + S_kj_e_data.samples)*0.5;
   
    kj_s_sample = kjs'; 
    
    [coeff,score,latent,tsquared,explained,mu]=pca(kj_s_sample);
    
%     for m = 1:length(latent)
%         if latent(m)>1
%             RD_eig_hsc = RD_eig_hsc+1;
%             myexplained_hsc=myexplained_hsc+explained(m);
%         end
%     end

    for n = 1:pcpc
        total_explained_kj=total_explained_kj+explained(n);
        
    end
    
    RD_var_kj = n;
    
%    RD_eff_hsc=100*RD_eig_hsc/myexplained_hsc;
    
    counter=counter+1;
  %  RDeig_hsc(counter,:)=RD_eig_hsc;
    kjj(counter,:)=total_explained_kj;
 %   RDeff_hsc(counter,:)=RD_eff_hsc;
    %% LSC组
    S_yy_s = fullfile(sub_path,'run7_yy_start_word.nii');
    S_yy_s_data = cosmo_fmri_dataset(S_yy_s,'mask',mask_fn);
    
    S_yy_e = fullfile(sub_path,'run7_yy_end_word.nii');
    S_yy_e_data = cosmo_fmri_dataset(S_yy_e,'mask',mask_fn);
    
    yys = (S_yy_s_data.samples + S_yy_e_data.samples)*0.5;
    yy_s_sample = yys'; 
    
    [coeff,score,latent,tsquared,explained,mu]=pca(yy_s_sample);
%     for w = 1:length(latent)
%         if latent(w)>1
%             RD_eig_lsc = RD_eig_lsc+1;
%             myexplained_lsc=myexplained_lsc+explained(w);
%         end
%     end
    
    for j = 1:pcpc
        total_explained_yy=total_explained_yy+explained(j);
    end
     RD_var_yy = j;
   %  RD_eff_lsc=100*RD_eig_lsc/myexplained_lsc;
    counter1=counter1+1;
 %   RDeig_lsc(counter1,:)=RD_eig_lsc;
    yyy(counter1,:)=total_explained_yy;
 %   RDeff_lsc(counter1,:)=RD_eff_lsc;
end
% 
[h,p,ci,stats]=ttest(kjj,yyy)
counterp = counterp + 1;
allp(counterp,:) = p;
alldata = [alldata,kjj,yyy];


