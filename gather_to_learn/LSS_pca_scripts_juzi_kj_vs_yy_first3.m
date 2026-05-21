
subjects = {'sub-01';'sub-02';'sub-03';'sub-04';'sub-05';'sub-06';'sub-07';'sub-08';'sub-09';'sub-10';'sub-11';'sub-12';'sub-13';'sub-14';'sub-15';'sub-16';'sub-17';'sub-18';'sub-19';'sub-20';'sub-21';'sub-22';'sub-23';'sub-24';'sub-25';'sub-26';'sub-27';'sub-28'};
masks = {'Hippocampus_L.nii'};% mask  
study_path='H:\metaphor\LSS\rsa\pattern_similarity_change\mvpa_pattern';
roi_path = 'H:\metaphor\mask'; 
n_subjects=numel(subjects);
n_masks=numel(masks);
msk = masks{1};
counter=0;
counter1=0;

for s = 1:length(subjects) 
    %% 
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
    F_yy_s = fullfile(sub_path,'F_kj_juzi.nii');
    F_yy_s_data = cosmo_fmri_dataset(F_yy_s,'mask',mask_fn);
    
    kj = F_yy_s_data.samples'; 
    
    
    [coeff,score,latent,tsquared,explained,mu]=pca(kj);
    
%     for m = 1:length(latent)
%         if latent(m)>1
%             RD_eig_hsc = RD_eig_hsc+1;
%             myexplained_hsc=myexplained_hsc+explained(m);
%         end
%     end
%     
    for n = 1:3
        total_explained_kj=total_explained_kj+explained(n);
        
    end
    
%    RD_eff_hsc=100*RD_eig_hsc/myexplained_hsc;
    
    counter=counter+1;
  %  RDeig_hsc(counter,:)=RD_eig_hsc;
    RDkj(counter,:)=total_explained_kj;
 %   RDeff_hsc(counter,:)=RD_eff_hsc;
    %% LSC组
    syyjuzi = fullfile(sub_path,'F_yy_juzi.nii');
    syyjuzi = cosmo_fmri_dataset(syyjuzi,'mask',mask_fn);
    yy = syyjuzi.samples'; 
    
    [coeff,score,latent,tsquared,explained,mu]=pca(yy);
%     for w = 1:length(latent)
%         if latent(w)>1
%             RD_eig_lsc = RD_eig_lsc+1;
%             myexplained_lsc=myexplained_lsc+explained(w);
%         end
%     end
    
    for j = 1:3
        total_explained_yy=total_explained_yy+explained(j);
        
    end
   
   %  RD_eff_lsc=100*RD_eig_lsc/myexplained_lsc;
    counter1 = counter1 + 1;
 %   RDeig_lsc(counter1,:)=RD_eig_lsc;
    RDyy(counter1,:) = total_explained_yy;
 %   RDeff_lsc(counter1,:)=RD_eff_lsc;
end
% 
[h,p,ci,stats] = ttest(RDkj,RDyy)

%save lIFG_juzi1_kj_vs_yy.mat allp alldata

