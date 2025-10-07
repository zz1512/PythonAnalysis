clear;clc;
subjects = {'sub-01';'sub-02';'sub-03';'sub-04';'sub-05';'sub-06';'sub-07';'sub-08';'sub-09';'sub-10';'sub-11';'sub-12';'sub-13';'sub-14';'sub-15';'sub-16';'sub-17';'sub-18';'sub-19';'sub-20';'sub-21';'sub-22';'sub-23';'sub-24';'sub-25';'sub-26';'sub-27';'sub-28'};
masks = {'IFG_L.nii'};% mask  
study_path='H:\metaphor\LSS\rsa\pattern_similarity_change\mvpa_pattern';
roi_path = 'H:\metaphor\mask'; 
n_subjects=numel(subjects);
n_masks=numel(masks);
msk = masks{1};
alldata = [];
counterp = 0;

for kkk = 80:90
    
counter=0;
counter1=0;

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
    F_yy_s = fullfile(sub_path,'F_yy_start_word.nii');
    F_yy_s_data = cosmo_fmri_dataset(F_yy_s,'mask',mask_fn);
    
    F_yy_e = fullfile(sub_path,'F_yy_end_word.nii');
    F_yy_e_data = cosmo_fmri_dataset(F_yy_e,'mask',mask_fn);
    
    yy_f=(F_yy_s_data.samples +F_yy_e_data.samples)*0.5;
    yy_f_sample = yy_f'; 
    
    
    [coeff,score,latent,tsquared,explained,mu]=pca(yy_f_sample);
    
%     for m = 1:length(latent)
%         if latent(m)>1
%             RD_eig_hsc = RD_eig_hsc+1;
%             myexplained_hsc=myexplained_hsc+explained(m);
%         end
%     end
%     
    for n = 1:length(explained)
        total_explained_kj=total_explained_kj+explained(n);
        if total_explained_kj > kkk
            break
        end
    end
    
    RD_var_kj = n;
    
%    RD_eff_hsc=100*RD_eig_hsc/myexplained_hsc;
    
    counter=counter+1;
  %  RDeig_hsc(counter,:)=RD_eig_hsc;
    RDF(counter,:)=RD_var_kj;
 %   RDeff_hsc(counter,:)=RD_eff_hsc;
    %% LSC组
    syyjuzi = fullfile(sub_path,'F_yy_juzi.nii');
    syyjuzi = cosmo_fmri_dataset(syyjuzi,'mask',mask_fn);
    juzi = syyjuzi.samples'; 
    
    [coeff,score,latent,tsquared,explained,mu]=pca(juzi);
%     for w = 1:length(latent)
%         if latent(w)>1
%             RD_eig_lsc = RD_eig_lsc+1;
%             myexplained_lsc=myexplained_lsc+explained(w);
%         end
%     end
    
    for j = 1:length(explained)
        total_explained_yy=total_explained_yy+explained(j);
        if total_explained_yy > kkk
            break
        end
    end
     RD_var_yy = j;
   %  RD_eff_lsc=100*RD_eig_lsc/myexplained_lsc;
    counter1=counter1+1;
 %   RDeig_lsc(counter1,:)=RD_eig_lsc;
    RDjuzi(counter1,:)=RD_var_yy;
 %   RDeff_lsc(counter1,:)=RD_eff_lsc;
end
% 
[h,p,ci,stats]=ttest(RDF,RDjuzi)
counterp = counterp + 1;
allp(counterp,:) = p;
alldata = [alldata,RDF,RDjuzi];
end

save lIFG_yy1_vs_juzi1.mat allp alldata

