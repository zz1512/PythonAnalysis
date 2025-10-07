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
    %%
    sub = subjects{s};
    sub_path=fullfile(study_path,sub);
    mask_fn=fullfile(roi_path,msk);
    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    kj = fullfile(sub_path,'F_kj_juzi.nii');
    kj = cosmo_fmri_dataset(kj,'mask',mask_fn);
    
    kj = kj.samples'; 
    
    yy = fullfile(sub_path,'F_yy_juzi.nii');
    yy = cosmo_fmri_dataset(yy,'mask',mask_fn);
    yy = yy.samples'; 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    true_length = 30;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %从矩阵d_HSC中随机取出true_length列，然后做1000次降维，取均值
        allrd = 0;
        for x = 1:1000
            da_kj = kj(:,randperm(size(kj,2),true_length)); %randperm(n,k) returns a row vector containing k unique integers selected randomly from 1 to n.
            [coeff,score,latent,tsquared,explained,mu]=pca(da_kj);
                total_explained_hsc=0;
                 RD_var_hsc=0;
            for n = 1:length(explained)
                total_explained_hsc=total_explained_hsc+explained(n);
                if total_explained_hsc > kkk
                    break
                end
            end
            
            RD_var_hsc = n;
            allrd = allrd + RD_var_hsc;
            
        end
        
        counter=counter+1;
        RDkj(counter,:)=allrd/1000;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% LSC组

        %从矩阵d_HSC中随机取出true_length列，然后做1000次降维，取均值
        allrd = 0;
        for x = 1:1000
            da_yy = yy(:,randperm(size(yy,2),true_length)); %randperm(n,k) returns a row vector containing k unique integers selected randomly from 1 to n.
            [coeff,score,latent,tsquared,explained,mu]=pca(da_yy);
            total_explained_lsc=0;
             RD_var_lsc=0;
            for n = 1:length(explained)
                total_explained_lsc=total_explained_lsc+explained(n);
                if total_explained_lsc > kkk
                    break
                end
            end
            
            RD_var_lsc = n;
            allrd = allrd + RD_var_lsc;
            
        end
        
        counter1=counter1+1;
        RDyy(counter1,:)=allrd/1000;
end
[h,p,ci,stats]=ttest(RDkj,RDyy)
counterp = counterp + 1;
allp(counterp,:) = p;
alldata = [alldata,RDkj,RDyy];
end


