% pattern similarity change
clear;clc;
subjects = {'sub-01';'sub-02';'sub-03';'sub-04';'sub-05';'sub-06';'sub-07';'sub-08';'sub-09';'sub-10';'sub-11';'sub-13';'sub-14';'sub-15';'sub-16';'sub-17';'sub-18';'sub-19';'sub-20';'sub-21';'sub-22';'sub-24';'sub-25';'sub-26';'sub-27';'sub-28'};
masks = {'parahippocampal_gyrus_L.nii'};
s_path = 'H:\metaphor\LSS\rsa\trial_level_second_view_pair_simlarity_corr_with_memory\mvpa_pattern_sortnat';
msk1 = masks{1};
counter = 0;
roi_path = 'H:\metaphor\mask';
onset_dir = 'H:\metaphor\events_memory'; 
yy_mem_all=[];
ps_all = [];
for s = 1:length(subjects)
    sub = subjects{s};
    sub_path=fullfile(s_path,sub);
    mask_fn=fullfile(roi_path,msk1);
%%
    sub_onset_path=fullfile(onset_dir,sub);
    run_con_all = dir([sub_onset_path,filesep,'sub*.txt']);
    run7_con=readtable([sub_onset_path,filesep,run_con_all(7).name]);
    run7_con=sortrows(run7_con,4);    %%%这个很关键
    yy_mem = run7_con.new_variable(string(run7_con.trial_type) == 'yyew');
    yy_pic = run7_con.pic_num(string(run7_con.trial_type) == 'yyw');
    yy_mem_all = [yy_mem_all;yy_mem];
%%
    data_shouci=fullfile(sub_path,'S_yy_start_word.nii');     %%%%%%%
    ds_shouci=cosmo_fmri_dataset(data_shouci,'mask',mask_fn);
    [number,~]=size(ds_shouci.samples);
    % 
    data_weici=fullfile(sub_path,'S_yy_end_word.nii');     %%%%%%
    ds_weici=cosmo_fmri_dataset(data_weici,'mask',mask_fn);
    % hsc的r
    
     for x = 1:number
          r = corr(ds_shouci.samples(x,:)',ds_weici.samples(x,:)');
          ps_all=[ps_all;r];
     end
    
end
data = [ps_all,yy_mem_all];
[h,p] = corr(ps_all,yy_mem_all)

zu1 = ps_all(yy_mem_all==1);
zu2 = ps_all(yy_mem_all<1);

[h1,p1,ci1,stats1] = ttest2(zu1,zu2)


