clear;clc;
data_path='J:\metaphor\out_fmri';
onset_path='J:\metaphor\events';
subjects_ids={'sub-01';'sub-02';'sub-03';'sub-04';'sub-05';'sub-06';'sub-07';'sub-08';'sub-09';'sub-10';'sub-11';'sub-12';'sub-13';'sub-14';'sub-15';'sub-16';'sub-17';'sub-18';'sub-19';'sub-20';'sub-21';'sub-22';'sub-23';'sub-24';'sub-25';'sub-26';'sub-27';'sub-28'};
nsubjects=numel(subjects_ids);
save_to='J:\metaphor\Pro_proc_data';

for i_subj=24:nsubjects
    subjects_id=subjects_ids{i_subj};
    sub_data_path=fullfile(data_path,subjects_id,'func');
    sub_onset_path=fullfile(onset_path,subjects_id);
    
    save_to_path=fullfile(save_to,subjects_id);
    
    gz_fn=dir([sub_data_path,'\*preproc_bold.nii.gz']);
    rp=dir([sub_data_path,'\*.tsv']);
    onset=dir([sub_onset_path,'\*.tsv']);
   
    for k=1:7
        gz_data=gz_fn(k).name;
        rp_data=rp(k).name;
        onset_data=onset(k).name;
        gz_data_fn=[sub_data_path,filesep,gz_data];
        rp_data_fn=[sub_data_path,filesep,rp_data];
        onset_data_fn=[sub_onset_path,filesep,onset_data];
        
        run=num2str(k);
        
        run_data_path=[save_to_path,filesep,'run',run];
         mkdir(run_data_path);
        save_onset_to=[save_to_path,filesep,'condition'];
        mkdir(save_onset_to);
       
        save_rp_to=[save_to_path,filesep,'multi_reg'];
         mkdir(save_rp_to);
         
        copyfile(gz_data_fn,run_data_path);  %  应该改成copyfile
        copyfile(onset_data_fn,save_onset_to);
        copyfile(rp_data_fn,save_rp_to);
        
    cd (run_data_path)
    gunzip *.gz
        
    end
end


