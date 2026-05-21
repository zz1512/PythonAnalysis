
%%get 
clear
clc

mydir = 'H:\metaphor\gppi\lIFG_seed\output_run3';

image_names = filenames(fullfile(mydir, 'con_PPI_yy-kj_sub-*nii'), 'absolute'); % 
indices_to_remove = [12,23];
% Remove elements at specified indices
image_names(indices_to_remove) = [];
image_obj = fmri_data(image_names); 

mask = 'H:\metaphor\LSS\gm.nii';    %

load H:\metaphor\events_memory\selfreportmemory34.mat

beh = memory_yy-memory_kj;

maskdat = fmri_data(mask, 'noverbose');
%%
image_obj = apply_mask(image_obj, maskdat);
image_obj.X = beh;
%%
out = regress(image_obj,'robust');

t = threshold(out.t, .05, 'fdr','k',3);
% t = select_one_image(t, 1);
% t.fullpath = fullfile(pwd, 'fdr05.nii');
% write(t,'overwrite')
%%
%t = threshold(out.t, .001, 'unc','k',10);
t = select_one_image(t, 1);
t.fullpath = fullfile(pwd, 'fdr.nii');
write(t,'overwrite')


% o2 = montage(t, 'trans', 'full');
% o2 = title_montage(o2, 2, '0.005 unc');
% snapnow
r = region(t);     
table(r);   



