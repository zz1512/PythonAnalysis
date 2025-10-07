mydir = 'H:\metaphor\LSS\rsa\pattern_similarity_change\searchlight_compare_first_with_second';
image_names = filenames(fullfile(mydir, '*second_yy-kj*nii'), 'absolute'); % NA组con图  包括全部的例子     %剔除sub06  sub19
indices_to_remove = [12,23];
% Remove elements at specified indices
image_names(indices_to_remove) = [];
image_obj = fmri_data(image_names); 
%%
% no 12 23
load H:\metaphor\events_memory\selfreportmemory34.mat
beh = memory_yy-memory_kj;
%%
mask = 'H:\metaphor\LSS\gm.nii';    %
maskdat = fmri_data(mask, 'noverbose');
%%
image_obj = apply_mask(image_obj, maskdat);
%%
image_obj.X = beh;
%%
out = regress(image_obj,'robust');
%t = threshold(out.t, .05, 'fdr','k',3);
% t = select_one_image(t, 1);
% t.fullpath = fullfile(pwd, 'fdr05.nii');
% write(t,'overwrite')
%%
t = threshold(out.t, .001, 'unc','k',20);
t = select_one_image(t, 1);
t.fullpath = fullfile(pwd, 'p001k20.nii');
write(t,'overwrite')


% o2 = montage(t, 'trans', 'full');
% o2 = title_montage(o2, 2, '0.005 unc');
% snapnow
r = region(t);     
table(r);   

% save r_p001k20.mat r
% save image_obj_p001k20.mat image_obj

%Make a montage showing each significant region
%montage(r, 'colormap', 'regioncenters');

