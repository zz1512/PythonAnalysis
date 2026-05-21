mydir = 'H:\metaphor\LSS\words_with_juzi\run4_with_run7';
image_names = filenames(fullfile(mydir, 'searchlight_yy*nii'), 'absolute'); % NA组con图  包括全部的例子     %剔除sub06  sub19
image_obj = fmri_data(image_names); 
%%
load H:\metaphor\events_memory\selfreportmemory1050.mat

%%
mask = 'H:\metaphor\LSS\gm.nii';    
maskdat = fmri_data(mask, 'noverbose');
%%
image_obj = apply_mask(image_obj, maskdat);
%%
image_obj.X = memory_yy;
%%
out = regress(image_obj,'robust');
%t = threshold(out.t, .05, 'fdr','k',3);
% t = select_one_image(t, 1);
% t.fullpath = fullfile(pwd, 'fdr05.nii');
% write(t,'overwrite')
%%
t = threshold(out.t, .001, 'unc','k',10);
t = select_one_image(t, 1);
t.fullpath = fullfile(pwd, 'p001k10.nii');
write(t,'overwrite')


% o2 = montage(t, 'trans', 'full');
% o2 = title_montage(o2, 2, '0.005 unc');
% snapnow
r = region(t);     
table(r);   

%Make a montage showing each significant region
%montage(r, 'colormap', 'regioncenters');

% o2 = canlab_results_fmridisplay([], 'multirow', 2);
% o2 = multi_threshold(t, o2, 'thresh', [.01 .05], 'sizethresh', [10 10], 'wh_montages', 1:2);
% o2 = title_montage(o2, 2, 'Behavioral predictor: Reappraisal success (p < .001 one-tailed and showing extent');
% 
% metaimg = 'J:\Incubation\masks\Precuneus_LR.nii';
% 
% r = region(metaimg);
% 
% % Add these to the bottom montages
% o2 = addblobs(o2, r, 'maxcolor', [1 0 0], 'mincolor', [.7 .2 .5], 'wh_montages', 3:4);
% 
% o2 = title_montage(o2, 4, 'Neurosynth mask: Emotion regulation');

