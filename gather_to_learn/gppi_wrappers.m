% gPPI Wrapper Script Template 

% Specify paths to relevant directories.    

analysis_dir = 'H:\metaphor\gppi\univariate_ppi_run3';  % where ever we are pulling from
gPPI_dir     = 'H:\metaphor\gppi\output_run3_reverse_precu';    % where ever we are outputting to

% Specify all subjects to run
subjects = { 'sub-01' 'sub-02' 'sub-03' 'sub-04' 'sub-05' 'sub-06' 'sub-07' 'sub-08' 'sub-09' 'sub-10' 'sub-11' 'sub-12' 'sub-13' 'sub-14' 'sub-15' 'sub-16' 'sub-17' 'sub-18' 'sub-19' 'sub-20' 'sub-21' 'sub-22' 'sub-23' 'sub-24' 'sub-25' 'sub-26' 'sub-27' 'sub-28'};

for s=1:length(subjects) % for each subject...
    
    if isdir([gPPI_dir filesep subjects{s}])    % if this subject's gPPI directory already exists, skip this part
    else                                        % if the subject's gPPI directory does NOT exist
        mkdir([gPPI_dir filesep subjects{s}])   % create new gPPI specific directory
    end
    cd(fullfile(gPPI_dir, subjects{s}))
    
    parameters(subjects{s}, gPPI_dir, analysis_dir) % run the gPPI analysis
end

function parameters(subject,gPPI_dir,analysis_dir)
% parameters    function designed to work with wrapper script.
%
% 
%
% See also: wrapper
%% Setting the gPPI Parameters
%%% For more details on the parameters below and what they mean, go to the
%%% gPPI website and download the guide: http://www.nitrc.org/projects/gppi
first_level_dir = fullfile(analysis_dir, subject); 
cd(fullfile(gPPI_dir, subject))
P.subject       = subject; % A string with the subjects id
P.directory     = first_level_dir; % path to the first level GLM directory
P.VOI           = 'H:\metaphor\gppi\output_run3\second_lhip\yy-kj\ClusterMask_spmT_0001_x=2_y=-50_z=52_130voxels.nii'; % path to the ROI image, created above
P.Region        = 'precu'; % string, basename of output folder
P.Estimate      = 1; % Yes, estimate this gPPI model
P.contrast      = 0; % contrast to adjust for. Default is zero for no adjustment
P.extract       = 'eig'; % method for ROI extraction. Default is eigenvariate
P.Tasks         = {'0' 'yy' 'kj'}; % Specify the tasks for this analysis. Think of these as trial types. Zero means "does not have to occur in every session"
P.Weights       = []; % Weights for each task. If you want to weight one more than another. Default is not to weight when left blank
P.maskdir       = fullfile(gPPI_dir, subject); % Where should we save the masks?
P.equalroi      = 1; % When 1, All ROI's must be of equal size. When 0, all ROIs do not have to be of equal size
P.FLmask        = 0; % restrict the ROI's by the first level mask. This is useful when ROI is close to the edge of the brain
P.analysis      = 'psy'; % for "psychophysiological interaction"
P.method        = 'cond'; % "cond" for gPPI and "trad" for traditional PPI
P.CompContrasts = 1; % 1 to estimate contrasts
P.Weighted      = 0; % Weight tasks by number of trials. Default is 0 for do not weight
P.outdir        = fullfile(gPPI_dir, subject); % Output directory
P.ConcatR       = 1; % Tells gPPI toolbox to concatenate runs

P.Contrasts(1).left      = {'yy'}; % left side or positive side of contrast
P.Contrasts(1).right     = {'kj'}; % right side or negative side of contrast
P.Contrasts(1).STAT      = 'T'; % T contrast
P.Contrasts(1).Weighted  = 0; % Wieghting contrasts by trials. Deafult is 0 for do not weight
P.Contrasts(1).MinEvents = 1; % min number of event need to compute this contrast
P.Contrasts(1).name      = 'yy-kj'; % Name of this contrast
  %prefix to the task name (optional), can be used to select each run 

P.Contrasts(2).left      = {'kj'}; % left side or positive side of contrast
P.Contrasts(2).right     = {'yy'}; % right side or negative side of contrast
P.Contrasts(2).STAT      = 'T'; % T contrast
P.Contrasts(2).Weighted  = 0; % Wieghting contrasts by trials
P.Contrasts(2).MinEvents = 1; % min number of event need to compute this contrast
P.Contrasts(2).name      = 'kj-yy'; % Name of this contrast


%%% Below are parameters for gPPI. All set to zero for do not use. See website
%%% for more details on what they do.
P.FSFAST           = 0;
P.peerservevarcorr = 0;
P.wb               = 0;
P.zipfiles         = 0;
P.rWLS             = 0;

%% Actually Run PPI
PPPI(P)
end