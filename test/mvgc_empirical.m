%% Parameters

if ~exist('momax',      'var'), momax       = 30;       end         % maximum model order
if ~exist('moregmode',  'var'), moregmode   = 'LWR';    end         % model order regression mode 
if ~exist('mosel',      'var'), mosel       = 'AIC';    end         % which model order to select
if ~exist('plotm',      'var'), plotm       = 0;        end         % for plotting on seperate figures
if ~exist('regmode',    'var'), regmode     = 'LWR';    end 
if ~exist('stat',       'var'), stat        = 'F';      end
if ~exist('alpha',      'var'), alpha       = 0.01;     end
if ~exist('mhtc',       'var'), mhtc        = 'FDRD';   end

%%%%%%%%%%%%%%% PREOPT

defvar('iseed',    0           ); % initialisation random seed (0 to use current rng state)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

defvar('nrunsp',   100         ); % pre-optimisation runs (restarts)
defvar('nitersp',  10000       ); % pre-optimisation iterations
defvar('sig0p',    1           ); % pre-optimisation (gradient descent) initial step size
defvar('gdlsp',    2           ); % gradient-descent "line search" parameters
defvar('gdtolp',   1e-10       ); % gradient descent convergence tolerance
defvar('histp',    true        ); % calculate optimisation history?
defvar('ppp',      false       ); % parallelise multiple runs?

%%%%%%%%%%%%%% OPTIMISATION

defvar('ctol',     1e-6        ); % hyperplane clustering tolerance

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

defvar('niterso',  10000       ); % pre-optimisation iterations
defvar('sig0o',    0.1         ); % optimisation (gradient descent) initial step size
defvar('gdlso',    2           ); % gradient-descent "line search" parameters
defvar('gdtolo',   1e-10       ); % gradient descent convergence tolerance
defvar('histo',    true        ); % calculate optimisation history?
defvar('ppo',      false       ); % parallelise multiple runs?



%% LOAD IN DATA
% Setting up the directory structure for saving files first

results_dir = '/mnt/data/bokim/restEEHHealthySubjects/AnesthesiaProjectEmergence/results/';
pwcgc_dir = [results_dir 'PWCGC/'];
ssdi_data_dir = [results_dir 'ssdiData/'];

% Create directories for results if they don't already exist
if exist([ssdi_data_dir]) == 0
    mkdir([ssdi_data_dir])
end


if exist([pwcgc_dir]) == 0
    mkdir([pwcgc_dir])
end

% Loading in data

source_data_dir = '/mnt/data/bokim/restEEGHealthySubjects/preprocessedData/sourceReconstructions/';
files = dir([source_data_dir '*.mat']);


%% MVGC

edgeWeightsMatrix = zeros(46, 46, length(files)); 


for fileNumber = 1:length(files)
    filename = files(fileNumber).name;
    folder = files(fileNumber).folder;
    
    
    tic;
    fprintf('..Starting on subject %s (%g / %g) \n', filename, fileNumber, length(files));
    
    load([folder filesep filename]);
    data = permute(source_ts, [2,3,1]);
    data = demean(data, true);

    % model order selection and building our VAR model 
    [moaic, mobic] = tsdata_to_varmo(data, momax, moregmode, [], [], []); % Calculate model order using AIC abd BIC criterion
    morder = moselect(sprintf('VAR model order selection, max = %d',momax),mosel, 'AIC', moaic, 'BIC', mobic);
    [A, V] = tsdata_to_var(data, morder, regmode);                        % Create VAR Model
    info = var_info(A, V);                                                % Check information of VAR model
    F = var_to_pwcgc(A, V);                                               % Run Pairwise Granger Causality
    tstat = var_to_pwcgc_tstat(data, V, morder, regmode, stat);           % Estimate Statistic


    % Significance testing
    nvars       = size(A, 1);
    nobs        = size(data, 2);
    ntrials     = size(data, 3);

    pval = mvgc_pval(tstat, stat, 1, 1, nvars-2, morder, nobs, ntrials);  % Calculate p-values
    sig = significance(pval, alpha, mhtc);    % Multiple comparisons test.

    
    edgeWeightsMatrix = F/nanmax(F(:)); 
    edgematrixfile = sprintf([ssdi_data_dir 'pwcgc_matrix_%s_%g-of-%g.mat'], filename(1:end-4), fileNumber, length(files));
    save(edgematrixfile, 'edgeWeightsMatrix');
    
    % Plotting
    pdata = {F, sig};
    ptitle = {'estimated PWCGC', [stat '-test']};
    if isnumeric(plotm), plotm = plotm+1; end
    plot_gc(pdata, ptitle, [], [], plotm, []);


    % Set up state-space representation of VAR model.
    % Could also attempt a SS-model for GC estimation above.

    [Ass, Vss] = transform_var(A, V);
    [A, C, K] = var_to_ss(Ass);
    [fres,ierr] = var2fres(Ass,Vss);
    CAK = Ass; % we need this
    H = var2trfun(Ass, fres); % and this

    % setting up model descripton
    r = morder;
    n = size(data,1);

    modfile = sprintf([pwcgc_dir 'pwcgc_%s_%g-of-%g.mat'], filename(1:end-4), fileNumber, length(files));
    fprintf('Saving PWCGC model ''%s''. ', modfile);
    save(modfile, 'V', 'CAK', 'H', 'Vss');
    fprintf('Saved..\n');
    toc;
end

% Edge weights (GC-graph) for each subject
%edgeWeights = reshape(edgeWeightsMatrix, [5, 2000]);  % the second digit is nodes x simulations
    














