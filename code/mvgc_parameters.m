%% 1. Parameters

if ~exist('momax',      'var'), momax       = 30;       end         % maximum model order
if ~exist('moregmode',  'var'), moregmode   = 'LWR';    end         % model order regression mode 
if ~exist('mosel',      'var'), mosel       = 'AIC';    end         % which model order to select
if ~exist('plotm',      'var'), plotm       = 0;        end         % for plotting on seperate figures
if ~exist('regmode',    'var'), regmode     = 'LWR';    end 
if ~exist('stat',       'var'), stat        = 'F';      end
if ~exist('alpha',      'var'), alpha       = 0.01;     end
if ~exist('mhtc',       'var'), mhtc        = 'FDRD';   end

%% 2. LOAD DATA
% Setting up the directory structure for saving files first

src_data = '/data/gpfs/projects/punim1761/di_eeg/data/preprocessed/lcmv/mat/';
pwcgc_dir = '/data/gpfs/projects/punim1761/di_eeg/data/pwcgc/lcmv/';

if exist([pwcgc_dir]) == 0
    mkdir([pwcgc_dir])
end

% list files in directory of source reconstructed data
files = dir([src_data '*.mat']);


%% MVGC

eweights = zeros(46, 46, length(files)); 


for fileNumber = 1:length(files)
    filename = files(fileNumber).name;
    folder = files(fileNumber).folder;
    
    
    tic;
    fprintf('..Starting on subject %s (%g / %g) \n', filename, fileNumber, length(files));
    
    load([folder filesep filename]);
    % for lcmv
    data = permute(source_ts, [2,3,1]);

    % for dspm
    % data = reshape(source_ts(:, 1:(floor(size(source_ts, 2) / 512) * 512)), 46, 512, []);

    data = demean(data, true);



    % model order selection and building our VAR model 

    [moaic, mobic] = tsdata_to_varmo(data, momax, moregmode, [], [], []); % Calculate model order using AIC abd BIC criterion
    morder = moselect(sprintf('VAR model order selection, max = %d',momax),mosel, 'AIC', moaic, 'BIC', mobic);
    [A, V] = tsdata_to_var(data, morder, regmode);                        % Fit VAR Model using LWR
    info = var_info(A, V);                                                % Check information of VAR model

    % The following two steps are not needed for Dynamical Independence
    % analysis

    F = var_to_pwcgc(A, V);                                               % Run Pairwise Granger Causality
    tstat = var_to_pwcgc_tstat(data, V, morder, regmode, stat);           % Estimate Statistic


    % Significance testing
    nvars       = size(A, 1);
    nobs        = size(data, 2);
    ntrials     = size(data, 3);

    pval = mvgc_pval(tstat, stat, 1, 1, nvars-2, morder, nobs, ntrials);  % Calculate p-values
    sig = significance(pval, alpha, mhtc);    % Multiple comparisons test.

    
    eweights = F/nanmax(F(:)); 
    eweights_file = sprintf([pwcgc_dir 'lcmv_pwcgc_matrix_%s.mat'], filename(1:6));
    save(eweights_file, 'eweights');

   % Convert to SS parameters -- necessary for DI analysis

    [Ass, Vss] = transform_var(A, V);
    [A, C, K] = var_to_ss(Ass);
    [fres,ierr] = var2fres(Ass,Vss);
    CAK = Ass;
    H = var2trfun(Ass, fres); % Transfer function

    % setting up model descripton
    r = morder;
    n = size(data,1);

    modfile = sprintf([pwcgc_dir 'lcmv_pwcgc_%s.mat'], filename(1:6));
    fprintf('Saving PWCGC model ''%s''. ', modfile);
    save(modfile, 'V', 'CAK', 'H', 'Vss');
    fprintf('Saved..\n');
    toc;
end


