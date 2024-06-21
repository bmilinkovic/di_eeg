function run_optimisation(participant_id)

    % Ensure participant_id is a string
    participant_id = string(participant_id);

    fprintf('running optimisation for participant : %s \n', participant_id);

    % Get participant ID from command line argument
    % args = string(argv());
    % if numel(args) < 1
    %    error('Participant ID must be provided as a command line argument.');
    % end
    % participant_id = char(args(1));

    % Initialization and configuration
    iseed = 0; % Initialisation random seed (0 to use current rng state)
    nrunsp = 100; % Pre-optimisation runs (restarts)
    nitersp = 10000; % Pre-optimisation iterations
    sig0p = 1; % Pre-optimisation (gradient descent) initial step size
    gdlsp = 2; % Gradient-descent "line search" parameters
    gdtolp = 1e-10; % Gradient descent convergence tolerance
    histp = true; % Calculate optimisation history?
    ppp = true; % Parallelise multiple runs?

    ctol = 1e-6; % Hyperplane clustering tolerance

    defvar('gdeso',     2          ); % gradient-descent ES version (1 or 2)
    defvar('sig0o',   0.1        ); % optimisation (gradient descent) initial step size
    defvar('gdlso',     2          ); % gradient-descent "line search" parameters
    niterso = 10000; % Optimisation iterations
    gdtolo = 1e-10; % Gradient descent convergence tolerance
    histo = true; % Calculate optimisation history?
    ppo = true; % Parallelise multiple runs?
    alpha = 0.05; % statistics for inference on beta statistics
    mhtc = true; % multiple comparisons

    % Directories
    data_dir = '/data/gpfs/projects/punim1761/di_eeg/data/pwcgc/dspm/';
    
    ssdi_results_dir = '/data/gpfs/projects/punim1761/di_eeg/results/ssdi_results/dspm/';
    if exist([ssdi_results_dir]) == 0
        mkdir([ssdi_results_dir])
    end 

    % Load the data
    load_filename = fullfile(data_dir, 'dspm_pwcgc_' + participant_id + '.mat');
    fprintf('loading file: %s \n', load_filename);

    if ischar(load_filename) || isstring(load_filename)
        load(load_filename);
    else
        error('load_filename must be a character array or string scalar.');
    end

    n = size(V,1);
    fres = size(H,3)-1;

    % Setup parallel processing
    pc = parcluster('local');
    pc.JobStorageLocation = getenv('SCRATCH');
    parpool(pc, str2num(getenv('SLURM_CPUS_PER_TASK')));

    % start time to record CPU time
    start_time = tic;
    
    % Optimisation loop
    for m = 2:9
        fprintf('Beginning pre-optimisation for %d-macro \n', m);

        % Initialise the optimisation runs
        rstate = rng_seed(iseed);
        L0p = rand_orthonormal(n,m,nrunsp); % Initial (orthonormalised) random linear projections
        rng_restore(rstate);

        % Run pre-optimisation
        [doptp, Lp, convp, ioptp, soptp, cputp, ohistp] = opt_gd_ddx_mruns(CAK, L0p, nitersp, sig0p, gdlsp, gdtolp, histp, ppp);
        Loptp = itransform_subspace(Lp, V); % Inverse-transform Lopto back for un-decorrelated residuals
        goptp = gmetrics(Loptp); % Preoptima distances

        % Proper optimisation
        [uidx, usiz, nrunso] = Lcluster(goptp, ctol, doptp); 
        L0o = Lp(:, :, uidx);

        % Run optimisation
        [dopto, Lo, convo, iopto, sopto, cputo, ohisto] = opt_gd_dds_mruns(H, L0o, niterso, gdeso, sig0o, gdlso, gdtolo, histo, ppo);
        Lopto = itransform_subspace(Lo, Vss);
        gopto = gmetrics(Lopto);

        % Node weights calculation
        node_weights = zeros(n, size(Lopto, 3));
        for k = 1:size(Lopto, 3)
            node_weights(:, k) = 1 - gmetricsx(Lopto(:, :, k));
        end

        % New node weight calculation:

         node_weights_new = zeros(n, size(Lopto, 3));
        for k = 1:size(Lopto, 3)
            node_weights_new(:, k) = 1 - gmetrics1(Lopto(:, :, k));
        end

        % Beta statistics
        for k = 1:size(Lopto, 3)
            beta_stat(:, k) = habeta(Lopto(:, :, k));
        end

        % Statistical inference on beta statistics only for the minimally dynamically independent low-rank Lopto
        [cval_cc, pval_cc, sig_cc] = habeta_statinf(beta_stat(:, 1), n, m,  alpha, 'both', mhtc);

        % grouped-node level n-macroscopic [UNDER CONSTRUCTION]
        %grouped_dist = zeros(size(Lopto, 1), size(Lopto, 3));

        %for k = 1:size(Lopto, 3)
        %    grouped_dist(:, k) = 1 - gmetricsxx(Lopto(:, :, k));
        %end

        fprintf('Saving results for %d-macro \n', m);

        % Create a structure for all results
        results.(['doptp_m' num2str(m)]) = doptp; % pre optimisation history dd values
        results.(['goptp_m' num2str(m)]) = goptp; % pre optimisation suboptima distances
        results.(['dopto_m' num2str(m)]) = dopto; % optimisation history dd values
        results.(['gopto_m' num2str(m)]) = gopto; % optimisation suboptima distances
        results.(['node_weights_m' num2str(m)]) = node_weights; % node contribution (old algo)
        results.(['node_weights_new_m' num2str(m)]) = node_weights_new; % node contribution (new algo)
        results.(['Loptp_m' num2str(m)]) = Loptp; % preoptimised L matrices
        results.(['Lopto_m' num2str(m)]) = Lopto; % optimised L matrices
        results.(['beta_m' num2str(m)]) = beta_stat; % beta statistics, i.e., nodal contribution under a beta distribution null
        results.(['beta_pvals_m' num2str(m)]) = pval_cc; % pvalues beta statistics for each node
        results.(['beta_sigs_m' num2str(m)]) = sig_cc; % significance values for each beta stat pval
        results.(['beta_cval_m' num2str(m)]) = cval_cc; % critical values for multiple comparisons (?)
        %results.(['grouped_nodes_m' num2str(m)]) = grouped_dist; % grouped-node analysis

        fprintf('Optimisation Completed for %d-macro \n' , m);
    end

    end_time = toc(start_time);

    fprintf('Participant optimisation time = %s \n',datestr(seconds(end_time),'HH:MM:SS.FFF'));

    % Save all results to a single file
    save_filename = fullfile(ssdi_results_dir, participant_id + '_ssdi_all_results_dspm.mat');
    save(save_filename, 'results');

    fprintf('All data saved to %s \n', save_filename);
end
