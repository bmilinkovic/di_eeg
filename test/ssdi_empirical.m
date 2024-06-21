%%%%%%%%%%%%%%% PREOPT

defvar('iseed',    0           ); % initialisation random seed (0 to use current rng state)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

defvar('nrunsp',   100         ); % pre-optimisation runs (restarts)
defvar('nitersp',  10000       ); % pre-optimisation iterations
defvar('sig0p',    1           ); % pre-optimisation (gradient descent) initial step size
defvar('gdlsp',    2           ); % gradient-descent "line search" parameters
defvar('gdtolp',   1e-10       ); % gradient descent convergence tolerance
defvar('histp',    true        ); % calculate optimisation history?
defvar('ppp',      true       ); % parallelise multiple runs?

%%%%%%%%%%%%%% OPTIMISATION

defvar('ctol',     1e-6        ); % hyperplane clustering tolerance

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

defvar('niterso',  10000       ); % pre-optimisation iterations
defvar('sig0o',    0.1         ); % optimisation (gradient descent) initial step size
defvar('gdlso',    2           ); % gradient-descent "line search" parameters
defvar('gdtolo',   1e-10       ); % gradient descent convergence tolerance
defvar('histo',    true        ); % calculate optimisation history?
defvar('ppo',      true       ); % parallelise multiple runs?




%% Setting directories

results_dir = '/Volumes/dataSets/restEEGHealthySubjects/restEEGHealthySubjects/AnesthesiaProjectEmergence/results/';
pwcgc_dir = [results_dir 'pwcgc/'];
ssdi_data_dir = [results_dir 'ssdiData/'];


%% Dynamical Independence Run across n-macros


pwcgcFiles = dir([pwcgc_dir filesep '*.mat']);
%doptp = cell(1, length(pwcgcFiles));





for fileNumber = 1:length(pwcgcFiles)
    filename = pwcgcFiles(fileNumber).name;
    folder = pwcgcFiles(fileNumber).folder;
    
    tic;
    % Load the data
    load([folder filesep filename]);
    
    n = size(V,1);
    fres = size(H,3)-1;
    
    fprintf('Beginning pre-optimisation for %d-macro on participant %g / %g \n', m, fileNumber, length(pwcgcFiles));
    
    % setting loop over macro variables
    for m = 2:42
    
        % Initialise the optimisation runs
        rstate = rng_seed(iseed);
        L0p = rand_orthonormal(n,m,nrunsp); % initial (orthonormalised) random linear projections
        rng_restore(rstate);

        % Run optimisation
        [doptp,Lp,convp,ioptp,soptp,cputp,ohistp] = opt_gd_ddx_mruns(CAK,L0p,nitersp,sig0p,gdlsp,gdtolp,histp,ppp);

        % Inverse-transform Lopto back for un-decorrelated residuals
        Loptp = transform_proj(Lp,V);

        % Preoptima distances
        goptp = gmetrics(Loptp);

        % Proper optimisation
        [uidx,usiz,nrunso] = Lcluster(goptp, ctol, doptp); 

        % initialise optimisation
        L0o = Lp(:,:,uidx);

        % Run this sucker!
        [dopto, Lo, convp, iopto, sopto, cputo, ohisto] = opt_gd_dds_mruns(H,L0o,niterso,sig0o,gdlso,gdtolo,histo,ppo);
        
        fprintf('Optimisation Completed for participant %d-macro %g / %g \n', m, fileNumber, length(pwcgcFiles));

        % transform to un-decorrelated again
        Lopto = transform_proj(Lo,Vss);

        %inter optima distance
        gopto = gmetrics(Lopto);

        % weighting of nodes in contributing to macro

        node_weights = zeros(n, size(Lopto, 3));        % the 3rd dimension is the number of runs. 
        for k = 1:size(Lopto, 3)
            node_weights(:, k) = 1-gmetricsx(Lopto(:, :, k));
        end
        
        % we don't need this, just use dopto
        %dynamical_dependence{fileNumber} = dopto{fileNumber};
        
        % Saving data:
        
        fprintf('Saving data for %d-macro for participant %g / %g \n', m, fileNumber, length(pwcgcFiles));
        ddx_file = fullfile(ssdi_data_dir, [filename(7:12) '_mdim_' num2str(m) '_preopt_dynamical_dependence.mat']);
        save(ddx_file, 'doptp');
        
        opt_dist_x_file = fullfile(ssdi_data_dir, [filename(7:12) '_mdim_' num2str(m) '_preopt_optima_dist.mat']);
        save(opt_dist_x_file, 'goptp');
        
        dd_file = fullfile(ssdi_data_dir, [filename(7:12) '_mdim_' num2str(m) '_dynamical_dependence.mat']);
        save(dd_file, 'dopto');
        
        opt_dist_file = fullfile(ssdi_data_dir, [filename(7:12) '_mdim_' num2str(m) '_optima_dist.mat']);
        save(opt_dist_file, 'gopto');
        
        node_weights_file = fullfile(ssdi_data_dir, [filename(7:12) '_mdim_' num2str(m) '_node_weights.mat']);
        save(node_weights_file, 'node_weights');  
        fprintf('Saving Complete!');
    end
end

%% Saving data
% Dynamical Dependence Vector
% dynamical_independence_matrix = reshape(dynamical_dependence, [20,20]);
% dynamical_independence_matrix = dynamical_independence_matrix.';

% This will save all of the dynamical dependence values for each subject
% across every optimisation run

% save(ssdi_data_dir, 'EEG_46Region_dynamical_dependence');
% 
% % Here we are selecting the nodes weights attribued to the maximally
% % dynamically independent macro _Y_ for each n-macro across the k
% % optimisation runs.
% 
% maximal_node_weights = cell(1, length(node_weights));
% for subject = 1:length(node_weights) % -> numel(cellmatrix)
%         maximal_node_weights{i} = node_weights{1,subject}(:,:,1);   % change 1 to 7 if you want to extract 7th column
% end
% maximal_node_weights = cell2mat(maximal_node_weights);
% save(ssdi_data_dir, 'EEG_46Region_maximal_node_weights');



