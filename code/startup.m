global emergenceDir;
emergenceDir = fileparts(mfilename('fullpath'));
addpath(emergenceDir);
fprintf('[Emergence Pipeline startup] Added path code directory: %s\n', emergenceDir);

%addpath(fullfile(emergenceDir, 'preprocessing'));
%addpath(genpath(fullfile(emergenceDir, 'mvgcfuncs')));
%addpath(fullfile(emergenceDir, 'results'));
%addpath(fullfile(emergenceDir, 'simulation'));
%addpath(fullfile(emergenceDir, 'src'));
%addpath(fullfile(emergenceDir, 'utils'));
%addpath(fullfile(emergenceDir, 'test'));

global ssdiDir;
ssdiDir = '/data/gpfs/projects/punim1761/di_eeg/code/ssdi-1'; % I just changed this to the new ssdi-1 toolbox might not work, so please test
addpath((ssdiDir));
cd(ssdiDir);
startup;
cd(emergenceDir);
fprintf('[Emergence Pipeline startup] Added path to State-Space Dynamical Indpendence toolbox: %s\n',ssdiDir);

global ceDir;
ceDir = '/data/gpfs/projects/punim1761/di_eeg/code/ReconcilingEmergences';
addpath(genpath(ceDir));
fprintf('[Emergence Pipeline startup] Added path to Strong Causal Emergence toolbox: %s\n',ceDir);


% adds fieldtrip to our path and sets default settings
%fieldtripDir = '/Users/borjanmilinkovic/Documents/toolboxes/fieldtrip';
%addpath(fieldtripDir);
%ft_defaults;

% adds Thomas' LSCP toolbox to our path
%LSCPtoolsDir = '/Users/borjanmilinkovic/Documents/toolboxes/LSCPtools';
%addpath(genpath(LSCPtoolsDir));

% adds paths to out raw data and the directory into which we will save our
% preprocessed data. Note, that these directories are on an external HDD.

%path_rawdata = '/Volumes/dataSets/restEEGHealthySubjects/rawData';
%path_preproc = '/Volumes/dataSets/restEEGHealthySubjects/preprocessedData';
%path_source = '/Volumes/dataSets/restEEGHealthySubjects/preprocessedData/sourceReconstructions';







