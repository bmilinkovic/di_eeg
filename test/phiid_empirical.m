% Script for computing PhiiD on empirical data

% to compute the synergy and redundancy for each region

source_data_dir = '/mnt/data/bokim/restEEGHealthySubjects/preprocessedData/sourceReconstructions/';
files = dir([source_data_dir '*.mat']);

phiid_data_dir = '/mnt/data/bokim/restEEHHealthySubjects/AnesthesiaProjectEmergence/results/phiid/data/';

for fileNumber = 1:length(files)
    filename = files(fileNumber).name;
    folder = files(fileNumber).folder;
    
    % this is code for loading in the data and constructing a time-series
    % properly for the synergistic and redundancy matrices.
    load([folder filesep filename]);
    time_series = permute(source_ts, [2,3,1]);


    %this is a good trip to concatenate trials into one continuous time series.
    time_series = time_series(:,:);  

    synergy_mat = zeros(size(time_series,1), size(time_series,1));
    redundancy_mat = zeros(size(time_series,1), size(time_series,1));

    for row = 1:size(time_series, 1)
        for col = 1:size(time_series, 1)
            if row == col
                continue
            else
            atoms = PhiIDFull([time_series(row,:); time_series(col, :)]);
            synergy_mat(row, col) = atoms.sts;
            redundancy_mat(row, col) = atoms.rtr;
            fprintf("Computing the PhiID Decomposition of row: %d and col: %d ... \n", row, col);
            end
        end
    end

    % Save synergy and redundancy matrices
    phiid_syn = sprintf([phiid_data_dir '%s_syn_matrix.mat'], filename(1:6))
    phiid_red = sprintf([phiid_data_dir '%s_red_matrix.mat'], filename(1:6))
    save(phiid_syn, 'synergy_mat');
    save(phiid_red, 'redundancy_mat');

%% obtaining the synergy-redundancy gradient. 

    gradient = floor(tiedrank(mean(synergy_mat))) - floor(tiedrank(mean(redundancy_mat)));
    gradient_file = sprintf([phiid_data_dir '%s_synred_gradient.mat'], filename(1:6))
    save(gradient_file, 'gradient');
end



