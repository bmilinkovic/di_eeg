## [Under Construction] Dynamical Independence analysis for Electroencephalographic (EEG) Data

This toolbox contains a full pipeline for computing Dynamical Independence from EEG data, amongst a suite of other measures.

External dependencies are also provided for:

1. [MVGC2 toolbox] Multivariate Granger Causality toolbox (with state space modelling)---originally forked from Dr. Lionel Barnett, https://github.com/lcbarnett/ssdi
2. [Reconciling Emergences] Approximates Causal Emergence---originally forked from Dr. Pedro Mediano, https://github.com/pmediano/ReconcilingEmergences
3. Whole-brain modelling with Stuart-Landau models.
4. [Fast LZc] Computation of Lempel-Ziv Complexity---originally forked from Dr. Lionel Barnett, https://github.com/lcbarnett/fLZc
5. [MNE data] For template fsl brain, etc.

Code begins from cleaning raw EEG data.

First, loading in the raw data and rejection of bad channels can be found in bad_channels.py

Second, source reconstruction is automated and can be performed for all subjects either using LCMV or dSPM, which can be found in LCMV_recon.py and dSPM_recon.py scripts, respectively. Automated bash scripts for running each of these on HPC clusters is also provided as run_lcmv.sh, and run_dspm.sh respectively.

Third, leveraging MVGC2 to obtain either VAR or SS model parameters and pair-wise conditional Granger-causal graphs is done through the mvgc_parameters.m routine.

Fourth run_optimisation.m performs the entire DI analysis discovering macroscopic variables across all hierarchical scales. The analysis saves the resulting data in a results structure that is very easy to use and extract all needed results. A bash script for implementation of HPC is also provided to run all optimisations.

For directory structure, I also have a results/ and data/ structure. I store the results of the GC and DI analysis in the results/ directory. And, raw, preprocessed, and source reconstructed data in the data/ directory.

Please feel free to contact me regarding any tips on how to make the toolbox better:
borjan.milinkovic@gmail.com
