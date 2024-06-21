#!/bin/bash
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --job-name=bm_lcmv_mvgc
#SBATCH --account="punim1761"
#SBATCH --output=/data/gpfs/projects/punim1761/di_eeg/test/%A_%a.out
#SBATCH --error=/data/gpfs/projects/punim1761/di_eeg/test/%A_%a.err
#SBATCH --time=2-24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=16G

# Load MATLAB module
module load MATLAB

matlab -nodisplay -nosplash -nodesktop -r "mvgc_parameters; quit"
