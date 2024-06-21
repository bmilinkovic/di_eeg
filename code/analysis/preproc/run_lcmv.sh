#!/bin/bash
#SBATCH --job-name=lcmv_analysis
#SBATCH --output=lcmv_analysis_%j.out
#SBATCH --error=lcmv_analysis_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8  # Adjust based on your CPU requirements
#SBATCH --mem=128G  # Adjust based on your memory requirements
#SBATCH --time=4:00:00  # Adjust based on your expected runtime

# Load necessary modules (replace these with your specific module loads)
source ~/miniconda3/bin/activate eeg_analysis

# Navigate to the directory containing the Python script
cd /data/gpfs/projects/punim1761/di_eeg/code/analysis/preproc/

# Execute the Python script
python LCMV_recon.py

# Deactivate the conda environment (optional)
conda deactivate
