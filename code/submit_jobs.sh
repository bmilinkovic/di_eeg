#!/bin/bash
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --job-name=bm_optimisation
#SBATCH --account="punim1761"
#SBATCH --output=/data/gpfs/projects/punim1761/di_eeg/results/output/%A_%a.out
#SBATCH --error=/data/gpfs/projects/punim1761/di_eeg/results/error/%A_%a.err
#SBATCH --time=20-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=4096
#SBATCH --array=1-34%17  # 34 participants, 10 concurrent jobs
#SBATCH --mail-user=bmilinkovic@student.unimelb.edu.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

# Load MATLAB module
module load MATLAB

# Set participants array
participants=("H0010P" "H0010W" "H0048P" "H0048W" "H0081W" "H0081X" "H0090W" "H0090X" "H0109K" "H0109W" "H0294K" "H0294W" "H0338K" "H0338W" "H0526S" "H0534K" "H0534W" "H0603P" "H0603W" "H0631W" "H0631X" "H0668P" "H0668W" "H0707S" "H0730S" "H0746P" "H0746W" "H0777W" "H0777X" "H0781S" "H0826K" "H0826W" "H0905W" "H0905X")

# Get the participant ID from the array using SLURM_ARRAY_TASK_ID
participant_id=${participants[$SLURM_ARRAY_TASK_ID-1]}

# Run MATLAB script
export SCRATCH="/data/gpfs/projects/punim1761/di_eeg/results/jobarrays/${SLURM_JOB_ID}"
mkdir -p $SCRATCH

matlab -nodisplay -nosplash -nodesktop -r "run_optimisation(\"$participant_id\"); quit"
