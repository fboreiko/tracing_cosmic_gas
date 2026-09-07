#!/bin/bash
#SBATCH --job-name=tau_xy_field
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --output=tau_xy_field_%j.out
#SBATCH --error=tau_xy_field_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fb635@cam.ac.uk

set -euo pipefail

# Activate your virtual environment
source /home/fb635/fedirfiles/venv/bin/activate

# Set environment variables for optimal performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export FFTW_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory: 120GB"
echo "Start time: $(date)"

# Print memory usage before starting
free -h

# Run the tau xy field script
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python -u tau_xy_field.py

# Print completion info
echo "Job completed at: $(date)"
echo "Final memory usage:"
free -h