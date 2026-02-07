#!/bin/bash
#SBATCH --job-name=field_manip
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=run_2d_recon%j.out
#SBATCH --error=run_2d_recon%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fb635@cam.ac.uk

# Activate your virtual environment
source /home/fb635/fedirfiles/venv/bin/activate

# Change to RDS working directory (where data and scripts are located)
cd /home/fb635/fedirfiles/tracing_cosmic_gas

# Set environment variables for optimal performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export FFTW_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Disable MPI initialization to avoid PMI errors
export HDF5_USE_FILE_LOCKING=FALSE
export OMPI_MCA_mpi_warn_on_fork=0

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory: 180GB"
echo "Start time: $(date)"

# Print memory usage before starting
free -h

# Run the field manipulation script
echo "Starting 2D reconstruction (8 CPUs)..."
python -u Tk_massdep_reconstruction.py

# Print completion info
echo "Job completed at: $(date)"
echo "Final memory usage:"
free -h