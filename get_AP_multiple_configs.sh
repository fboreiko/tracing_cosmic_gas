#!/bin/bash
#SBATCH --job-name=get_AP_mult_conf
#SBATCH --partition=icelake-himem
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=2:00:00
#SBATCH --output=get_AP_mult_conf_%j.out
#SBATCH --error=get_AP_mult_conf_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fb635@cam.ac.uk

# Activate your virtual environment
source /home/fb635/fedirfiles/venv/bin/activate

# Change to working directory (where data and scripts are located)
cd /home/fb635/fedirfiles/tracing_cosmic_gas

# Set environment variables for optimal performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export FFTW_NUM_THREADS=$SLURM_CPUS_PER_TASK

# JAX configuration for CPU execution
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu

# Print job information
echo "========================================================================"
echo "MULTI-CONFIGURATION BATCH JOB"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "Number of tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Total CPUs: $(($SLURM_NTASKS * $SLURM_CPUS_PER_TASK))"
echo "Memory: 30GB"
echo "Time limit: 2 hours"
echo "Start time: $(date)"
echo "========================================================================"

# Print memory usage before starting
free -h

# Run the multi-configuration script with MPI
echo ""
echo "Starting multi-configuration aperture photometry processing..."
echo "This will process multiple gas types and configurations sequentially..."
echo ""

mpiexec -n $SLURM_NTASKS python -u get_AP_multiple_configs.py

# Print completion info
echo ""
echo "========================================================================"
echo "Job completed at: $(date)"
echo "========================================================================"
