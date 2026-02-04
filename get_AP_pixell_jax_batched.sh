#!/bin/bash
#SBATCH --job-name=ap_jax_batch
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --output=ap_jax_batch_%j.out
#SBATCH --error=ap_jax_batch_%j.err
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
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "Number of tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Total CPUs: $(($SLURM_NTASKS * $SLURM_CPUS_PER_TASK))"
echo "Memory: 240GB"
echo "Start time: $(date)"

# Print memory usage before starting
free -h

# Run the aperture photometry script with MPI
echo "Starting JAX batched aperture photometry with MPI ($SLURM_NTASKS processes)..."
echo "This will compute aperture photometry on tau maps using batched JAX operations..."

mpiexec -n $SLURM_NTASKS python -u get_AP_pixell_jax_batched.py

# Check exit status
if [ $? -ne 0 ]; then
    echo "ERROR: Aperture photometry script failed!"
    exit 1
fi

# Print completion info
echo "Job completed at: $(date)"
echo "Final memory usage:"
free -h

# Check output files
echo "Checking output files:"
OUTPUT_PATTERN="data/pixell_*CAP_code/fullFT_tau_reconstruction*/tau_apertures_*/tau_apertures_z_*.npz"
if ls $OUTPUT_PATTERN 1> /dev/null 2>&1; then
    echo "Output files found:"
    ls -lh $OUTPUT_PATTERN
else
    echo "WARNING: No output files matching pattern $OUTPUT_PATTERN"
fi

echo "JAX batched aperture photometry workflow complete!"
