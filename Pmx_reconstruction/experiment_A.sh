#!/bin/bash
#SBATCH --job-name=experiment_A
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=4:00:00
#SBATCH --output=logs/experiment_A_%j.out
#SBATCH --error=logs/experiment_A_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fb635@cam.ac.uk

set -euo pipefail

# ---------------------------------------------------------------------------
# Usage:
#   sbatch experiment_A_from_Phx.sh [extra args passed straight to the script]
#
# Examples:
#   sbatch experiment_A_from_Phx.sh
#
# The script will be executed with the specified arguments.
# Anything after the script name is forwarded to the Python argparse parser,
# so you don't need to edit this file for one-off parameter changes.
# ---------------------------------------------------------------------------

WORKDIR=/home/fb635/fedirfiles/tracing_cosmic_gas
VENV=/home/fb635/fedirfiles/venv
# -m, not a path: running the file directly would put
# Pmx_reconstruction/ on sys.path[0] instead of the repo root, which
# breaks both "import utils" and "from Pmx_reconstruction.pmxlib ...".
MODULE=Pmx_reconstruction.experiment_A

# Activate the virtual environment
source "${VENV}/bin/activate"

# Change to RDS working directory (where data and scripts are located)
cd "${WORKDIR}"
mkdir -p logs

# Set environment variables for optimal performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export FFTW_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Disable MPI initialization to avoid PMI errors
export HDF5_USE_FILE_LOCKING=FALSE
export OMPI_MCA_mpi_warn_on_fork=0
export MPI4PY_RC_INITIALIZE=0

# Print job information
echo "Job ID:            $SLURM_JOB_ID"
echo "Job Name:          $SLURM_JOB_NAME"
echo "Node:              $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "CPUs per task:     $SLURM_CPUS_PER_TASK"
echo "Memory requested:  $(scontrol show job "$SLURM_JOB_ID" | grep -o 'mem=[^,]*' | head -1)"
echo "Script arguments:  $*"
echo "Start time:        $(date)"

# Print memory usage before starting
free -h

echo "Starting Experiment A..."
srun --ntasks="${SLURM_NTASKS:-1}" \
     --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
     python -u -m "${MODULE}" "$@"

# Print completion info
echo "Job completed at: $(date)"
echo "Final memory usage:"
free -h