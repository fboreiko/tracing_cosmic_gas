#!/bin/bash
#SBATCH --job-name=measure_utilde
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=8:00:00
#SBATCH --output=logs/measure_utilde_%j.out
#SBATCH --error=logs/measure_utilde_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fb635@cam.ac.uk

set -euo pipefail

# ---------------------------------------------------------------------------
# Usage:
#   sbatch measure_u_tilde.sh [extra args passed straight to the script]
#
# Examples:
#   sbatch measure_u_tilde.sh
#   sbatch measure_u_tilde.sh --recompute
#   sbatch measure_u_tilde.sh --nbins 30 --logm-min 11 --logm-max 15
# ---------------------------------------------------------------------------

WORKDIR=/home/fb635/fedirfiles/tracing_cosmic_gas
VENV=/home/fb635/fedirfiles/ox_env
SCRIPT=measure_u_tilde.py

# Activate the virtual environment
source "${VENV}/bin/activate"

# Change to working directory
cd "${WORKDIR}"
mkdir -p logs

# Threading environment
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export FFTW_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Avoid MPI init warnings/errors in non-MPI jobs
export HDF5_USE_FILE_LOCKING=FALSE
export OMPI_MCA_mpi_warn_on_fork=0
export MPI4PY_RC_INITIALIZE=0

# Forward user args; inject --nthread if user did not provide it explicitly
HAS_NTHREAD=0
for arg in "$@"; do
    if [[ "$arg" == "--nthread" || "$arg" == --nthread=* ]]; then
        HAS_NTHREAD=1
        break
    fi
done

SCRIPT_ARGS=("$@")
if [[ ${HAS_NTHREAD} -eq 0 ]]; then
    SCRIPT_ARGS=(--nthread "${SLURM_CPUS_PER_TASK:-1}" "${SCRIPT_ARGS[@]}")
fi

# Print job information
echo "Job ID:            $SLURM_JOB_ID"
echo "Job Name:          $SLURM_JOB_NAME"
echo "Node:              $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "CPUs per task:     $SLURM_CPUS_PER_TASK"
echo "Memory requested:  $(scontrol show job "$SLURM_JOB_ID" | grep -o 'mem=[^,]*' | head -1)"
echo "Script arguments:  ${SCRIPT_ARGS[*]}"
echo "Start time:        $(date)"

free -h

echo "Starting u_tilde measurement..."
srun --ntasks="${SLURM_NTASKS:-1}" \
     --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
     python -u "${SCRIPT}" "${SCRIPT_ARGS[@]}"

echo "Job completed at: $(date)"
echo "Final memory usage:"
free -h
