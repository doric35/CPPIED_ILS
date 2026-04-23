#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM submission script – ILS configuration local search on Compute Canada
#
# The search terminates after 24 h or when the incumbent did not change
# over a full iteration.  Wall time is set to 25 h to allow for a clean
# end-of-iteration flush before the job is killed.
#
# Usage:
#   sbatch submit_ejor_ls_config_cc.sh \
#       --executable /path/to/cppied  \
#       --lkh        /path/to/LKH
#
#   Or set CPPIED_EXE and LKH_EXE as environment variables before submitting:
#   export CPPIED_EXE=/path/to/cppied
#   export LKH_EXE=/path/to/LKH
#   sbatch submit_ejor_ls_config_cc.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=ejor_ls_config
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --exclusive                # guarantee sole occupancy of the node; SLURM_CPUS_ON_NODE reflects all CPUs
#SBATCH --mem=0                    # use all available RAM on the node
#SBATCH --time=75:00:00
#SBATCH --output=output/%x_%j.out
#SBATCH --error=output/%x_%j.err
# Uncomment and set the appropriate allocation account:
#SBATCH --account=def-mmorin-ab

# ── Parse optional --executable / --lkh arguments passed to sbatch ───────────
# (sbatch forwards extra arguments after the script name)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --executable|-e) CPPIED_EXE="$2"; shift 2 ;;
        --lkh)           LKH_EXE="$2";   shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Validate required paths ───────────────────────────────────────────────────
if [[ -z "${CPPIED_EXE:-}" ]]; then
    echo "ERROR: CPPIED executable not set." \
         "Use --executable or export CPPIED_EXE." >&2
    exit 1
fi
if [[ -z "${LKH_EXE:-}" ]]; then
    echo "ERROR: LKH executable not set." \
         "Use --lkh or export LKH_EXE." >&2
    exit 1
fi

# ── Resolve paths ─────────────────────────────────────────────────────────────
# SLURM copies the script to a spool directory, so BASH_SOURCE[0] points there.
# SLURM_SUBMIT_DIR is the directory from which sbatch was called — use that instead.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
LS_CONFIG_SCRIPT="${SCRIPT_DIR}/run_ejor_ls_config_parallel.py"
LOG_DIR="${SCRIPT_DIR}/../results/logs"
mkdir -p "${LOG_DIR}"

# ── Environment setup ─────────────────────────────────────────────────────────
# Load required modules (adjust module names for your cluster)
module load python/3.11 2>/dev/null || true
module load scipy-stack 2>/dev/null || true  # provides numpy, scipy, pandas
module load gurobi 2>/dev/null || true
module load eigen 2>/dev/null || true
module load opencv 2>/dev/null || true

# Activate a virtual environment if one exists alongside the experiments folder
VENV="${SCRIPT_DIR}/../../../GurobiEnv"
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
fi

# ── Diagnostics ───────────────────────────────────────────────────────────────
echo "================================================================"
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Node          : $(hostname)"
echo "CPUs on node  : ${SLURM_CPUS_ON_NODE}"
echo "CPPIED exe    : ${CPPIED_EXE}"
echo "LKH exe       : ${LKH_EXE}"
echo "Python        : $(which python3) ($(python3 --version))"
echo "Start time    : $(date)"
echo "================================================================"

# ── Run ILS configuration local search ───────────────────────────────────────
python3 "${LS_CONFIG_SCRIPT}" \
    --executable "${CPPIED_EXE}" \
    --lkh        "${LKH_EXE}"    \
    --workers    "${SLURM_CPUS_ON_NODE}"

EXIT_CODE=$?

echo "================================================================"
echo "End time : $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "================================================================"

exit ${EXIT_CODE}
