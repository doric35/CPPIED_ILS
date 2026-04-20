#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM submission script – EJOR s1616 experiments on Compute Canada
#
# Workload estimate (worst case, all 840 pairs pending):
#   1680 pairs × 5 runs × 1 min = 8 400 CPU-minutes serial
#   With a 32-core node          ≈  263 min  →  wall time request: 6 h
#   With a 64-core node (Narval) ≈   132 min  →  wall time request: 3 h
# Workload estimate (worst case, all 840 pairs pending):
#   1680 pairs × 5 runs × 10 min = 84 000 CPU-minutes serial
#   With a 32-core node          ≈  2625 min  →  wall time request: 50 h
#   With a 64-core node (Narval) ≈   1313 min  →  wall time request: 25 h
# Workload estimate (worst case, all 840 pairs pending):
#   1680 pairs × 5 runs × 20 min = 168 000 CPU-minutes serial
#   With a 32-core node          ≈  5250 min  →  wall time request: 100 h
#   With a 64-core node (Narval) ≈   2625 min  →  wall time request: 50 h
#
# Usage:
#   sbatch submit_ejor_cc.sh \
#       --executable /path/to/cppied  \
#       --lkh        /path/to/LKH  \
#       --solver     ID
#
#   Or set CPPIED_EXE and LKH_EXE as environment variables before submitting:
#   export CPPIED_EXE=/path/to/cppied
#   export LKH_EXE=/path/to/LKH
#   sbatch submit_ejor_cc.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=ejor_cppied
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --exclusive                # guarantee sole occupancy of the node; SLURM_CPUS_ON_NODE reflects all CPUs
#SBATCH --mem=0                    # use all available RAM on the node
#SBATCH --time=160:00:00
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
        --solver)        SOLVER_ID="$2"; shift 2 ;;
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
if [[ -z "${SOLVER_ID:-}" ]]; then
    echo "WARNING: Solver ID not set. Using auto solver configuration." >&2
    SOLVER_ID="auto"
fi
# ── Resolve paths ─────────────────────────────────────────────────────────────
# SLURM copies the script to a spool directory, so BASH_SOURCE[0] points there.
# SLURM_SUBMIT_DIR is the directory from which sbatch was called — use that instead.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PARALLEL_SCRIPT="${SCRIPT_DIR}/run_ejor_experiments_parallel.py"
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
echo "SOLVER        : ${SOLVER_ID}"
echo "Python        : $(which python3) ($(python3 --version))"
echo "Start time    : $(date)"
echo "================================================================"

# ── Run experiments ───────────────────────────────────────────────────────────
python3 "${PARALLEL_SCRIPT}" \
    --executable "${CPPIED_EXE}" \
    --lkh        "${LKH_EXE}"    \
    --workers    "${SLURM_CPUS_ON_NODE}" \
    --solver     "${SOLVER_ID}"

EXIT_CODE=$?

echo "================================================================"
echo "End time : $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "================================================================"

exit ${EXIT_CODE}
