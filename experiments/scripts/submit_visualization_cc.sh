#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM submission script – visualization runs on Compute Canada
#
# Runs run_visualization_experiments.py for each pair listed in a pairs file.
# One solve per pair, with SOLUTION and VISUALIZE output enabled.
# Results are written to experiments/scratch/ only.
#
# Usage:
#   sbatch submit_visualization_cc.sh \
#       --executable /path/to/Cppied  \
#       --lkh        /path/to/LKH     \
#       --pairs      /path/to/pairs.txt
#
#   Or set environment variables before submitting:
#   export CPPIED_EXE=/path/to/Cppied
#   export LKH_EXE=/path/to/LKH
#   export PAIRS_FILE=/path/to/pairs.txt
#   sbatch submit_visualization_cc.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=viz_cppied
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive                # guarantee sole occupancy of the node; SLURM_CPUS_ON_NODE reflects all CPUs
#SBATCH --mem=0                    # use all available RAM on the node
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Uncomment and set the appropriate allocation account:
##SBATCH --account=<your_account>

# ── Parse optional arguments passed via sbatch ────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --executable|-e) CPPIED_EXE="$2"; shift 2 ;;
        --lkh)           LKH_EXE="$2";    shift 2 ;;
        --pairs|-p)      PAIRS_FILE="$2"; shift 2 ;;
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
if [[ -z "${PAIRS_FILE:-}" ]]; then
    echo "ERROR: Pairs file not set." \
         "Use --pairs or export PAIRS_FILE." >&2
    exit 1
fi
if [[ ! -f "${PAIRS_FILE}" ]]; then
    echo "ERROR: Pairs file not found: ${PAIRS_FILE}" >&2
    exit 1
fi

# ── Resolve paths ─────────────────────────────────────────────────────────────
# SLURM copies the script to a spool directory, so BASH_SOURCE[0] points there.
# SLURM_SUBMIT_DIR is the directory from which sbatch was called — use that instead.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VIZ_SCRIPT="${SCRIPT_DIR}/run_visualization_experiments.py"
SCRATCH_DIR="${SCRIPT_DIR}/../scratch"
mkdir -p "${SCRATCH_DIR}"

# ── Environment setup ─────────────────────────────────────────────────────────
module load python/3.11       2>/dev/null || true
module load scipy-stack       2>/dev/null || true
module load gurobi            2>/dev/null || true
module load eigen             2>/dev/null || true
module load opencv            2>/dev/null || true

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
echo "Pairs file    : ${PAIRS_FILE}"
echo "Scratch dir   : ${SCRATCH_DIR}"
echo "Python        : $(which python3) ($(python3 --version))"
echo "Start time    : $(date)"
echo "================================================================"

# ── Run visualization experiments ─────────────────────────────────────────────
python3 "${VIZ_SCRIPT}"      \
    --executable "${CPPIED_EXE}" \
    --lkh        "${LKH_EXE}"    \
    --pairs      "${PAIRS_FILE}" \
    --workers    "${SLURM_CPUS_ON_NODE}"

EXIT_CODE=$?

echo "================================================================"
echo "End time : $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "================================================================"

exit ${EXIT_CODE}
