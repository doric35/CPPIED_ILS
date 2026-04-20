#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM submission script – Google Test suite on Compute Canada
#
# Runs the compiled test binary (build/test_cppied).  All tests are
# executed sequentially in a single process; the ILP fixture requires
# a Gurobi licence, and several metaheuristic fixtures require LKH.
#
# Wall-time estimate:
#   Fast unit tests               < 1 min
#   ILP small fixture (Gurobi)    ~ 2 min
#   ILP large fixture (60 s/run)  ~ 5 min
#   Total (conservative)          ~ 30 min   → wall-time request: 1 h
#
# Usage:
#   # ensure the output directory exists before submitting
#   mkdir -p output
#   sbatch submit_tests_cc.sh \
#       --executable /path/to/build/test_cppied \
#       --lkh        /path/to/LKH
#
#   # run only a subset of tests (GoogleTest filter syntax)
#   sbatch submit_tests_cc.sh \
#       --executable /path/to/build/test_cppied \
#       --lkh        /path/to/LKH \
#       --filter     "ilp_*:dp_sweeper*"
#
#   # or export the paths as environment variables first:
#   export TEST_EXE=/path/to/build/test_cppied
#   export LKH_EXE=/path/to/LKH
#   sbatch submit_tests_cc.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=cppied_tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=output/%x_%j.out
#SBATCH --error=output/%x_%j.err
# Uncomment and set the appropriate allocation account:
#SBATCH --account=def-mmorin-ab

# ── Parse optional arguments passed to sbatch ─────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --executable|-e) TEST_EXE="$2";    shift 2 ;;
        --lkh)           LKH_EXE="$2";    shift 2 ;;
        --filter|-f)     GTEST_FILTER="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Validate required paths ───────────────────────────────────────────────────
if [[ -z "${TEST_EXE:-}" ]]; then
    echo "ERROR: test executable not set." \
         "Use --executable or export TEST_EXE." >&2
    exit 1
fi
if [[ ! -x "${TEST_EXE}" ]]; then
    echo "ERROR: test executable not found or not executable: ${TEST_EXE}" >&2
    exit 1
fi
if [[ -z "${LKH_EXE:-}" ]]; then
    echo "WARNING: LKH executable not set. Tests requiring LKH will fail." >&2
fi

# ── Resolve paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_ROOT="${SCRIPT_DIR}/../.."
RESULTS_DIR="${SCRIPT_DIR}/../results"
mkdir -p "${RESULTS_DIR}"

# GTest XML report — written next to the other experiment results
GTEST_XML="${RESULTS_DIR}/gtest_${SLURM_JOB_ID:-local}.xml"

# ── Environment setup ─────────────────────────────────────────────────────────
module load python/3.11   2>/dev/null || true
module load scipy-stack   2>/dev/null || true
module load gurobi        2>/dev/null || true
module load eigen         2>/dev/null || true
module load opencv        2>/dev/null || true

VENV="${SCRIPT_DIR}/../../../GurobiEnv"
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
fi

# ── Diagnostics ───────────────────────────────────────────────────────────────
echo "================================================================"
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Node          : $(hostname)"
echo "Test binary   : ${TEST_EXE}"
echo "LKH binary    : ${LKH_EXE:-<not set>}"
echo "GTest filter  : ${GTEST_FILTER:-<all tests>}"
echo "GTest XML     : ${GTEST_XML}"
echo "Start time    : $(date)"
echo "================================================================"

# ── Build the GTest argument list ─────────────────────────────────────────────
GTEST_ARGS=(
    "--gtest_output=xml:${GTEST_XML}"
    "--gtest_color=no"
)
if [[ -n "${GTEST_FILTER:-}" ]]; then
    GTEST_ARGS+=("--gtest_filter=${GTEST_FILTER}")
fi

# ── Inject LKH path into the environment expected by the test fixture ─────────
# The test fixture reads LKH_EXECUTABLE from ctx.config; passing it via an
# environment variable avoids hardcoding a path inside the test binary.
if [[ -n "${LKH_EXE:-}" ]]; then
    export LKH_EXECUTABLE="${LKH_EXE}"
fi

# ── Run the test suite ────────────────────────────────────────────────────────
"${TEST_EXE}" "${GTEST_ARGS[@]}"
EXIT_CODE=$?

# ── Summary ───────────────────────────────────────────────────────────────────
echo "================================================================"
echo "End time  : $(date)"
echo "Exit code : ${EXIT_CODE}"
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "Result    : ALL TESTS PASSED"
else
    echo "Result    : SOME TESTS FAILED (see XML report and output above)"
fi
echo "XML report: ${GTEST_XML}"
echo "================================================================"

exit ${EXIT_CODE}
