"""
Parallel version of run_ejor_experiments.py.

Distributes all (algorithm_config × instance) pairs across worker
processes (one pair per worker at a time).  Already-completed pairs are
skipped by checking the existing results CSV before launching any work.

I/O strategy
────────────
During execution every write goes to the scratch filesystem:
  • Per-experiment working directories (config, local CSV, solution) are
    created under SCRATCH_DIR/<exp_name>/.
  • Completed results and errors are buffered in memory and also streamed
    to scratch-side buffer CSVs (results_buffer.csv / errors_buffer.csv)
    so that progress survives a job cancellation.

At the very end the master process performs a single append to the
permanent result files (experiments/results/), keeping write traffic
outside scratch to a minimum.

SCRATCH_DIR is resolved as:
  1. $SCRATCH/cppied_ejor/   (Compute Canada standard env var)
  2. ~/scratch/cppied_ejor/  (fallback)

Usage:
    python run_ejor_experiments_parallel.py \\
        -e /path/to/cppied_executable \\
        --lkh /path/to/LKH \\
        [--workers N]          # default: all available CPUs
"""

import sys
import subprocess
import argparse
import csv
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Resolve layout relative to this script ───────────────────────────────────
SCRIPT_DIR    = Path(__file__).resolve().parent
EXPERIMENTS   = SCRIPT_DIR.parent
DATA_DIR      = EXPERIMENTS / "data"
RESULTS_DIR   = EXPERIMENTS / "results"

CONFIGS_FILE  = DATA_DIR / "algorithm_configuration" / "random_configurations.txt"
INSTANCES_DIR = DATA_DIR / "ejor_tests"
INSTANCES_FILE= INSTANCES_DIR / "run_instances.txt"
RESULTS_FILE  = RESULTS_DIR / "ejor_results.csv"
ERRORS_FILE   = RESULTS_DIR / "ejor_errors.csv"
K_AVG_SCRIPT  = SCRIPT_DIR / "k_avg_performance.py"
K             = 5

RESULTS_COLS = ["name", "solver", "length", "turns", "time"]
ERRORS_COLS  = ["name", "instance", "configuration", "returncode", "stdout", "stderr"]

TIME_MAP = {"s1616_" : 60, "s6464_" : 600, "s128128_" : 1200}


def resolve_scratch_dir() -> Path:
    """Return the scratch working directory, preferring $SCRATCH if set."""
    base = os.environ.get("SCRATCH")
    if base:
        return Path(base) / "cppied_ejor"
    return Path("~/scratch/cppied_ejor").expanduser()


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_configs(path: Path) -> list[str]:
    """Return non-empty, non-comment lines from the configuration file."""
    configs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                configs.append(line)
    return configs

def read_existing_names(path: Path) -> set[str]:
    """Return the set of experiment names already in the results CSV."""
    if not path.exists():
        return set()
    names = set()
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            names.add(row["name"])
    return names


def append_csv_rows(path: Path, fieldnames: list[str], rows: list[dict]):
    """Append *rows* to *path*, writing the header only when the file is new."""
    write_header = not path.exists()
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def write_config_file(
    exp_dir: Path,
    exp_name: str,
    instance_dir: Path,
    algorithm_config: str,
    lkh_exe: str,
) -> Path:
    """Write the CPPIED config file into *exp_dir* and return its path."""
    local_csv   = exp_dir / "results.csv"
    config_path = exp_dir / "config.txt"
    prefix = exp_name.split("_")[0] + "_"
    time = TIME_MAP.get(prefix, 60)
    if prefix not in TIME_MAP:
        import warnings
        warnings.warn(f"Unknown instance prefix '{prefix}'; defaulting TIME to {time}s.")
    config_path.write_text(
        f"NAME = {exp_name}\n"
        f"TIME = {time}\n"
        f"WORKING_DIRECTORY = {exp_dir}\n"
        f"SEABED_FILE = {instance_dir / 'cppied_problem.txt'}\n"
        f"POD_FILE = {instance_dir / 'cppied_pod.txt'}\n"
        f"REQ_COVERAGE_FILE = {instance_dir / 'cppied_req.txt'}\n"
        f"CSV_LOG_FILE = {local_csv}\n"
        f"SOLUTION_FILE = {exp_dir / 'solution.txt'}\n"
        f"LKH_EXECUTABLE = {lkh_exe}\n"
        f"SOLVER = ILS\n"
        f"ALGORITHM_CONFIG = {algorithm_config}\n"
        f"VERBOSE = SUMMARY\n"
    )
    return config_path


def ensure_local_csv(path: Path):
    """Create an empty results CSV (header only) if it does not exist yet."""
    if not path.exists():
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=RESULTS_COLS)
            writer.writeheader()


# ── Worker function (runs in a separate process) ──────────────────────────────

def run_experiment(
    exp_name: str,
    instance_dir: str,      # str so it survives pickling cleanly
    algorithm_config: str,
    lkh_exe: str,
    cppied_exe: str,
    scratch_dir: str,       # str for the same reason
) -> dict:
    """
    Run one (instance, config) pair via k_avg_performance.py.
    All file I/O stays inside *scratch_dir*.

    Returns a dict with keys:
      success    bool
      exp_name   str
      rows       list[dict]  – K result rows on success, empty on failure
      error      dict | None – error metadata on failure, None on success
    """
    instance_dir = Path(instance_dir)
    exp_dir = Path(scratch_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    config_path = write_config_file(
        exp_dir, exp_name, instance_dir, algorithm_config, lkh_exe
    )
    local_csv = exp_dir / "results.csv"
    ensure_local_csv(local_csv)

    proc = subprocess.run(
        [sys.executable, str(K_AVG_SCRIPT), "-e", cppied_exe, "-c", str(config_path)],
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        return {
            "success":  False,
            "exp_name": exp_name,
            "rows":     [],
            "error": {
                "name":          exp_name,
                "instance":      instance_dir.name,
                "configuration": algorithm_config,
                "returncode":    proc.returncode,
                "stdout":        proc.stdout.strip()[:4000],
                "stderr":        proc.stderr.strip()[:4000],
            },
        }

    # Read the K rows appended by k_avg_performance.py
    with open(local_csv, newline="") as fh:
        all_rows = list(csv.DictReader(fh))
    rows = [{col: row[col] for col in RESULTS_COLS} for row in all_rows[-K:]]

    return {
        "success":  True,
        "exp_name": exp_name,
        "rows":     rows,
        "error":    None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parallel CPPIED EJOR experiments (all configs × listed instances)."
    )
    parser.add_argument("-e", "--executable", required=True,
                        help="Absolute path to the CPPIED executable.")
    parser.add_argument("--lkh", required=True,
                        help="Absolute path to the LKH executable.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Number of parallel worker processes (default: all CPUs).")
    args = parser.parse_args()

    scratch_dir = resolve_scratch_dir()
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # Scratch-side buffer CSVs — updated as futures complete so that
    # progress is not lost if the job is cancelled before the final flush.
    scratch_results = scratch_dir / "results_buffer.csv"
    scratch_errors  = scratch_dir / "errors_buffer.csv"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    configs   = read_configs(CONFIGS_FILE)
    run_instances = set(read_configs(INSTANCES_FILE))

    instances = sorted(
        d for d in INSTANCES_DIR.iterdir()
        if d.is_dir() and d.name in run_instances
    )

    if not configs:
        sys.exit("No configurations found – check random_configurations.txt.")
    if not instances:
        sys.exit("No instances found – check the ejor_tests directory.")

    existing = read_existing_names(RESULTS_FILE)

    # Build the pending work list
    pending = []
    for algorithm_config in configs:
        for instance_dir in instances:
            exp_name = f"{instance_dir.name}_{algorithm_config}"
            if exp_name not in existing:
                pending.append((exp_name, str(instance_dir), algorithm_config))

    total_all  = len(configs) * len(instances)
    total_skip = total_all - len(pending)
    total_missed = len(run_instances) - len(instances)
    print(
        f"Configurations : {len(configs)}\n"
        f"Listed instances: {len(instances)}\n"
        f"Not found instances: {total_missed}\n"
        f"Total pairs    : {total_all}\n"
        f"Already done   : {total_skip}\n"
        f"To run         : {len(pending)}\n"
        f"Workers        : {args.workers}\n"
        f"Scratch dir    : {scratch_dir}\n"
    )

    if not pending:
        print("Nothing to do.")
        return

    # In-memory accumulators — flushed to permanent storage once at the end
    accumulated_results: list[dict] = []
    accumulated_errors:  list[dict] = []

    done   = 0
    errors = 0
    width  = len(str(len(pending)))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_experiment,
                exp_name, instance_dir_str, algorithm_config,
                args.lkh, args.executable, str(scratch_dir),
            ): exp_name
            for exp_name, instance_dir_str, algorithm_config in pending
        }

        for future in as_completed(futures):
            exp_name = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"  [EXCEPTION] {exp_name}: {exc}")
                errors += 1
                continue

            done += 1
            if result["success"]:
                accumulated_results.extend(result["rows"])
                # Stream to scratch buffer so progress survives cancellation
                append_csv_rows(scratch_results, RESULTS_COLS, result["rows"])
                print(f"  [{done:{width}}/{len(pending)}] OK    {exp_name}")
            else:
                accumulated_errors.append(result["error"])
                append_csv_rows(scratch_errors, ERRORS_COLS, [result["error"]])
                errors += 1
                rc = result["error"]["returncode"]
                print(f"  [{done:{width}}/{len(pending)}] ERROR {exp_name}  (rc={rc})")

    # ── Single flush to permanent storage ────────────────────────────────────
    print("\nFlushing results to permanent storage …")
    if accumulated_results:
        append_csv_rows(RESULTS_FILE, RESULTS_COLS, accumulated_results)
        print(f"  Written {len(accumulated_results)} result rows → {RESULTS_FILE}")
    if accumulated_errors:
        append_csv_rows(ERRORS_FILE, ERRORS_COLS, accumulated_errors)
        print(f"  Written {len(accumulated_errors)} error rows  → {ERRORS_FILE}")

    print(
        f"\nDone.  Successful: {done - errors}/{len(pending)}  "
        f"Errors: {errors}/{len(pending)}"
    )


if __name__ == "__main__":
    main()
