"""
Parallel version of run_ejor_experiments.py.

Distributes all (algorithm_config × s1616 instance) pairs across worker
processes (one pair per worker at a time).  Already-completed pairs are
skipped by checking the existing results CSV before launching any work.

Each worker writes its K rows to an isolated per-task CSV file under
experiments/scratch/<exp_name>/results.csv; after all futures finish,
the master process appends every successful result to the global CSV in
one go, avoiding any concurrent-write race conditions.

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
SCRATCH_DIR   = EXPERIMENTS / "scratch"
RESULTS_DIR   = EXPERIMENTS / "results"

CONFIGS_FILE  = DATA_DIR / "algorithm_configuration" / "random_configurations.txt"
INSTANCES_DIR = DATA_DIR / "ejor_tests"
RESULTS_FILE  = RESULTS_DIR / "ejor_results.csv"
ERRORS_FILE   = RESULTS_DIR / "ejor_errors.csv"
K_AVG_SCRIPT  = SCRIPT_DIR / "k_avg_performance.py"
K             = 5

RESULTS_COLS = ["name", "solver", "length", "turns", "time"]
ERRORS_COLS  = ["name", "instance", "configuration", "returncode", "stdout", "stderr"]


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


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict],
                   write_header: bool):
    """Append a list of row dicts to *path*."""
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
    config_path.write_text(
        f"NAME = {exp_name}\n"
        f"TIME = 60\n"
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
    instance_dir: str,      # passed as str so it survives pickling cleanly
    algorithm_config: str,
    lkh_exe: str,
    cppied_exe: str,
) -> dict:
    """
    Run one (instance, config) pair via k_avg_performance.py.

    Returns a dict with keys:
      success    bool
      exp_name   str
      rows       list[dict]  – K result rows on success, empty on failure
      error      dict | None – error metadata on failure, None on success
    """
    instance_dir = Path(instance_dir)
    exp_dir = SCRATCH_DIR / exp_name
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
    rows = []
    with open(local_csv, newline="") as fh:
        all_rows = list(csv.DictReader(fh))
    for row in all_rows[-K:]:
        rows.append({col: row[col] for col in RESULTS_COLS})

    return {
        "success":  True,
        "exp_name": exp_name,
        "rows":     rows,
        "error":    None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parallel CPPIED EJOR experiments (all configs × all s1616 instances)."
    )
    parser.add_argument("-e", "--executable", required=True,
                        help="Absolute path to the CPPIED executable.")
    parser.add_argument("--lkh", required=True,
                        help="Absolute path to the LKH executable.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Number of parallel worker processes (default: all CPUs).")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    configs   = read_configs(CONFIGS_FILE)
    instances = sorted(
        d for d in INSTANCES_DIR.iterdir()
        if d.is_dir() and d.name.startswith("s1616_")
    )

    if not configs:
        sys.exit("No configurations found – check random_configurations.txt.")
    if not instances:
        sys.exit("No s1616 instances found – check the ejor_tests directory.")

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
    print(
        f"Configurations : {len(configs)}\n"
        f"s1616 instances: {len(instances)}\n"
        f"Total pairs    : {total_all}\n"
        f"Already done   : {total_skip}\n"
        f"To run         : {len(pending)}\n"
        f"Workers        : {args.workers}\n"
    )

    if not pending:
        print("Nothing to do.")
        return

    results_exist = RESULTS_FILE.exists()
    errors_exist  = ERRORS_FILE.exists()

    done = 0
    errors = 0
    width = len(str(len(pending)))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_experiment,
                exp_name, instance_dir_str, algorithm_config,
                args.lkh, args.executable,
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
                write_csv_rows(
                    RESULTS_FILE, RESULTS_COLS, result["rows"],
                    write_header=not results_exist,
                )
                results_exist = True
                print(f"  [{done:{width}}/{len(pending)}] OK    {exp_name}")
            else:
                write_csv_rows(
                    ERRORS_FILE, ERRORS_COLS, [result["error"]],
                    write_header=not errors_exist,
                )
                errors_exist = True
                errors += 1
                rc = result["error"]["returncode"]
                print(f"  [{done:{width}}/{len(pending)}] ERROR {exp_name}  (rc={rc})")

    print(
        f"\nDone.  Successful: {done - errors}/{len(pending)}  "
        f"Errors: {errors}/{len(pending)}"
    )


if __name__ == "__main__":
    main()
