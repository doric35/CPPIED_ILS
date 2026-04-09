"""
Run CPPIED K-avg experiments for every algorithm configuration
crossed with every small (s1616) EJOR test instance.

Usage:
    python run_ejor_experiments.py \\
        -e /path/to/cppied_executable \\
        --lkh /path/to/LKH

Outputs:
    experiments/results/ejor_results.csv  – appended with new K=200 rows per experiment
    experiments/results/ejor_errors.csv   – one row per failed experiment
"""

import sys
import subprocess
import argparse
from pathlib import Path

import pandas as pd

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
K             = 5          # must match the K defined in k_avg_performance.py

RESULTS_COLS  = ["name", "solver", "length", "turns", "time"]
ERRORS_COLS   = ["name", "instance", "configuration", "returncode", "stdout", "stderr"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def read_configs(path: Path) -> list:
    """Return non-empty, non-comment lines from the configuration file."""
    configs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                configs.append(line)
    return configs


def read_existing_names(path: Path) -> set:
    """Return the set of experiment names already present in the results CSV."""
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty or "name" not in df.columns:
        return set()
    return set(df["name"].unique())


def append_to_csv(path: Path, df: pd.DataFrame, columns: list):
    """Append *df* to *path*, writing the header only when the file is new."""
    write_header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=write_header, columns=columns)


def write_config(
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
    """Create an empty results CSV (with header) if it does not yet exist."""
    if not path.exists():
        pd.DataFrame(columns=RESULTS_COLS).to_csv(path, index=False)


def run_experiment(
    exp_name: str,
    instance_dir: Path,
    algorithm_config: str,
    lkh_exe: str,
    cppied_exe: str,
) -> tuple:
    """
    Set up the experiment directory, call k_avg_performance.py, and return
    (new_rows_df, proc) where *new_rows_df* is None on failure.
    """
    exp_dir = SCRATCH_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    config_path = write_config(exp_dir, exp_name, instance_dir, algorithm_config, lkh_exe)
    local_csv   = exp_dir / "results.csv"
    ensure_local_csv(local_csv)

    proc = subprocess.run(
        [sys.executable, str(K_AVG_SCRIPT), "-e", cppied_exe, "-c", str(config_path)],
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        return None, proc

    # k_avg appends K rows; since we started from an empty CSV the whole file
    # contains exactly the new results – but we take tail(K) defensively.
    new_df = pd.read_csv(local_csv).tail(K)
    return new_df, proc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run CPPIED EJOR experiments (all configs × all s1616 instances)."
    )
    parser.add_argument(
        "-e", "--executable",
        required=True,
        help="Absolute path to the CPPIED executable.",
    )
    parser.add_argument(
        "--lkh",
        required=True,
        help="Absolute path to the LKH executable.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
    total    = len(configs) * len(instances)

    print(
        f"Configurations : {len(configs)}\n"
        f"s1616 instances: {len(instances)}\n"
        f"Total planned  : {total}\n"
        f"Already done   : {len(existing)} experiment name(s) in {RESULTS_FILE.name}\n"
    )

    counter = 0
    for algorithm_config in configs:
        for instance_dir in instances:
            counter += 1
            exp_name = f"{instance_dir.name}_{algorithm_config}"

            if exp_name in existing:
                print(f"[{counter:>{len(str(total))}}/{total}] SKIP  {exp_name}")
                continue

            print(f"[{counter:>{len(str(total))}}/{total}] RUN   {exp_name} ...", end=" ", flush=True)

            new_df, proc = run_experiment(
                exp_name,
                instance_dir,
                algorithm_config,
                args.lkh,
                args.executable,
            )

            if new_df is None:
                print(f"ERROR (rc={proc.returncode})")
                err_row = pd.DataFrame([{
                    "name"          : exp_name,
                    "instance"      : instance_dir.name,
                    "configuration" : algorithm_config,
                    "returncode"    : proc.returncode,
                    "stdout"        : proc.stdout.strip()[:4000],
                    "stderr"        : proc.stderr.strip()[:4000],
                }])
                append_to_csv(ERRORS_FILE, err_row, ERRORS_COLS)
            else:
                print(f"OK ({len(new_df)} rows)")
                append_to_csv(RESULTS_FILE, new_df, RESULTS_COLS)
                existing.add(exp_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
