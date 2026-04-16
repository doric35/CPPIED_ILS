"""
This file contains ejor experiment helpers that are used
to run experiments.
"""

import sys
import subprocess
import argparse
import csv
from collections import defaultdict
import os
from pathlib import Path
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Resolve layout relative to this script ───────────────────────────────────
SCRIPT_DIR    = Path(__file__).resolve().parent
EXPERIMENTS   = SCRIPT_DIR.parent
DATA_DIR      = EXPERIMENTS / "data"
RESULTS_DIR   = EXPERIMENTS / "results"

INSTANCES_DIR = DATA_DIR / "ejor_tests"
INSTANCES_FILE= INSTANCES_DIR / "run_instances.txt"
K_AVG_SCRIPT  = SCRIPT_DIR / "k_avg_performance.py"
K             = 5

RESULTS_COLS = ["name", "solver", "length", "turns", "time"]
ERRORS_COLS  = ["name", "instance", "configuration", "returncode", "stdout", "stderr"]

TIME_MAP   = {"s1616_": 60,    "s6464_": 600,  "s128128_": 1200}
# Default solver per instance-size prefix.  Small instances are solved
# exactly with GUROBI; larger ones use the ILS metaheuristic.
SOLVER_MAP = {"s1616_": "GUROBI", "s6464_": "ILS", "s128128_": "ILS"}

#------Solvers configurations helpers--------------
def resolve_solver(prefix: str, override: str) -> str:
    """Return the solver to use for a given instance prefix.

    *override* is the value of --solver: "auto", "ILS", or "GUROBI".
    When "auto", the default from SOLVER_MAP is used; unknown prefixes
    fall back to ILS with a warning.
    """
    if override != "auto":
        return override
    if prefix not in SOLVER_MAP:
        import warnings
        warnings.warn(f"Unknown instance prefix '{prefix}'; defaulting solver to ILS.")
        return "ILS"
    return SOLVER_MAP[prefix]


def configs_for_solver(solver: str,
                       ils_configs: list[str],
                       gurobi_configs: list[str]) -> list[str]:
    """Return the configuration list to iterate for a given solver."""
    return gurobi_configs if solver == "GUROBI" else ils_configs

#--------File system helpers---------------------------

def resolve_scratch_dir() -> Path:
    """Return the scratch working directory, preferring $SCRATCH if set."""
    base = os.environ.get("SCRATCH")
    if base:
        return Path(base) / "cppied_ejor"
    return Path("~/scratch/cppied_ejor").expanduser()
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
        solver: str,
) -> Path:
    """Write the CPPIED config file into *exp_dir* and return its path.

    ALGORITHM_CONFIG is always written with *algorithm_config*, whose
    format depends on the solver:
      ILS    — multi-bit string from random_configurations.txt
      GUROBI — single-bit string from gurobi_configurations.txt (e.g. "c0", "c1")
    """
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
        f"SOLVER = {solver}\n"
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


#-------Executable launch and post treatments--------------------------
# ── Worker function (runs in a separate process) ──────────────────────────────

def run_experiment(
        exp_name: str,
        instance_dir: str,      # str so it survives pickling cleanly
        algorithm_config: str,
        lkh_exe: str,
        cppied_exe: str,
        scratch_dir: str,       # str for the same reason
        solver: str = "ILS",
) -> dict:
    """
    Run one experiment via k_avg_performance.py.
    All file I/O stays inside *scratch_dir*.

    For ILS the pair is (instance × algorithm_config);.

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
        exp_dir, exp_name, instance_dir, algorithm_config, lkh_exe, solver
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

def run_statistics(results : list[dict], bks: dict[str, (int, int)], instance : str):
    """
    :param instance: Instance name as string
    :param bks: dictionnary mapping instance to best known (length, turns) of an instance
    :param results: [name,solver,length,turns,time]
    :return: [RAD length, RAD turns]
    """
    values = [float(r["length"]) for r in results]
    avg_val = statistics.mean(values)
    l_rad = (avg_val - bks[instance][0]) / bks[instance][0] * 100
    values = [float(r["turns"]) for r in results]
    avg_val = statistics.mean(values)
    t_rad = (avg_val - bks[instance][1]) / bks[instance][1] * 100
    return l_rad, t_rad

def get_bks(csv_results : Path):
    """Return {instance: [best_length, best_turns]} for the given metric field."""
    rows = []
    with open(csv_results, newline="") as fh:
        for row in csv.DictReader(fh):
            # name = {size}_{typeRID}_{lrcRID}_{config}
            # e.g.  s1616_ir0_lrc031_c11111111111
            parts    = row["name"].split("_")
            instance = "_".join(parts[:3])   # s1616_ir0_lrc031
            config   = parts[3]              # c11111111111
            rows.append({
                "instance": instance,
                "config":   config,
                "length":   int(row["length"]),
                "turns":    int(row["turns"]),
            })
    bks: dict[str, (int, int)] = defaultdict(lambda: [float("inf"), float("inf")])

    for r in rows:
        if r["length"] < bks[r["instance"]][0]:
            bks[r["instance"]][0] = r["length"]
        if r["turns"] < bks[r["instance"]][1]:
            bks[r["instance"]][1] = r["turns"]

    return bks
