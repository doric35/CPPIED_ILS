"""
run_visualization_experiments.py

Run one solve per (instance, config) pair from a list file, with full
SOLUTION and VISUALIZE output enabled.  Results are written to
experiments/scratch/<exp_name>/ and are never appended to the shared
results CSV.

Input file format (one pair per line, blank lines and #-comments ignored):
    s1616_is0_lrc181_c01111011010
    s1616_ir2_lrc053_c10101011010
    ...

The instance part is everything up to the trailing _c[01]{11} token,
which is the algorithm configuration.

Usage:
    python run_visualization_experiments.py \\
        -e /path/to/Cppied        \\
        --lkh /path/to/LKH        \\
        --pairs pairs.txt         \\
        [--workers N]             # default: 1 (sequential)
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ── Layout relative to this script ───────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).resolve().parent
EXPERIMENTS   = SCRIPT_DIR.parent
DATA_DIR      = EXPERIMENTS / "data"
SCRATCH_DIR   = EXPERIMENTS / "scratch"
INSTANCES_DIR = DATA_DIR / "ejor_tests"

# Regex that splits  s1616_is0_lrc181_c01111011010
#                 → ("s1616_is0_lrc181", "c01111011010")
_PAIR_RE = re.compile(r"^(.+)_(c[01]{11})$")


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_pairs_file(path: Path) -> list[tuple[str, str]]:
    """
    Read a pairs file and return a list of (instance_name, algorithm_config).
    Blank lines and lines starting with '#' are ignored.
    Raises ValueError on lines that don't match the expected format.
    """
    pairs = []
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = _PAIR_RE.match(line)
            if not m:
                raise ValueError(
                    f"{path}:{lineno}: cannot parse pair {line!r} — "
                    "expected format  <instance>_c<11 binary digits>"
                )
            pairs.append((m.group(1), m.group(2)))
    return pairs


def write_config_file(
    exp_dir: Path,
    exp_name: str,
    instance_dir: Path,
    algorithm_config: str,
    lkh_exe: str,
) -> Path:
    """Write the CPPIED config file into *exp_dir* and return its path."""
    config_path = exp_dir / "config.txt"
    config_path.write_text(
        f"NAME = {exp_name}\n"
        f"TIME = 60\n"
        f"WORKING_DIRECTORY = {exp_dir}\n"
        f"SEABED_FILE = {instance_dir / 'cppied_problem.txt'}\n"
        f"POD_FILE = {instance_dir / 'cppied_pod.txt'}\n"
        f"REQ_COVERAGE_FILE = {instance_dir / 'cppied_req.txt'}\n"
        f"CSV_LOG_FILE = {exp_dir / 'results.csv'}\n"
        f"SOLUTION_FILE = {exp_dir / 'solution.txt'}\n"
        f"VIZ_FILE = {exp_dir / 'visualisation.avi'}\n"
        f"LKH_EXECUTABLE = {lkh_exe}\n"
        f"SOLVER = ILS\n"
        f"ALGORITHM_CONFIG = {algorithm_config}\n"
        f"VERBOSE = SUMMARY,SOLUTION,VISUALIZE\n"
    )
    return config_path


# ── Worker (runs in a separate process when --workers > 1) ───────────────────

def run_experiment(
    instance_name: str,
    algorithm_config: str,
    lkh_exe: str,
    cppied_exe: str,
    scratch_dir: str,       # str for clean pickling
    instances_dir: str,
) -> dict:
    """
    Set up and execute one (instance, config) pair.
    Returns a status dict with keys: success, exp_name, stdout, stderr, returncode.
    """
    scratch  = Path(scratch_dir)
    inst_dir = Path(instances_dir) / instance_name
    exp_name = f"{instance_name}_{algorithm_config}"
    exp_dir  = scratch / exp_name

    if not inst_dir.is_dir():
        return {
            "success":    False,
            "exp_name":   exp_name,
            "returncode": None,
            "stdout":     "",
            "stderr":     f"Instance directory not found: {inst_dir}",
        }

    exp_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_config_file(
        exp_dir, exp_name, inst_dir, algorithm_config, lkh_exe
    )

    proc = subprocess.run(
        [cppied_exe, str(config_path)],
        capture_output=True,
        text=True,
    )

    return {
        "success":    proc.returncode == 0,
        "exp_name":   exp_name,
        "returncode": proc.returncode,
        "stdout":     proc.stdout.strip(),
        "stderr":     proc.stderr.strip(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run one visualization solve per (instance, config) pair. "
            "Results are written to experiments/scratch/ only."
        )
    )
    parser.add_argument(
        "-e", "--executable", required=True,
        help="Absolute path to the CPPIED executable.",
    )
    parser.add_argument(
        "--lkh", required=True,
        help="Absolute path to the LKH executable.",
    )
    parser.add_argument(
        "--pairs", required=True,
        help="Path to a text file listing one instance_config pair per line.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1 — sequential).",
    )
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        sys.exit(f"ERROR: pairs file not found: {pairs_path}")

    pairs = parse_pairs_file(pairs_path)
    if not pairs:
        sys.exit("No pairs found in the input file.")

    # Ensure scratch directory exists
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Pairs to run  : {len(pairs)}\n"
        f"Workers       : {args.workers}\n"
        f"Scratch dir   : {SCRATCH_DIR}\n"
    )

    done = errors = 0
    width = len(str(len(pairs)))

    def _dispatch(pool: ProcessPoolExecutor | None):
        futures = {}
        for instance_name, algorithm_config in pairs:
            kwargs = dict(
                instance_name=instance_name,
                algorithm_config=algorithm_config,
                lkh_exe=args.lkh,
                cppied_exe=args.executable,
                scratch_dir=str(SCRATCH_DIR),
                instances_dir=str(INSTANCES_DIR),
            )
            if pool is not None:
                futures[pool.submit(run_experiment, **kwargs)] = (
                    f"{instance_name}_{algorithm_config}"
                )
            else:
                # Sequential: run inline and yield immediately
                yield None, run_experiment(**kwargs)
        # Parallel: yield futures as they complete
        if pool is not None:
            for future in as_completed(futures):
                yield future, futures[future]

    nonlocal_done   = [0]
    nonlocal_errors = [0]

    def handle(result: dict):
        nonlocal_done[0] += 1
        tag = f"[{nonlocal_done[0]:{width}}/{len(pairs)}]"
        if result["success"]:
            print(f"  {tag} OK    {result['exp_name']}")
            print(f"           → {SCRATCH_DIR / result['exp_name']}")
        else:
            nonlocal_errors[0] += 1
            rc = result["returncode"]
            print(f"  {tag} ERROR {result['exp_name']}  (rc={rc})")
            if result["stderr"]:
                for line in result["stderr"].splitlines()[:10]:
                    print(f"           {line}")

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future, name in _dispatch(pool):
                try:
                    handle(future.result())
                except Exception as exc:
                    nonlocal_done[0] += 1
                    nonlocal_errors[0] += 1
                    print(f"  [EXCEPTION] {name}: {exc}")
    else:
        for _, result in _dispatch(None):
            handle(result)

    done   = nonlocal_done[0]
    errors = nonlocal_errors[0]
    print(
        f"\nDone.  Successful: {done - errors}/{len(pairs)}  "
        f"Errors: {errors}/{len(pairs)}"
    )


if __name__ == "__main__":
    main()
