"""
Ablation analysis (local machine version) following the greedy procedure
of Hoos et al.

Starting from Θ_s = c00000000000 toward a target configuration Θ_t read
from ablation_configurations.txt (first entry), the script performs five
iterations.  At each iteration every bit that is 0 in the current
configuration and 1 in the target is a candidate flip.  The candidate
that yields the best lexicographic mean RAD across all instances is
selected, the configuration is updated, and the process repeats.

The final report lists the full sequence Θ_s … Θ_t with their mean RADs.

Differences from the Compute Canada version
───────────────────────────────────────────
  • Working directories live under experiments/scratch/ (no $SCRATCH).
  • Results are written to permanent storage immediately as each future
    completes (no end-of-phase flush).
  • No scratch-side buffer CSVs.

Usage:
    python run_ablation_analysis_local.py \\
        -e /path/to/cppied_executable \\
        --lkh /path/to/LKH \\
        [--workers N]              # default: all CPUs
        [--instances FILE]         # default: run_instances.txt
        [--ablation-configs FILE]  # default: ablation_configurations.txt
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ── Layout (mirrors the rest of the experiment scripts) ───────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
EXPERIMENTS = SCRIPT_DIR.parent
DATA_DIR    = EXPERIMENTS / "data"
RESULTS_DIR = EXPERIMENTS / "results"
SCRATCH_DIR = EXPERIMENTS / "scratch"

ABLATION_CONFIGS_FILE = DATA_DIR / "algorithm_configuration" / "ablation_configurations.txt"
INSTANCES_DIR         = DATA_DIR / "ejor_tests"
DEFAULT_INSTANCES_FILE = INSTANCES_DIR / "run_instances.txt"

RESULTS_FILE        = RESULTS_DIR / "ejor_ablation_results.csv"
ERRORS_FILE         = RESULTS_DIR / "ejor_ablation_errors.csv"
GLOBAL_RESULTS_FILE = RESULTS_DIR / "ejor_results.csv"

SOURCE_CONFIG = "c00000000000"
N_ITERATIONS  = 5

# Import helpers from the sibling module (works when scripts/ is on sys.path).
sys.path.insert(0, str(SCRIPT_DIR))
from ejor_experiments_helpers import (
    RESULTS_COLS,
    ERRORS_COLS,
    append_csv_rows,
    get_bks,
    read_configs,
    read_existing_names,
    run_experiment,
    run_statistics
)


# ── Ablation helpers ──────────────────────────────────────────────────────────

def read_ablation_configs(path: Path) -> list[str]:
    """Return config strings from ablation_configurations.txt.

    Each non-empty, non-comment line is tab-separated: <index>\\t<config>.
    """
    configs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            configs.append(parts[-1].strip())
    return configs


def candidate_flips(current: str, target: str) -> list[int]:
    """Return bit indices (1-based in string) that are 0 in current and 1 in target."""
    return [i for i in range(1, len(current)) if current[i] == "0" and target[i] == "1"]


def apply_flip(config: str, index: int) -> str:
    return config[:index] + "1" + config[index + 1:]


def read_global_rows(path: Path) -> dict[str, list[dict]]:
    if not path.exists():
        return {}
    rows_by_name: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows_by_name[row["name"]].append(
                {col: row.get(col, "") for col in RESULTS_COLS}
            )
    return dict(rows_by_name)


def run_configs_for_instances(
        cfgs: list[str],
        instances: list[Path],
        existing: set[str],
        args,
        global_rows: dict[str, list[dict]],
        results_rows: dict[str, list[dict]],
) -> list[dict]:
    """Run all (cfg × instance) pairs, reusing cached results when available."""
    accumulated: list[dict] = []
    pending = []

    for instance_dir in instances:
        solver = "ILS"
        for cfg in cfgs:
            exp_name = f"{instance_dir.name}_{cfg}"
            if exp_name in existing:
                if exp_name in results_rows:
                    accumulated.extend(results_rows[exp_name])
                    print(f"  [RESULTS] {exp_name}")
                continue
            existing.add(exp_name)
            if exp_name in global_rows:
                rows = global_rows[exp_name]
                accumulated.extend(rows)
                append_csv_rows(RESULTS_FILE, RESULTS_COLS, rows)
                results_rows[exp_name] = rows
                print(f"  [CACHED]  {exp_name}")
            else:
                pending.append((exp_name, str(instance_dir), cfg, solver))

    if not pending:
        print("  Nothing to run (all pairs cached).")
        return accumulated

    done = 0
    errors = 0
    width = len(str(len(pending)))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_experiment,
                exp_name, instance_dir_str, cfg,
                args.lkh, args.executable, str(SCRATCH_DIR), solver,
            ): exp_name
            for exp_name, instance_dir_str, cfg, solver in pending
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
                accumulated.extend(result["rows"])
                append_csv_rows(RESULTS_FILE,        RESULTS_COLS, result["rows"])
                append_csv_rows(GLOBAL_RESULTS_FILE, RESULTS_COLS, result["rows"])
                global_rows[exp_name]  = result["rows"]
                results_rows[exp_name] = result["rows"]
                print(f"  [{done:{width}}/{len(pending)}] OK    {exp_name}")
            else:
                append_csv_rows(ERRORS_FILE, ERRORS_COLS, [result["error"]])
                errors += 1
                rc = result["error"]["returncode"]
                print(f"  [{done:{width}}/{len(pending)}] ERROR {exp_name}  (rc={rc})")

    print(f"  Done. OK: {done - errors}/{len(pending)}  Errors: {errors}/{len(pending)}")
    return accumulated


def compute_mean_rad(
        config: str,
        all_rows: list[dict],
        bks: dict,
) -> tuple[float, float]:
    """Return (mean_rad_length, mean_rad_turns) for *config* across all instances."""
    pair_values: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        parts = r["name"].split("_")
        if len(parts) < 4:
            continue
        instance = "_".join(parts[:3])
        cfg      = parts[3]
        if cfg == config and instance in bks:
            pair_values[instance].append(r)

    rads = []
    for instance, rows in pair_values.items():
        rad = run_statistics(rows, bks, instance)
        if not any(v != v for v in rad):   # skip NaN
            rads.append(rad)

    if not rads:
        return float("inf"), float("inf")

    arr = np.array(rads)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def best_lexico(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    if a[0] < b[0]:
        return a
    if a[0] == b[0] and a[1] <= b[1]:
        return a
    return b


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ablation analysis (Hoos et al. greedy procedure)."
    )
    parser.add_argument("-e", "--executable", required=True,
                        help="Path to the CPPIED executable.")
    parser.add_argument("--lkh", required=True,
                        help="Path to the LKH executable.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Parallel worker processes (default: all CPUs).")
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES_FILE,
                        help="Instance list file (default: run_instances.txt).")
    parser.add_argument("--ablation-configs", type=Path, default=ABLATION_CONFIGS_FILE,
                        help="Ablation configurations file.")
    args = parser.parse_args()

    # ── Load target configuration ─────────────────────────────────────────────
    ablation_cfgs = read_ablation_configs(args.ablation_configs)
    if not ablation_cfgs:
        sys.exit(f"No configurations found in {args.ablation_configs}")
    target_config = ablation_cfgs[0]

    print(f"Source config  : {SOURCE_CONFIG}")
    print(f"Target config  : {target_config}")
    print(f"Instances file : {args.instances}")
    print(f"Workers        : {args.workers}")

    # Validate configs have the same length
    if len(SOURCE_CONFIG) != len(target_config):
        sys.exit(
            f"Config length mismatch: source={len(SOURCE_CONFIG)}, "
            f"target={len(target_config)}"
        )

    flippable = candidate_flips(SOURCE_CONFIG, target_config)
    print(f"Bits to flip   : {len(flippable)} at positions {flippable}")
    if len(flippable) != N_ITERATIONS:
        print(
            f"[WARN] Expected {N_ITERATIONS} bits to flip, found {len(flippable)}. "
            "Continuing anyway."
        )

    # ── Load instances ────────────────────────────────────────────────────────
    if not GLOBAL_RESULTS_FILE.exists():
        sys.exit(
            f"Global results file not found: {GLOBAL_RESULTS_FILE}\n"
            "Run run_ejor_experiments_parallel_local.py first to generate BKS data."
        )
    BKS = get_bks(GLOBAL_RESULTS_FILE)

    run_instance_names = set(read_configs(args.instances))
    instances = sorted(
        d for d in INSTANCES_DIR.iterdir()
        if d.is_dir() and d.name in run_instance_names
    )
    if not instances:
        sys.exit(f"No instances found under {INSTANCES_DIR} matching {args.instances}")
    print(f"Instances      : {len(instances)}\n")

    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    existing     = read_existing_names(RESULTS_FILE)
    results_rows = read_global_rows(RESULTS_FILE)
    global_rows  = read_global_rows(GLOBAL_RESULTS_FILE)

    # ── Evaluate source configuration ─────────────────────────────────────────
    print(f"── Evaluating source configuration {SOURCE_CONFIG} ──")
    source_rows = run_configs_for_instances(
        [SOURCE_CONFIG], instances, existing, args, global_rows, results_rows
    )
    source_rad = compute_mean_rad(SOURCE_CONFIG, source_rows, BKS)
    print(f"  Source RAD: length={source_rad[0]:.4f}  turns={source_rad[1]:.4f}\n")

    # Track the ablation path: list of (config, (rad_length, rad_turns))
    sequence = [(SOURCE_CONFIG, source_rad)]

    current_config = SOURCE_CONFIG

    # ── Greedy ablation iterations ────────────────────────────────────────────
    for iteration in range(1, N_ITERATIONS + 1):
        if iteration == 1:
            first_target = target_config[:8] + "0000"
            candidates = candidate_flips(current_config, first_target)
        else:
            candidates = candidate_flips(current_config, target_config)
        if not candidates:
            print(f"Iteration {iteration}: no bits left to flip.  Stopping early.")
            break

        candidate_configs = [apply_flip(current_config, i) for i in candidates]

        print(
            f"── Iteration {iteration}/{N_ITERATIONS}  "
            f"(current: {current_config}) ──\n"
            f"  Candidate configs: {candidate_configs}"
        )

        all_rows = run_configs_for_instances(
            candidate_configs, instances, existing, args, global_rows, results_rows
        )

        # Select the candidate with the best lexicographic mean RAD
        best_cfg = None
        best_rad = (float("inf"), float("inf"))
        for cfg in candidate_configs:
            rad = compute_mean_rad(cfg, all_rows, BKS)
            print(f"    {cfg}  rad_length={rad[0]:.4f}  rad_turns={rad[1]:.4f}")
            if best_lexico(rad, best_rad) == rad and rad != best_rad:
                best_cfg = cfg
                best_rad = rad

        if best_cfg is None:
            print("  [WARN] No valid RAD computed for any candidate; stopping.")
            break

        print(f"\n  Selected: {best_cfg}  RAD=({best_rad[0]:.4f}, {best_rad[1]:.4f})\n")
        current_config = best_cfg
        sequence.append((current_config, best_rad))

    # ── Evaluate target configuration (if not already reached) ───────────────
    if current_config != target_config:
        print(f"── Evaluating target configuration {target_config} ──")
        target_rows = run_configs_for_instances(
            [target_config], instances, existing, args, global_rows, results_rows
        )
        target_rad = compute_mean_rad(target_config, target_rows, BKS)
        sequence.append((target_config, target_rad))
        print(f"  Target RAD: length={target_rad[0]:.4f}  turns={target_rad[1]:.4f}\n")

    # ── Report ────────────────────────────────────────────────────────────────
    col_w = max(len(c) for c, _ in sequence)
    header = f"{'Configuration':<{col_w}}  {'RAD length':>12}  {'RAD turns':>12}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print("Ablation path")
    print(sep)
    print(header)
    print(sep)
    for i, (cfg, (rl, rt)) in enumerate(sequence):
        label = cfg
        if i == 0:
            label += "  (Θ_s)"
        elif i == len(sequence) - 1 and cfg == target_config:
            label += "  (Θ_t)"
        print(f"{label:<{col_w + 8}}  {rl:>12.4f}  {rt:>12.4f}")
    print(sep)

    # Save ablation path to CSV
    ablation_summary = RESULTS_DIR / "ejor_ablation_path.csv"
    with open(ablation_summary, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["step", "config", "rad_length", "rad_turns"])
        writer.writeheader()
        for i, (cfg, (rl, rt)) in enumerate(sequence):
            writer.writerow({"step": i, "config": cfg, "rad_length": rl, "rad_turns": rt})
    print(f"\nAblation path saved → {ablation_summary}")


if __name__ == "__main__":
    main()
