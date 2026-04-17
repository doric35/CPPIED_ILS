"""
ILS configuration local search — local machine version.

Runs a local search over ILS algorithm configurations.  At each
iteration three phases are applied in order to the current incumbent:

  1. Neighborhood — each 0-bit in positions 1–7 is flipped to 1 (add move).
  2. Perturbation  — each bit in positions 8–10 is flipped (flip move).
  3. Restart       — the last bit is flipped (flip move).

Each phase generates a set of candidate configurations, runs every
(candidate × instance) pair that has not already been evaluated, and
proposes a new incumbent as the lexicographically best mean RAD across
all instances.  The incumbent is replaced only when a strictly better
candidate is found.

The search terminates after 24 h or when the incumbent did not change
over a full iteration.

Differences from the Compute Canada version
───────────────────────────────────────────
  • Working directories live under experiments/scratch/ (no $SCRATCH).
  • Results are written to permanent storage immediately as each future
    completes (no end-of-phase flush).
  • No scratch-side buffer CSVs.

Usage:
    python run_ejor_ls_config_parallel_local.py \\
        -e /path/to/cppied_executable \\
        --lkh /path/to/LKH \\
        [--workers N]          # default: all available CPUs
"""
import numpy as np

from ejor_experiments_helpers import *
import time

SCRATCH_DIR      = EXPERIMENTS / "scratch"

ILS_CONFIGS_FILE = DATA_DIR / "algorithm_configuration" / "ils_configuration.txt"
RESULTS_FILE     = RESULTS_DIR / "ejor_configuration_results.csv"
ERRORS_FILE      = RESULTS_DIR / "ejor_configuration_errors.csv"
BEST_CONFIG_FILE = RESULTS_DIR / "ejor_best_config.txt"

GLOBAL_RESULTS_FILE = RESULTS_DIR / "ejor_results.csv"


# ── Search helpers ────────────────────────────────────────────────────────────

def neighborhood_experiments_configs(source_config: str) -> list[str]:
    """Return configs obtained by flipping each 0-bit in positions 1-7 to 1."""
    algo_cfgs = []
    for i in range(1, 8):
        if int(source_config[i]) == 0:
            new_config = source_config[:i] + "1" + source_config[i + 1:]
            algo_cfgs.append(new_config)
    return algo_cfgs


def perturbation_experiments_configs(source_config: str) -> list[str]:
    """Return configs obtained by flipping each bit in positions 8-10."""
    algo_cfgs = []
    for i in range(8, 11):
        bit = "0" if int(source_config[i]) == 1 else "1"
        new_config = source_config[:i] + bit + source_config[i + 1:]
        algo_cfgs.append(new_config)
    return algo_cfgs


def restart_experiments_configs(source_config: str) -> list[str]:
    """Return the config obtained by flipping the last bit."""
    bit = "0" if int(source_config[-1]) == 1 else "1"
    return [source_config[:-1] + bit]


def stopping_criteria(start: float, prev_incumbent: str, incumbent: str) -> bool:
    """Return True when time budget is exhausted or the search has converged."""
    return time.time() - start >= 86400 or prev_incumbent == incumbent


def run_and_save_configurations(
        cfgs: list[str],
        instances,
        existing: set[str],
        solver: str,
        args,
) -> list[dict]:
    """Run all (cfg × instance) pairs not yet in *existing*.

    Results and errors are written to permanent storage immediately as
    each future completes.  Returns the accumulated result rows (may be
    empty).
    """
    pending = []
    for instance_dir in instances:
        for algorithm_config in cfgs:
            exp_name = f"{instance_dir.name}_{algorithm_config}"
            if exp_name not in existing:
                pending.append((exp_name, str(instance_dir), algorithm_config, solver))
                existing.add(exp_name)

    if not pending:
        print("Nothing to do.")
        return []

    accumulated_results: list[dict] = []
    done   = 0
    errors = 0
    width  = len(str(len(pending)))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_experiment,
                exp_name, instance_dir_str, algorithm_config,
                args.lkh, args.executable, str(SCRATCH_DIR), solver,
            ): exp_name
            for exp_name, instance_dir_str, algorithm_config, solver in pending
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
                # Write immediately so progress survives interruption
                append_csv_rows(RESULTS_FILE, RESULTS_COLS, result["rows"])
                print(f"  [{done:{width}}/{len(pending)}] OK    {exp_name}")
            else:
                append_csv_rows(ERRORS_FILE, ERRORS_COLS, [result["error"]])
                errors += 1
                rc = result["error"]["returncode"]
                print(f"  [{done:{width}}/{len(pending)}] ERROR {exp_name}  (rc={rc})")

    print(
        f"\nDone.  Successful: {done - errors}/{len(pending)}  "
        f"Errors: {errors}/{len(pending)}"
    )
    return accumulated_results


def best_lexico(
        a: list,   # [config_str, (rad_length, rad_turns)]
        b: list,
) -> list:
    if a[1][0] < b[1][0]:
        return a
    if a[1][0] == b[1][0] and a[1][1] <= b[1][1]:
        return a
    return b


def incumbent_configuration(
        results: list[dict],
        bks: dict[str, list],
) -> list | None:
    """Return [best_config_str, (mean_rad_length, mean_rad_turns)], or None.

    *results* rows have keys from RESULTS_COLS:
      name, solver, length, turns, time
    The instance and config are extracted by splitting ``name``.
    Returns None when no valid RAD can be computed (all rows malformed,
    missing BKS, or producing NaN).
    """
    # Group rows by (config, instance); skip malformed names
    pair_values: dict[tuple, list] = defaultdict(list)
    for r in results:
        parts = r["name"].split("_")
        if len(parts) < 4:
            print(f"  [WARN] Skipping malformed result name: {r['name']!r}")
            continue
        instance = "_".join(parts[:3])   # e.g. s6464_ir0_lrc031
        config   = parts[3]              # e.g. c10110001011
        if instance not in bks:
            print(f"  [WARN] No BKS entry for instance {instance!r}; skipping.")
            continue
        pair_values[(config, instance)].append(r)

    if not pair_values:
        return None

    # Aggregate per-instance RADs for each config
    config_rads: dict[str, list] = defaultdict(list)
    for (config, instance), values in pair_values.items():
        rad = run_statistics(values, bks, instance)
        if any(v != v for v in rad):   # NaN check
            print(f"  [WARN] NaN RAD for config={config} instance={instance}; skipping.")
            continue
        config_rads[config].append(rad)

    if not config_rads:
        return None

    # Pick the lexicographically best config across all instances
    best = ["", (float("inf"), float("inf"))]
    for config, rads in config_rads.items():
        np_rads  = np.array(rads)
        mean_rad = (float(np_rads[:, 0].mean()), float(np_rads[:, 1].mean()))
        best = best_lexico(best, [config, mean_rad])

    return best if best[0] else None


def save_best_config(config: str, rad: tuple, path: Path):
    """Persist the best configuration and its mean RAD to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# Best ILS configuration found by local search\n"
        f"config          = {config}\n"
        f"mean_rad_length = {rad[0]:.6f}\n"
        f"mean_rad_turns  = {rad[1]:.6f}\n"
    )
    print(f"  Best configuration saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Local ILS configuration search (local machine version)."
    )
    parser.add_argument("-e", "--executable", required=True,
                        help="Absolute path to the CPPIED executable.")
    parser.add_argument("--lkh", required=True,
                        help="Absolute path to the LKH executable.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Number of parallel worker processes (default: all CPUs).")
    args = parser.parse_args()

    # Guard: read configs before accessing ils_configs[0]
    ils_configs = read_configs(ILS_CONFIGS_FILE)
    if not ils_configs:
        sys.exit("No ILS configurations found – check ils_configuration.txt.")

    if not GLOBAL_RESULTS_FILE.exists():
        sys.exit(
            f"Global results file not found: {GLOBAL_RESULTS_FILE}\n"
            "Run run_ejor_experiments_parallel_local.py first to generate BKS data."
        )
    BKS = get_bks(GLOBAL_RESULTS_FILE)

    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_instances = set(read_configs(INSTANCES_FILE))
    instances = sorted(
        d for d in INSTANCES_DIR.iterdir()
        if d.is_dir() and d.name in run_instances
    )

    if not instances:
        sys.exit("No instances found – check the ejor_tests directory.")

    existing = read_existing_names(RESULTS_FILE)

    solver           = "ILS"
    start_time       = time.time()
    prev_incumbent   = ""
    incumbent_config = [ils_configs[0], (float("inf"), float("inf"))]

    print(
        f"Listed instances     : {len(instances)}\n"
        f"Workers              : {args.workers}\n"
        f"Scratch dir          : {SCRATCH_DIR}\n"
        f"Starting config      : {incumbent_config[0]}\n"
    )

    while not stopping_criteria(start_time, prev_incumbent, incumbent_config[0]):
        prev_incumbent = incumbent_config[0]

        # Neighborhood phase — flip 0-bits one at a time in positions 1-7
        print(f"\n── Neighborhood phase  (incumbent: {incumbent_config[0]}) ──")
        cfgs = neighborhood_experiments_configs(incumbent_config[0])
        results = run_and_save_configurations(cfgs, instances, existing, solver, args)
        candidate = incumbent_configuration(results, BKS) if results else None
        if candidate:
            incumbent_config = candidate

        # Perturbation phase — flip each bit in positions 8-10
        print(f"\n── Perturbation phase  (incumbent: {incumbent_config[0]}) ──")
        cfgs = perturbation_experiments_configs(incumbent_config[0])
        results = run_and_save_configurations(cfgs, instances, existing, solver, args)
        candidate = incumbent_configuration(results, BKS) if results else None
        if candidate:
            incumbent_config = candidate

        # Restart phase — flip the last bit
        print(f"\n── Restart phase       (incumbent: {incumbent_config[0]}) ──")
        cfgs = restart_experiments_configs(incumbent_config[0])
        results = run_and_save_configurations(cfgs, instances, existing, solver, args)
        candidate = incumbent_configuration(results, BKS) if results else None
        if candidate:
            incumbent_config = candidate

        print(
            f"\nEnd of iteration.  Incumbent: {incumbent_config[0]}  "
            f"RAD=({incumbent_config[1][0]:.4f}, {incumbent_config[1][1]:.4f})\n"
        )

    print(
        f"\nSearch finished.  Best config: {incumbent_config[0]}  "
        f"RAD=({incumbent_config[1][0]:.4f}, {incumbent_config[1][1]:.4f})"
    )
    save_best_config(incumbent_config[0], incumbent_config[1], BEST_CONFIG_FILE)


if __name__ == "__main__":
    main()
