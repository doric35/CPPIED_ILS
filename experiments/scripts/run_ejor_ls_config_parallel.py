"""
Distributes all (local_algorithm_config × instance) pairs across worker
processes (one pair per worker at a time).  Already-completed pairs are
skipped by checking the existing results CSV before launching any work.

ILS selection
────────────────
The ILS config is chosen automatically at each iteration based on
its size prefix, following SOLVER_MAP.

ILS experiments iterate over each instances in the configuration set.
Each ILS configxinstance is run K times; the experiment name is  <instance>_<config>.

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
    python run_ejor_ls_config_parallel.py \\
        -e /path/to/cppied_executable \\
        --lkh /path/to/LKH \\
        [--workers N]          # default: all available CPUs
"""
import numpy as np

from ejor_experiments_helpers import *
import time

ILS_CONFIGS_FILE    = DATA_DIR / "algorithm_configuration" / "ils_configuration.txt"
RESULTS_FILE        = RESULTS_DIR / "ejor_configuration_results.csv"
ERRORS_FILE         = RESULTS_DIR / "ejor_configuration_errors.csv"
BEST_CONFIG_FILE    = RESULTS_DIR / "ejor_best_config.txt"

GLOBAL_RESULTS_FILE = RESULTS_DIR / "ejor_results.csv"

# ── Main ──────────────────────────────────────────────────────────────────────

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
        scratch_dir,
        solver: str,
        scratch_results: Path,
        scratch_errors: Path,
        args,
) -> list[dict]:
    """Run all (cfg × instance) pairs that are not yet in *existing*.

    Returns the accumulated result rows (may be empty).
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
                args.lkh, args.executable, str(scratch_dir), solver,
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
                append_csv_rows(scratch_results, RESULTS_COLS, result["rows"])
                print(f"  [{done:{width}}/{len(pending)}] OK    {exp_name}")
            else:
                accumulated_errors.append(result["error"])
                append_csv_rows(scratch_errors, ERRORS_COLS, [result["error"]])
                errors += 1
                rc = result["error"]["returncode"]
                print(f"  [{done:{width}}/{len(pending)}] ERROR {exp_name}  (rc={rc})")

    # ── Flush to permanent storage ─────────────────────────────────────────────
    print("\nFlushing results to permanent storage…")
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
) -> list:
    """Return [best_config_str, (mean_rad_length, mean_rad_turns)].

    *results* rows have keys from RESULTS_COLS:
      name, solver, length, turns, time
    The instance and config are extracted by splitting ``name``.
    """
    # Group rows by (config, instance)
    pair_values: dict[tuple, list] = defaultdict(list)
    for r in results:
        parts    = r["name"].split("_")
        instance = "_".join(parts[:3])   # e.g. s6464_ir0_lrc031
        config   = parts[3]              # e.g. c10110001011
        pair_values[(config, instance)].append(r)

    # Aggregate per-instance RADs for each config
    config_rads: dict[str, list] = defaultdict(list)
    for (config, instance), values in pair_values.items():
        rad = run_statistics(values, bks, instance)
        config_rads[config].append(rad)

    # Pick the lexicographically best config across all instances
    best = ["", (float("inf"), float("inf"))]
    for config, rads in config_rads.items():
        np_rads = np.array(rads)
        mean_rad = (float(np_rads[:, 0].mean()), float(np_rads[:, 1].mean()))
        best = best_lexico(best, [config, mean_rad])

    return best


def save_best_config(config: str, rad: tuple, path: Path):
    """Persist the best configuration and its mean RAD to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# Best ILS configuration found by local search\n"
        f"config    = {config}\n"
        f"mean_rad_length = {rad[0]:.6f}\n"
        f"mean_rad_turns  = {rad[1]:.6f}\n"
    )
    print(f"  Best configuration saved → {path}")


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

    # Guard: read configs before using ils_configs[0]
    ils_configs = read_configs(ILS_CONFIGS_FILE)
    if not ils_configs:
        sys.exit("No ILS configurations found – check ils_configuration.txt.")

    BKS = get_bks(GLOBAL_RESULTS_FILE)

    scratch_dir = resolve_scratch_dir()
    scratch_dir.mkdir(parents=True, exist_ok=True)

    scratch_results = scratch_dir / "results_buffer.csv"
    scratch_errors  = scratch_dir / "errors_buffer.csv"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_instances = set(read_configs(INSTANCES_FILE))
    instances = sorted(
        d for d in INSTANCES_DIR.iterdir()
        if d.is_dir() and d.name in run_instances
    )

    if not instances:
        sys.exit("No instances found – check the ejor_tests directory.")

    existing = read_existing_names(RESULTS_FILE)

    solver         = "ILS"
    start_time     = time.time()
    prev_incumbent = ""
    incumbent_config = [ils_configs[0], (float("inf"), float("inf"))]

    print(
        f"Listed instances     : {len(instances)}\n"
        f"Workers              : {args.workers}\n"
        f"Scratch dir          : {scratch_dir}\n"
        f"Starting config      : {incumbent_config[0]}\n"
    )

    while not stopping_criteria(start_time, prev_incumbent, incumbent_config[0]):
        prev_incumbent = incumbent_config[0]

        # Neighborhood phase — flip 0-bits one at a time in positions 1-7
        cfgs = neighborhood_experiments_configs(incumbent_config[0])
        results = run_and_save_configurations(
            cfgs, instances, existing, scratch_dir, solver,
            scratch_results, scratch_errors, args,
        )
        if results:
            incumbent_config = incumbent_configuration(results, BKS)

        # Perturbation phase — flip each bit in positions 8-10
        cfgs = perturbation_experiments_configs(incumbent_config[0])
        results = run_and_save_configurations(
            cfgs, instances, existing, scratch_dir, solver,
            scratch_results, scratch_errors, args,
        )
        if results:
            incumbent_config = incumbent_configuration(results, BKS)

        # Restart phase — flip the last bit
        cfgs = restart_experiments_configs(incumbent_config[0])
        results = run_and_save_configurations(
            cfgs, instances, existing, scratch_dir, solver,
            scratch_results, scratch_errors, args,
        )
        if results:
            incumbent_config = incumbent_configuration(results, BKS)

        print(
            f"\nEnd of iteration.  Incumbent: {incumbent_config[0]}  "
            f"RAD=({incumbent_config[1][0]:.4f}, {incumbent_config[1][1]:.4f})\n"
        )

    print(f"\nSearch finished.  Best config: {incumbent_config[0]}  "
          f"RAD=({incumbent_config[1][0]:.4f}, {incumbent_config[1][1]:.4f})")
    save_best_config(incumbent_config[0], incumbent_config[1], BEST_CONFIG_FILE)


if __name__ == "__main__":
    main()
