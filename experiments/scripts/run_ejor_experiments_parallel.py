"""
Parallel version of run_ejor_experiments.py.

Distributes all (algorithm_config × instance) pairs across worker
processes (one pair per worker at a time).  Already-completed pairs are
skipped by checking the existing results CSV before launching any work.

Solver selection
────────────────
The solver (ILS or ILP) is chosen automatically per instance based on
its size prefix, following SOLVER_MAP.  Use --solver to override for
all instances:

  --solver auto    Size-based default (s1616_ → GUROBI, larger → ILS) [default]
  --solver ILS     Force ILS for every instance
  --solver GUROBI  Force GUROBI for every instance

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
        [--solver {auto,ILS,GUROBI}]
"""

from ejor_experiments_helpers import *

RESULTS_FILE  = RESULTS_DIR / "ejor_results.csv"
ERRORS_FILE   = RESULTS_DIR / "ejor_errors.csv"

ILS_CONFIGS_FILE    = DATA_DIR / "algorithm_configuration" / "random_configurations.txt"
GUROBI_CONFIGS_FILE = DATA_DIR / "algorithm_configuration" / "gurobi_configurations.txt"

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
    parser.add_argument(
        "--solver", choices=["auto", "ILS", "GUROBI"], default="auto",
        help=(
            "Solver to use. 'auto' (default) picks GUROBI for small instances "
            "(s1616_) and ILS for larger ones. 'ILS' or 'GUROBI' forces a single "
            "solver for all instances regardless of size."
        ),
    )
    args = parser.parse_args()

    scratch_dir = resolve_scratch_dir()
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # Scratch-side buffer CSVs — updated as futures complete so that
    # progress is not lost if the job is cancelled before the final flush.
    scratch_results = scratch_dir / "results_buffer.csv"
    scratch_errors  = scratch_dir / "errors_buffer.csv"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ils_configs    = read_configs(ILS_CONFIGS_FILE)
    gurobi_configs = read_configs(GUROBI_CONFIGS_FILE)
    run_instances  = set(read_configs(INSTANCES_FILE))

    instances = sorted(
        d for d in INSTANCES_DIR.iterdir()
        if d.is_dir() and d.name in run_instances
    )

    if not ils_configs:
        sys.exit("No ILS configurations found – check random_configurations.txt.")
    if not gurobi_configs:
        sys.exit("No GUROBI configurations found – check gurobi_configurations.txt.")
    if not instances:
        sys.exit("No instances found – check the ejor_tests directory.")

    existing = read_existing_names(RESULTS_FILE)

    # Build the pending work list.
    # Both ILS and GUROBI iterate over their respective config lists.
    # ILS   entry: (instance × ils_config)    — exp_name = <instance>_<config>
    # GUROBI entry: (instance × gurobi_config) — exp_name = <instance>_<config>
    # pending items: (exp_name, instance_dir_str, algorithm_config, solver)
    pending: list[tuple[str, str, str, str]] = []
    total_ils_all = total_gurobi_all = 0

    for instance_dir in instances:
        prefix = instance_dir.name.split("_")[0] + "_"
        solver = resolve_solver(prefix, args.solver)
        cfgs   = configs_for_solver(solver, ils_configs, gurobi_configs)

        if solver == "GUROBI":
            total_gurobi_all += len(cfgs)
        else:
            total_ils_all += len(cfgs)

        for algorithm_config in cfgs:
            exp_name = f"{instance_dir.name}_{algorithm_config}"
            if exp_name not in existing:
                pending.append((exp_name, str(instance_dir), algorithm_config, solver))

    total_all   = total_ils_all + total_gurobi_all
    total_skip  = total_all - len(pending)
    total_missed = len(run_instances) - len(instances)
    n_ils_inst    = total_ils_all    // max(len(ils_configs),    1)
    n_gurobi_inst = total_gurobi_all // max(len(gurobi_configs), 1)
    print(
        f"ILS configurations   : {len(ils_configs)}\n"
        f"GUROBI configurations: {len(gurobi_configs)}\n"
        f"Listed instances     : {len(instances)}  "
        f"(ILS: {n_ils_inst}, GUROBI: {n_gurobi_inst})\n"
        f"Not found instances  : {total_missed}\n"
        f"Total pairs          : {total_all}\n"
        f"Already done         : {total_skip}\n"
        f"To run               : {len(pending)}\n"
        f"Workers              : {args.workers}\n"
        f"Scratch dir          : {scratch_dir}\n"
        f"Solver selection     : {args.solver}\n"
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
