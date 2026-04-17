"""
Parallel local version of run_ejor_experiments.py.

Distributes all (algorithm_config × instance) pairs across worker
processes (one pair per worker at a time).  Already-completed pairs are
skipped by checking the existing results CSV before launching any work.

Solver selection
────────────────
The solver (ILS or GUROBI) is chosen automatically per instance based on
its size prefix, following SOLVER_MAP.  Use --solver to override for
all instances:

  --solver auto    Size-based default (s1616_ → GUROBI, larger → ILS) [default]
  --solver ILS     Force ILS for every instance
  --solver GUROBI  Force GUROBI for every instance

Working directories for each experiment are created under:
    experiments/scratch/<exp_name>/

Results are written to experiments/results/ incrementally as each future
completes, so progress is preserved if the script is interrupted.

Usage:
    python run_ejor_experiments_parallel_local.py \\
        -e /path/to/cppied_executable \\
        --lkh /path/to/LKH \\
        [--workers N]                    # default: all available CPUs
        [--solver {auto,ILS,GUROBI}]
"""

from ejor_experiments_helpers import *

SCRATCH_DIR = EXPERIMENTS / "scratch"

ILS_CONFIGS_FILE    = DATA_DIR / "algorithm_configuration" / "random_configurations.txt"
GUROBI_CONFIGS_FILE = DATA_DIR / "algorithm_configuration" / "gurobi_configurations.txt"

RESULTS_FILE = RESULTS_DIR / "ejor_results.csv"
ERRORS_FILE  = RESULTS_DIR / "ejor_errors.csv"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parallel local CPPIED EJOR experiments (all configs × all instances)."
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
            "'auto' (default) picks GUROBI for small instances (s1616_) and ILS "
            "for larger ones. 'ILS' or 'GUROBI' forces a single solver for all "
            "instances regardless of size."
        ),
    )
    args = parser.parse_args()

    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ils_configs    = read_configs(ILS_CONFIGS_FILE)
    gurobi_configs = read_configs(GUROBI_CONFIGS_FILE)

    instances = sorted(
        d for d in INSTANCES_DIR.iterdir()
        if d.is_dir() and d.name.startswith("s1616_")
    )

    if not ils_configs:
        sys.exit("No ILS configurations found – check random_configurations.txt.")
    if not gurobi_configs:
        sys.exit("No GUROBI configurations found – check gurobi_configurations.txt.")
    if not instances:
        sys.exit("No s1616 instances found – check the ejor_tests directory.")

    existing = read_existing_names(RESULTS_FILE)

    # Build the pending work list.
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

    total_all  = total_ils_all + total_gurobi_all
    total_skip = total_all - len(pending)
    n_ils_inst    = total_ils_all    // max(len(ils_configs),    1)
    n_gurobi_inst = total_gurobi_all // max(len(gurobi_configs), 1)
    print(
        f"ILS configurations   : {len(ils_configs)}\n"
        f"GUROBI configurations: {len(gurobi_configs)}\n"
        f"s1616 instances      : {len(instances)}  "
        f"(ILS: {n_ils_inst}, GUROBI: {n_gurobi_inst})\n"
        f"Total pairs          : {total_all}\n"
        f"Already done         : {total_skip}\n"
        f"To run               : {len(pending)}\n"
        f"Workers              : {args.workers}\n"
        f"Scratch dir          : {SCRATCH_DIR}\n"
        f"Solver selection     : {args.solver}\n"
    )

    if not pending:
        print("Nothing to do.")
        return

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
                # Write results immediately so progress survives interruption
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


if __name__ == "__main__":
    main()
