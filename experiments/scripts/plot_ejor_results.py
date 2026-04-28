"""
Plot horizontal box plots of the relative average deviation (RAD) per configuration
for two metrics: path length and number of turns.

For each (configuration, instance) pair the 5-run average is computed.
RAD (%) = (avg_metric - BKS) / BKS * 100
where BKS = best value ever recorded for that instance across all rows.

Boxes are sorted by ascending mean RAD.  The mean is marked with a blue diamond.
"""

from ejor_experiments_helpers import *
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#Get result filename as input

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
DEFAULT_RESULTS_FILE = "ejor_results.csv"
DEFAULT_CONFIGS_FILE = "plot_configurations.txt"

parser = argparse.ArgumentParser(
    description="Ablation analysis (Hoos et al. greedy procedure)."
)
parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_FILE,
                    help="Results list file (default: run_instances.txt).")
parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGS_FILE,
                    help="Results list file (default: run_instances.txt).")
args = parser.parse_args()

RESULTS_CSV = RESULTS_DIR / args.results
GLOBAL_RESULTS_CSV = RESULTS_DIR / "ejor_results.csv"
CONFIGS_FILE = DATA_DIR / "algorithm_configuration" / args.configs

configurations = read_configs(CONFIGS_FILE)

# ── Load data ─────────────────────────────────────────────────────────────────

def read_results(results_csv : Path):
    loc_rows = []
    with open(results_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            # Skip infeasible runs — their length/turns are not meaningful and
            # would corrupt the BKS baseline and produce division-by-zero RADs.
            if row.get("status", "") == TIME_LIMIT_INFEASIBLE_STATUS:
                continue
            # name = {size}_{typeRID}_{lrcRID}_{config}
            # e.g.  s1616_ir0_lrc031_c11111111111
            parts    = row["name"].split("_")
            instance = "_".join(parts[:3])   # s1616_ir0_lrc031
            config   = parts[3]  # c11111111111
            loc_size = parts[0]  # s1616, s6464, s128128
            if config in configurations:
                loc_rows.append({
                    "instance": instance,
                    "size":     loc_size,
                    "config":   config,
                    "length":   int(row["length"]),
                    "turns":    int(row["turns"]),
                })
    return loc_rows

rows = read_results(RESULTS_CSV)
unique_instances = set([r["instance"] for r in rows])

global_rows = read_results(GLOBAL_RESULTS_CSV)
global_rows = [r for r in global_rows if r["instance"] in unique_instances]

def config_label_map(c : str):
    if c == "c10":
        return "Gurobi"
    if c == "c11":
        return "Gurobi(DpS)"
    return c

def compute_config_rads(loc_rows, loc_metric):
    """Return {config: [rad, ...]} for the given metric field."""
    bks: dict[str, int] = defaultdict(lambda: float("inf"))
    for r in loc_rows:
        if r[loc_metric] < bks[r["instance"]]:
            bks[r["instance"]] = r[loc_metric]

    pair_values: dict[tuple, list] = defaultdict(list)
    for r in loc_rows:
        pair_values[(r["config"], r["instance"])].append(r[loc_metric])

    config_rads: dict[str, list] = defaultdict(list)
    for (config, instance), values in pair_values.items():
        avg_val = statistics.mean(values)
        rad = (avg_val - bks[instance]) / bks[instance] * 100.0
        config_rads[config_label_map(config)].append(rad)

    return config_rads

def get_ylim(config_rads):
    max_ils_rad = 0.0
    for k, v in config_rads.items():
        if config_label_map(k) != "Gurobi":
            max_ils_rad = max(max_ils_rad, max(v))
    return max_ils_rad


def make_boxplot(config_rads, loc_metric_label, output_pdf):
    ils_ylim = get_ylim(config_rads)

    configs_sorted = sorted(
        config_rads.keys(),
        key=lambda c: statistics.mean(config_rads[c]),
    )

    data   = [config_rads[c] for c in configs_sorted]
    labels = configs_sorted
    means  = [statistics.mean(d) for d in data]

    n = len(configs_sorted)
    fig_height = max(6, n * 0.25 + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    ax.boxplot(
        data,
        whis = (5, 95),
        vert=False,
        patch_artist=True,
        positions=range(n),
        widths=0.25,
        boxprops=dict(facecolor="#d6e4f0", color="#2c5f8a"),
        medianprops=dict(color="#e05c2a", linewidth=2),
        whiskerprops=dict(color="#2c5f8a"),
        capprops=dict(color="#2c5f8a"),
        flierprops=dict(marker="o", color="#2c5f8a", markersize=4, alpha=0.6),
    )

    ax.scatter(
        means,
        range(n),
        marker="D",
        color="#1a6fb5",
        s=25,
        zorder=5,
    )

    plt.xlim(0, ils_ylim + 5)

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Average Deviation (%)", fontsize=11)
    #ax.set_title(f"RAD per Configuration — {metric_label} (sorted by mean)", fontsize=13, pad=12)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#1a6fb5",
                   markersize=8, label="Mean AvgD"),
    ], loc="lower right", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_pdf, dpi=150, format='pdf')
    plt.close(fig)
    print(f"Saved: {output_pdf}")


SIZES = sorted({r["size"] for r in rows})

SIZE_LABELS = {
    "s1616":   r"Small instances ($16\times16$)",
    "s6464":   r"Medium instances ($64\times64$)",
    "s128128": r"Large instances ($128\times128$)",
}

METRICS = [
    ("length", "Path Length"),
    ("turns",  "Turns"),
]

rows = rows + global_rows

def compute_config_table_stats(loc_rows, loc_metric):
    """Return {config_label: stat_dict} for all configs present in loc_rows.

    stat_dict keys: avgd, std, mediand, maxd, mind, n_best, n_lt5, n_worst, n_na.
    BKS and worst-per-instance are derived from all configs in loc_rows.
    """
    all_instances = sorted({r["instance"] for r in loc_rows})

    bks = defaultdict(lambda: float("inf"))
    for r in loc_rows:
        if r[loc_metric] < bks[r["instance"]]:
            bks[r["instance"]] = r[loc_metric]

    pair_values: dict[tuple, list] = defaultdict(list)
    for r in loc_rows:
        pair_values[(r["config"], r["instance"])].append(r[loc_metric])

    pair_avg = {k: statistics.mean(v) for k, v in pair_values.items()}

    worst_val: dict[str, float] = defaultdict(lambda: -float("inf"))
    for (config, instance), avg_val in pair_avg.items():
        worst_val[instance] = max(worst_val[instance], avg_val)

    stats: dict[str, dict] = {}
    for config in sorted({r["config"] for r in loc_rows}):
        rads, n_best, n_lt5, n_worst, n_na = [], 0, 0, 0, 0
        for instance in all_instances:
            if (config, instance) not in pair_avg:
                n_na += 1
                continue
            avg_val = pair_avg[(config, instance)]
            rad = (avg_val - bks[instance]) / bks[instance] * 100.0
            rads.append(rad)
            if rad == 0.0:
                n_best += 1
            if rad < 5.0:
                n_lt5 += 1
            if avg_val >= worst_val[instance]:
                n_worst += 1

        label = config_label_map(config)
        stats[label] = {
            "avgd":    statistics.mean(rads) if rads else float("nan"),
            "std":     statistics.stdev(rads) if len(rads) > 1 else 0.0,
            "mediand": statistics.median(rads) if rads else float("nan"),
            "maxd":    max(rads) if rads else float("nan"),
            "mind":    min(rads) if rads else float("nan"),
            "n_best":  n_best,
            "n_lt5":   n_lt5,
            "n_worst": n_worst,
            "n_na":    n_na,
        }
    return stats


def _fmt(val, decimals=2):
    return "---" if val != val else f"{val:.{decimals}f}"


def make_latex_table(all_rows, sizes, loc_metric, loc_metric_label, output_tex):
    N_COLS = 10  # label + AvgD StD MedianD MaxD MinD #Best #<5% #Worst #NA

    lines = [
        r"\begin{table}",
        r"    \setlength{\tabcolsep}{1pt}",
        r"    \centering",
        rf"    \caption{{Empirical performance analysis — {loc_metric_label}}}",
        r"    \label{tab:empirical}",
        r"    \begin{tabular}{ldddddrrrr}",
        r"    \toprule",
        r"     & AvgD & StD & MedianD & MaxD & MinD & \#Best & \#$<5$\% & \#Worst & \#NA \\",
        r"    \midrule",
    ]

    for i, loc_size in enumerate(sizes):
        size_label = SIZE_LABELS.get(loc_size, loc_size)
        lines.append(
            rf"    \multicolumn{{{N_COLS}}}{{l}}{{\textit{{{size_label}}}}} \\"
        )
        loc_size_rows = [r for r in all_rows if r["size"] == loc_size]
        stats = compute_config_table_stats(loc_size_rows, loc_metric)
        for label in sorted(stats, key=lambda c: stats[c]["avgd"]):
            s = stats[label]
            lines.append(
                f"    {config_label_map(label)} & "
                f"{_fmt(s['avgd'])} & {_fmt(s['std'])} & {_fmt(s['mediand'])} & "
                f"{_fmt(s['maxd'])} & {_fmt(s['mind'])} & "
                f"{s['n_best']} & {s['n_lt5']} & {s['n_worst']} & {s['n_na']} \\\\"
            )
        if i < len(sizes) - 1:
            lines.append(r"    \midrule")

    lines += [
        r"    \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ]

    with open(output_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {output_tex}")


for metric, metric_label in METRICS:
    for size in SIZES:
        size_rows = [r for r in rows if r["size"] == size]
        make_boxplot(
            compute_config_rads(size_rows, metric),
            loc_metric_label=f"{metric_label} — {size}",
            output_pdf=RESULTS_DIR / f"ejor_rad_boxplot_{metric}_{size}.pdf",
        )
    make_latex_table(
        rows, SIZES, metric, metric_label,
        output_tex=RESULTS_DIR / f"ejor_table_{metric}.tex",
    )
