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
from scipy import stats as scipy_stats
import re

_TYPE_RE = re.compile(r"^i([a-zA-Z]+)\d+$")

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
parser.add_argument(
    "--sort",
    choices=["rad", "file"],
    default="rad",
    help=(
        "Box ordering: 'rad' — ascending mean RAD (default); "
        "'file' — preserve the order configs appear in the configs file."
    ),
)
parser.add_argument(
    "--group",
    choices=["none", "size", "type", "both"],
    default="size",
    help=(
        "Grouping strategy for box plots: "
        "'none' — one plot for all instances; "
        "'size' — one plot per instance size (default); "
        "'type' — one plot per instance type (the letter(s) between 'i' and the number, e.g. 'r', 's'); "
        "'both' — one plot per (size, type) combination."
    ),
)
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
                    "type":     _TYPE_RE.match(parts[1]).group(1),  # e.g. "r" from "ir0", "s" from "is1"
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

def _non_gurobi_xlim(config_rads):
    """Return the max RAD value across all non-Gurobi configs."""
    return max(
        (max(v) for c, v in config_rads.items() if c != "Gurobi"),
        default=0.0,
    )


def _draw_panel(ax, config_rads, configs_sorted, metric_label):
    """Draw one boxplot panel onto *ax* using the pre-sorted config order."""
    data  = [config_rads[c] for c in configs_sorted]
    means = [statistics.mean(d) for d in data]
    n     = len(configs_sorted)

    ax.boxplot(
        data,
        whis=(5, 95),
        vert=False,
        patch_artist=True,
        positions=range(n),
        widths=0.4,
        boxprops=dict(facecolor="#d6e4f0", color="#2c5f8a"),
        medianprops=dict(color="#e05c2a", linewidth=7),
        whiskerprops=dict(color="#2c5f8a"),
        capprops=dict(color="#2c5f8a"),
        flierprops=dict(marker="o", color="#2c5f8a", markersize=6, alpha=0.6),
    )
    ax.scatter(means, range(n), marker="D", color="#1a6fb5", s=50, zorder=5)
    ax.set_xlim(0, _non_gurobi_xlim(config_rads) + 5)
    ax.set_xlabel("Average Deviation (\%)", fontsize=40)
    ax.set_title(metric_label, fontsize=40)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.grid(axis="x", linestyle=":", alpha=0.5)


def make_combined_boxplot(length_rads, turns_rads, output_pdf, sort_by="rad", file_order=None):
    """Side-by-side box plots: path length (left) and turns (right).

    Config order is determined by *sort_by*:
      'rad'  — ascending mean length RAD (default)
      'file' — order configs appear in the configs file (*file_order* required)
    Only configs present in both metric dicts are shown.
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",  # Use serif fonts to match LaTeX defaults
        "font.serif": ["Computer Modern"], # Specify Computer Modern
        "font.size": 40,
    })
    both = {c for c in length_rads if c in turns_rads}
    if sort_by == "file" and file_order is not None:
        configs_sorted = [config_label_map(c) for c in file_order
                          if config_label_map(c) in both]
        configs_sorted = configs_sorted[::-1]
        # append any labelled configs not covered by the file order
        configs_sorted += sorted(c for c in both if c not in configs_sorted)
    else:
        configs_sorted = sorted(both, key=lambda c: statistics.mean(length_rads[c]))

    n          = len(configs_sorted)
    fig_height = 3.9
    fig, axes  = plt.subplots(1, 2, figsize=(30, fig_height), sharey=True)

    _draw_panel(axes[0], length_rads, configs_sorted, "Path Length")
    _draw_panel(axes[1], turns_rads,  configs_sorted, "Turns")

    # Y-tick labels only on the left panel (sharey handles the right panel).
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(configs_sorted, fontsize=40)

    axes[1].legend(
        handles=[plt.Line2D([0], [0], marker="D", color="w",
                            markerfacecolor="#1a6fb5", markersize=8,
                            label="AvgD")],
        loc="lower right",
        fontsize=30,
    )

    plt.tight_layout()
    fig.savefig(output_pdf, dpi=150, format="pdf")
    plt.close(fig)
    print(f"Saved: {output_pdf}")


def get_groups(all_rows, group_mode):
    """Return [(group_key, group_rows), ...] according to *group_mode*.

    group_mode values:
        'none' — single group containing all rows
        'size' — one group per instance size  (s1616, s6464, s128128)
        'type' — one group per instance type  (ir0, ir1, …)
        'both' — one group per (size, type) combination
    """
    if group_mode == "none":
        return [("all", all_rows)]
    if group_mode == "size":
        keys = sorted({r["size"] for r in all_rows})
        return [(k, [r for r in all_rows if r["size"] == k]) for k in keys]
    if group_mode == "type":
        keys = sorted({r["type"] for r in all_rows})
        return [(k, [r for r in all_rows if r["type"] == k]) for k in keys]
    # both
    combos = sorted({(r["size"], r["type"]) for r in all_rows})
    return [
        (f"{s}_{t}", [r for r in all_rows if r["size"] == s and r["type"] == t])
        for s, t in combos
    ]


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
            # Count the instance if at least one individual run matched the BKS.
            if min(pair_values[(config, instance)]) <= bks[instance]:
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


ALPHA = 0.05  # significance level for normality tests


def normality_tests(values):
    """Run Shapiro-Wilk and D'Agostino-Pearson tests on *values*.

    Returns a dict with keys sw_stat, sw_p, sw_normal, dp_stat, dp_p, dp_normal.
    D'Agostino-Pearson requires at least 8 observations; if fewer are available
    its fields are set to None.
    """
    n = len(values)
    sw_stat, sw_p = scipy_stats.shapiro(values)
    result = {
        "n":         n,
        "sw_stat":   sw_stat,
        "sw_p":      sw_p,
        "sw_normal": sw_p >= ALPHA,
    }
    if n >= 8:
        dp_stat, dp_p = scipy_stats.normaltest(values)
        result.update({
            "dp_stat":   dp_stat,
            "dp_p":      dp_p,
            "dp_normal": dp_p >= ALPHA,
        })
    else:
        result.update({"dp_stat": None, "dp_p": None, "dp_normal": None})
    return result


def _sig_stars(p):
    """Return significance stars for a p-value (or 'N/A' if None)."""
    if p is None:
        return "N/A"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < ALPHA:
        return "*"
    return "ns"


# ── Pairwise Wilcoxon signed-rank test ────────────────────────────────────────

def compute_pairwise_wilcoxon(size_rows, loc_metric):
    """Return a list of pairwise Wilcoxon results for all config pairs.

    Each observation fed to Wilcoxon is the per-instance mean RAD so that
    paired comparisons share the same instance ordering between any two configs.
    Only instances present in *both* configs are used (intersection).

    Each entry in the returned list is a dict with:
        config_a, config_b  — compared labels (sorted by mean RAD)
        n_common            — number of matched instances
        w_stat, p_value     — Wilcoxon W and two-sided p-value (None if untestable)
        stars               — significance stars
        better              — label of the config with lower mean RAD when significant,
                              "tie" when all differences are zero, or "ns" otherwise
    """
    # ── build BKS over this size slice ───────────────────────────────────────
    bks: dict[str, float] = defaultdict(lambda: float("inf"))
    for r in size_rows:
        if r[loc_metric] < bks[r["instance"]]:
            bks[r["instance"]] = r[loc_metric]

    # ── per-(config_label, instance) mean RAD ────────────────────────────────
    pair_values: dict[tuple, list] = defaultdict(list)
    for r in size_rows:
        pair_values[(r["config"], r["instance"])].append(r[loc_metric])

    # config_label → instance → mean_rad
    config_instance_rad: dict[str, dict[str, float]] = defaultdict(dict)
    for (config, instance), values in pair_values.items():
        avg_val = statistics.mean(values)
        rad = (avg_val - bks[instance]) / bks[instance] * 100.0
        config_instance_rad[config_label_map(config)][instance] = rad

    # sort configs by their overall mean RAD (ascending)
    configs = sorted(
        config_instance_rad.keys(),
        key=lambda c: statistics.mean(config_instance_rad[c].values()),
    )

    results = []
    for i, ca in enumerate(configs):
        for cb in configs[i + 1:]:
            common = sorted(
                set(config_instance_rad[ca]) & set(config_instance_rad[cb])
            )
            n_common = len(common)

            if n_common < 2:
                results.append({
                    "config_a": ca, "config_b": cb,
                    "n_common": n_common,
                    "w_stat": None, "p_value": None,
                    "stars": "N/A", "better": "N/A",
                })
                continue

            x = [config_instance_rad[ca][inst] for inst in common]
            y = [config_instance_rad[cb][inst] for inst in common]

            diffs = [xi - yi for xi, yi in zip(x, y)]
            if all(d == 0.0 for d in diffs):
                results.append({
                    "config_a": ca, "config_b": cb,
                    "n_common": n_common,
                    "w_stat": 0.0, "p_value": 1.0,
                    "stars": "ns", "better": "tie",
                })
                continue

            try:
                w_stat, p_value = scipy_stats.wilcoxon(x, y, alternative="two-sided")
                stars = _sig_stars(p_value)
                if p_value < ALPHA:
                    better = ca if statistics.mean(x) < statistics.mean(y) else cb
                else:
                    better = "ns"
                results.append({
                    "config_a": ca, "config_b": cb,
                    "n_common": n_common,
                    "w_stat": w_stat, "p_value": p_value,
                    "stars": stars, "better": better,
                })
            except Exception as exc:
                results.append({
                    "config_a": ca, "config_b": cb,
                    "n_common": n_common,
                    "w_stat": None, "p_value": None,
                    "stars": "ERR", "better": str(exc),
                })

    return results


# ── Combined statistics report (normality + pairwise Wilcoxon) ───────────────

def write_statistics_report(all_rows, sizes, loc_metric, loc_metric_label, output_path):
    """Write normality tests then pairwise Wilcoxon results to *output_path*."""

    # ════════════════════════════════════════════════════════════════════════
    # Section 1 — Normality tests
    # ════════════════════════════════════════════════════════════════════════
    lines = [
        f"Normality analysis — {loc_metric_label}",
        f"Significance level α = {ALPHA}",
        f"Tests: Shapiro-Wilk (SW) | D'Agostino-Pearson K² (DP, requires n≥8)",
        "=" * 72,
    ]

    for loc_size in sizes:
        size_label = SIZE_LABELS.get(loc_size, loc_size)
        lines.append(f"\n{size_label}")
        lines.append("-" * 72)

        size_rows = [r for r in all_rows if r["size"] == loc_size]
        config_rads = compute_config_rads(size_rows, loc_metric)

        header = (
            f"{'Config':<30} {'n':>4}  "
            f"{'SW stat':>9} {'SW p':>9} {'SW normal?':>10}  "
            f"{'DP stat':>9} {'DP p':>9} {'DP normal?':>10}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for cfg_label in sorted(config_rads, key=lambda c: statistics.mean(config_rads[c])):
            values = config_rads[cfg_label]
            r = normality_tests(values)

            dp_stat_s = f"{r['dp_stat']:9.4f}" if r["dp_stat"] is not None else f"{'N/A':>9}"
            dp_p_s    = f"{r['dp_p']:9.4f}"    if r["dp_p"]    is not None else f"{'N/A':>9}"
            dp_norm_s = (
                ("Yes" if r["dp_normal"] else "No") if r["dp_normal"] is not None else "N/A"
            )

            lines.append(
                f"{cfg_label:<30} {r['n']:>4}  "
                f"{r['sw_stat']:9.4f} {r['sw_p']:9.4f} {'Yes' if r['sw_normal'] else 'No':>10}  "
                f"{dp_stat_s} {dp_p_s} {dp_norm_s:>10}"
            )

    # ════════════════════════════════════════════════════════════════════════
    # Section 2 — Pairwise Wilcoxon signed-rank tests
    # ════════════════════════════════════════════════════════════════════════
    lines += [
        "",
        "",
        f"Pairwise Wilcoxon Signed-Rank Test — {loc_metric_label}",
        f"Significance level α = {ALPHA}  |  ns: p≥α  *: p<0.05  **: p<0.01  ***: p<0.001",
        "Observation = per-instance mean RAD; only instances present in both configs used.",
        "Configs ordered by ascending mean RAD within each size group.",
        "=" * 72,
    ]

    col_a  = 30
    col_b  = 30
    header = (
        f"{'Config A':<{col_a}} {'Config B':<{col_b}}"
        f" {'n':>4}  {'W stat':>9} {'p-value':>9} {'sig':>4}  Better"
    )

    def _append_wilcoxon_block(label, subset_rows):
        pairs = compute_pairwise_wilcoxon(subset_rows, loc_metric)
        lines.append(f"\n{label}")
        lines.append("-" * 72)
        lines.append(header)
        lines.append("-" * len(header))
        for p in pairs:
            w_s  = f"{p['w_stat']:9.2f}"  if p["w_stat"]  is not None else f"{'N/A':>9}"
            pv_s = f"{p['p_value']:9.4f}" if p["p_value"] is not None else f"{'N/A':>9}"
            lines.append(
                f"{p['config_a']:<{col_a}} {p['config_b']:<{col_b}}"
                f" {p['n_common']:>4}  {w_s} {pv_s} {p['stars']:>4}  {p['better']}"
            )

    for loc_size in sizes:
        _append_wilcoxon_block(
            SIZE_LABELS.get(loc_size, loc_size),
            [r for r in all_rows if r["size"] == loc_size],
        )

    _append_wilcoxon_block("All sizes combined", all_rows)

    lines.append("")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {output_path}")


# ── Box plots (side-by-side, grouped) ────────────────────────────────────────
for group_key, group_rows in get_groups(rows, args.group):
    make_combined_boxplot(
        length_rads=compute_config_rads(group_rows, "length"),
        turns_rads =compute_config_rads(group_rows, "turns"),
        output_pdf =RESULTS_DIR / f"ejor_rad_boxplot_{group_key}.pdf",
        sort_by    =args.sort,
        file_order =configurations,
    )

# ── LaTeX tables and statistics reports (always per size) ────────────────────
for metric, metric_label in METRICS:
    make_latex_table(
        rows, SIZES, metric, metric_label,
        output_tex=RESULTS_DIR / f"ejor_table_{metric}.tex",
    )
    write_statistics_report(
        rows, SIZES, metric, metric_label,
        output_path=RESULTS_DIR / f"{metric}_statistics.txt",
    )
