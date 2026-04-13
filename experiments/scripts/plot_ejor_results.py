"""
Plot horizontal box plots of the relative average deviation (RAD) per configuration
for two metrics: path length and number of turns.

For each (configuration, instance) pair the 5-run average is computed.
RAD (%) = (avg_metric - BKS) / BKS * 100
where BKS = best value ever recorded for that instance across all rows.

Boxes are sorted by ascending mean RAD.  The mean is marked with a blue diamond.
"""

import csv
from collections import defaultdict
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_CSV = RESULTS_DIR / "ejor_results.csv"

# ── Load data ─────────────────────────────────────────────────────────────────
rows = []
with open(RESULTS_CSV, newline="") as fh:
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


def compute_config_rads(rows, metric):
    """Return {config: [rad, ...]} for the given metric field."""
    bks: dict[str, int] = defaultdict(lambda: float("inf"))
    for r in rows:
        if r[metric] < bks[r["instance"]]:
            bks[r["instance"]] = r[metric]

    pair_values: dict[tuple, list] = defaultdict(list)
    for r in rows:
        pair_values[(r["config"], r["instance"])].append(r[metric])

    config_rads: dict[str, list] = defaultdict(list)
    for (config, instance), values in pair_values.items():
        avg_val = statistics.mean(values)
        rad = (avg_val - bks[instance]) / bks[instance] * 100
        config_rads[config].append(rad)

    return config_rads


def make_boxplot(config_rads, metric_label, output_png):
    configs_sorted = sorted(
        config_rads.keys(),
        key=lambda c: statistics.mean(config_rads[c]),
    )

    data   = [config_rads[c] for c in configs_sorted]
    labels = configs_sorted
    means  = [statistics.mean(d) for d in data]

    n = len(configs_sorted)
    fig_height = max(6, n * 0.55 + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    ax.boxplot(
        data,
        vert=False,
        patch_artist=True,
        positions=range(n),
        widths=0.55,
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
        s=55,
        zorder=5,
    )

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Relative Average Deviation (%)", fontsize=11)
    ax.set_title(f"RAD per Configuration — {metric_label} (sorted by mean)", fontsize=13, pad=12)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#1a6fb5",
                   markersize=8, label="Mean RAD"),
    ], loc="lower right", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_png}")


# ── Length plot ───────────────────────────────────────────────────────────────
make_boxplot(
    compute_config_rads(rows, "length"),
    metric_label="Path Length",
    output_png=RESULTS_DIR / "ejor_rad_boxplot_length.png",
)

# ── Turns plot ────────────────────────────────────────────────────────────────
make_boxplot(
    compute_config_rads(rows, "turns"),
    metric_label="Turns",
    output_png=RESULTS_DIR / "ejor_rad_boxplot_turns.png",
)
