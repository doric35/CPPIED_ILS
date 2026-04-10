"""
Plot a horizontal box plot of the relative average deviation (RAD) per configuration.

For each (configuration, instance) pair the 5-run average length is computed.
RAD (%) = (avg_length - BKS) / BKS * 100
where BKS = best length ever recorded for that instance across all rows.

Boxes are sorted by ascending mean RAD.  The mean is marked with a blue diamond.
"""

import csv
from collections import defaultdict
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_CSV = RESULTS_DIR / "ejor_results.csv"
OUTPUT_PNG  = RESULTS_DIR / "ejor_rad_boxplot.png"

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
        })

# ── Best known solution per instance ─────────────────────────────────────────
bks: dict[str, int] = defaultdict(lambda: float("inf"))
for r in rows:
    if r["length"] < bks[r["instance"]]:
        bks[r["instance"]] = r["length"]

# ── Average length per (config, instance) pair → one RAD value ───────────────
pair_lengths: dict[tuple, list] = defaultdict(list)
for r in rows:
    pair_lengths[(r["config"], r["instance"])].append(r["length"])

# RAD per (config, instance)
config_rads: dict[str, list] = defaultdict(list)
for (config, instance), lengths in pair_lengths.items():
    avg_len = statistics.mean(lengths)
    rad = (avg_len - bks[instance]) / bks[instance] * 100
    config_rads[config].append(rad)

# ── Sort configurations by ascending mean RAD ────────────────────────────────
configs_sorted = sorted(
    config_rads.keys(),
    key=lambda c: statistics.mean(config_rads[c]),
)

# ── Build data list in sorted order ──────────────────────────────────────────
data   = [config_rads[c] for c in configs_sorted]
labels = configs_sorted
means  = [statistics.mean(d) for d in data]

# ── Plot ──────────────────────────────────────────────────────────────────────
n = len(configs_sorted)
fig_height = max(6, n * 0.55 + 1.5)
fig, ax = plt.subplots(figsize=(10, fig_height))

bp = ax.boxplot(
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

# Blue diamond for the mean
ax.scatter(
    means,
    range(n),
    marker="D",
    color="#1a6fb5",
    s=55,
    zorder=5,
    label="Mean RAD",
)

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_yticks(range(n))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Relative Average Deviation (%)", fontsize=11)
ax.set_title("RAD per Configuration (sorted by mean)", fontsize=13, pad=12)
ax.axvline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
ax.grid(axis="x", linestyle=":", alpha=0.5)

diamond = mpatches.Patch(color="#1a6fb5", label="Mean RAD")
ax.legend(handles=[
    plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#1a6fb5",
               markersize=8, label="Mean RAD"),
], loc="lower right", fontsize=9)

plt.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=150)
print(f"Saved: {OUTPUT_PNG}")
