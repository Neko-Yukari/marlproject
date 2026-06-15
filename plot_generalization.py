"""Plot generalization comparison chart."""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
JSON_PATH = RESULTS_DIR / "generalization_report.json"
OUTPUT_PATH = RESULTS_DIR / "generalization_comparison.png"

# Load data
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Parse model names: {algo}_{network}_{nmd}md{nes}es_YYYYMMDD_HHMMSS
configs = ["2ES-3MD", "2ES-5MD", "3ES-7MD"]


def parse_model_name(name: str):
    """Parse model name into algorithm, network, train config."""
    parts = name.split("_")
    # e.g. explaboff_gnn_7md3es_20260615_023657
    algo = parts[0]  # ippo or explaboff
    network = parts[1]  # standard, gnn, hyper
    # train config from parts[2]: 7md3es -> 3ES-7MD
    cfg_part = parts[2]
    md = int(cfg_part.split("md")[0])
    es = int(cfg_part.split("md")[1].split("es")[0])
    train_cfg = f"{es}ES-{md}MD"
    return algo, network, train_cfg


# Build matrix: (algo, network, train_cfg, test_cfg) -> best cost
matrix = {}
for model_name, results in data.items():
    algo, network, train_cfg = parse_model_name(model_name)
    key = (algo, network, train_cfg)
    if key not in matrix:
        matrix[key] = {}
    for test_cfg, r in results.items():
        if not r.get("compatible", False):
            continue
        current = matrix[key].get(test_cfg)
        if current is None or r["cost_mean"] < current["cost_mean"]:
            matrix[key][test_cfg] = r

# Select the key comparison: train on 3ES-7MD, test on all 3 configs
TRAIN_CFG = "3ES-7MD"
selected = []
for algo in ["ippo", "explaboff"]:
    for network in ["standard", "gnn", "hyper"]:
        key = (algo, network, TRAIN_CFG)
        if key in matrix:
            costs = [matrix[key].get(cfg, {}).get("cost_mean", np.nan) for cfg in configs]
            stds = [matrix[key].get(cfg, {}).get("cost_std", 0.0) for cfg in configs]
            comps = [matrix[key].get(cfg, {}).get("completion", 0.0) * 100 for cfg in configs]
            selected.append({
                "label": f"{algo.upper()} + {network.capitalize()}",
                "costs": costs,
                "stds": stds,
                "comps": comps,
                "network": network,
                "algo": algo,
            })

# Also add IPPO+GNN trained on 2ES-3MD and 2ES-5MD for comparison
extra = []
for train_cfg in ["2ES-3MD", "2ES-5MD"]:
    key = ("ippo", "gnn", train_cfg)
    if key in matrix:
        costs = [matrix[key].get(cfg, {}).get("cost_mean", np.nan) for cfg in configs]
        stds = [matrix[key].get(cfg, {}).get("cost_std", 0.0) for cfg in configs]
        comps = [matrix[key].get(cfg, {}).get("completion", 0.0) * 100 for cfg in configs]
        extra.append({
            "label": f"IPPO+GNN trained {train_cfg}",
            "costs": costs,
            "stds": stds,
            "comps": comps,
        })

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: Cost comparison for models trained on 3ES-7MD
ax1 = axes[0]
x = np.arange(len(configs))
width = 0.12
multiplier = 0

colors = {
    ("ippo", "standard"): "#5B9BD5",
    ("ippo", "gnn"): "#4472C4",
    ("ippo", "hyper"): "#2F5597",
    ("explaboff", "standard"): "#ED7D31",
    ("explaboff", "gnn"): "#C55A11",
    ("explaboff", "hyper"): "#843C0C",
}

for entry in selected:
    offset = width * multiplier
    color = colors.get((entry["algo"], entry["network"]), "gray")
    bars = ax1.bar(x + offset, entry["costs"], width, yerr=entry["stds"],
                   label=entry["label"], color=color, capsize=3, edgecolor="white", linewidth=0.5)
    multiplier += 1

ax1.set_ylabel("Average Cost", fontsize=12)
ax1.set_title("Generalization: Trained on 3ES-7MD, Tested on All Configs", fontsize=13, fontweight="bold")
ax1.set_xticks(x + width * (multiplier - 1) / 2)
ax1.set_xticklabels(configs)
ax1.legend(loc="upper left", fontsize=9)
ax1.set_ylim(0, 1.0)
ax1.grid(axis="y", alpha=0.3)
ax1.axhline(y=0.45, color="red", linestyle="--", alpha=0.4, label="target threshold")

# Right: Completion rate comparison
ax2 = axes[1]
multiplier = 0
for entry in selected:
    offset = width * multiplier
    color = colors.get((entry["algo"], entry["network"]), "gray")
    ax2.bar(x + offset, entry["comps"], width,
            label=entry["label"], color=color, edgecolor="white", linewidth=0.5)
    multiplier += 1

ax2.set_ylabel("Task Completion Rate (%)", fontsize=12)
ax2.set_title("Task Completion: Trained on 3ES-7MD", fontsize=13, fontweight="bold")
ax2.set_xticks(x + width * (multiplier - 1) / 2)
ax2.set_xticklabels(configs)
ax2.legend(loc="upper right", fontsize=9)
ax2.set_ylim(0, 110)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved generalization comparison chart to {OUTPUT_PATH}")

# Also create a focused GNN-only comparison chart
fig2, ax3 = plt.subplots(figsize=(8, 5))
gnn_entries = [e for e in selected if e["network"] == "gnn"]
x = np.arange(len(configs))
width = 0.25

for i, entry in enumerate(gnn_entries):
    offset = width * (i - 0.5)
    color = "#4472C4" if entry["algo"] == "ippo" else "#C55A11"
    ax3.bar(x + offset, entry["costs"], width, yerr=entry["stds"],
            label=entry["label"], color=color, capsize=4, edgecolor="white", linewidth=1)

ax3.set_ylabel("Average Cost", fontsize=12)
ax3.set_title("GNN Generalization: IPPO vs ExplabOff (Trained on 3ES-7MD)", fontsize=13, fontweight="bold")
ax3.set_xticks(x)
ax3.set_xticklabels(configs)
ax3.legend(fontsize=10)
ax3.set_ylim(0, 0.7)
ax3.grid(axis="y", alpha=0.3)
plt.tight_layout()
gnn_output = RESULTS_DIR / "generalization_gnn_ippo_vs_explaboff.png"
plt.savefig(gnn_output, dpi=300, bbox_inches="tight")
print(f"Saved GNN comparison chart to {gnn_output}")
