import os
import glob
import torch
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("results")

def is_es_aware(policy_path: Path) -> bool:
    """Check if a saved policy uses the ES-aware architecture."""
    try:
        state = torch.load(policy_path, map_location="cpu")
        return "local_score_head.weight" in state
    except Exception:
        return False


def classify_model_dirs():
    """Scan results/ and classify each model directory."""
    legacy_gnn = []
    legacy_hyper = []
    standard_mlp = []
    es_aware_valid = []
    unknown = []

    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        policy_pt = d / "policy.pt"
        if not policy_pt.exists():
            continue

        if "_gnn_" in name:
            if is_es_aware(policy_pt):
                es_aware_valid.append(name)
            else:
                legacy_gnn.append(name)
        elif "_hyper_" in name:
            legacy_hyper.append(name)
        elif "_standard_" in name:
            standard_mlp.append(name)
        else:
            unknown.append(name)

    return legacy_gnn, legacy_hyper, standard_mlp, es_aware_valid, unknown


def mark_legacy_dirs(legacy_gnn, legacy_hyper, standard_mlp):
    """Create LEGACY_WARNING.txt in problematic model directories."""
    warnings = {
        "gnn": """LEGACY MODEL WARNING
====================
This model uses the old index-based GNN policy head.

Issue:
  The actor network outputs discrete action indices (0=local, 1=ES1, 2=ES2, ...)
  instead of ES-semantic scores. During training on one (M, E) configuration,
  the model can overfit to the action-index ordering rather than learning the
  actual edge-server capabilities.

Consequence for cross-config evaluation:
  Results from cross-config tests (e.g., a model trained on 3ES-7MD evaluated
  on 2ES-3MD) are unreliable. The model may simply pick the highest valid
  action index, overloading the fastest edge server in the new config.

When this model is still useful:
  In-domain evaluation (same M and E as training) is still valid.

For reliable cross-config results, use ES-aware GNN models saved after
2026-06-17 11:00, which contain `local_score_head` and `es_score_head`.
""",
        "hyper": """LEGACY MODEL WARNING
====================
This model uses an older HyperNetwork implementation.

Issue:
  HyperNetwork generates policy weights from (M, E) embeddings, but the
  generated weight matrices are tied to fixed observation/action dimensions.
  Cross-config evaluation requires careful `set_config(M, E)` calls and may
  silently fall back to a default configuration if not handled correctly.

Consequence for cross-config evaluation:
  Cross-config numbers from this checkpoint should be treated with caution.
  Verify that `set_config()` was called and the generated weight shapes match
  the target environment.

For reliable cross-config results, retrain with the latest HyperNetwork code
and explicitly validate `set_config()` behavior.
""",
        "standard": """NOT CROSS-CONFIG COMPATIBLE
============================
This model is a standard fully-connected MLP policy.

Issue:
  The MLP has fixed input and output dimensions that depend on the training
  environment's observation size (1 + num_es) and action size (1 + num_es).
  It cannot be loaded for environments with a different number of edge servers.

Consequence for cross-config evaluation:
  This model is only valid for the exact (M, E) configuration it was trained on.
""",
    }

    for name in legacy_gnn:
        (RESULTS_DIR / name / "LEGACY_WARNING.txt").write_text(
            warnings["gnn"], encoding="utf-8"
        )
    for name in legacy_hyper:
        (RESULTS_DIR / name / "LEGACY_WARNING.txt").write_text(
            warnings["hyper"], encoding="utf-8"
        )
    for name in standard_mlp:
        (RESULTS_DIR / name / "LEGACY_WARNING.txt").write_text(
            warnings["standard"], encoding="utf-8"
        )


def generate_markdown(legacy_gnn, legacy_hyper, standard_mlp, es_aware_valid, unknown):
    """Generate LEGACY_MODELS.md for the repository root."""
    lines = [
        "# Legacy / Problematic Model Warning",
        "",
        "This document lists saved model checkpoints in `results/` that have known",
        "limitations for cross-config generalization evaluation.",
        "",
        "## Why some models are problematic",
        "",
        "### 1. Index-based GNN policy head (legacy GNN models)",
        "",
        "The original GNN implementation used a single linear actor head that output",
        "discrete action indices: 0=local, 1=ES1, 2=ES2, ..., E=ES_E.",
        "Because the highest action index happened to map to the fastest edge server",
        "in the training config, models learned a relative index heuristic instead of",
        "reasoning about actual edge-server CPU/bandwidth/load. When evaluated on a",
        "config with fewer edge servers, the model still selects the highest valid",
        "action index and overloads that server.",
        "",
        "**Impact:** Cross-config generalization numbers from these models are unreliable.",
        "",
        "### 2. Standard MLP",
        "",
        "Standard MLP policies have fixed input/output dimensions. They are inherently",
        "incompatible with environments that have a different number of edge servers.",
        "",
        "### 3. Older HyperNetwork",
        "",
        "HyperNetwork can generate weights for different configs, but older checkpoints",
        "were trained before rigorous validation of `set_config(M, E)` at evaluation time.",
        "",
        "## Valid ES-aware GNN models",
        "",
        "The ES-aware GNN (implemented after 2026-06-17 11:00) computes per-edge-server",
        "semantic scores using `mlp(concat(md_embedding, es_embedding))`. These models",
        "genuinely learn edge-server capabilities, verified by the reversed-ES-order test.",
        "",
        "| Directory | Status |",
        "| --- | --- |",
    ]
    for name in sorted(es_aware_valid):
        lines.append(f"| `{name}` | ✅ ES-aware, valid for cross-config |")
    lines.append("")
    lines.append("## Legacy index-based GNN models")
    lines.append("")
    lines.append("| Directory | Status |")
    lines.append("| --- | --- |")
    for name in sorted(legacy_gnn):
        lines.append(f"| `{name}` | ❌ Legacy index-based GNN, cross-config unreliable |")
    lines.append("")
    lines.append("## Legacy HyperNetwork models")
    lines.append("")
    lines.append("| Directory | Status |")
    lines.append("| --- | --- |")
    for name in sorted(legacy_hyper):
        lines.append(f"| `{name}` | ⚠️ Older HyperNetwork, verify set_config() |")
    lines.append("")
    lines.append("## Standard MLP models (not cross-config compatible)")
    lines.append("")
    lines.append("| Directory | Status |")
    lines.append("| --- | --- |")
    for name in sorted(standard_mlp):
        lines.append(f"| `{name}` | ❌ Fixed obs/action dim, in-domain only |")
    if unknown:
        lines.append("")
        lines.append("## Unclassified directories")
        lines.append("")
        for name in sorted(unknown):
            lines.append(f"- `{name}`")
    lines.append("")
    lines.append("## How this list was generated")
    lines.append("")
    lines.append("Run `python mark_legacy_models.py` to regenerate both this file and the")
    lines.append("`LEGACY_WARNING.txt` markers inside each affected `results/` subdirectory.")
    lines.append("")
    lines.append(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    Path("LEGACY_MODELS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    legacy_gnn, legacy_hyper, standard_mlp, es_aware_valid, unknown = classify_model_dirs()
    mark_legacy_dirs(legacy_gnn, legacy_hyper, standard_mlp)
    generate_markdown(legacy_gnn, legacy_hyper, standard_mlp, es_aware_valid, unknown)

    print(f"Marked {len(legacy_gnn)} legacy GNN directories")
    print(f"Marked {len(legacy_hyper)} legacy HyperNetwork directories")
    print(f"Marked {len(standard_mlp)} standard MLP directories")
    print(f"Listed {len(es_aware_valid)} valid ES-aware directories")
    print("Generated LEGACY_MODELS.md")
