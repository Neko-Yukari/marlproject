# Legacy / Problematic Model Warning

This document lists saved model checkpoints in `results/` that have known
limitations for cross-config generalization evaluation.

## Why some models are problematic

### 1. Index-based GNN policy head (legacy GNN models)

The original GNN implementation used a single linear actor head that output
discrete action indices: 0=local, 1=ES1, 2=ES2, ..., E=ES_E.
Because the highest action index happened to map to the fastest edge server
in the training config, models learned a relative index heuristic instead of
reasoning about actual edge-server CPU/bandwidth/load. When evaluated on a
config with fewer edge servers, the model still selects the highest valid
action index and overloads that server.

**Impact:** Cross-config generalization numbers from these models are unreliable.

### 2. Standard MLP

Standard MLP policies have fixed input/output dimensions. They are inherently
incompatible with environments that have a different number of edge servers.

### 3. Older HyperNetwork

HyperNetwork can generate weights for different configs, but older checkpoints
were trained before rigorous validation of `set_config(M, E)` at evaluation time.

## Valid ES-aware GNN models

The ES-aware GNN (implemented after 2026-06-17 11:00) computes per-edge-server
semantic scores using `mlp(concat(md_embedding, es_embedding))`. These models
genuinely learn edge-server capabilities, verified by the reversed-ES-order test.

| Directory | Status |
| --- | --- |
| `explaboff_gnn_7md3es_20260617_133935` | ✅ ES-aware, valid for cross-config |
| `ippo_gnn_7md3es_20260617_110813` | ✅ ES-aware, valid for cross-config |
| `ippo_gnn_7md3es_20260617_112910` | ✅ ES-aware, valid for cross-config |
| `ippo_gnn_7md3es_20260617_122103` | ✅ ES-aware, valid for cross-config |

## Legacy index-based GNN models

| Directory | Status |
| --- | --- |
| `explaboff_gnn_7md3es_20260615_023657` | ❌ Legacy index-based GNN, cross-config unreliable |
| `explaboff_gnn_7md3es_20260617_042116` | ❌ Legacy index-based GNN, cross-config unreliable |
| `explaboff_gnn_7md3es_20260617_050536` | ❌ Legacy index-based GNN, cross-config unreliable |
| `ippo_gnn_3md2es_20260608_203154` | ❌ Legacy index-based GNN, cross-config unreliable |
| `ippo_gnn_3md2es_20260608_205641` | ❌ Legacy index-based GNN, cross-config unreliable |
| `ippo_gnn_3md2es_20260609_001449` | ❌ Legacy index-based GNN, cross-config unreliable |
| `ippo_gnn_3md2es_20260609_012920` | ❌ Legacy index-based GNN, cross-config unreliable |
| `ippo_gnn_3md2es_20260609_023634` | ❌ Legacy index-based GNN, cross-config unreliable |
| `ippo_gnn_3md2es_20260612_141204` | ❌ Legacy index-based GNN, cross-config unreliable |
| `ippo_gnn_5md2es_20260609_025315` | ❌ Legacy index-based GNN, cross-config unreliable |
| `ippo_gnn_5md2es_20260609_032446` | ❌ Legacy index-based GNN, cross-config unreliable |
| `ippo_gnn_7md3es_20260609_040308` | ❌ Legacy index-based GNN, cross-config unreliable |
| `ippo_gnn_7md3es_20260609_043432` | ❌ Legacy index-based GNN, cross-config unreliable |
| `ippo_gnn_7md3es_20260617_033932` | ❌ Legacy index-based GNN, cross-config unreliable |

## Legacy HyperNetwork models

| Directory | Status |
| --- | --- |
| `explaboff_hyper_7md3es_20260615_031854` | ⚠️ Older HyperNetwork, verify set_config() |
| `ippo_hyper_3md2es_20260608_203202` | ⚠️ Older HyperNetwork, verify set_config() |
| `ippo_hyper_3md2es_20260609_015739` | ⚠️ Older HyperNetwork, verify set_config() |
| `ippo_hyper_3md2es_20260609_025042` | ⚠️ Older HyperNetwork, verify set_config() |
| `ippo_hyper_5md2es_20260609_031649` | ⚠️ Older HyperNetwork, verify set_config() |
| `ippo_hyper_5md2es_20260609_034814` | ⚠️ Older HyperNetwork, verify set_config() |
| `ippo_hyper_7md3es_20260609_043433` | ⚠️ Older HyperNetwork, verify set_config() |
| `ippo_hyper_7md3es_20260609_045237` | ⚠️ Older HyperNetwork, verify set_config() |

## Standard MLP models (not cross-config compatible)

| Directory | Status |
| --- | --- |
| `explaboff_standard_7md3es_20260615_020100` | ❌ Fixed obs/action dim, in-domain only |
| `ippo_standard_3md2es_20260608_203145` | ❌ Fixed obs/action dim, in-domain only |
| `ippo_standard_3md2es_20260608_203228` | ❌ Fixed obs/action dim, in-domain only |
| `ippo_standard_3md2es_20260608_205853` | ❌ Fixed obs/action dim, in-domain only |
| `ippo_standard_3md2es_20260608_210800` | ❌ Fixed obs/action dim, in-domain only |
| `ippo_standard_3md2es_20260609_004556` | ❌ Fixed obs/action dim, in-domain only |
| `ippo_standard_3md2es_20260609_010811` | ❌ Fixed obs/action dim, in-domain only |
| `ippo_standard_3md2es_20260609_022445` | ❌ Fixed obs/action dim, in-domain only |
| `ippo_standard_3md2es_20260612_141052` | ❌ Fixed obs/action dim, in-domain only |
| `ippo_standard_5md2es_20260609_023512` | ❌ Fixed obs/action dim, in-domain only |
| `ippo_standard_5md2es_20260609_030641` | ❌ Fixed obs/action dim, in-domain only |
| `ippo_standard_7md3es_20260609_033857` | ❌ Fixed obs/action dim, in-domain only |
| `ippo_standard_7md3es_20260609_041012` | ❌ Fixed obs/action dim, in-domain only |

## How this list was generated

Run `python mark_legacy_models.py` to regenerate both this file and the
`LEGACY_WARNING.txt` markers inside each affected `results/` subdirectory.

Last updated: 2026-06-17 15:27:30
