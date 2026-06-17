# MARL Edge Computing Task Offloading

This repository contains a modular implementation of multi-agent reinforcement learning (MARL) algorithms for task offloading in mobile edge computing (MEC).

## Highlights

- **Orthogonal architecture**: decouple `network`, `algorithm`, and `environment` so that any network (Standard MLP / GNN / HyperNetwork) can be paired with any algorithm (IPPO / ExplabOff) on any MEC configuration.
- **Paper-based environment**: faithful reproduction of the system model from the ExplabOff (INFOCOM 2025) paper, supporting 2ES-3MD, 2ES-5MD, and 3ES-7MD scenarios.
- **Cross-config generalization**: GNN and HyperNetwork policies adapt to different numbers of mobile devices (MDs) and edge servers (ESs) without retraining.
- **ES-aware GNN policy head**: fixes the action-index overfitting problem by scoring each edge server using pairwise MD-ES features.
- **ExplabOff MI reward**: implements the mutual-information enhanced reward from the paper with dual MI buffers (B+/B-).

## Repository Structure

```
agents/           # Policy networks and PPO agents
configs/          # YAML training configs (network × algorithm × env)
envs/             # PettingZoo-compatible MEC environment
greedy_baseline.py# Hand-written heuristic for comparison
results/          # Training outputs (not uploaded to HF)
train_unified.py  # Single entry point for all experiments
upload_to_hf.py   # Helper to upload code to Hugging Face
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train IPPO + ES-aware GNN on 3ES-7MD
python train_unified.py --config configs/ippo_gnn_7md3es.yaml --episodes 10000

# Train ExplabOff + Standard MLP on 2ES-3MD
python train_unified.py --config configs/explaboff_standard_3md2es.yaml --episodes 10000
```

## Selected Results

| Algorithm | Network | Config | Best Cost | Completion |
|-----------|---------|--------|-----------|------------|
| IPPO      | Standard MLP | 2ES-3MD | 0.472 | 66.7% |
| IPPO      | GNN (ES-aware) | 3ES-7MD | 0.401 | 86.0% |
| ExplabOff | GNN (ES-aware) | 3ES-7MD | 0.407 | 85.1% |
| Greedy    | - | 2ES-5MD | 0.389 | 99.6% |

See `docs/KNOWLEDGE_MANUAL.md` for detailed background and results.

## Citation

If you use this code, please cite the original ExplabOff paper:

```bibtex
@inproceedings{explaboff2025,
  title={ExplabOff: Towards Explorative and Collaborative Task Offloading via Mutual Information-Enhanced Multi-Agent Reinforcement Learning},
  booktitle={IEEE INFOCOM 2025}
}
```

## License

MIT
