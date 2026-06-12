# MARL Edge Computing - Unified Training Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

多智能体强化学习边缘计算任务卸载框架。基于 INFOCOM 2025 论文 "ExplabOff" 的思想，实现**正交化架构**：网络结构、训练算法、环境配置三者自由组合。

## 核心特性

- **正交架构**: 网络结构(标准MLP/超网络) × 训练算法(IPPO/ExplabOff) × 环境配置(任意MD/ES) 自由组合
- **统一入口**: 单个 `train_unified.py` 通过命令行参数配置任意组合
- **互信息增强**: ExplabOff 的 InfoNCE + L1Out MI 估计器作为可选插件
- **论文参数**: 环境完全匹配 ExplabOff 论文 Table I 参数
- **GPU加速**: 支持 CUDA，大 batch 累积训练

## 快速开始

### 安装依赖

```bash
pip install torch numpy pettingzoo gymnasium matplotlib pyyaml
```

### YAML 配置训练（推荐）

```bash
# IPPO + GNN + 3MD-2ES
python train_unified.py --config configs/ippo_gnn_3md2es.yaml

# IPPO + HyperNetwork + 7MD-3ES
python train_unified.py --config configs/ippo_hyper_7md3es.yaml

# ExplabOff + Standard + 3MD-2ES
python train_unified.py --config configs/explaboff_standard_3md2es.yaml

# 覆盖参数：训练20000 episodes
python train_unified.py --config configs/ippo_gnn_3md2es.yaml --episodes 20000
```

### 命令行组合训练

```bash
# IPPO + 标准网络 + 3MD-2ES
python train_unified.py --network standard --algorithm ippo --md 3 --es 2

# ExplabOff + 超网络 + 5MD-2ES
python train_unified.py --network hyper --algorithm explaboff --md 5 --es 2

# IPPO + GNN + 7MD-3ES
python train_unified.py --network gnn --algorithm ippo --md 7 --es 3
```

### 运行完整对比实验（9 实验）

```bash
python run_all_comparisons.py
```

### 参数说明

| 参数 | 选项 | 说明 |
|------|------|------|
| `--config` | YAML 文件路径 | **推荐**：从配置文件加载全部参数 |
| `--network` | `standard`, `hyper`, `gnn` | 网络架构 |
| `--algorithm` | `ippo`, `explaboff` | 训练算法 |
| `--md` | 整数 | 移动设备数量 |
| `--es` | 整数 | 边缘服务器数量 |
| `--episodes` | 整数 | 训练回合数（可覆盖 YAML） |
| `--seed` | 整数 | 随机种子 |
| `--device` | `cpu`, `cuda` | 计算设备 |

### 可用的 YAML 配置

| 文件 | 网络 | 算法 | 配置 |
|------|------|------|------|
| `ippo_standard_3md2es.yaml` | Standard MLP | IPPO | 2ES-3MD |
| `ippo_gnn_3md2es.yaml` | GNN | IPPO | 2ES-3MD |
| `ippo_hyper_3md2es.yaml` | HyperNetwork | IPPO | 2ES-3MD |
| `ippo_standard_5md2es.yaml` | Standard MLP | IPPO | 2ES-5MD |
| `ippo_gnn_5md2es.yaml` | GNN | IPPO | 2ES-5MD |
| `ippo_hyper_5md2es.yaml` | HyperNetwork | IPPO | 2ES-5MD |
| `ippo_standard_7md3es.yaml` | Standard MLP | IPPO | 3ES-7MD |
| `ippo_gnn_7md3es.yaml` | GNN | IPPO | 3ES-7MD |
| `ippo_hyper_7md3es.yaml` | HyperNetwork | IPPO | 3ES-7MD |
| `explaboff_standard_3md2es.yaml` | Standard MLP | ExplabOff | 2ES-3MD |
| `explaboff_gnn_3md2es.yaml` | GNN | ExplabOff | 2ES-3MD |
| `explaboff_hyper_3md2es.yaml` | HyperNetwork | ExplabOff | 2ES-3MD |

### 评估已训练模型

```bash
python train_unified.py --load results/your_model --mode eval --md 3 --es 2
```

### 验证环境

```bash
# 验证 PettingZoo API
python test_pettingzoo_api.py

# 验证完整训练流水线
python validate_pipeline.py
```

## 项目结构

```
.
├── train_unified.py           # 统一训练入口（YAML + CLI）
├── run_all_comparisons.py     # 全量对比实验脚本
├── validate_pipeline.py       # 端到端验证
├── test_pettingzoo_api.py     # PettingZoo API 测试
├── requirements.txt           # 依赖列表
├── configs/                   # YAML 配置文件（12个）
│   ├── ippo_standard_3md2es.yaml
│   ├── ippo_gnn_3md2es.yaml
│   ├── ippo_hyper_3md2es.yaml
│   ├── ippo_standard_5md2es.yaml
│   ├── ippo_gnn_5md2es.yaml
│   ├── ippo_hyper_5md2es.yaml
│   ├── ippo_standard_7md3es.yaml
│   ├── ippo_gnn_7md3es.yaml
│   ├── ippo_hyper_7md3es.yaml
│   ├── explaboff_standard_3md2es.yaml
│   ├── explaboff_gnn_3md2es.yaml
│   └── explaboff_hyper_3md2es.yaml
├── agents/                    # 智能体实现（正交架构）
│   ├── policy_interface.py    # 网络抽象接口
│   ├── standard_policy.py     # 标准 MLP 策略网络
│   ├── gnn_policy.py          # GNN 图神经网络策略
│   ├── hyper_policy.py        # 超网络策略（跨配置）
│   ├── hypernetwork.py        # 超网络核心实现
│   ├── ppo_agent.py           # 统一 PPO 训练器
│   ├── mi_plugin.py           # MI 增强插件（ExplabOff）
│   └── networks/              # 神经网络组件
│       ├── actor_critic.py    # Actor-Critic 网络
│       └── mi_estimator.py    # InfoNCE / L1Out 估计器
├── envs/                      # 环境实现
│   └── paper_accurate_env.py  # 论文参数环境（支持任意 M/E）
├── docs/                      # 文档
│   └── KNOWLEDGE_MANUAL.md    # 知识手册（PPT制作参考）
├── evaluation/                # 评估工具
│   └── compare.py             # 结果对比
├── utils/                     # 工具函数
│   ├── helpers.py             # 辅助函数
│   ├── metrics.py             # 评估指标
│   ├── reporter.py            # 训练报告
│   └── task_device.py         # 任务/设备数据结构
├── results/                   # 实验结果（自动保存）
│   └── baseline/              # 基线结果
└── papers/                    # 论文资料
```

## 架构设计

### 正交分离

```
训练入口 ──┬── 网络层 ──┬── StandardPolicy (标准MLP)
            │            ├── GNNPolicy (图神经网络)
            │            └── HyperPolicy (超网络)
            │
            ├── 算法层 ──┬── IPPO (无MI)
            │            └── ExplabOff (有MI)
            │
            └── 环境层 ── 任意 (M, E) 组合
```

### 核心类关系

```
PolicyNetwork (接口)
├── StandardPolicy: MLP(obs_dim, action_dim) → logits, value
└── HyperPolicy: HyperNetwork(config) → 动态生成网络权重

PPOAgent
├── policy: PolicyNetwork (上述任一)
├── mi_plugin: Optional[MIPlugin] (ExplabOff时启用)
└── update(): PPO clipped surrogate objective

MIPlugin (可选)
├── InfoNCEEstimator: I(s; a) 下界估计
├── L1OutEstimator: I(s; a|B-) 上界估计
└── compute_reward(): r_MI = μ·I_NCE - ν·I_L1Out
```

## 环境参数

匹配 ExplabOff 论文 Table I:

| 参数 | 值 | 说明 |
|------|-----|------|
| MD CPU | 1.0 GHz | 移动设备计算能力 |
| ES CPU | 6.0–30.0 GHz | 边缘服务器计算能力（按配置） |
| 任务大小 | 2.5–5.5 Mb | 随机轮廓，每 episode 变化 |
| 时隙数 | 10 | 每 episode 10 秒 |
| 时隙长度 | 1 s | 每秒一个决策点 |
| 带宽 | 10 Mbps | 传输速率 |
| Cost | η·latency + (1-η)·energy | η=0.5 |

## 算法原理

### IPPO

每个设备独立运行 PPO，目标函数:

```
L_CLIP(θ) = E[min(r·A, clip(r, 1-ε, 1+ε)·A)]
```

### ExplabOff

在 IPPO 基础上增加 MI 奖励:

```
r_total = r_env + r_MI
r_MI = μ·I_NCE(s; a) - ν·I_L1Out(s; a|B-)
```

- **I_NCE**: 鼓励探索（高熵动作分布）
- **I_L1Out**: 限制过度探索（在差 episode 上惩罚高 MI）

## 实验结果

### 三网络全面对比（10K episodes, 最优Cost）

| 配置    | Standard MLP | GNN (V4) | HyperNetwork |         优胜          |
| ------- | :----------: | :------: | :----------: | :-------------------: |
| **2ES-3MD** |    0.446     |  0.426   |    **0.423**     |       **Hyper** 🥇       |
| **2ES-5MD** |    0.385     |  0.385   |    0.394     | **GNN** 🥇 (仅0.0002优势) |
| **3ES-7MD** |    0.407     |  0.405   |    **0.400**     |       **Hyper** 🥇       |

### 最终评估（100 episodes 均值）

| 配置    | Standard MLP |   GNN (V4)   |  HyperNetwork   |
| ------- | :----------: | :----------: | :-------------: |
| **2ES-3MD** | 0.472 ± 0.01 | 0.481 ± 0.03 |  0.481 ± 0.05   |
| **2ES-5MD** | 0.412 ± 0.01 | 0.411 ± 0.01 | 0.859 ± 0.16 ⚠️ |
| **3ES-7MD** | 0.427 ± 0.01 | 0.426 ± 0.01 |  0.434 ± 0.02   |

### 单配置最优结果（IPPO/ExplabOff）

| 算法 | 平均 Cost | 完成率 |
|------|-----------|--------|
| **IPPO (3ES-7MD)** | **0.390** | **98.6%** |
| ExplabOff (2ES-5MD) | 0.380 | 100% |

## 关键发现

1. **IPPO 在复杂环境胜出**: 确定性环境不需要额外探索
2. **GNN 过平滑修复**: 1-layer GNN + agent embedding 将 2ES-3MD cost 从 0.998 降至 0.426（-57%）
3. **三网络各有地盘**: Hyper 擅长小/大配置，GNN 擅长中配置，Standard 最稳定
4. **超网络跨配置**: 单模型服务多配置，牺牲 ~10% 性能换通用性
5. **最优 checkpoint 在 10K**: ep9999 (cost=0.4096) 优于最终模型
6. **Trajectory buffer 清空**: 忘记清空导致数据累积 100K+，训练失效

## 引用

```bibtex
@inproceedings{ren2025explaboff,
  title={ExplabOff: Towards Explorative and Collaborative Task Offloading 
         via Mutual Information-Enhanced MARL},
  author={Ren, Jinbo and others},
  booktitle={IEEE INFOCOM},
  year={2025}
}
```

## 许可证

MIT License

---

**项目地址**: github.com/Neko-Yukari/marlproject
