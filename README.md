# MARL Edge Computing - ExplabOff Replication & Extension

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

复现并扩展 INFOCOM 2025 论文 "ExplabOff: Towards Explorative and Collaborative Task Offloading via Mutual Information-Enhanced MARL" 的多智能体强化学习边缘计算任务卸载项目。

## 📋 项目概述

本项目实现了多种多智能体强化学习（MARL）算法，用于解决移动设备（MD）到边缘服务器（ES）的任务卸载问题。通过对比 IPPO、ExplabOff（MI增强）以及多种启发式基线算法，验证了强化学习在复杂边缘计算环境中的优势。

### 核心特性

- **多种MARL算法**: IPPO、ExplabOff（InfoNCE + L1Out MI估计）、MB-MERL
- **多环境配置**: 2ES-3MD、2ES-5MD、3ES-7MD（支持论文Table I参数）
- **随机任务轮廓**: 每episode随机选择任务大小配置，增强泛化性
- **GPU加速**: 支持CUDA加速，大batch累积训练（180x提速）
- **完整基线对比**: Size_Based、Greedy、Random、All_Local、All_ES、Round_Robin

## 🚀 快速开始

### 环境要求

```bash
# Python >= 3.9
# PyTorch >= 2.0 (with CUDA support recommended)
# PettingZoo, Gymnasium

pip install torch numpy pettingzoo gymnasium matplotlib
```

### 运行训练

```bash
# IPPO训练（3ES-7MD，20K episodes）
python trainers/stage1_train.py --config configs/3es7md_ippo.yaml

# ExplabOff训练（3ES-7MD，20K episodes）
python trainers/stage1_train.py --config configs/3es7md_explaboff.yaml

# 多环境benchmark（对比所有算法）
python trainers/stage2_train.py --benchmark all
```

### 运行基线对比

```bash
python evaluation/baseline_comparison.py --env-config configs/3es7md_env.yaml
```

## 📊 算法原理

### 1. IPPO (Independent Proximal Policy Optimization)

每个设备独立运行PPO算法，通过 clipped surrogate objective 更新策略：

```
L^{CLIP}(θ) = E_t [min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
```

其中 `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` 是重要性采样比率，Â_t 是GAE优势估计。

### 2. ExplabOff (MI-Enhanced PPO)

在IPPO基础上引入互信息（Mutual Information）增强奖励：

```
r_MI = μ·I(s; a) - ν·I(s; a|B-)
```

- **I(s; a)** (InfoNCE): 状态和动作之间的互信息，鼓励探索
- **I(s; a|B-)** (L1Out): 在差episode上的条件互信息，限制过度探索

**关键组件**:
- `InfoNCEEstimator`: 使用噪声对比估计学习互信息下界
- `L1OutEstimator`: 使用留一法估计互信息上界
- 双缓冲机制 B+ (优秀episode) 和 B- (差episode)

### 3. 环境模型

#### 任务处理
- **本地执行**: t_loc = (task_cycles) / (MD_CPU)
- **边缘执行**: t_edge = t_tx + t_wait + t_exe
  - t_tx = data_size / bandwidth
  - t_wait = 同ES上先前任务的执行时间之和
  - t_exe = task_cycles / ES_CPU

#### 成本函数（论文Eq.13）
```
cost = η·latency + (1-η)·energy
```

其中能量计算：
- 本地: E = κ·(MD_CPU)²·cycles
- 边缘传输: E_tx = P_tx·t_tx

### 4. MB-MERL (Model-Based Meta-RL)

元学习框架，结合模型预测控制：
- **CostPredictor**: 预测(task_size, es_load, es_cpu) → cost
- **MAML式适应**: 内循环快速适应新任务轮廓，外循环更新元参数
- **贪婪规划**: 基于适应后的cost predictor选择最优动作

## 📁 项目结构

```
.
├── agents/                 # 智能体实现
│   ├── ippo_agent.py       # IPPO算法
│   ├── explaboff_agent.py  # ExplabOff (MI增强)
│   ├── mbmerl_agent.py     # 基于模型的元学习
│   └── networks/           # 神经网络组件
│       ├── actor_critic.py # Actor-Critic网络
│       └── mi_estimator.py # MI估计器 (InfoNCE, L1Out)
├── envs/                   # 环境实现
│   ├── paper_accurate_env_v3.py  # 主环境 (支持多配置)
│   └── vectorized_env.py   # 并行环境包装器
├── trainers/               # 训练脚本
│   ├── stage1_train.py     # 论文复现训练
│   ├── stage2_train.py     # 改进算法训练
│   └── stage3_train.py     # 新算法探索
├── evaluation/             # 评估工具
│   ├── baseline_comparison.py    # 基线对比
│   ├── cross_evaluation.py       # 跨环境/跨种子评估
│   └── behavior_analysis.py      # 行为分析
├── utils/                  # 工具函数
│   ├── reporter.py         # 训练报告生成
│   ├── metrics.py          # 评估指标
│   └── helpers.py          # 辅助函数
├── configs/                # 配置文件
│   ├── 2es3md_*.yaml       # 2ES-3MD配置
│   ├── 2es5md_*.yaml       # 2ES-5MD配置
│   └── 3es7md_*.yaml       # 3ES-7MD配置
├── docs/                   # 文档
│   ├── MIGRATION_PLAN.md   # PettingZoo迁移计划
│   └── SCALABLE_POLICY_CORRECTED.md  # 可扩展策略设计
├── results/                # 实验结果
│   └── [实验名称]/         # 包含results.json, curves.csv, checkpoints/
└── papers/                 # 论文PDF和笔记
    └── ExplabOff_.../      # ExplabOff论文相关文件
```

## 📈 实验结果

### 3ES-7MD（最复杂配置）

| 算法 | 平均Cost | 完成率 | 相对改进 |
|------|----------|--------|----------|
| **IPPO** (最佳) | **0.390** | **98.6%** | — |
| ExplabOff | 0.394 | 97.1% | +1.0% vs IPPO |
| Greedy | 0.452 | 83.5% | +15.9% vs IPPO |
| Size_Based | 3.587 | 0.0% | 完全失败 |

### 2ES-5MD

| 算法 | 平均Cost | 完成率 |
|------|----------|--------|
| **ExplabOff** (最佳) | **0.380** | **100%** |
| IPPO | 0.391 | 100% |
| Size_Based | 0.404 | 100% |

### 关键发现

1. **IPPO在复杂环境胜出**: 3ES-7MD中IPPO（0.390）优于ExplabOff（0.394），因为确定性环境不需要额外探索
2. **ExplabOff在中等环境有效**: 2ES-5MD中MI探索帮助找到更好策略
3. **启发式基线局限性**: Size_Based在3ES-7MD完全失败（3.587 cost），验证了RL的必要性
4. **LR Decay关键**: StepLR防止5K+ episodes后的过拟合
5. **GPU大batch**: update_every=500累积策略使10K训练从30小时→10分钟

## 🔧 配置说明

### 环境配置示例 (configs/3es7md_env.yaml)

```yaml
environment:
  num_md: 7          # 移动设备数量
  num_es: 3          # 边缘服务器数量
  num_slots: 10      # 每episode时隙数
  slot_duration: 1.0 # 时隙长度(秒)
  deadline: 1.0      # 任务截止时间
  eta: 0.5           # 时间-能量权重
  
  # 设备CPU (Hz)
  md_cpu: 1.0e9
  es_cpus: [6.0e9, 12.0e9, 12.0e9]
  
  # 任务配置 (Mb)
  task_sizes: [5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5]
  task_variation: 0.05  # ±5%随机扰动
  
  # 传输
  bandwidth: 10.0e6  # 10 Mbps
  tx_power: 0.1      # 传输功率(W)
```

### 训练配置示例 (configs/3es7md_ippo.yaml)

```yaml
algorithm: IPPO
hidden_dim: 128
num_layers: 2
lr: 5.0e-5
lr_step: 5000      # LR衰减步数
lr_gamma: 0.5      # LR衰减率

training:
  episodes: 20000
  update_every: 500    # 累积500 episodes后更新
  batch_size: 4096     # GPU大batch
  num_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio: 0.2
  entropy_coeff: 0.01
  
action_masking: true   # 启用动作掩码
```

## 🐛 已修复的关键Bug

1. **任务一致性**: `_get_obs()`和`step()`生成不同任务大小 → 添加`self._slot_tasks`缓存
2. **L1OutEstimator公式**: `log(avg(logs))` → `avg(logs) - log(B)`
3. **InfoNCE负采样**: 可能包含正样本 → 排除自身索引
4. **等待时间计算**: 只计算lower-index设备 → 计算所有同ES设备

## 📝 引用

如果您使用本项目，请引用原始论文：

```bibtex
@inproceedings{ren2025explaboff,
  title={ExplabOff: Towards Explorative and Collaborative Task Offloading via Mutual Information-Enhanced MARL},
  author={Ren, Jinbo and others},
  booktitle={IEEE INFOCOM},
  year={2025}
}
```

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献

欢迎提交Issue和PR！主要改进方向：
- PettingZoo API完整迁移（进行中）
- Transformer-based Critic
- 课程学习（简单→复杂环境）
- 真实世界数据验证

## 📧 联系方式

项目地址: [github.com/Neko-Yukari/marlproject](https://github.com/Neko-Yukari/marlproject)

---

**状态**: 论文复现完成 (Stage 1: 100%) | 优化探索完成 (Stage 2: 100%) | 新方法原型 (Stage 3: 50%)
