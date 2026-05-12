# MARL Edge Offloading — 完整架构设计文档

> **框架**: PettingZoo ParallelEnv + PyTorch + NetworkX
> **核心参考**: ExplabOff (INFOCOM 2025), Yang et al. (IEEE TNSM 2023), IPPO/MAPPO (NeurIPS 2022), MIPI (NeurIPS 2023), GNNComm-MARL (2024)

---

## 1. 项目概述

### 1.1 问题定义

在边缘计算场景中，多台移动设备 (MD) 的计算任务需要决定"本地执行还是卸载到边缘服务器"。当设备数量多、任务动态变化时，这是一个**去中心化多智能体调度问题**。

### 1.2 需求分解

| 阶段 | 权重 | 要求 | 本设计 |
|------|------|------|--------|
| **Stage 1** | 30% | 仿真环境 + IPPO 基线 (无通信) | PettingZoo ParallelEnv + 参数共享 IPPO |
| **Stage 2** | 35% | Explaboff 互信息方案 + 评估 | 严格按 INFOCOM 2025 论文的 MI 机制 |
| **Stage 3** | 35% | 大规模场景优化 | GNN 通信 + 参数共享 + 课程学习 |
| **Stage 4** | +10% | GUI 实时演示 | 训练过程 Web Dashboard |

### 1.3 三个核心挑战

1. **可扩展性** — 设备数量动态变化 (5→50+)，算法需保持收敛
2. **隐式协作** — D2D 通信不可用，只能通过 CTDE 训练隐式协调
3. **公平性与可解释性** — 资源分配不能倾斜，决策过程需可追溯

---

## 2. 学术基础

| # | 论文 | 来源 | 用途 | 本地文件 |
|---|------|------|------|---------|
| 1 | **ExplabOff**: Ren et al. "ExplabOff: Towards Explorative and Collaborative Task Offloading via Mutual Information-Enhanced MARL" | INFOCOM 2025 (CCF-A) | Stage 2 核心 | `papers/ExplabOff_...pdf` |
| 2 | **IPPO**: de Witt et al. "Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?" | ICLR 2020 Workshop | Stage 1 算法 | `papers/IPPO_deWitt2020.pdf` |
| 3 | **MAPPO**: Yu et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" | NeurIPS 2022 (CCF-A) | Stage 1 对比 | `papers/MAPPO_Yu2022.pdf` |
| 4 | **MIPI**: Wang et al. "Mutual-Information Regularized Multi-Agent Policy Iteration" | NeurIPS 2023 (CCF-A) | MI 正则化参考 | `papers/MIPI_Wang2023_NeurIPS.pdf` |
| 5 | **GNNComm-MARL**: "Graph Neural Network Meets Multi-Agent Reinforcement Learning" | IEEE Wireless Comm. 2024 | Stage 3 GNN 架构 | `papers/GNNComm-MARL_2024.pdf` |
| 6 | **TapFinger**: Li et al. "Task Placement and Resource Allocation for Edge ML: A GNN-based MARL Paradigm" | IEEE TPDS 2023 (CCF-A) | Stage 3 异构 GAT | `papers/TapFinger_Li2023.pdf` |
| 7 | **FGNN-MADRL**: Wu et al. "Optimizing AoI in VEC with Federated GNN MARL" | arXiv:2407.02342 | Stage 3 GNN 实现 | `papers/FGNN-MADRL_Wu2024.pdf` |
| 8 | **Com-DDPG**: "Com-DDPG: A Multiagent RL-based Offloading Strategy for MEC" | arXiv:2012.05105 | 任务模型补充 | `papers/Com-DDPG_Li2020.pdf` |
| 9 | **Yang et al.**: "Cooperative Task Offloading for MEC Based on MADRL" | IEEE TNSM 2023 (CCF-B) | 环境建模参考 | 需机构访问 |

---

## 3. 系统模型

> 主要基于 ExplabOff (INFOCOM 2025) 的系统模型定义。

### 3.1 网络拓扑

- `M` 个移动设备 (MD)，`E` 个边缘服务器 (ES)，`N` 个时隙
- ES 部署在基站，通过无线链路服务 MD
- MD 之间**无 D2D 直接通信**

```
MD m:
   CPU: f_m (cycles/s)
   能量: ε_m (J)
   每时隙任务: j_n = {d_n, c, t_max}
     d_n: 数据量 (bits), c: CPU周期/bit, t_max: 延迟约束 (s)

ES e:
   CPU: f_e (cycles/s)
   队列: q_e
```

### 3.2 部分卸载动作

每个 MD 的动作 `a_m = {ρ_m, ϵ_m}`:

| 参数 | 范围 | 含义 |
|------|------|------|
| `ρ_m` | [0, 1] | 卸载比例: 0=全部本地, 1=全部卸载 |
| `ϵ_m` | {0..E} | 0=本地, 1..E=目标 ES |

### 3.3 信道模型

**块衰落模型** (Jakes + LTE):

```
增益: g_{m,e,n} = |f̃_{m,e,n}|² · f̂_{m,e,n}
  小尺度 f̃: Jakes 一阶 Gauss-Markov [28]
  大尺度 f̂: LTE 对数正态 shadowing [3GPP TR 36.814]

干扰: I_{m,e,n} = Σ_{m'≠m} 1(ρ_{m'}≠0)·p_tran·g_{m',e,n}
速率: υ_{m,e,n} = B·log₂(1 + p_tran·g/(I + σ²))
```

### 3.4 延迟模型

```
边缘: t_edge = t_tran + t_wait + t_start + t_exe
  传输: t_tran = ρ·d / υ
  排队: t_wait = (前序总数据量)·c / f_e
  启动: t_start = 首次卸载到该 ES 的初始化开销
  计算: t_exe = ρ·d·c / f_e

本地: t_loc = (1-ρ)·d·c / f_m

总延迟: t = max(t_loc, t_edge)  [本地和边缘并行]
```

### 3.5 能耗模型

```
边缘能耗: e_edge = p_tran · ρ·d / υ       [仅传输]
本地能耗: e_loc = ξ·f_m²·(1-ρ)·d·c        [CMOS 动态功耗, ξ≈10⁻²⁸]
总能耗:   e = e_edge + e_loc
```

### 3.6 系统成本

```
单任务: c_m = η·t_m + (1-η)·e_m,   η∈[0,1]
系统:   C = 1/N · Σ_n 1/M · Σ_m c_{m,n}
```

问题 `min_{ρ,ϵ} C` 受限于能量非负、延迟不超 t_max，**已证明 NP-hard** (归约至广义分配问题)。

---

## 4. MDP 形式化

> 基于 ExplabOff 论文 Section III。

### 4.1 全局 MDP

5-元组: `(s_n, a_n, P, γ, r_n)`

| 元素 | 定义 |
|------|------|
| **s_n** | `{d_{m,n}, q_{e,n}, ε_{m,n} | m∈M, e∈E}` |
| **a_n** | `{ρ_{m,n}, ϵ_{m,n} | m∈M}` 联合动作 |
| **P** | 未知转移概率 |
| **r_n** | 与 C 反相关 + 约束违反惩罚 |
| **γ** | 0.99 (折扣因子) |

### 4.2 去中心化执行 (Dec-POMDP)

CTDE 范式: critic 训练时可访问全局状态，actor 执行时只用**局部观测**:

```
s_{m,n} = {
    d_{m,n},           # 当前任务数据量
    t_max,             # 延迟约束
    ε_{m,n},           # 剩余能量
    q̂_{e,n},           # 各 ES 队列摘要 (近似, 可能过时)
    υ̂_{m,e,n},         # 到各 ES 估算速率
}
```

---

## 5. 环境架构 (PettingZoo)

### 5.1 接口设计

```python
class EdgeOffloadEnv(ParallelEnv):
    agents = ['device_0', ..., 'device_{M-1}']
    
    def step(self, actions: Dict[str, Dict]) -> Tuple:
        """
        actions = {'device_0': {'offload_ratio': 0.7, 'target_es': 2},
                   'device_1': {'offload_ratio': 0.0, 'target_es': 0}}
        Returns: observations, rewards, terms, truncs, infos
        """
```

### 5.2 每时隙流程

```
1. 每 MD 观察局部状态
2. 每 MD 独立决定动作
3. 环境并行收集所有动作
4. 计算信道增益、干扰、传输速率
5. 分发任务到 ES 队列 + 本地执行
6. 推进 ES 队列, 完成计算, 检查超时
7. 奖励: r_n = -C_n - penalty(违反约束)
8. 能量更新: ε_{n+1} = ε_n - e_n
9. 生成新任务
```

---

## 6. 算法管线

### 6.1 Stage 1: IPPO 基线

**目标**: 无显式通信的独立 PPO，性能下界。

```
Architecture:
  Actor: π_φ(s_m) → a_m
  Critic: V_θ(s_m) → value
  所有 MD 共享参数 + agent_id embedding

Training:
  For episode:
    For t=1..T:
      每 MD 用局部观测选动作
      环境执行联合动作 → 全局 reward
      存储 (s_m, a_m, r, s'_m)
    计算 GAE → PPO loss → 更新
```

**对比基线**: MAPPO (集中式 critic)。

### 6.2 Stage 2: ExplabOff

> 严格按 INFOCOM 2025 论文实现。

**核心**: critic 目标引入 `I(s_n; a_n)` — 全局状态与联合动作的互信息。

**数学推导** (论文 Section IV-B):

三项设计目标的代数恒等变换:
```
+H(a)                    ← 探索 (最大化联合动作熵)
-H(a^{-m} | a^m, s)     ← 协作 (确定他人行为)
-H(a^m | s)             ← 确定性 (给定状态动作应确定)

三者之和 = H(a) - H(a, s) + H(s) = I(a; s)
```

**Critic 目标**: `J(Q) = E[γ^{n-1}·(r + I(a; s))]`

**优质/劣质 MI 区分**:

```
B+ (优质 buffer): episode reward > 历史最优
  → 最大化 MI  (InfoNCE 估计下界)
  → 强化最优协作

B- (劣质 buffer): FIFO, 最新低 reward episode
  → 最小化 MI  (L1Out 估计上界)
  → 打破次优协作

合成: Î(s; a) = μ·I_NCE - ν·I_L1Out
增强奖励: r̂ = r + Î(s; a)
```

### 6.3 Stage 3: 大规模优化

**三层设计**:

```
Layer 1 — 参数共享 + Embedding:
  Actor/Critic 全 MD 共享权重
  agent_id embedding 区分个体

Layer 2 — GNN 邻域通信 (GNNComm-MARL):
  图: 节点=MD, 边=物理距离+共享ES竞争
  邻接矩阵: Gaussian 核 + 阈值截断
  GAT 聚合 top-k 邻域消息

Layer 3 — 课程学习:
  5→10→20→50 设备, 继承权重逐步扩展
  每阶段冻结底层 GNN, 只训练策略头
```

---

## 7. 公平性与可解释性

### 7.1 公平性

1. **虚拟公平队列**: MD 维护历史卸载 debt，补偿高 debt 设备
2. **Lagrangian 约束**: Jain 公平指数为约束，乘子调控
3. **加权公平调度**: ES 按历史资源使用加权

### 7.2 可解释性

1. **反事实分析**: "如果选了其他 ES 会怎样?"
2. **GAT 权重追溯**: 显示哪条邻域消息影响决策
3. **成本分解**: 传输/排队/计算/能耗各占多少
4. **协作标注**: B+/B- 标记当前协作质量

---

## 8. 目录结构

```
E:\MARL-IPPOAndMore\
├── DESIGN_DOCUMENT.md
├── papers/                    # 参考文献 + 论文 PDF
├── envs/
│   ├── edge_offload_env.py   # PettingZoo ParallelEnv
│   └── network_model.py      # 信道、干扰、拓扑
├── agents/
│   ├── ippo_agent.py
│   ├── mappo_agent.py
│   ├── explaboff_agent.py
│   └── networks/
│       ├── actor_critic.py
│       ├── mi_estimator.py   # InfoNCE + L1Out
│       └── gat_module.py
├── trainers/
│   ├── stage1_train.py
│   ├── stage2_train.py
│   └── stage3_train.py
├── evaluation/
│   ├── compare.py
│   └── fairness.py
├── configs/
│   ├── default.yaml
│   ├── explaboff.yaml
│   └── large_scale.yaml
└── utils/
    ├── task_device.py
    ├── helpers.py
    └── metrics.py
```

---

## 9. 参数与来源

### 9.1 物理参数 (基于 ExplabOff)

| 参数 | 值 | 来源 |
|------|-----|------|
| MD CPU | 1×10⁹ cyc/s | ExplabOff Fig.2 |
| ES CPU | 7~14×10⁹ cyc/s | ExplabOff Fig.4 |
| 发射功率 | 0.1 W | 移动设备典型值 |
| 能效系数 ξ | 10⁻²⁸ | CMOS [33] |
| 带宽 B | 10 MHz | LTE |
| 噪声 σ² | -114 dBm | 热噪声 |
| 衰落 | Jakes + LTE | [28][29] + 3GPP TR 36.814 |
| 任务数据 | 0.1~1 Mbits | AR/VR 帧 |
| CPU 周期 c | 1000 cyc/bit | 视频压缩 |
| 延迟约束 | 100~500 ms | 实时交互 |

### 9.2 算法参数

| 参数 | 值 | 来源 |
|------|-----|------|
| lr | 5×10⁻⁵ | ExplabOff |
| γ | 0.99 | 标准 |
| λ (GAE) | 0.95 | 标准 |
| clip ε | 0.2 | 标准 |
| MI 权重 μ,ν | 0.01 | ExplabOff |
