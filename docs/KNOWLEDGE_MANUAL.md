# MARL 边缘计算任务卸载 —— 团队知识手册

> 面向零基础成员的入门指南，涵盖从问题背景到实验结果的完整知识体系，配套 PPT 制作建议。

---

## 目录

1. [问题背景](#1-问题背景)
2. [环境模型](#2-环境模型)
3. [算法详解](#3-算法详解)
4. [我们的改进](#4-我们的改进)
5. [关键实验结果](#5-关键实验结果)
6. [快速使用指南](#6-快速使用指南)
7. [PPT 制作建议](#7-ppt-制作建议)

---

## 1. 问题背景

### 1.1 什么是边缘计算？

想象一个场景：你的手机正在运行一个人工智能应用，它需要实时分析视频画面。手机本身的算力非常有限，如果把视频数据传到千里之外的云数据中心处理，来回的网络延迟可能高达几百毫秒，根本无法满足"实时"的要求。

**边缘计算（Edge Computing）** 解决的就是这个问题：在网络边缘（靠近用户的地方）部署计算服务器，让数据处理在"家门口"完成，从而大幅降低延迟。

```
传统模式：手机 →（几百毫秒）→ 云数据中心 →（几百毫秒）→ 手机
边缘模式：手机 →（几十毫秒）→ 附近边缘服务器 →（几十毫秒）→ 手机
```

### 1.2 什么是 MEC（多接入边缘计算）？

**MEC（Multi-access Edge Computing，多接入边缘计算）** 是边缘计算的一种标准化架构。在这个模型中：

- **移动设备（MD, Mobile Device）**：手机、IoT 传感器、自动驾驶汽车等，算力有限（CPU 通常 1 GHz 级别）。
- **边缘服务器（ES, Edge Server）**：部署在基站附近的小型数据中心，算力较强（CPU 6-30 GHz）。
- **核心网络**：连接到云数据中心的骨干网络。

MEC 的核心思想是：**让移动设备把计算密集型任务"卸载"到附近的边缘服务器上执行，而不是自己硬算，也不是传到遥远的云端。**

### 1.3 任务卸载决策问题

边缘计算并非万能。卸载一个任务到边缘服务器需要经历以下流程：

```
1. 移动端把任务数据通过无线网络发送到边缘服务器（传输时间）
2. 边缘服务器执行计算（执行时间）
3. 边缘服务器把结果返回给移动端（传输时间，通常忽略，因为结果很小）
```

因此，卸载的总时间 = **传输时间 + 执行时间**。

关键问题是：**每个任务到底该在本地算，还是该卸载到边缘？如果卸载，该选哪一台边缘服务器？**

这个决策面临以下挑战：

| 挑战 | 说明 |
|------|------|
| **任务大小随机变化** | 每个时隙的任务大小在 2.5-5.5 Mb 之间随机波动 |
| **多设备竞争** | 多个移动设备同时卸载到同一台服务器时，会产生排队等待 |
| **硬性截止时间** | 每个任务必须在 1 秒内完成，否则任务失败 |
| **服务器异构** | 不同边缘服务器的 CPU 算力不同（6-30 GHz） |
| **多维代价** | 需要在**延迟（latency）**和**能耗（energy）**之间权衡 |

### 1.4 为什么用多智能体强化学习（MARL）？

这个决策问题用传统方法很难解决：

- **数学优化**：需要对未来的任务大小、设备行为做完美预测，实际不可能。
- **单智能体 RL**：所有移动设备共享一个大脑，维度爆炸，且设备间无法协调。
- **固定规则（如贪心算法）**：总是选当前最快的服务器，导致所有设备挤在同一台服务器上排队，反而变慢。

**MARL（Multi-Agent Reinforcement Learning，多智能体强化学习）** 是理想的方案：

- 每个移动设备作为一个独立的**智能体（Agent）**，拥有自己的策略网络。
- 智能体通过与环境交互学习：观察当前状态 → 做出决策 → 获得奖励/惩罚 → 更新策略。
- 经过大量训练后，智能体学会在"选最快的服务器"和"避免拥挤"之间找到平衡。

---

## 2. 环境模型

> 本章严格基于 INFOCOM 2025 论文 "ExplabOff" 的 Table I 参数。代码实现位于 `envs/paper_accurate_env.py`。

### 2.1 系统架构

```
┌─────────────────────────────────────────────────┐
│                  核心网络（云）                    │
└──────────┬──────────┬──────────┬─────────────────┘
           │          │          │
    ┌──────▼──┐ ┌─────▼───┐ ┌───▼──────┐
    │  ES₀   │ │  ES₁    │ │  ES₂    │     边缘层
    │ 6 GHz  │ │ 12 GHz  │ │  ...    │     (E台)
    └──▲──▲───┘ └──▲──▲──┘ └──▲──▲───┘
       │  │        │  │       │  │
    ┌──┴──┴┐   ┌──┴──┴┐  ┌──┴──┴┐
    │MD₀ MD₁│  │MD₂ MD₃│  │MD₄ MD₅│          设备层
    └───────┘  └───────┘  └───────┘          (M台)
```

我们支持三种标准配置：

| 配置名 | 移动设备数 M | 边缘服务器数 E | ES 算力配置 (GHz) |
|--------|:-----------:|:------------:|-------------------|
| 2ES-3MD | 3 | 2 | [6, 12] |
| 2ES-5MD | 5 | 2 | [15, 26] |
| 3ES-7MD | 7 | 3 | [10, 19, 26] |

### 2.2 时间模型

环境以**时隙（Slot）**为单位推进，每个时隙对应现实世界 1 秒。

- 每个 episode 包含 **10 个时隙**（10 秒进行 10 轮决策）。
- 在每个时隙开始时，**每台移动设备恰好产生 1 个计算任务**。
- 每个智能体观察环境状态，做出决策：0 = 本地执行，1 = 卸载到 ES₀，2 = 卸载到 ES₁，3 = 卸载到 ES₂，以此类推。
- 关键设定：**所有设备在同一时隙并行执行**，本地计算和边缘执行同时进行。

### 2.3 任务模型

每个移动设备在每个时隙产生的任务包含以下属性：

| 属性 | 说明 |
|------|------|
| **任务大小** | 2.5-5.5 Mb（随机采样，每个 episode 不同） |
| **计算量** | 任务大小 × 900 cycles/bit |
| **数据量** | 任务大小 × 10⁶ bits |

任务大小采用 **episode 级别随机化**：
- 代码为每种配置预定义了 5-8 组合法的任务大小组合（验证了存在可行解）。
- 每个 episode 开始时随机选择一组，并添加 ±5% 噪声。
- 这确保了每个 episode 的任务难度略有不同，测试智能体的泛化能力。

**合法配置的约束条件**（数学推导）：

1. **全部本地执行必然失败**：每个任务单独计算 > 1 秒（否则本地永远最优，无需卸载）。
2. **全部卸载到最快服务器必然失败**：5 个任务排队，最后一个必定超时（否则贪婪算法就是最优解）。
3. **存在最优分配方案**：聪明的调度能在 1 秒内完成所有任务（确保问题有解）。

### 2.4 延迟计算

本地执行的延迟：

```
t_local = 计算量 / MD_CPU
       = (任务大小 × 10⁶ × 900) / 10⁹
       = 任务大小(Mb) × 0.9 秒
```

边缘执行的延迟（假设卸载到第 e 台 ES）：

```
t_tx   = 数据量 / 带宽
       = (任务大小 × 10⁶) / 10⁷
       = 任务大小(Mb) × 0.1 秒

t_exe  = 计算量 / ES_CPU[e]
       = (任务大小 × 10⁶ × 900) / (ES算力)

t_wait = 排在当前任务前面的其他任务的总执行时间

t_edge = t_tx + t_wait + t_exe
```

**最终延迟** = max(t_local, t_edge)（同一时隙内本地和边缘并行）。

**关键约束**：如果最终延迟 > 1 秒（DEADLINE），则该任务**失败**。

### 2.5 能耗计算

| 执行方式 | 能耗公式 | 说明 |
|---------|---------|------|
| 本地执行 | `E = 10⁻²⁷ × (MD_CPU)² × 计算量` | 与 CPU 频率平方成正比 |
| 边缘卸载 | `E = 0.1 × t_tx` | 仅传输能耗，执行能耗由服务器承担 |

### 2.6 代价函数（Cost）

最终的优化目标是**最小化代价**，代价是延迟和能耗的加权和：

```
Cost = η × latency + (1-η) × energy
```

其中 η = 0.5（延迟和能耗各占一半权重）。

**奖励（Reward）设计**：
- 任务成功完成：reward = **-Cost**（负代价，代价越小奖励越大）
- 任务失败（超时）：reward = **-Cost - 10**（额外惩罚 -10）

### 2.7 观察空间

每个智能体在每个时隙观察到的状态向量：

| 维度 | 内容 | 归一化方式 |
|------|------|-----------|
| obs[0] | 当前任务大小 | size_mb / 10.0（截断到 1.0） |
| obs[1:E+1] | 各 ES 的负载比例 | 该 ES 接收的任务数 / 总卸载任务数 |
| obs[E+1:2E+1] | 各 ES 的 CPU 算力 | ES_CPU / 30 GHz（截断到 1.0） |

总维度 = 1 + 2×E，例如 3ES 配置的观察向量长度为 7。

### 2.8 动作空间

动作是一个离散整数：

| 动作值 | 含义 |
|:------:|------|
| 0 | 本地执行 |
| 1 | 卸载到 ES₀ |
| 2 | 卸载到 ES₁ |
| 3 | 卸载到 ES₂ |
| ... | 以此类推 |

动作空间大小 = E + 1（例如 2ES 配置有 3 个可选动作）。

---

## 3. 算法详解

### 3.1 IPPO（Independent PPO）

#### 3.1.1 核心思想

IPPO（Independent Proximal Policy Optimization）是多智能体强化学习中**最简洁有效的算法**之一。

它的核心思想非常简单：
- 每个智能体（移动设备）各自运行独立的 PPO 算法。
- 智能体之间**不共享网络参数**，也不直接通信。
- 但通过共享环境状态（ES 负载、CPU 信息），智能体学会了**隐式协调**。

#### 3.1.2 PPO 算法原理

PPO 是一种策略梯度方法，它的目标是逐步改进策略，同时防止"改得太猛"导致训练崩溃。

**核心公式 —— PPO Clipped Objective：**

```
L_CLIP(θ) = E[ min( r_t(θ) × A_t,  clip(r_t(θ), 1-ε, 1+ε) × A_t ) ]
```

其中：
- **r_t(θ)** = π_θ(a|s) / π_old(a|s)：新旧策略在相同状态下的动作概率之比。r > 1 表示新策略更喜欢这个动作。
- **A_t**（Advantage，优势函数）：在状态 s 下采取动作 a 比平均水平好多少。
  - A > 0：这个动作比预期好，应该增加概率。
  - A < 0：这个动作比预期差，应该减少概率。
- **clip(r, 1-ε, 1+ε)**：ε = 0.2，将概率比限制在 [0.8, 1.2] 之间。
  - 如果 r > 1.2 且 A > 0（想增加但已经增加太多了），截断，不再增加。
  - 如果 r < 0.8 且 A < 0（想减少但已经减少太多了），截断，不再减少。

**直觉理解**：PPO 像是一个谨慎的教练，每次只微调策略，确保不会因为一次训练就忘掉之前学到的好经验。

**补充损失项：**

```
Total Loss = Policy Loss + 0.5 × Value Loss - 0.01 × Entropy

Policy Loss = -L_CLIP（最小化负的目标 = 最大化目标）
Value Loss  = 0.5 × (预测价值 - 真实回报)²
Entropy     = 策略分布的熵（鼓励探索，防止过早收敛）
```

#### 3.1.3 IPPO 训练流程

```
For each episode:
    1. 环境 reset，随机选择任务大小配置
    2. For each slot (0~9):
        a. 每个智能体观察当前状态 obs
        b. 策略网络输出动作概率分布，采样得到动作
        c. 所有智能体同步执行动作
        d. 环境计算每个智能体的代价和奖励
        e. 存储 (obs, action, reward, value, log_prob) 到 trajectory buffer
    3. 每 100 个 episode，使用 trajectory buffer 更新策略：
        a. 计算 GAE（广义优势估计）
        b. 随机打乱，mini-batch 训练 4 个 epoch
        c. 使用 PPO clipped objective 更新参数

每 10K episodes 保存一次模型。
```

#### 3.1.4 GAE（广义优势估计）

GAE 解决了"如何准确估计每个动作的长期价值"的问题。

```
δ_t = r_t + γ × V(s_{t+1}) × (1 - done_t) - V(s_t)
A_t = δ_t + γλ × (1 - done_t) × A_{t+1}
```

参数：
- γ = 0.99：折扣因子，远期奖励的重要性
- λ = 0.95：GAE 参数，平衡偏差和方差

最终优势函数标准化：`A = (A - mean(A)) / (std(A) + 1e-8)`

#### 3.1.5 策略网络结构（Standard MLP）

```
输入层: obs_dim → 128 (Linear + ReLU)
隐藏层: 128 → 128 (Linear + ReLU)
输出层:
  ├── Actor:   128 → action_dim (动作 logits)
  └── Critic:  128 → 1 (状态价值估计)
```

### 3.2 ExplabOff（MI 增强的 MARL）

> 基于 INFOCOM 2025 论文 "Towards Explorative and Collaborative Task Offloading via Mutual Information-Enhanced MARL"。

#### 3.2.1 核心问题

IPPO 在训练中可能有如下问题：
- **探索不足**：所有智能体很快收敛到某个次优策略，比如都选同一台最快的服务器。
- **缺乏多样性**：没有机制鼓励智能体尝试不同的卸载组合。

ExplabOff 的解决方案：**在环境奖励的基础上，额外添加互信息（Mutual Information, MI）奖励**，鼓励智能体进行更多样化的探索。

#### 3.2.2 互信息奖励

```
r_total = r_env + r_MI

r_MI = μ × I_NCE(s; a) - ν × I_L1Out(s; a | B⁻)
```

其中：
- **I_NCE（InfoNCE）**：使用 InfoNCE 估计器估计状态 s 和动作 a 之间的互信息下界。鼓励高互信息，即智能体根据当前状态做出"有信息量"的动作，而非总选同一个。
- **I_L1Out（L1Out）**：使用 L1Out 估计器估计状态-动作在"差 episode"上的互信息上界。**减去**这一项，惩罚在差 episode 上的过度探索。
- **μ = 3.5, ν = 1.0**：平衡参数。

#### 3.2.3 双缓冲机制（Dual Buffers）

ExplabOff 维护两个经验缓冲池：

| 缓冲区 | 内容 | 用途 |
|--------|------|------|
| **B⁺（正缓冲区）** | 奖励超过历史最佳 episode 的经验 | 训练 InfoNCE 估计器，鼓励好行为 |
| **B⁻（负缓冲区）** | 奖励未超过历史最佳 episode 的经验 | 训练 L1Out 估计器，惩罚差行为 |

每个 episode 结束后：
- 如果该 episode 的总奖励 > 历史最高奖励 → 经验加入 B⁺
- 否则 → 经验加入 B⁻

#### 3.2.4 MI 估计器更新

- **InfoNCE 估计器**：在 B⁺ 上训练，最大化状态和动作之间的互信息。
- **L1Out 估计器**：在 B⁻ 上训练，学习哪些 (s, a) 组合在差 episode 中常见，以便惩罚它们。

两个估计器共享主网络的优化器，与 PPO 损失联合优化。

#### 3.2.5 IPPO vs ExplabOff 对比

| 维度 | IPPO | ExplabOff |
|------|------|-----------|
| 基础算法 | PPO | PPO |
| 奖励来源 | 纯环境奖励（-Cost） | 环境奖励 + MI 奖励 |
| 探索机制 | 熵正则化 | 互信息驱动探索 |
| 额外组件 | 无 | InfoNCE + L1Out 估计器 |
| 训练复杂度 | 较低 | 较高（额外估计器更新） |
| 适用场景 | 难度适中的环境 | 需要强探索的复杂环境 |

---

## 4. 我们的改进

> 本章介绍本项目的核心技术创新，对比 ExplabOff 论文原版实现。

### 4.1 正交架构（Orthogonal Architecture）

#### 4.1.1 设计理念

传统 MARL 框架将"网络结构"和"训练算法"耦合在一起。例如：
- IPPO 只能用 MLP 网络
- ExplabOff 只能用特定网络结构

**我们的创新**：将三件事彻底解耦，像乐高积木一样自由组合：

```
          ┌──────────────────┐
          │   训练入口        │
          │   train_unified   │
          └────────┬─────────┘
                   │
      ┌────────────┼────────────┐
      │            │            │
      ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ 网络层    │ │ 算法层    │ │ 环境层    │
│          │ │          │ │          │
│ · Standard│ │ · IPPO   │ │ · 3MD-2ES│
│ · Hyper  │ │ · ExplabOff│ │ · 5MD-2ES│
│ · GNN    │ │          │ │ · 7MD-3ES│
└──────────┘ └──────────┘ └──────────┘
```

**12 种配置组合** = 3 种网络 × 2 种算法 × 3 种环境，任意搭配。

#### 4.1.2 架构实现

**抽象接口 —— PolicyNetwork：**

```python
class PolicyNetwork(nn.Module):
    def forward(obs, action_mask=None, **kwargs) -> (action_probs, value):
        """所有策略网络必须实现此接口"""
    def get_action_dim() -> int: ...
    def get_obs_dim() -> int: ...
```

**统一训练器 —— PPOAgent：**

```python
class PPOAgent:
    def __init__(agent_id, policy_network, mi_plugin=None, ...):
        # policy_network 可以是 StandardPolicy / HyperPolicy / GNNPolicy
        # mi_plugin 非 None 时即为 ExplabOff 模式
```

**核心类关系图：**

```
PolicyNetwork (抽象接口)
├── StandardPolicy: MLP 网络，固定 (M, E) 配置
├── HyperPolicy:   超网络，单模型服务所有配置
└── GNNPolicy:     图神经网络，处理可变 M/E 数量

PPOAgent (统一训练器)
├── policy: PolicyNetwork (上述任意一种)
├── mi_plugin: Optional[MIPlugin] (ExplabOff 时启用)
└── update(): PPO clipped surrogate objective

MIPlugin (可选插件)
├── InfoNCEEstimator: I(s; a) 下界估计
├── L1OutEstimator:  I(s; a|B⁻) 上界估计
└── compute_reward(): r_MI = μ·I_NCE - ν·I_L1Out
```

#### 4.1.3 配置驱动设计

通过 YAML 配置文件或命令行参数，一行命令即可启动任意组合：

```bash
# YAML 配置模式（推荐）
python train_unified.py --config configs/ippo_gnn_3md2es.yaml

# 命令行模式
python train_unified.py --network gnn --algorithm ippo --md 3 --es 2 --episodes 10000

# ExplabOff + 超网络
python train_unified.py --network hyper --algorithm explaboff --md 5 --es 2
```

### 4.2 跨配置泛化（Cross-Config Generalization）

#### 4.2.1 问题：固定维度 MLP 的局限

标准 MLP 网络要求固定的输入维度和输出维度：

| 配置 | obs_dim | action_dim | 需要不同的网络 |
|------|:------:|:----------:|:------------:|
| 2ES-3MD | 5 | 3 | Network A |
| 2ES-5MD | 5 | 3 | Network B（obs 含义不同）|
| 3ES-7MD | 7 | 4 | Network C |

如果 M=3 训练的模型拿到 M=7 的环境，输入维度不匹配，根本无法运行。

#### 4.2.2 方案一：HyperNetwork（超网络）

**核心思想**：不直接训练 MLP 权重，而是训练一个"生成器网络"（HyperNetwork），它接收 (M, E) 配置码，输出适合该配置的 MLP 权重。

```
                    ┌─────────────┐
    M=3, E=2  ───►  │  Config     │  ───►  config_vec [64]
                    │  Encoder    │
                    └─────────────┘
                           │
                    ┌──────▼──────┐
                    │ Weight      │
                    │ Generator   │  ───►  W1, b1, W2, b2 (动态生成)
                    └─────────────┘

    obs ──────────►  生成的 MLP    ───►  action_probs, value
```

**Config Encoder 结构**：
- M 嵌入：max_M=10，32 维嵌入向量
- E 嵌入：max_E=5，32 维嵌入向量
- 拼接 → Linear(64, 64) → ReLU → Linear(64, 64) → config_vec [64]

**优势**：一个 HyperNetwork 模型可以服务 2ES-3MD、2ES-5MD、3ES-7MD 三种配置，无需重新训练。

**代价**：性能略低于专用 MLP（约 8% 损失），但换来了通用性。

#### 4.2.3 方案二：GNN Policy（图神经网络）

**核心思想**：将 MEC 系统建模为**二分图（Bipartite Graph）**，用图神经网络处理可变数量的 MD/ES 节点。

```
    MD₀ ──────────── ES₀
         ╲         ╱
    MD₁ ───╳───── ES₁
         ╱         ╲
    MD₂ ──────────── ES₂

    节点特征：
    - MD节点: [task_size, deadline_margin, relative_power, slot_progress]
    - ES节点: [cpu_norm, queue_fill, service_cap, wait_proxy]

    边：全连接二分图（MD ↔ ES，无 MD↔MD 或 ES↔ES连接）
```

**优势**：自然支持任意数量的 MD 和 ES 节点，不需要固定维度。

### 4.3 GNN 过平滑修复（Over-Smoothing Fix）

#### 4.3.1 问题发现

最初实现使用 2 层 GCN（图卷积网络）在全连接二分图上：

```
结果：2ES-3MD 的 Cost 高达 0.998（几乎等于随机策略的 1.0）
```

原因分析：全连接二分图中，经过 2 层消息传递后：
- **所有 MD 节点的嵌入向量几乎完全相同**（因为它们通过 ES 节点间接连接）。
- 智能体失去了区分不同设备的能力，无法做出差异化决策。
- 这被称为 **过平滑（Over-Smoothing）** 问题，是深度 GCN 的经典缺陷。

#### 4.3.2 修复方案

**V4 修复（两个关键改动）：**

1. **减少 GNN 层数：2 层 → 1 层**
   - 1 层 GNN 的 MD 节点只能直接接收 ES 的信息，不会通过其他 MD 间接混合。
   - 从根本上阻止了 MD-MD 之间的特征平滑。

2. **添加 Per-MD Agent Embedding（可学习的设备身份向量）**
   - 新增 `nn.Embedding(max_md, hidden_dim)`，为每个 MD 分配独立的可学习身份向量。
   - MD₀ 和 MD₁ 即使 GNN 消息传递相同，身份嵌入也不同。
   - ES 节点使用零向量作为身份嵌入。

```
输入特征 = [原始节点特征, 类型嵌入, Agent身份嵌入]
    ↓
输入投影: Linear → LayerNorm → ReLU
    ↓
GNN层 (×1): 消息传递 + 残差连接
    ↓
输出头: Actor(Linear) + Critic(Linear)
```

**修复效果：**

| 阶段 | GNN 2ES-3MD Cost |
|:-----|:-----------------:|
| 修复前 (2层GCN) | 0.998 |
| 修复后 (1层+AgentEmbed) | **0.426** |

代价恢复到合理范围，与 Standard MLP（0.446）相当。

### 4.4 配置系统（Config System）

#### 4.4.1 YAML 配置文件

项目提供 12 个预配置 YAML 文件，覆盖所有常见组合：

```
configs/
├── ippo_standard_3md2es.yaml    # IPPO + 标准MLP + 2ES-3MD
├── ippo_standard_5md2es.yaml    # IPPO + 标准MLP + 2ES-5MD
├── ippo_standard_7md3es.yaml    # IPPO + 标准MLP + 3ES-7MD
├── ippo_gnn_3md2es.yaml         # IPPO + GNN + 2ES-3MD
├── ippo_gnn_5md2es.yaml         # IPPO + GNN + 2ES-5MD
├── ippo_gnn_7md3es.yaml         # IPPO + GNN + 3ES-7MD
├── ippo_hyper_3md2es.yaml       # IPPO + Hyper + 2ES-3MD
├── ippo_hyper_5md2es.yaml       # IPPO + Hyper + 2ES-5MD
├── ippo_hyper_7md3es.yaml       # IPPO + Hyper + 3ES-7MD
├── explaboff_standard_3md2es.yaml
├── explaboff_gnn_3md2es.yaml
└── explaboff_hyper_3md2es.yaml
```

#### 4.4.2 配置文件结构

```yaml
network:
  type: gnn                    # standard | gnn | hyper
  hidden_dim: 128              # 隐藏层维度
  gnn_layers: 1                # GNN专属
  node_dim: 4                  # GNN专属
  max_action_dim: 4            # GNN/Hyper专属

algorithm:
  name: ippo                   # ippo | explaboff
  use_mi: false               # 是否启用 MI 插件
  lr: 0.00005                 # 学习率
  gamma: 0.99                 # 折扣因子
  gae_lambda: 0.95            # GAE λ
  clip_ratio: 0.2             # PPO 裁剪范围
  entropy_coeff: 0.01         # 熵系数
  value_coeff: 0.5            # 价值损失系数
  max_grad_norm: 0.5          # 梯度裁剪
  update_every: 100           # 多少 episode 更新一次
  num_epochs: 4               # 每次更新训练轮数
  batch_size: 128             # Mini-batch 大小

environment:
  num_md: 3                   # 移动设备数
  num_es: 2                   # 边缘服务器数
  slots: 10                   # 每 episode 时隙数
  randomize_profile: true     # 是否随机任务大小
  profile_noise: 0.05         # 任务大小噪声

training:
  num_episodes: 10000         # 总训练 episode 数
  log_interval: 100           # 日志间隔
  seed: 42                    # 随机种子

device: cuda                  # cpu | cuda
```

#### 4.4.3 CLI 参数覆盖

配置文件中的任何参数都可以通过命令行直接覆盖：

```bash
python train_unified.py --config configs/ippo_gnn_3md2es.yaml --episodes 5000 --seed 123
```

---

## 5. 关键实验结果

### 5.1 三种网络架构性能对比

> 以下数据来自 10,000 episodes 训练后的最佳 checkpoint，以平均 Cost 为评价指标（越低越好）。

| 配置 | Standard MLP | GNN (V4修复后) | HyperNetwork | 优胜者 |
|------|:-----------:|:------------:|:------------:|:------:|
| 2ES-3MD | 0.446 | 0.426 | **0.423** | Hyper |
| 2ES-5MD | **0.385** | **0.385** | 0.394 | GNN/Standard |
| 3ES-7MD | 0.407 | 0.405 | **0.400** | Hyper |

**数据解读**：

1. **GNN 在 2ES-3MD 的表现（0.426）**：经过过平滑修复后，GNN 在小型配置上与 MLP 旗鼓相当，甚至略优。
2. **HyperNetwork 在 2ES-3MD 和 3ES-7MD 表现最佳**：说明超网络在边缘配置（最小和最大）受益于共享学习。
3. **2ES-5MD 三者难分伯仲**：中等配置下各自能力相当，差距在统计误差范围内。

### 5.2 对比基线：IPPO vs ExplabOff

| 配置 | IPPO | ExplabOff | 分析 |
|------|:----:|:---------:|------|
| 2ES-3MD | 0.446 | 0.447 | 几乎相同，该环境不需要额外探索 |
| 2ES-5MD | 0.385 | 0.380 | ExplabOff 略优，MI 探索在中等环境有帮助 |
| 3ES-7MD | 0.390 | 0.394 | IPPO 反超，确定性环境不需要 MI |

### 5.3 关键发现

**发现 1：IPPO 在所有配置下都收敛到接近最优**

无需复杂的 MI 机制，独立 PPO 智能体已经能通过隐式协调找到很好的卸载策略。三组配置的平均 Cost 均在 0.40 左右，相比贪心算法（0.452）提升约 12%。

**发现 2：GNN 经过修复后匹配标准 MLP**

过平滑修复前 GNN Cost 高达 0.998，修复后降至 0.426，**修复带来了 57% 的 Cost 下降**。这验证了 1 层 GNN + Per-MD Agent Embedding 方案的有效性。

**发现 3：HyperNetwork 在边界配置最优，中等配置不稳定**

HyperNetwork 在 2ES-3MD（最小）和 3ES-7MD（最大）都排第一，但在 2ES-5MD（中等）表现最差。可能原因是中等配置的参数空间处于过度区域，权重生成不够精确。

**发现 4：所有方法在 3K-5K episodes 内趋于稳定**

训练曲线通常在 3,000 episodes 左右开始收敛，到 5,000 episodes 基本稳定。10,000 episodes 的长时间训练主要起 fine-tune 作用。

**发现 5：训练中的重要 Bug 修复**

| Bug | 影响 | 修复 |
|-----|------|------|
| Trajectory Buffer 未清空 | 跨 episode 的经验混在一起，训练不稳定 | 每个 episode 结束时调用 `clear_trajectory()` |
| 任务生成不一致 | `_get_obs()` 和 `step()` 中各自生成任务，导致观察和实际执行任务不同 | 在 `step()` 中统一生成并缓存到 `_slot_tasks` |
| GNN 嵌入未缓存 | 每个智能体调用 forward 时重新跑一遍 GNN，速度慢且梯度丢失 | 添加 `set_graph()` 一次性计算所有嵌入，forward 直接取缓存 |

---

## 6. 快速使用指南

### 6.1 环境安装

```bash
# 克隆项目
git clone <repo-url>
cd MARL-IPPOAndMore

# 安装依赖
pip install torch numpy pettingzoo gymnasium matplotlib pyyaml
```

### 6.2 训练

#### 单次训练

```bash
# YAML 配置模式（推荐，参数可追溯）
python train_unified.py --config configs/ippo_gnn_3md2es.yaml

# CLI 快捷模式
python train_unified.py --network gnn --algorithm ippo --md 3 --es 2 --episodes 10000

# IPPO + 标准MLP + 5MD-2ES
python train_unified.py --network standard --algorithm ippo --md 5 --es 2 --episodes 10000

# ExplabOff + HyperNetwork + 3MD-2ES
python train_unified.py --network hyper --algorithm explaboff --md 3 --es 2 --episodes 10000
```

#### 批量对比实验

```bash
# 运行全部 12 组配置的对比实验（耗时较长，建议 GPU）
python run_all_comparisons.py
```

### 6.3 评估

```bash
# 加载已训练模型进行评估
python train_unified.py --load results/your_model --mode eval --md 3 --es 2
```

### 6.4 验证环境

```bash
# 验证 PettingZoo API
python test_pettingzoo_api.py

# 端到端训练流水线验证
python validate_pipeline.py
```

### 6.5 参数速查表

| 参数 | 选项 | 默认值 | 说明 |
|------|------|:------:|------|
| `--config` | YAML 文件路径 | — | 配置文件（优先级最高） |
| `--network` | `standard`, `gnn`, `hyper` | `standard` | 网络架构 |
| `--algorithm` | `ippo`, `explaboff` | `ippo` | 训练算法 |
| `--md` | 整数 | 3 | 移动设备数量 |
| `--es` | 整数 | 2 | 边缘服务器数量 |
| `--episodes` | 整数 | 10000 | 训练 episode 数 |
| `--seed` | 整数 | 42 | 随机种子 |
| `--device` | `cpu`, `cuda` | 自动检测 | 计算设备 |
| `--load` | 模型路径 | — | 加载已有模型 |
| `--mode` | `train`, `eval` | `train` | 运行模式 |

### 6.6 项目依赖

| 包 | 版本 | 用途 |
|----|------|------|
| PyTorch | >= 2.0.0 | 深度学习框架 |
| NumPy | >= 1.24.0 | 数值计算 |
| PettingZoo | >= 1.24.0 | 多智能体环境接口 |
| Gymnasium | >= 0.29.0 | 强化学习环境标准 |
| Matplotlib | >= 3.7.0 | 训练曲线绘制 |
| PyYAML | >= 6.0 | 配置文件解析 |

---

## 7. PPT 制作建议

> 以下为面向 10 分钟技术分享的标准结构。每页建议 3-5 个要点，配合图表而非大段文字。

### Slide 1：标题页 + 问题陈述

**标题**：基于 MARL 的边缘计算任务卸载优化

**要点**：
- 边缘计算：在靠近用户的地方处理数据，降低延迟
- 核心挑战：多设备竞争有限边缘资源，如何在延迟和能耗间取得平衡
- 我们的方案：多智能体强化学习 + 正交化可组合架构

### Slide 2：MEC 系统模型（配图）

**要点**：
- M 台移动设备（MD，1 GHz CPU），E 台边缘服务器（ES，6-30 GHz）
- 每秒产生一个任务，大小 2.5-5.5 Mb 随机波动
- 每个任务必须在 1 秒内完成（硬性 deadline）
- 本地执行 vs 卸载到 ES：并行进行，最终延迟 = max(t_local, t_edge)

**建议配图**：画 MD → 无线网络 → ES → 网络的层次结构图。

### Slide 3：任务卸载决策

**要点**：
- 智能体在每个时隙需要决策：0=本地 / 1=ES₀ / 2=ES₁ / ...
- 观察空间：任务大小 + ES 负载 + ES 算力
- 奖励 = -Cost（Cost = 0.5×延迟 + 0.5×能耗）
- 失败惩罚：超时额外 -10

### Slide 4：为什么用多智能体强化学习？

**要点**：
- 传统方法局限：数学优化需要完美预测，贪心算法导致全部挤在最快 ES
- MARL 优势：每个设备独立学习，通过环境状态隐式协调
- IPPO：每个设备独立运行 PPO，简洁有效
- ExplabOff：在 IPPO 基础上增加互信息驱动探索

### Slide 5：IPPO 算法简介

**要点**：
- PPO = 策略梯度 + 裁剪约束，防止训练崩溃
- 核心公式：L = E[min(r×A, clip(r,0.8,1.2)×A)]
- GAE 估计每个动作的长期优势
- 每 100 episodes 更新一次，mini-batch 训练

**建议配图**：PPO clip 机制示意图（r-A 平面对比）。

### Slide 6：ExplabOff 互信息增强

**要点**：
- 在环境奖励上叠加 MI 奖励：r_total = r_env + μ·I_NCE - ν·I_L1Out
- I_NCE：鼓励智能体根据状态做出有信息量的决策
- I_L1Out：惩罚在差 episode 上的无效探索
- 双缓冲机制：B⁺ 存好经验，B⁻ 存差经验

### Slide 7：我们的正交架构（核心创新，配架构图）

**要点**：
- 三大模块解耦：网络层 × 算法层 × 环境层，自由组合
- PolicyNetwork 抽象接口 → StandardPolicy / GNNPolicy / HyperPolicy
- PPOAgent 通用训练器，MIPlugin 可选插件
- 12 种配置组合 = 3×2×3，YAML 驱动

**建议配图**：三层架构 + 可选插件的组合关系图。

### Slide 8：跨配置泛化

**要点**：
- 问题：固定维度 MLP 无法适配不同 M/E 配置
- GNN 方案：将 MEC 建模为二分图，1 层 GNN + Agent 身份嵌入
- HyperNetwork 方案：Config Encoder 生成动态权重
- GNN 关键修复：过平滑问题（2层→1层），Cost 从 0.998 降至 0.426

### Slide 9：实验结果

| 配置 | Standard MLP | GNN | HyperNetwork | 优胜 |
|------|:-----------:|:----:|:------------:|:----:|
| 2ES-3MD | 0.446 | 0.426 | **0.423** | Hyper |
| 2ES-5MD | 0.385 | 0.385 | 0.394 | GNN/Standard |
| 3ES-7MD | 0.407 | 0.405 | **0.400** | Hyper |

**要点**：
- 所有方法收敛到 Cost≈0.40，远优于贪心算法（0.452）
- GNN 修复后性能大幅提升
- HyperNetwork 在边界配置最优

### Slide 10：关键发现与结论

**要点**：
1. IPPO 在所有配置下收敛到接近最优，隐式协调有效
2. GNN 匹配标准 MLP 性能（过平滑修复关键）
3. HyperNetwork 在边界配置最优，中等配置不稳定
4. 正交架构设计可推广到其他 MARL 场景
5. 开源代码 + 12 个预配置 YAML，一行命令即可复现

---

## 附录

### A. 术语对照表

| 英文缩写 | 全称 | 中文 |
|:--------:|------|------|
| MEC | Multi-access Edge Computing | 多接入边缘计算 |
| MD | Mobile Device | 移动设备 |
| ES | Edge Server | 边缘服务器 |
| MARL | Multi-Agent Reinforcement Learning | 多智能体强化学习 |
| PPO | Proximal Policy Optimization | 近端策略优化 |
| IPPO | Independent PPO | 独立 PPO |
| MI | Mutual Information | 互信息 |
| GAE | Generalized Advantage Estimation | 广义优势估计 |
| GNN | Graph Neural Network | 图神经网络 |
| GCN | Graph Convolutional Network | 图卷积网络 |
| InfoNCE | Information Noise-Contrastive Estimation | 信息噪声对比估计 |

### B. 数学符号表

| 符号 | 含义 | 典型值 |
|:----:|------|:------:|
| M | 移动设备数量 | 3, 5, 7 |
| E | 边缘服务器数量 | 2, 3 |
| η | 延迟-能耗权重 | 0.5 |
| γ | 折扣因子 | 0.99 |
| λ | GAE 参数 | 0.95 |
| ε | PPO 裁剪范围 | 0.2 |
| μ | MI 奖励中 I_NCE 权重 | 3.5 |
| ν | MI 奖励中 I_L1Out 权重 | 1.0 |
| B⁺ | 正样本缓冲池（好 episode） | max 1000 |
| B⁻ | 负样本缓冲池（差 episode） | max 1000 |

### C. 参考文献

```bibtex
@inproceedings{ren2025explaboff,
  title={ExplabOff: Towards Explorative and Collaborative Task Offloading
         via Mutual Information-Enhanced MARL},
  author={Ren, Jinbo and others},
  booktitle={IEEE INFOCOM},
  year={2025}
}
```

---

> **文档版本**: v1
> **最后更新**: 2026-06-12
> **维护者**: MARL-IPPOAndMore 项目组
