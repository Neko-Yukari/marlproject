# 泛化失败原因分析报告

## 实验设置

- **训练配置**: 3ES-7MD（7个MD，3个ES）
- **测试配置**: 2ES-3MD、2ES-5MD、3ES-7MD
- **网络**: GNNPolicy（1层GNN + agent embedding）
- **算法**: IPPO 和 ExplabOff
- **检查点**: ep1000、ep5000、ep10000、final（共10K episodes）

## 核心发现

### 1. 跨配置评估：所有模型在2ES配置上都选择 action=2

| 模型 | 检查点 | 2ES-3MD cost/comp | 2ES-5MD cost/comp | 3ES-7MD cost/comp |
|------|--------|-------------------|-------------------|-------------------|
| **IPPO+GNN** | ep1000 | 0.4720 / 66.6% | 0.4486 / 83.8% | 0.6388 / 20.9% |
| **IPPO+GNN** | ep5000 | 0.4720 / 66.6% | 0.4486 / 83.8% | 0.4164 / 94.2% |
| **IPPO+GNN** | ep10000 | 0.4720 / 66.6% | 0.4486 / 83.8% | 0.5257 / 42.8% |
| **IPPO+GNN** | final | 0.4720 / 66.6% | 0.4486 / 83.8% | 0.5257 / 42.8% |
| **ExplabOff+GNN** | ep1000 | 0.4720 / 66.6% | 0.4486 / 83.8% | 0.4251 / 71.7% |
| **ExplabOff+GNN** | ep5000 | 0.4720 / 66.6% | 0.4486 / 83.8% | 0.4251 / 71.7% |
| **ExplabOff+GNN** | ep10000 | 0.4720 / 66.6% | 0.4486 / 83.8% | **0.4107 / 99.6%** |
| **ExplabOff+GNN** | final | 0.4720 / 66.6% | 0.4486 / 83.8% | **0.4107 / 99.6%** |

### 2. 动作分布演变

**IPPO+GNN 在 3ES-7MD 上：**
- ep1000: 所有设备选择 action=2（随机）
- ep5000: 大多数设备选择 action=3（ES3，最快）
- ep10000/final: 所有设备选择 action=3（过度集中，完成率降至42.8%）

**ExplabOff+GNN 在 3ES-7MD 上：**
- ep10000/final: 设备分化——部分选择 action=3，部分选择 action=2（负载均衡，完成率99.6%）

**在 2ES-3MD / 2ES-5MD 上（所有检查点）：**
- 所有设备、所有检查点都选择 action=2（ES2）

## 失败原因分析

### 根本原因：GNN Policy Head 输出的是绝对 action index，而非 ES 语义

GNNPolicy 的 forward() 输出 `max_action_dim` 个 logits：
```
action 0 = local
action 1 = ES1
action 2 = ES2
action 3 = ES3
```

模型学到的是**动作索引的映射关系**，而不是"根据 ES 的 CPU/负载/任务大小选择最优 ES"。

### 具体表现

1. **在训练域（3ES-7MD）**：
   - action=3 通常对应最快的 ES3
   - 模型学会"选择最高索引 action"或"某些设备选3、某些选2"
   - IPPO 后期过度拟合到全选 action=3，导致 ES3 拥堵
   - ExplabOff 的 MI 奖励保持了动作多样性，因此表现更好

2. **在跨配置（2ES）**：
   - 最高可用 action 变成 2
   - 模型延续"选择最高可用索引"的启发式
   - 所有设备都选择 action=2（ES2）
   - 但 2ES-3MD 中 ES2 比 ES1 慢，所有任务涌向 ES2 导致严重拥堵

### 为什么 GNN 没有解决跨配置问题？

GNN 确实让模型能处理不同数量的节点（MD/ES），但：
- **节点嵌入是位置/索引相关的**：每个 MD 有独立的 agent_embed，每个 ES 在图中位置固定
- **Policy head 仍然基于绝对 action index 做决策**
- 模型没有学到"比较 ES1 和 ES2 的 CPU 速度"这种语义泛化

### 与 ExplabOff 的对比

- **ExplabOff 在训练域显著优于 IPPO**：99.6% vs 42.8% 完成率（ep10000）
- **但跨配置两者完全相同**：都卡在 action=2
- 结论：MI 奖励改善了 in-domain 的探索和负载均衡，但**没有改善跨配置语义泛化**

## 改进方向

### 方案 1：ES-aware Policy Head（推荐）

让 policy head 输出**每个 ES 的分数**，而不是 action index：
```python
# 当前
logits = actor_head(md_embedding)  # shape [max_action_dim]

# 改进
es_scores = score_head(md_embedding, es_embeddings)  # shape [num_es]
# 选择最高分的 ES，local 用单独的二元决策
```

这样当 ES 数量变化时，模型仍然可以比较每个 ES 的得分，而不是依赖 action index。

### 方案 2：输出 local/ES1/ES2/ES3 的概率，但基于 ES 属性排序

保留当前输出格式，但在训练时加入**ES 排序损失**：
- 让模型学会 ES3 的得分应高于 ES2，ES2 高于 ES1（基于 CPU 速度）
- 跨配置时，只保留存在的 ES 的概率并重新归一化

### 方案 3：在输入中加入 ES 相对能力编码

当前 ES 节点特征只有 cpu_norm、queue_fill_ratio 等。可以加入：
- 相对于最快 ES 的速度比例
- 相对于任务需求的处理能力
- 让模型显式比较 ES 能力

### 方案 4：混合训练 + Curriculum Learning

同时在多个配置（2ES、3ES）上训练，让模型看到不同数量的 ES：
- 每个 batch 随机采样 (M, E) 配置
- 迫使模型学习 ES 数量无关的策略

## 结论

当前 GNNPolicy 的跨配置失败**不是训练问题，也不是 GNN 架构问题，而是输出层设计问题**。Policy head 输出离散的 action index，天然导致模型学习到与 ES 数量相关的启发式（"选最高索引"），无法在 ES 数量变化时做出正确决策。

要真正实现跨配置泛化，必须将 action 输出从"索引"改为"ES 语义评分"，让模型根据 ES 的实际能力而非位置索引做选择。
