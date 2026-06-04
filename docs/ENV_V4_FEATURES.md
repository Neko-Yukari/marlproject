# Environment V4 - Advanced Dynamic MEC Environment

## 新特性概览

Environment V4 相比 V3 引入了10大改进，使训练/测试环境更加复杂和真实：

---

## 1. 动态任务到达 (Poisson Process)

**V3**: 每个slot每个MD**固定生成**一个任务（10 tasks/slot）

**V4**: 任务按**泊松过程随机到达**（arrival_rate=0.8，平均0.8 tasks/slot/MD）

```python
# V4: 每个MD每个slot有80%概率生成任务
if np.random.random() > self.arrival_rate:
    return None  # 没有任务到达
```

**影响**: 
- 任务数量不确定，需要处理"无任务"状态
- 需要学习何时保持空闲（do nothing）

---

## 2. 异构移动设备 (Heterogeneous MDs)

**V3**: 所有MD CPU相同（1GHz）

**V4**: MD CPU能力不同（0.4-2.0 GHz）

```python
MD_CPU_DB = {
    3: [0.8e9, 1.0e9, 1.5e9],  # 慢、中、快
    5: [0.6e9, 0.8e9, 1.0e9, 1.2e9, 1.5e9],
    10: [0.4e9, ..., 2.0e9],
}
```

**影响**:
- 不同MD的本地计算能力差异巨大
- 某些MD更适合本地计算，某些更适合卸载

---

## 3. 动态ES能力 (Time-Varying CPU)

**V3**: ES CPU固定不变

**V4**: ES CPU随时间波动（±20%）

```python
def _update_es_cpu(self):
    noise = np.random.uniform(0.8, 1.2)
    self.es_cpu_current[e] = self.es_cpu_base[e] * noise
```

**影响**:
- 无法依赖固定的ES性能
- 需要实时监控ES负载

---

## 4. 动态无线信道 (Time-Varying Bandwidth)

**V3**: 固定10 Mbps

**V4**: 带宽随时间变化（2-50 Mbps），具有时间相关性

```python
def _update_bandwidth(self):
    # Markov-like variation
    noise = np.random.uniform(0.5, 1.5)
    self.bw_current = prev_bw * 0.8 + base * 0.2 * noise
```

**影响**:
- 信道质量差时，卸载成本增加
- 需要动态决策：信道好时多卸载，差时少卸载

---

## 5. 任务队列累积 (Task Queuing)

**V3**: 每个slot独立，任务不累积

**V4**: 任务可以排队等待处理

```python
self.es_queues = [[] for _ in range(self.E)]
self.local_queues = [[] for _ in range(self.M)]
```

**影响**:
- 需要管理队列长度
- 任务可能在队列中等待多个slots
- 需要避免队列溢出

---

## 6. 任务优先级 (Task Priorities)

**V3**: 所有任务同等重要

**V4**: 任务有3个优先级（1=普通, 2=重要, 3=关键）

```python
priority = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
deadline = 1.0 + (priority - 1) * 0.5  # 1.0, 1.5, 2.0 slots
```

**影响**:
- 高优先级任务应优先处理
- 错过高优先级任务的惩罚更大（-20×priority）

---

## 7. 更多样化的配置

**V3**: 仅3种配置（2ES-3MD, 2ES-5MD, 3ES-7MD）

**V4**: 9种配置，覆盖更广的规模

| 配置      | M (MDs) | E (ESs) | 复杂度 |
|-----------|---------|---------|--------|
| 1ES-2MD   | 2       | 1       | 极简   |
| 1ES-3MD   | 3       | 1       | 简单   |
| 2ES-2MD   | 2       | 2       | 简单   |
| 2ES-3MD   | 3       | 2       | 中等   |
| 2ES-5MD   | 5       | 2       | 中等   |
| 3ES-5MD   | 5       | 3       | 中等   |
| 3ES-7MD   | 7       | 3       | 复杂   |
| 4ES-8MD   | 8       | 4       | 复杂   |
| 5ES-10MD  | 10      | 5       | 极复杂 |

---

## 8. 任务大小多样性

**V3**: 任务大小从预定义profile中选择

**V4**: 任务大小按分布随机生成

```python
task_type = np.random.choice(['small', 'medium', 'large'], p=[0.5, 0.35, 0.15])
size_mb = {
    'small': np.random.uniform(1.0, 3.0),
    'medium': np.random.uniform(3.0, 6.0),
    'large': np.random.uniform(6.0, 10.0),
}
```

---

## 9. 更长的Episode长度

**V3**: 10 slots/episode

**V4**: 20 slots/episode（允许队列动态发展）

---

## 10. 增强的观察空间

**V3**: obs_dim = 1 + E + E = 5-7

**V4**: obs_dim = 4 + E×3 + 3 = 13-22

```python
# V4 Observation:
obs[0] = task_size_norm          # 任务大小
obs[1] = priority / 3.0          # 优先级
obs[2] = deadline / 2.0          # 截止时间
obs[3] = local_queue_len / 5.0   # 本地队列长度
obs[4:4+E] = es_load             # ES负载
obs[4+E:4+2E] = es_cpu_norm      # ES CPU
obs[4+2E:4+3E] = es_queue_len    # ES队列长度
obs[4+3E] = md_cpu_norm          # MD CPU
obs[4+3E+1] = bw_quality         # 带宽质量
obs[4+3E+2] = slot_progress      # 进度
```

---

## V3 vs V4 对比总结

| 特性           | V3       | V4           | 复杂度提升 |
| -------------- | -------- | ------------ | ---------- |
| 任务到达       | 固定     | 泊松随机     | +++        |
| MD CPU         | 同质     | 异构         | ++         |
| ES CPU         | 固定     | 时变         | ++         |
| 带宽           | 固定     | 时变         | ++         |
| 队列           | 无       | 有           | +++        |
| 优先级         | 无       | 有           | ++         |
| 配置数         | 3        | 9            | +++        |
| Episode长度    | 10       | 20           | ++         |
| 观察维度       | 5-7      | 13-22        | ++         |
| **总体复杂度** | **简单** | **非常复杂** | **+++++**  |

---

## 使用示例

```python
from envs.paper_accurate_env_v4 import make_env_v4

# 创建环境
env = make_env_v4('3ES-7MD', 
                  episode_length=30,
                  arrival_rate=0.9,
                  hetero_mds=True,
                  dynamic_es=True,
                  dynamic_bw=True,
                  task_priorities=True,
                  queue_enabled=True)

# 运行一个episode
obs, _ = env.reset(seed=42)
for step in range(30):
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    if all(terms.values()):
        break

metrics = env.get_episode_metrics()
print(f"Completion: {metrics['completion_rate']:.1%}")
print(f"Avg Cost: {metrics['avg_cost']:.3f}")
print(f"Queue Rate: {metrics['queue_rate']:.1%}")
```

---

## 训练建议

V4环境复杂度大幅提升，建议：

1. **更长的训练时间**: 20K-50K episodes（vs V3的10K）
2. **更大的网络**: hidden_dim=512+（vs V3的128-256）
3. **课程学习**: 从简单配置（1ES-2MD）开始，逐步增加复杂度
4. **优先级奖励塑形**: 为高优先级任务完成提供额外奖励
5. **队列管理奖励**: 奖励保持队列短的行为

---

## 待实现功能

- [ ] 可视化（render）显示队列状态
- [ ] 任务依赖关系（DAG任务图）
- [ ] ES能量约束（绿色计算）
- [ ] 多episode任务（跨episode执行）
- [ ] 突发流量模式（非泊松到达）
