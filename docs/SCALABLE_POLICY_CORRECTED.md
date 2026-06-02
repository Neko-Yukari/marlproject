# O(1) Parameter Scalable MARL — Corrected Design

## 用户纠正 (2026-06-01)
**错误**: 假设设备同质，针对当前测试环境定制
**正确**: 算法必须通用，支持异构设备，不限于当前环境

## 核心需求
1. **O(1) 参数**（不随M增长）
2. **支持异构设备**（不同CPU、不同能力）
3. **通用性**（不限于边缘计算）
4. **可扩展性**（M=7到M=1000+）

## 正确方案：Permutation Equivariant Policy with Device Features

### 核心思想
**网络参数固定，但通过输入特征区分设备**

```python
class ScalablePolicy(nn.Module):
    """O(1) parameters, handles heterogeneous devices."""
    
    def __init__(self, obs_dim, action_dim, device_feat_dim, hidden_dim=256):
        # 设备特征维度（CPU速度、内存、位置等）
        self.device_feat_dim = device_feat_dim
        
        # 网络参数：O(1)，与M无关
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + device_feat_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 全局聚合（Permutation Invariant）
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 策略头
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, observations, device_features):
        """
        Args:
            observations: [batch, M, obs_dim] - 局部观测（任务大小等）
            device_features: [batch, M, device_feat_dim] - 设备能力（CPU速度等）
        
        Returns:
            logits: [batch, M, action_dim] - 每个设备的动作分布
            values: [batch, M, 1] - 每个设备的价值
        """
        batch_size, M, _ = observations.shape
        
        # 1. 局部编码 [batch, M, hidden]
        local_input = torch.cat([observations, device_features], dim=-1)
        local_feat = self.encoder(local_input)
        
        # 2. 全局聚合（Permutation Invariant）
        # 对所有设备取平均/最大/和 → [batch, hidden]
        global_feat = local_feat.mean(dim=1)  # Permutation invariant!
        global_feat = self.global_pool(global_feat)
        
        # 3. 广播到每个设备 [batch, M, hidden]
        global_expanded = global_feat.unsqueeze(1).expand(-1, M, -1)
        
        # 4. 结合局部+全局 [batch, M, hidden*2]
        combined = torch.cat([local_feat, global_expanded], dim=-1)
        
        # 5. 输出（Permutation Equivariant）
        # 如果交换设备i和j，输出也交换 → 物理正确
        logits = self.policy(combined)  # [batch, M, action_dim]
        
        return logits
```

### 为什么这是通用的

**1. 异构设备支持**
```python
# 设备特征可以是：
device_features = [
    [1.0, 4GB, 0.1],   # Device 0: 1GHz, 4GB内存, 0.1W功耗
    [2.0, 8GB, 0.2],   # Device 1: 2GHz, 8GB内存, 0.2W功耗
    [0.5, 2GB, 0.05],  # Device 2: 0.5GHz, 2GB内存, 0.05W功耗
]
# 网络自动学习不同设备的策略！
```

**2. Permutation Equivariance**
```python
# 交换Device 0和Device 1：
obs = [obs_0, obs_1, obs_2] → [obs_1, obs_0, obs_2]
feat = [feat_0, feat_1, feat_2] → [feat_1, feat_0, feat_2]

# 输出也自动交换：
actions = [a_0, a_1, a_2] → [a_1, a_0, a_2]

# 这是物理正确的：设备没有固定ID，只有特征区别
```

**3. O(1) 参数**
```
网络大小 = encoder + global_pool + policy
         = O(hidden_dim²)  ← 与M无关！

无论M=7还是M=1000，参数量相同
```

### 数学性质

**Permutation Equivariance定理**：
对于任意排列π：
$$f(\pi(x)) = \pi(f(x))$$

即：输入设备重新排序，输出动作也重新排序。

**这是关键**：因为设备之间没有固有顺序，只有特征区别。

### 对比错误方案

| 特性 | 错误方案（MFSP同质） | 正确方案（Equivariant） |
|------|---------------------|------------------------|
| 设备假设 | 必须同质 | 支持异构 |
| 参数 | O(1) | O(1) ✓ |
| 通用性 | 仅限同质环境 | 任何多智能体环境 |
| 扩展性 | M可变 | M可变 ✓ |
| 物理正确性 | 假设设备可交换 | 排列等变性保证 |

### 实现计划

**Phase 1: 基础实现**（2-3小时）
- ScalablePolicy网络
- 设备特征提取（CPU速度等）
- 训练循环（累积梯度，大batch）
- 测试2ES-3MD

**Phase 2: 验证**（1-2小时）
- 对比IPPO（M=7）
- 测试M=20（合成环境）
- 验证O(1)参数

**Phase 3: 大规模测试**（overnight）
- M=100（合成环境）
- M=1000（如果可能）
- 对比Baseline

### 预期优势

1. **参数效率**: M=100时，参数量是IPPO的1/100
2. **训练速度**: 单次forward处理所有设备，GPU利用率100%
3. **泛化能力**: 见过M=10训练后，可直接用于M=100（zero-shot!）
4. **物理正确性**: 排列等变性保证算法不依赖设备ID

## 总结

**正确设计**: 
- 输入：局部观测 + 设备特征
- 处理：Permutation Equivariant网络
- 输出：每个设备的动作
- 参数：O(1)，与M无关

**vs 错误设计**:
- ❌ 假设同质设备
- ✅ 支持异构设备
- ❌ 针对特定环境
- ✅ 通用多智能体算法
