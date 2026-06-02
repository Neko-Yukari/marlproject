# O(log M) Scalable MARL Algorithm Design

## Problem Analysis

Current IPPO/ExplabOff complexity:
- **Parameters**: O(M) - each device has its own network
- **Inference**: O(M) - sequential or parallel but M separate forwards
- **Training**: O(M * batch_size) - M agents updated separately
- **Action space**: (E+1)^M - exponential!

Target for M=1000 devices:
- Current: 1000 networks, 1000 forward passes
- Goal: O(log M) ≈ 10 layers, O(1) parameters

## Proposed Solutions

### Option 1: Hierarchical Coordinator (HC) - Strict O(log M)
**Architecture**: Binary tree over devices

```
Level 0 (leaves):    M devices [obs encode]
Level 1:             M/2 aggregators [aggregate 2 children]
Level 2:             M/4 aggregators
...                    
Level log M:         1 root coordinator [global decision]
```

**Decision flow**:
1. **Upward aggregation** (log M parallel layers):
   - Each node computes: `f(child1_feature, child2_feature)`
   - Shared network across all nodes
   - Latency: O(log M) (tree depth)

2. **Downward decision** (log M parallel layers):
   - Root decides coarse allocation to subtrees
   - Each internal node refines for its children
   - Leaves output final actions

**Complexity**:
- Latency: O(log M) ✓
- Parameters: O(1) ✓ (single shared network)
- Communication: O(M) total but tree-structured

**Pros**: Strict O(log M), naturally handles any M
**Cons**: New architecture, tree structure fixed by device IDs

---

### Option 2: Mean Field Shared Policy (MFSP) - O(1) Parameters
**Core idea**: All devices share ONE network

**Input per device**: `[local_obs, mean_field_state]`
- `mean_field_state`: Aggregated statistics of ALL devices
  - Mean task size, ES load distribution, completion rate
  - Dimension: O(1) regardless of M

**Output per device**: Action probability distribution

**Complexity**:
- Parameters: O(1) ✓ (single network)
- Per-device inference: O(1) ✓
- Total inference: O(M) but embarrassingly parallel → O(1) on GPU
- Mean field update: O(M) aggregation (can be done in O(log M) with tree)

**Pros**: Simple, proven in mean-field MARL literature
**Cons**: Assumes device homogeneity, may miss fine-grained interactions

---

### Option 3: Centralized Attention Policy (CAP) - O(1) Parameters
**Core idea**: One transformer processes ALL devices simultaneously

**Architecture**:
```python
class CAP(nn.Module):
    def forward(self, observations):  # [M, obs_dim]
        # Self-attention across devices
        attended = self.set_transformer(observations)  # [M, hidden]
        # Decode actions for all devices
        actions = self.decoder(attended)  # [M, action_dim]
        return actions
```

**Complexity**:
- Parameters: O(1) ✓
- Forward pass: O(M^2) for full attention, O(M) for linear attention
- With linear attention + GPU: effectively O(1) latency!

**Pros**: Handles variable M (like transformer handles variable sequence length)
**Cons**: O(M^2) attention cost, need linear attention variant

---

### Option 4: Task-Priority Scheduler (TPS) - Hybrid RL + Algorithm
**Core idea**: RL learns priorities, deterministic scheduler assigns

**RL Network**:
- Input: Global task queue statistics
- Output: Priority weights for ES assignment rules
- Parameters: O(1) ✓

**Scheduler** (O(M log M) deterministic):
1. Sort tasks by size: O(M log M)
2. Sort ES by speed: O(E log E)  
3. Assign greedily: O(M log E) with binary search

**Complexity**:
- RL part: O(1) ✓
- Scheduler: O(M log M) (but this is deterministic code, not learned)

**Pros**: Best of both worlds - RL for adaptation, algorithm for efficiency
**Cons**: Not end-to-end learned

---

## Recommendation

For **strict O(log M)**: Option 1 (Hierarchical Coordinator)
For **practical scalability**: Option 3 (CAP with linear attention) or Option 2 (MFSP)

My recommendation: **Start with MFSP** (simplest, proven) then upgrade to **Hierarchical** if needed.

## Implementation Plan

### Phase 1: MFSP (2-3 hours)
- Single shared policy network
- Mean field state computation
- Test on 2ES-3MD → 2ES-20MD

### Phase 2: Hierarchical (1-2 days)
- Binary tree structure
- Upward/downward network
- Test scalability to M=100+

### Phase 3: Hybrid (optional)
- Combine MFSP + deterministic scheduler
- Best performance + efficiency

## Expected Results

| M Devices | IPPO (Current) | MFSP | Hierarchical | Speedup |
|-----------|---------------|------|--------------|---------|
| 7         | 0.390         | 0.39x| 0.39x        | 1x      |
| 20        | ~0.5 (slow)   | 0.40x| 0.40x        | 10x     |
| 100       | Infeasible    | 0.42x| 0.42x        | 100x    |
| 1000      | Infeasible    | 0.45x| 0.45x        | 1000x   |
