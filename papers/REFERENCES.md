# MARL Edge Offloading — 参考文献索引

## 已下载论文 (7篇)

### 1. IPPO (de Witt et al., 2020)
- **标题**: Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?
- **作者**: Christian Schroeder de Witt, Tarun Gupta, Denys Makoviichuk, Viktor Makoviychuk, Philip H.S. Torr, Mingfei Sun, Shimon Whiteson
- **来源**: ICLR 2020 Workshop (arXiv:2011.09533)
- **文件**: `IPPO_deWitt2020.pdf`
- **用途**: Stage 1 算法基础 — IPPO 的理论和实验验证
- **关键结论**: 独立 PPO (IPPO) 在 SMAC 等多个 MARL 基准上可匹敌甚至超过 CTDE 方法。PPO 的策略裁剪 (policy clipping) 对非平稳环境中的稳定性至关重要。

### 2. MAPPO (Yu et al., 2022)
- **标题**: The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games
- **作者**: Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, Yi Wu
- **来源**: NeurIPS 2022 (arXiv:2103.01955)
- **文件**: `MAPPO_Yu2022.pdf`
- **用途**: Stage 1 对比基线 — CTDE 范式的标准 PPO 实现
- **关键结论**: MAPPO (Multi-Agent PPO) 在 SMAC、Google Football、Hanabi 等任务上取得 SOTA。关键设计选择：集中值函数、参数共享、recurrent policy。

### 3. MIPI (Wang et al., 2023)
- **标题**: Mutual-Information Regularized Multi-Agent Policy Iteration
- **作者**: Jiangxing Wang, Deheng Ye, Zongqing Lu
- **来源**: NeurIPS 2023 (CCF-A)
- **文件**: `MIPI_Wang2023_NeurIPS.pdf`
- **用途**: Stage 2 互信息设计 — 使用 MI 作为正则化项防止对团队信息的过度依赖
- **关键结论**: 最小化 I(π_i(a|s); s_{team}) 鼓励智能体学习在多变团队组合中鲁棒的策略，实现强零样本泛化。

### 4. Com-DDPG (2020)
- **标题**: Com-DDPG: A Multiagent Reinforcement Learning-based Offloading Strategy for Mobile Edge Computing
- **作者**: Li et al.
- **来源**: arXiv:2012.05105
- **文件**: `Com-DDPG_Li2020.pdf`
- **用途**: 环境建模补充 — 任务 DAG 依赖建模、能耗模型、多设备多服务器异构 MEC 框架
- **关键结论**: 引入 BRNN 作为 agent 间通信层，LSTM 捕捉内部状态。完整的状态空间包含任务依赖模型、优先级模型、能耗模型、延迟模型。

### 5. GNNComm-MARL (2024)
- **标题**: Graph Neural Network Meets Multi-Agent Reinforcement Learning: Fundamentals, Applications, and Future Directions
- **作者**: [IEEE Wireless Communications 论文]
- **来源**: arXiv:2404.04898
- **文件**: `GNNComm-MARL_2024.pdf`
- **用途**: Stage 3 GNN 架构 — GAT 辅助通信的系统性设计
- **关键结论**: GNNComm-MARL 通过图注意力网络 (GAT) 有效采样邻域、选择性聚合消息，实现高性能低开销通信。系统性地涵盖了通信模式、类型、调度器、整合器等六个设计维度。

### 6. FGNN-MADRL (Qiong Wu et al., 2024)
- **标题**: Optimizing Age of Information in Vehicular Edge Computing with Federated Graph Neural Network Multi-Agent Reinforcement Learning
- **作者**: Qiong Wu et al.
- **来源**: arXiv:2407.02342
- **文件**: `FGNN-MADRL_Wu2024.pdf`
- **用途**: Stage 3 GNN 实现参考 — 道路场景构建为图数据结构，GNN 用于提取车辆-道路图特征
- **关键结论**: 首次将道路场景构建为图数据结构用于 GNN-based FL 框架。GNN 提取车辆网络特征后联合多智能体 SAC 算法做协作卸载。

### 7. TapFinger (Li et al., 2023)
- **标题**: Task Placement and Resource Allocation for Edge Machine Learning: A GNN-based Multi-Agent Reinforcement Learning Paradigm
- **作者**: Yihong Li et al.
- **来源**: IEEE TPDS 2023 (arXiv:2302.00571)
- **文件**: `TapFinger_Li2023.pdf`
- **用途**: Stage 3 异构 GAT 参考 — 异构图注意力网络作为 MARL 骨干
- **关键结论**: 异构 GAT 作为 MARL backbone，配合贝叶斯定理和 masking schemes，有效缓解扩展决策空间。任务完成时间降低 54.9%。

---

## 网络可访问论文 (1篇)

### 8. Yang et al. (2023)
- **标题**: Cooperative Task Offloading for Mobile Edge Computing Based on Multi-Agent Deep Reinforcement Learning
- **作者**: Jian Yang, Qifeng Yuan, Shuangwu Chen, Huasen He, Xiaofeng Jiang, Xiaobin Tan
- **来源**: IEEE Transactions on Network and Service Management, 20(3):3205-3219, September 2023
- **DOI**: 10.1109/TNSM.2023.3240415
- **访问**: IEEE Xplore https://ieeexplore.ieee.org/document/10032423 (需机构订阅)
- **用途**: 环境建模主参考 — 多用户多服务器协作卸载 MDP 完整建模
- **关键结论**: 将资源分配和任务卸载形式化为多目标 MDP，每个 MES 作为独立 agent，CTDE 框架。任务卸载率提升 17%，延迟降低 53%。

---

## 快速对照

| # | 论文 | Stage | 角色 |
|---|------|-------|------|
| 1 | IPPO (de Witt 2020) | Stage 1 | 算法理论基础 |
| 2 | MAPPO (Yu 2022) | Stage 1 | 对比基线 |
| 3 | MIPI (Wang 2023) | Stage 2 | 互信息正确用法 |
| 4 | Com-DDPG (2020) | Env | 任务模型补充 |
| 5 | GNNComm-MARL (2024) | Stage 3 | GNN 通信架构 |
| 6 | FGNN-MADRL (2024) | Stage 3 | GNN 实现参考 |
| 7 | TapFinger (2023) | Stage 3 | 异构 GAT 参考 |
| 8 | Yang et al. (2023) | Env | 环境建模主参考 |
