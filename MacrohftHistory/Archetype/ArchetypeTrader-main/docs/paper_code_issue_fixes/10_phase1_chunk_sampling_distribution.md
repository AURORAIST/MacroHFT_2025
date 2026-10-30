# 问题 10: Phase I chunk 采样分布与论文描述不完全一致

## 问题描述

论文叙述为采样 n 个 chunk。当前实现先遍历全部 horizon，再循环复用到目标数量，本质更接近 deterministic repeat。

## 可能修复方案

1. 随机采样 horizon（推荐）
- 按均匀或分层策略随机采样 num_trajectories 个 horizon 索引。
- 可允许重复，但由随机过程决定。

2. 全量遍历 + 随机重排
- 每个 epoch 对 horizon 列表随机打乱，再截断或补齐。

## 推荐修改位置

- src/phase1/dp_planner.py（generate_trajectories）

## 原因与收益

- 更符合论文描述和统计假设。
- 避免固定复用模式导致过拟合某些 horizon。

## 风险与代价

- 不同随机种子下结果波动可能增大，需要固定 seed。
