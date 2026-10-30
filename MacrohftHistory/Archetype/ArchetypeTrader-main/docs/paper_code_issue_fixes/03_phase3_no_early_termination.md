# 问题 03: Phase III 非零调整后未提前终止 episode

## 问题描述

论文定义中，一旦 adapter 输出非零调整，本 horizon episode 立即结束。当前实现继续执行到 horizon 结束，仅通过标志位阻止再次覆盖。

## 可能修复方案

1. 严格终止（推荐）
- 在 run_horizon_with_refinement 中，当 a_ref != 0 时记录奖励后立即 break。
- 后续 return/advantage 仅基于终止前轨迹。

2. 保持当前实现但改论文对齐声明
- 明确这是 one-shot override but full-horizon rollout 的变体。

## 推荐修改位置

- scripts/train_phase3.py（run_horizon_with_refinement）

## 原因与收益

- 状态分布、奖励归因、回报定义都会与论文 MDP 设定对齐。
- 训练目标更清晰，减少非必要噪声梯度。

## 风险与代价

- 轨迹变短后，value 学习可能不稳定，需要适当调学习率或熵正则。
