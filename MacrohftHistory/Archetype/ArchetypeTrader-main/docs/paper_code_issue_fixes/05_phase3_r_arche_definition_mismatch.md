# 问题 05: s_ref2 中 R_arche 定义与论文不一致

## 问题描述

论文定义 R_arche 为 base archetype policy 的累计收益。当前实现使用了实际执行路径累计收益，可能已混入 refinement 后动作结果。

## 可能修复方案

1. 双累计量并行维护（推荐）
- cumulative_reward_base: 严格按 base_actions 累计。
- cumulative_reward_actual: 按 final actions 累计，仅用于统计。
- s_ref2 只使用 cumulative_reward_base。

2. 近似替代
- 使用可微 surrogate 估计 base 累计收益（不建议首选）。

## 推荐修改位置

- scripts/train_phase3.py（run_horizon_with_refinement）

## 原因与收益

- 保证状态定义与论文一致，避免 agent 在状态输入中读取到自身动作造成的信息泄漏。
- 提升实验可解释性。

## 风险与代价

- 需要额外一次 base 回放或同步计算逻辑。
