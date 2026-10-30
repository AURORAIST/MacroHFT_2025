# 问题 04: Regret 奖励在 step 维度的分配偏离 Eq.(8)

## 问题描述

Eq.(8) 语义是仅在发生非零调整时给予 regret-aware reward，否则为 0。当前实现对每个 step 的 a_ref 循环计算，和未提前终止逻辑叠加后会产生重复或混杂归因。

## 可能修复方案

1. 单点奖励（推荐）
- 只在首次非零调整步赋值 r_ref。
- 其余步统一为 0。
- 若本 horizon 无调整，则全 0。

2. 稀疏奖励 + baseline 扩展
- 保持单点奖励，同时在 value 目标加入 baseline-centered 归一化。

## 推荐修改位置

- scripts/train_phase3.py（step_rewards 构造部分）
- src/phase3/regret_reward.py（明确输入为 single intervention）

## 原因与收益

- 直接贴合 Eq.(8) 的数学定义。
- 减少奖励泄漏导致的过估计。

## 风险与代价

- 奖励更稀疏，训练初期探索难度会增加。
