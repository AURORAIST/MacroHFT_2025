# 问题 06: Eq.(9) 监督仅使用 top-1，未体现 top-5 结构

## 问题描述

论文先构建 top-5 hindsight-optimal 集合，再定义监督目标。当前实现仅采用 top5[0] 作为唯一监督标签。

## 可能修复方案

1. Top-k 软标签监督（推荐）
- 将 top-5 转换为概率分布（按收益 softmax 或 rank 权重）。
- 用 KLDivLoss 或 soft cross-entropy 训练 refinement 策略。

2. 多正样本监督
- top-5 每个样本单独构造监督项并平均。

## 推荐修改位置

- scripts/train_phase3.py（optimal_actions 构造与 ce_loss）

## 原因与收益

- 更符合论文中 top-5 的设计初衷。
- 缓解只跟随 top-1 造成的监督高方差问题。

## 风险与代价

- 需要重新标定 beta2，避免监督项过强。
