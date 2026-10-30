# 问题 07: Phase I single-trade 约束实现与论文形式化存在差异

## 问题描述

论文 Algorithm 1 的约束写法与正文“exactly one timestep changed”描述较强。当前代码允许 flat->nonflat->flat 的 round-trip，即会出现两次动作变化。

## 可能修复方案

1. 以“单次动作变化”为准（严格公式）
- 将约束改为每个 horizon 仅允许一次 i!=j 转移。
- 一旦变化后，后续动作固定。

2. 以“单次交易机会”为准（推荐）
- 保留开仓+平仓的 round-trip。
- 同步修订论文表述为 one round-trip trade。

## 推荐修改位置

- src/phase1/dp_planner.py（_compute_next_constraint 及注释）
- AAAI26_ArchetypeTrader.md（若选择方案2）

## 原因与收益

- 该点属于公式和文字描述冲突，需要先统一规范再改代码。
- 统一后才能保证实验复现解释一致。

## 风险与代价

- 方案1会显著改变示范数据分布；方案2需修改论文文本。
