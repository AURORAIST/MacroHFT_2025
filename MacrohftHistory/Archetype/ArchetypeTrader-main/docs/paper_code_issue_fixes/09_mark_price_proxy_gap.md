# 问题 09: mark price 与训练价格代理可能不一致

## 问题描述

论文奖励基于 mark price。当前三阶段脚本统一使用状态向量第1列作为价格代理，未验证该列是否等于 mark price。

## 可能修复方案

1. 明确价格字段映射（推荐）
- 在 FeaturePipeline 显式输出 mark_price 序列。
- TradingEnv 统一接收该字段，禁止隐式用 states[:,0]。

2. 配置化价格列
- 在 Config 增加 price_column_name 或 price_column_idx。
- 训练时强制日志打印所用价格字段。

## 推荐修改位置

- src/data/feature_pipeline.py
- src/config.py
- scripts/train_phase1.py
- scripts/train_phase2.py
- scripts/train_phase3.py

## 原因与收益

- 减少指标漂移和实验不确定性。
- 方便不同数据源复现。

## 风险与代价

- 若历史结果都基于旧代理，切换后需要重新训练和重跑评估。
