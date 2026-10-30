# 论文-代码不一致问题修复清单

本目录为上一轮审查识别出的 10 个问题，逐条提供：

- 问题定义
- 可能修复方案
- 方案原因与取舍
- 推荐实施路径

文件列表：

1. 01_phase2_label_source_mismatch.md
2. 02_phase2_missing_demo_supervision.md
3. 03_phase3_no_early_termination.md
4. 04_phase3_regret_reward_assignment.md
5. 05_phase3_r_arche_definition_mismatch.md
6. 06_phase3_top5_supervision_scope.md
7. 07_phase1_single_trade_constraint_gap.md
8. 08_execution_cost_model_gap.md
9. 09_mark_price_proxy_gap.md
10. 10_phase1_chunk_sampling_distribution.md

建议分析顺序：1 -> 2 -> 7 -> 3 -> 4 -> 5 -> 6 -> 8 -> 9 -> 10。
