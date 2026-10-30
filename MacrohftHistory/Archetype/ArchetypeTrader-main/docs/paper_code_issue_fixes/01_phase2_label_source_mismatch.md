# 问题 01: Phase II 监督标签来源与论文不一致

## 问题描述

论文中 Eq.(5) 的 KL 项依赖示范轨迹对应的真实原型标签。当前实现使用了基于 decoder 自重建的近似打标流程，而非 DP 示范轨迹经过 VQ encoder 的标签。

## 可能修复方案

1. 严格对齐论文（推荐）
- 在 Phase II 训练中，为每个 horizon 构造或读取对应的示范轨迹 (s_demo, a_demo, r_demo)。
- 用冻结的 encoder + codebook 直接量化得到 gt_label。
- KL 项仅使用该 gt_label 计算。

2. 轻量折中方案
- 保留现有近似标签逻辑。
- 在文档中明确说明这是 weak supervision，不再宣称严格对应 Eq.(5)。

## 推荐修改位置

- scripts/train_phase2.py
- src/data/dataset.py（可扩展为支持按 horizon 索引读取 demo）

## 原因与收益

- 论文核心卖点之一是 demonstration-guided selection。
- 使用真实示范标签可降低目标漂移，避免 selector 学习到由近似标签带来的偏差。
- 有助于后续 ablation 结果与论文叙述一致。

## 风险与代价

- 需要维护 horizon 与 demo chunk 的对齐关系。
- 训练开销会增加（编码示范序列）。



完整提示词和回复
1.请根据我提供的论文markdown版本， 审查项目代码， 看算法和数学公式已经逻辑是否与论文一致。 不一一致的问题点请列出。后续再进行逐个分析。给我一个问题列表
answer：我会先对照两份论文说明文档提取“应实现的算法与关键公式”，再逐模块检查代码中的实现与训练流程，最后给你一个“仅列问题、不展开分析”的不一致清单。先从论文 markdown 和核心算法代码同时读取开始。

我已拿到论文里的核心公式(1)-(9)和三阶段定义，下一步会在代码中定位这些公式对应实现（DP单次交易、VQ损失、选择器KL约束、Refine单次覆盖与regret奖励），快速筛出不一致点。

已完成第一轮定位：论文公式都能在注释中找到对应意图，但训练脚本里已经出现几个“实现疑点”（尤其 Phase II/III 目标函数与监督项）。我接下来会把关键模块完整读完并给出带文件定位的问题清单。

接下来我会补齐环境与Phase II/III核心模块（agent、adapter、regret、环境奖励定义），这样就能把“公式层”和“实现层”一一对照并输出问题列表。

我已经锁定了主要偏差点，正在补全精确定位行号并整理成可追踪的问题清单；同时我会检查是否有你历史偏好记录可复用审查格式。

我已经完成论文-代码逐项对照，正在把不一致点压缩成“后续可逐条深挖”的问题列表，并附每条的论文与代码定位。下一条消息直接给你清单。

问题列表如下，按严重性从高到低排列，先用于后续逐条深挖。

Phase II 的监督标签来源与论文定义不一致（核心偏差）
论文要求：选择器的 KL 项使用 VQ 编码器对该 horizon 的示范轨迹给出的真实标签 â_sel（见 AAAI26_ArchetypeTrader.md:157）。
代码实现：未使用 DP 示范轨迹，而是对每个原型先用 decoder 生成动作，再用“自重建损失最小”选 best_label（见 train_phase2.py:333、train_phase2.py:353、train_phase2.py:364、train_phase2.py:370）。
影响：KL 监督目标不再是论文中的“示范标签约束”。
Phase II 实际没有按论文使用“示范 chunk 标签监督”
论文目标函数明确依赖 demonstration chunk 的标签（见 AAAI26_ArchetypeTrader.md:157）。
代码训练环节只从 train_env 当前 horizon 状态构造标签，未加载/对齐示范数据集进行监督（见 train_phase2.py:333、train_phase2.py:370）。
影响：第二阶段“demonstration-guided selection”逻辑被弱化为自举式近似。
Phase III 没有实现“非零调整即终止 episode”
论文明确：每个 horizon 仅一次适配，且一旦 adapter 选择非零动作，episode 立即终止（见 AAAI26_ArchetypeTrader.md:193）。
代码：仍遍历完整 horizon，仅靠 adapter 内部标志阻止后续覆盖，不终止采样（见 train_phase3.py:238、train_phase3.py:292、policy_adapter.py:45）。
影响：训练轨迹分布与论文 MDP 设定不一致。
Phase III 的 regret reward 分配方式与 Eq.(8) 语义不一致
论文：r_ref 仅在执行非零调整时给出，否则为 0（见 AAAI26_ArchetypeTrader.md:191）。
代码：对每个 step 的 a_ref 都调用 regret 公式；在“未终止 episode”的实现下，会出现多步非零 a_ref 对同一 R_actual 反复计奖（见 train_phase3.py:481、train_phase3.py:482）。
影响：奖励信用分配与论文定义偏离。
Phase III 的 s_ref2 中 R_arche 使用了“实际执行累计收益”，非“base policy 累计收益”
论文定义：R_arche_tau 是 base archetype policy 的累计收益（见 AAAI26_ArchetypeTrader.md:163）。
代码：context 中 cumulative_reward 来自最终动作执行过程（可能已被 refinement 改写），不是纯 base 路径（见 train_phase3.py:254、train_phase3.py:279）。
影响：状态定义与论文 M_ref 不一致。
Eq.(9) 的监督目标只用了 top-1，而非论文给出的 top-5 结构
论文先定义 top-5 hindsight-optimal 集合并据此构造监督（见 AAAI26_ArchetypeTrader.md:180、AAAI26_ArchetypeTrader.md:196）。
代码：仅取 top5[0] 作为监督标签来源（见 train_phase3.py:464、train_phase3.py:473）。
影响：监督信号覆盖度低于论文叙述。
Phase I 的 single-trade 约束与论文公式存在冲突
论文算法约束写法是 c + 1[i != j] <= 1（见 AAAI26_ArchetypeTrader.md:97），并文字说明“exactly one timestep changed”（见 AAAI26_ArchetypeTrader.md:110）。
代码约束允许 flat->nonflat 后再回 flat（round-trip），也就是允许两次动作变化（见 dp_planner.py:305、dp_planner.py:315）。
影响：示范轨迹约束与论文形式化定义不一致。
执行损失 O_t 的实现简化过强，未体现论文中的 C(|ΔP|) 结构
论文将执行损失写为含 C(|ΔP|) 的项（见 AAAI26_ArchetypeTrader.md:61）。
代码中 fill cost 直接返回 0，仅保留佣金项（见 trading_env.py:185、trading_env.py:212、trading_env.py:243）。
影响：奖励模型与论文金融微观结构假设不一致。
价格定义与论文的 mark price 语义不对齐（实现近似）
论文奖励使用 p_mark（见 AAAI26_ArchetypeTrader.md:61）。
代码在三阶段训练都直接用 states 第一列做价格代理（见 train_phase1.py:72、train_phase2.py:243、train_phase3.py:336）。
影响：若第一列不等于 mark price，会引入系统性偏差。
论文说“sample n chunks”，代码是“遍历 horizon 后循环复用”
论文描述为采样 n 个 chunk（见 AAAI26_ArchetypeTrader.md:84）。
代码先全量遍历 horizon，再按 i % num_horizons 循环复用到 30k（见 dp_planner.py:224、dp_planner.py:249）。
影响：数据分布与“随机采样 chunk”叙述不完全一致。
如果你同意，我下一步可以从第1条开始，逐条给出：

论文公式到代码的精确映射差异
为什么会影响训练行为
最小改动修正方案（含改哪些函数/脚本）


2.针对你提出的每一个问题，建立一个markdown文件，给出可能的修复方案及原因
answer：接下来我会一次性写入11个文档（10个问题文件+1个总览索引），这样你可以按编号逐个推进分析与改造。

我已写完索引和第1个问题文档，继续批量写入其余9个问题文档，保持同一模板便于后续逐条展开。

我已经完成到第5个问题文件，下面继续补齐第6到第10个，完成后给你完整文件路径。

已按你要求，把每个问题分别建立为独立 markdown 文件，并写入了“可能修复方案 + 原因 + 推荐路径”。

总览索引：

README.md
逐项文件：

01_phase2_label_source_mismatch.md
02_phase2_missing_demo_supervision.md
03_phase3_no_early_termination.md
04_phase3_regret_reward_assignment.md
05_phase3_r_arche_definition_mismatch.md
06_phase3_top5_supervision_scope.md
07_phase1_single_trade_constraint_gap.md
08_execution_cost_model_gap.md
09_mark_price_proxy_gap.md
10_phase1_chunk_sampling_distribution.md
如果你愿意，我下一步可以直接从第1个文件开始，按“最小改动可落地”的方式逐条提交对应代码修复。