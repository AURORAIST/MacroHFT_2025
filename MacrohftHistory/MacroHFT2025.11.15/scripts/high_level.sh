#!/bin/bash

# 创建日志目录
mkdir -p ./logs/high_level/ETHUSDT

# 高层策略训练
# 参数说明：
# --num_actions: 高层策略的动作数量（通常使用2个动作：持有或不持有）
# --exp: 实验名称
# --alpha: 记忆误差权重
# --beta: KL散度权重
# --epoch_number: 训练轮数
#
# 注意：低层模型的加载路径需要在 RL/agent/high_level.py 中的 low_level_config 手动配置
#       因为6个子代理可能来自不同的训练配置（不同的pos, seed, alpha等）
# 实验1: 基础配置
nohup python -u RL/agent/high_level.py \
    --dataset ETHUSDT \
    --device cuda:0 \
    --num_actions 2 \
    --exp exp1_k2_reward30000_epo3 \
    --alpha 0.5 \
    --beta 3 \
    --epoch_number 3 \
    > ./logs/high_level/ETHUSDT/exp1_k2_reward30000_epo3.log 2>&1 &

# 实验2: 不同超参数
# nohup python -u RL/agent/high_level.py \
#     --dataset ETHUSDT \
#     --device cuda:0 \
#     --num_actions 2 \
#     --exp exp2 \
#     --alpha 1.0 \
#     --beta 10 \
#     --epoch_number 15 \
#     > ./logs/high_level/ETHUSDT/exp2.log 2>&1 &

# 实验3: 多仓位高层策略（实验性）
# nohup python -u RL/agent/high_level.py \
#     --dataset ETHUSDT \
#     --device cuda:0 \
#     --num_actions 3 \
#     --exp exp3_pos3 \
#     --alpha 0.5 \
#     --beta 5 \
#     --epoch_number 15 \
#     > ./logs/high_level/ETHUSDT/exp3_pos3.log 2>&1 &

echo "High-level training scripts started. Check logs in ./logs/high_level/ETHUSDT/"
echo "Use 'tail -f ./logs/high_level/ETHUSDT/exp1.log' to monitor progress"
