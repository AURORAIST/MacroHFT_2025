#!/bin/bash

# 创建日志目录
mkdir -p ./logs/low_level/ETHUSDT/k2

COMMON="--dataset ETHUSDT --device cuda:0 --num_actions 2 --epoch_number 10 --lr 1e-4"

# Slope 分类器训练
nohup python -u RL/agent/low_level.py $COMMON --alpha 1 --clf slope --label label_1 --exp 131test --reward_scale 1\
      > ./logs/low_level/ETHUSDT/k2/slope_1/slope_1_epoch131test.log 2>&1 &

nohup python -u RL/agent/low_level.py $COMMON --alpha 4 --clf slope --label label_2 --exp 131test --reward_scale 1\
    > ./logs/low_level/ETHUSDT/k2/slope_2/slope_2_epoch131test.log 2>&1 &

nohup python -u RL/agent/low_level.py $COMMON --alpha 0 --clf slope --label label_3 --exp 131test --reward_scale 1\
    > ./logs/low_level/ETHUSDT/k2/slope_3/slope_3_epoch131test.log 2>&1 &

# # # # Vol 分类器训练
nohup python -u RL/agent/low_level.py $COMMON --alpha 4 --clf vol --label label_1 --exp 131test --reward_scale 1\
    > ./logs/low_level/ETHUSDT/k2/vol_1/vol_1_epoch131test.log 2>&1 &

nohup python -u RL/agent/low_level.py $COMMON --alpha 1 --clf vol --label label_2 --exp 131test --reward_scale 1\
    > ./logs/low_level/ETHUSDT/k2/vol_2/vol_2_epoch131test.log 2>&1 &

nohup python -u RL/agent/low_level.py $COMMON --alpha 1 --clf vol --label label_3 --exp 131test --reward_scale 1\
    > ./logs/low_level/ETHUSDT/k2/vol_3/vol_3_epoch131test.log 2>&1 &
