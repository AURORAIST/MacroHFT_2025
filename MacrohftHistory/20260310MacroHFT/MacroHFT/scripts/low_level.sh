#!/bin/bash

# 创建日志目录
mkdir -p ./logs/low_level/ETHUSDT/k3/reset1

COMMON="--dataset ETHUSDT --device cuda:0 --n_action 2 --epoch_number 35 --lr 1e-5 --reward_scale 1000"

# # Slope 分类器训练
nohup python -u RL/agent/low_level.py $COMMON --alpha 1 --clf slope --label label_1 --exp reset1 \
    > ./logs/low_level/ETHUSDT/k3/reset1/slope_1.log 2>&1 &

nohup python -u RL/agent/low_level.py $COMMON --alpha 4 --clf slope --label label_2 --exp reset1 \
    > ./logs/low_level/ETHUSDT/k3/reset1/slope_2.log 2>&1 &

nohup python -u RL/agent/low_level.py $COMMON --alpha 0 --clf slope --label label_3 --exp reset1 \
    > ./logs/low_level/ETHUSDT/k3/reset1/slope_3.log 2>&1 &
# Vol 分类器训练
nohup python -u RL/agent/low_level.py $COMMON --alpha 4 --clf vol --label label_1 --exp reset1 \
    > ./logs/low_level/ETHUSDT/k3/reset1/vol_1.log 2>&1 &

nohup python -u RL/agent/low_level.py $COMMON --alpha 1 --clf vol --label label_2 --exp reset1 \
    > ./logs/low_level/ETHUSDT/k3/reset1/vol_2.log 2>&1 &

nohup python -u RL/agent/low_level.py $COMMON --alpha 1 --clf vol --label label_3 --exp reset1 \
    > ./logs/low_level/ETHUSDT/k3/reset1/vol_3.log 2>&1 &