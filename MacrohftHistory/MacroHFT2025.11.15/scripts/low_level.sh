#!/bin/bash

# 创建日志目录
mkdir -p ./logs/low_level/ETHUSDT/k2

COMMON="--dataset ETHUSDT --device cuda:0 --num_actions 2 --epoch_number 35"

# Slope 分类器训练
nohup python -u RL/agent/low_level.py $COMMON --alpha 1 --clf slope --label label_1 --exp reward2000epo35 --reward_scale 2000\
      > ./logs/low_level/ETHUSDT/k2/slope_1epo_35reward_scale2000.log 2>&1 &

nohup python -u RL/agent/low_level.py $COMMON --alpha 4 --clf slope --label label_2 --exp reward1000epo35 --reward_scale 1000\
    > ./logs/low_level/ETHUSDT/k2/slope_2epo_35reward_scale1000.log 2>&1 &

nohup python -u RL/agent/low_level.py $COMMON --alpha 0 --clf slope --label label_3 --exp reward1000epo35 --reward_scale 1000\
    > ./logs/low_level/ETHUSDT/k2/slope_3epo_35reward_scale1000.log 2>&1 &

# # # Vol 分类器训练
nohup python -u RL/agent/low_level.py $COMMON --alpha 4 --clf vol --label label_1 --exp reward1000epo35 --reward_scale 1000\
    > ./logs/low_level/ETHUSDT/k2/vol_1epo_35reward_scale1000.log 2>&1 &

nohup python -u RL/agent/low_level.py $COMMON --alpha 1 --clf vol --label label_2 --exp reward2000epo35 --reward_scale 2000\
    > ./logs/low_level/ETHUSDT/k2/vol_2epo_35reward_scale2000.log 2>&1 &

nohup python -u RL/agent/low_level.py $COMMON --alpha 1 --clf vol --label label_3 --exp reward2000epo35 --reward_scale 2000\
    > ./logs/low_level/ETHUSDT/k2/vol_3epo_35reward_scale2000.log 2>&1 &
