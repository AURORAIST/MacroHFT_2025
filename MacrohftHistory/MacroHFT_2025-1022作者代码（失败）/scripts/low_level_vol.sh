#!/bin/bash
# low_level.sh - Execute training tasks with CUDA GPU using nohup

mkdir -p logs/low_level/ETHUSDT/exp3/vol/label_1
mkdir -p logs/low_level/ETHUSDT/exp3/vol/label_2
mkdir -p logs/low_level/ETHUSDT/exp3/vol/label_3

# 启动训练任务（使用 nohup 在后台运行）
nohup python -u RL/agent/low_level.py --alpha 4 --clf=vol --dataset=ETHUSDT --label=label_1 --device=cuda:0 --exp exp3 > logs/low_level/ETHUSDT/exp3/vol/label_1/slope_1.log 2>&1 &
PID1=$!

nohup python -u RL/agent/low_level.py --alpha 1 --clf=vol --dataset=ETHUSDT --label=label_2 --device=cuda:0 --exp exp3 > logs/low_level/ETHUSDT/exp3/vol/label_2/slope_2.log 2>&1 &
PID2=$!

nohup python -u RL/agent/low_level.py --alpha 1 --clf=vol --dataset=ETHUSDT --label=label_3 --device=cuda:0 --exp exp3 > logs/low_level/ETHUSDT/exp3/vol/label_3/slope_3.log 2>&1 &
PID3=$!

echo "ps -p $PID1,$PID2,$PID3"