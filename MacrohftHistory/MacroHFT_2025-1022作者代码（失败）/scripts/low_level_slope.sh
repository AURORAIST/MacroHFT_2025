#!/bin/bash
# low_level.sh - Execute training tasks with CUDA GPU (并行启动)

mkdir -p logs/low_level/ETHUSDT/exp3/slope/label_1
mkdir -p logs/low_level/ETHUSDT/exp3/slope/label_2
mkdir -p logs/low_level/ETHUSDT/exp3/slope/label_3

nohup python -u RL/agent/low_level.py --alpha 1 --clf=slope --dataset=ETHUSDT --label=label_1 --device=cuda:0 --exp exp3 > logs/low_level/ETHUSDT/exp3/slope/label_1/vol_1.log 2>&1 &
PID1=$!

nohup python -u RL/agent/low_level.py --alpha 4 --clf=slope --dataset=ETHUSDT --label=label_2 --device=cuda:0 --exp exp3 > logs/low_level/ETHUSDT/exp3/slope/label_2/vol_2.log 2>&1 &
PID2=$!

nohup python -u RL/agent/low_level.py --alpha 0 --clf=slope --dataset=ETHUSDT --label=label_3 --device=cuda:0 --exp exp3 > logs/low_level/ETHUSDT/exp3/slope/label_3/_3.log 2>&1 &
PID3=$!

echo "  ps -p $PID1,$PID2,$PID3"
