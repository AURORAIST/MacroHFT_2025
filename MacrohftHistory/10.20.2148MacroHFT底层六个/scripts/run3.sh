#!/bin/bash
#SBATCH --job-name=low-level-rl
#SBATCH -N 1
#SBATCH --gres=gpu:3
#SBATCH -p GPUFEE04
#SBATCH --constraint="Python"
#SBATCH -e log/%j.err
#SBATCH -o log/%j.log

source /gpfs/home/Y42314104/anaconda3/bin/activate
conda activate macrohft

nohup python -u RL/agent/high_level.py --dataset 'ETHUSDT' --device 'cuda:0' \
    >./logs/high_level/ETHUSDT.log 2>&1 &
wait

echo "所有low_level.py实例运行完成"
