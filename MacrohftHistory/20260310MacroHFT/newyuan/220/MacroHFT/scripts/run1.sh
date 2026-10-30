#!/bin/bash
#SBATCH --job-name=orignmodel
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p GPUFEE02
#SBATCH --constraint="Python"
#SBATCH -e log/%j.err
#SBATCH -o log/%j.log

source /gpfs/home/Y42314104/anaconda3/bin/activate
conda activate macrohft

mkdir -p logs/high_level

python -u RL/agent/high_level.py --dataset 'ETHUSDT' --device 'cuda:0' \
    >./logs/high_level/ETHUSDT.log 2>&1


