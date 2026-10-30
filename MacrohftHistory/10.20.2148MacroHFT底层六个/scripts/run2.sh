#!/bin/bash

#SBATCH --job-name=lowmacrohft
#SBATCH -N 1
#SBATCH --gres=gpu:3
#SBATCH -p GPUFEE04
#SBATCH --constraint="Python"
#SBATCH -e result/%j.err
#SBATCH -o result/%j.log
source /gpfs/home/Y42314104/anaconda3/bin/activate
conda activate macrohft

srun python -u RL/agent/low_level.py --alpha 1 --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' \
    --label label_1 >./logs/low_level/ETHUSDT/slope_1.log 2>&1 --epoch_number 1 &

srun python -u RL/agent/low_level.py --alpha 4 --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:1' \
    --label label_2 >./logs/low_level/ETHUSDT/slope_2.log 2>&1 --epoch_number 1 &

srun python -u RL/agent/low_level.py --alpha 0 --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:2' \
    --label label_3 >./logs/low_level/ETHUSDT/slope_3.log 2>&1 &

srun python -u RL/agent/low_level.py --alpha 4 --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:0' \
    --label label_1 >./logs/low_level/ETHUSDT/vol_1.log 2>&1 --epoch_number 1 &

srun python -u RL/agent/low_level.py --alpha 1 --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:1' \
    --label label_2 >./logs/low_level/ETHUSDT/vol_2.log 2>&1 &

srun python -u RL/agent/low_level.py --alpha 1 --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:2' \
    --label label_3 >./logs/low_level/ETHUSDT/vol_3.log 2>&1 &


