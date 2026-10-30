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

# nohup python -u preprocess/decomposition.py > ./logs/nohup.log 2>&1 

# nohup python -u RL/agent/low_level.py --alpha 1 --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' \
#     --label label_1 >./logs/low_level/ETHUSDT/slope_1.log 2>&1 --epoch_number 5 &
# nohup python -u RL/agent/low_level.py --alpha 4 --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:1' \
#     --label label_2 >./logs/low_level/ETHUSDT/slope_2.log 2>&1 --epoch_number 5 &
# nohup python -u RL/agent/low_level.py --alpha 0 --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:2' \
#     --label label_3 >./logs/low_level/ETHUSDT/slope_3.log 2>&1 --epoch_number 5 &

# nohup python -u RL/agent/low_level.py --alpha 4 --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:3' \
#     --label label_1 >./logs/low_level/ETHUSDT/vol_1.log 2>&1  &
# nohup python -u RL/agent/low_level.py --alpha 1 --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:4' \
#     --label label_2 >./logs/low_level/ETHUSDT/vol_2.log 2>&1 &
# nohup python -u RL/agent/low_level.py --alpha 1 --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:5' \
#     --label label_3 >./logs/low_level/ETHUSDT/vol_3.log 2>&1 &
# wait
nohup python -u RL/agent/high_level.py --dataset 'ETHUSDT' --device 'cuda:0' \
    >./logs/high_level/ETHUSDT.log 2>&1 &
wait

echo "所有low_level.py实例运行完成"
