#!/bin/bash
set -e

# 低层检查点验证脚本
# 使用 low_leveltest.py 对指定的 checkpoint 进行验证，输出到对应的 seed 目录下的 val/

# 通用参数
COMMON="--dataset ETHUSDT --device cuda:0 --n_action 2 --label label_1 --clf slope"

# 作者检查点根目录与你自己的检查点根目录（可按需修改）
AUTHOR_ROOT="/home/ustc1958/lxy/MacroHFT/result/low_level/ETHUSDT/pos_2/best_model"
MY_ROOT="/home/ustc1958/lxy/MacroHFT/result/low_level/ETHUSDT/pos_2/best_modelreward1"

mkdir -p ./logs/low_level/ETHUSDT/k2/reset1/test

# # 验证作者检查点（slope 与 vol，label_1/2/3）
# # slope
# nohup python -u RL/agent/low_leveltest.py --dataset ETHUSDT --device cuda:0 --n_action 2 --clf slope --label label_1 \
#   --checkpoint ${AUTHOR_ROOT}/slope/1/best_model.pkl \
#   > ./logs/low_level/ETHUSDT/k2/reset1/test/author_slope_1.val.log 2>&1 &
# nohup python -u RL/agent/low_leveltest.py --dataset ETHUSDT --device cuda:0 --n_action 2 --clf slope --label label_2 \
#   --checkpoint ${AUTHOR_ROOT}/slope/2/best_model.pkl \
#   > ./logs/low_level/ETHUSDT/k2/reset1/test/author_slope_2.val.log 2>&1 &
# nohup python -u RL/agent/low_leveltest.py --dataset ETHUSDT --device cuda:0 --n_action 2 --clf slope --label label_3 \
#   --checkpoint ${AUTHOR_ROOT}/slope/3/best_model.pkl \
#   > ./logs/low_level/ETHUSDT/k2/reset1/test/author_slope_3.val.log 2>&1 &

# # vol
# nohup python -u RL/agent/low_leveltest.py --dataset ETHUSDT --device cuda:0 --n_action 2 --clf vol --label label_1 \
#   --checkpoint ${AUTHOR_ROOT}/vol/1/best_model.pkl \
#   > ./logs/low_level/ETHUSDT/k2/reset1/test/author_vol_1.val.log 2>&1 &
# nohup python -u RL/agent/low_leveltest.py --dataset ETHUSDT --device cuda:0 --n_action 2 --clf vol --label label_2 \
#   --checkpoint ${AUTHOR_ROOT}/vol/2/best_model.pkl \
#   > ./logs/low_level/ETHUSDT/k2/reset1/test/author_vol_2.val.log 2>&1 &
# nohup python -u RL/agent/low_leveltest.py --dataset ETHUSDT --device cuda:0 --n_action 2 --clf vol --label label_3 \
#   --checkpoint ${AUTHOR_ROOT}/vol/3/best_model.pkl \
#   > ./logs/low_level/ETHUSDT/k2/reset1/test/author_vol_3.val.log 2>&1 &

# # 验证你自己的检查点（slope 与 vol，label_1/2/3）
# slope
nohup python -u RL/agent/low_leveltest.py --dataset ETHUSDT --device cuda:0 --n_action 2 --clf slope --label label_1 \
  --checkpoint ${MY_ROOT}/slope/1/best_model.pkl \
  > ./logs/low_level/ETHUSDT/k2/reset1/test/reward1my_slope_1.val.log 2>&1 &
nohup python -u RL/agent/low_leveltest.py --dataset ETHUSDT --device cuda:0 --n_action 2 --clf slope --label label_2 \
  --checkpoint ${MY_ROOT}/slope/2/best_model.pkl \
  > ./logs/low_level/ETHUSDT/k2/reset1/test/reward1my_slope_2.val.log 2>&1 &
nohup python -u RL/agent/low_leveltest.py --dataset ETHUSDT --device cuda:0 --n_action 2 --clf slope --label label_3 \
  --checkpoint ${MY_ROOT}/slope/3/best_model.pkl \
  > ./logs/low_level/ETHUSDT/k2/reset1/test/reward1my_slope_3.val.log 2>&1 &

# vol
nohup python -u RL/agent/low_leveltest.py --dataset ETHUSDT --device cuda:0 --n_action 2 --clf vol --label label_1 \
  --checkpoint ${MY_ROOT}/vol/1/best_model.pkl \
  > ./logs/low_level/ETHUSDT/k2/reset1/test/reward1my_vol_1.val.log 2>&1 &
nohup python -u RL/agent/low_leveltest.py --dataset ETHUSDT --device cuda:0 --n_action 2 --clf vol --label label_2 \
  --checkpoint ${MY_ROOT}/vol/2/best_model.pkl \
  > ./logs/low_level/ETHUSDT/k2/reset1/test/reward1my_vol_2.val.log 2>&1 &
nohup python -u RL/agent/low_leveltest.py --dataset ETHUSDT --device cuda:0 --n_action 2 --clf vol --label label_3 \
  --checkpoint ${MY_ROOT}/vol/3/best_model.pkl \
  > ./logs/low_level/ETHUSDT/k2/reset1/test/reward1my_vol_3.val.log 2>&1 &

echo "Started low-level checkpoint validations. Check logs under ./logs/low_level/ETHUSDT/k2/reset1/test"
