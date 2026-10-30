# MacroHFT
This is the official implementation of the KDD 2024 "MacroHFT: Memory Augmented Context-aware Reinforcement Learning On High Frequency Trading".
https://arxiv.org/abs/2406.14537

To run the demo code:

You may first download the dataset from Google Drive:

https://drive.google.com/drive/folders/1AYHy-wUV0IwPoA7E1zvMRPL3wK0tPNiY?usp=drive_link

and put the folder under data folder.

## Step 1
Run scripts/decomposition.sh for data decomposition and labeling. 
## Step 2
Run scripts/low_level.sh for low-level policy optimization. 

Update: We now provide trained model checkpoints for sub-agents, which can be directly used to train meta-policy.
## Step 3
Run scripts/high_level.sh for meta-policy optimization. 

> python tools/test_qtable_max_profit.py --data_path ./data/ETHUSDT/whole/val.feather --num_actions 2 --max_holding 0.2 --commission_fee 0.001 --reward_scale 1500  
Q表最优策略最大收益模拟结果:
最终资金: 9932.27
最大收益率: -0.0068
the portfit margine is 0.9932
累计手续费: 4837.56
动作分布: [164295   8145]
(C:\ProgramData\anaconda3\macrohft) PS D:\Code\gitstorehouse\新建文件夹\MacroHFT> python tools/test_qtable_max_profit.py --data_path ./data/ETHUSDT/whole/test.feather --num_actions 2 --max_holding 0.2 --commission_fee 0.001 --reward_scale 1500
Q表最优策略最大收益模拟结果:
最终资金: 9746.63
最大收益率: -0.0253
the portfit margine is 0.9747
累计手续费: 2869.17
动作分布: [215290   4670]
(C:\ProgramData\anaconda3\macrohft) PS D:\Code\gitstorehouse\新建文件夹\MacroHFT> python tools/test_qtable_max_profit.py --data_path ./data/ETHUSDT/whole/train.feather --num_actions 2 --max_holding 0.2 --commission_fee 0.001 --reward_scale 1500
Q表最优策略最大收益模拟结果:
最终资金: 15417.82
最大收益率: 0.5418
the portfit margine is 1.5418
累计手续费: 33445.22
动作分布: [469252  55988]

q表最优策略达到54%

MacroHFT2025.11.17高层实验，rewardhigh=1