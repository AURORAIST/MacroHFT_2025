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


支持多仓，reward_scale对低层代理有效果，高层出现问题，alpha=4时reward_scale=1000,alpha-2,reward_scale=2000

场景	slope1(α=1)（%）	slope2(α=4)（%）	slope3(α=0)（%）	vol1(α=4)（%）	vol2(α=1)（%）	vol3(α=1)（%）
检查点验证集	0.20	0.85	5.60	0.17	2.31	15.17
低层 k=2（35epoch）	0.20	0.85	5.56	0.17	2.33	15.17
reward_scale=30000(35epoch)	0.38	0.75	5.56	0.49	2.20	12.46