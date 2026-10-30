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


## lance  78599194@qq.com
After 4 months of unremitting efforts, mainly due to the lack of computing power equipment, I finally successfully reproduced the test results of low-level and high-level in your paper. Thank you for releasing all the code, algorithms, and data. Sincere tribute.
I will continue to try to improve some parts of this framework, including:

stage 1
- More actions, more position levels  
```
# 汇总：
- 条件：5倍杠杆， 万5费率， ETH， 7个月训练 2个验证， 3个月测试
- 做单方向多头   20-35%之间， 交易频率低。 原作者收益率在40%左右。
- 做单方向空头   70-100%之间， 交易频率高 
- 做双方向空头和多头   35-45%之间， 交易频率中
# 似乎加密货币空头策略更为有效。可能和训练数据集有关系。不能下定论。双方向策略会综合多头和空头的收益率，显然多头拖累了收益率
20260106
```

stage 2

- I tend to remove order data from the micro market for testing purposes
- Incorporate spot prices into the factor,
- Train Data by 2022-2025

stage 3

- backtest with fredtrade

stage 4
- More sub-agents pre volume
- Use LSTM or Transformer to observe the effect at the model level 
- Attempt to investigate the impact of ranbowdqn on the model


- Test buying spot goods （放弃，由于现货交易的手续费较高0.3%，而我在测试中只关注了期货交易的收益，所以放弃了测试现货交易的效果，之后会考虑）
