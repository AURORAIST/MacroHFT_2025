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

After 4 months of unremitting efforts, mainly due to the lack of computing power equipment, I finally successfully reproduced the test results of low-level and high-level in your paper. Thank you for releasing all the code, algorithms, and data. Sincere tribute.
I will continue to try to improve some parts of this framework, including:
--More actions, more position levels
--Due to the low trading frequency in my testing results, I tend to remove order data from the micro market for testing purposes
--Use LSTM or Transformer to observe the effect at the model level
--Attempt to investigate the impact of ranbowdqn on the model
--Incorporate spot prices into the factor,
--Short selling strategy, one-way short selling or as protection for long positions
--Test buying spot goods