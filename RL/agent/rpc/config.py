import numpy as np
import pathlib
import sys
ROOT = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
sys.path.insert(0, ".") 

seed=12345                            # 随机种子
replay_buffer_capacity=2000000        # 经验回放缓冲区的容量
replay_buffer_sample_batch_size=1024  # 从经验回放缓冲区中采样的批量大小
global_workersize=1                   # 一共有多少个worker


tech_indicator_list = np.load('./data/feature_list/single_features.npy', allow_pickle=True).tolist()
tech_indicator_list_trend = np.load('./data/feature_list/trend_features.npy', allow_pickle=True).tolist()
n_state_1_dim = len(tech_indicator_list)
n_state_2_dim = len(tech_indicator_list_trend)          
 
action_dim=2


