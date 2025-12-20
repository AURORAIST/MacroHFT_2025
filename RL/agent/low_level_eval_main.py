import time  # Import the time module
import pathlib
import sys
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import concurrent.futures
import pickle
import os
import logging as log
from torch.utils.tensorboard import SummaryWriter
import warnings

warnings.filterwarnings("ignore")

ROOT = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
sys.path.insert(0, ".")

from model.net import *
from env.low_level_env import Testing_Env, Training_Env
from RL.util.utili import get_ada, get_epsilon, LinearDecaySchedule
from RL.util.replay_buffer import ReplayBuffer
from RL.agent.low_level_eval import DQN_EVAL
from logging import StreamHandler, FileHandler, Formatter
import logging as log
import os
import time

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--buffer_size",type=int,default=3000000)  # 经验缓冲区大小 / Replay buffer capacity
parser.add_argument("--dataset",type=str,default="ETHUSDT")  # 数据集名称 / Dataset name
parser.add_argument("--q_value_memorize_freq",type=int, default=20)  # Q值记忆频率 / Q-value logging frequency
parser.add_argument("--batch_size",type=int,default=1024)  # 批次大小 / Mini-batch size
parser.add_argument("--eval_update_freq",type=int,default=50)  # 网络更新频率 / Network update frequency
parser.add_argument("--lr", type=float, default=1e-4)  # 学习率 / Learning rate
parser.add_argument("--epsilon_start",type=float,default=0.7)  # 初始探索率 / Initial exploration rate
parser.add_argument("--epsilon_end",type=float,default=0.3)  # 最小探索率 / Minimum exploration rate
parser.add_argument("--decay_length",type=int,default=5)  # 探索衰减周期 / Exploration decay length
parser.add_argument("--update_times",type=int,default=20)  # 单步更新次数 / Update times per step
parser.add_argument("--gamma", type=float, default=0.99)  # 折扣因子 / Discount factor
parser.add_argument("--tau", type=float, default=0.005)  # 软更新系数 / Soft update coefficient
parser.add_argument("--transcation_cost",type=float,default=2.0 / 10000)  # 交易成本（注意拼写） / Transaction cost (typo preserved)
parser.add_argument("--back_time_length",type=int,default=1)  # 历史窗口长度 / Historical window length
parser.add_argument("--seed",type=int,default=12345)  # 随机种子 / Random seed
parser.add_argument("--n_step",type=int,default=1)  # n-step TD目标 / N-step TD target
parser.add_argument("--epoch_number",type=int,default=20)  # 训练轮次数 / Training epochs
parser.add_argument("--label",type=str,default="label_2")  # 标签列名称 / Label column name
parser.add_argument("--clf",type=str,default="slope")  # 分类器类型 / Classifier type
parser.add_argument("--alpha",type=float,default=4)  # KL损失权重系数 / KL loss weight coefficient
parser.add_argument("--exp",type=str,default="exp_default_1")
parser.add_argument("--device",type=str,default="cuda:0")  # 计算设备 / Computation device

        
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

def config_log(logs_dir,pfx=''):
    today = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    file_name = f'{today}.log'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)  # 确保目录存在
    file_path = os.path.join(logs_dir, pfx+file_name)

    # 创建一个日志格式化器
    formatter = Formatter('%(asctime)s %(levelname)s: %(message)s')

    # 创建文件处理器并设置格式化器
    file_handler = FileHandler(file_path, encoding='utf-8')
    file_handler.setFormatter(formatter)

    # 创建流处理器（控制台）并设置格式化器
    stream_handler = StreamHandler()
    stream_handler.setFormatter(formatter)

    # 获取根记录器并添加处理器
    logger = log.getLogger()
    logger.setLevel(log.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
if __name__ == "__main__":
    args = parser.parse_args()
    clf = args.clf
    
    # Create log directory
    logs_dir = os.path.join("./logs/low_level", '{}'.format(args.dataset), args.exp, '{}'.format(clf), args.label, str(int(args.alpha)))
    os.makedirs(logs_dir, exist_ok=True) 
        
    config_log(logs_dir,pfx='train-')
    log.info(args) 
    seed_torch(args.seed)
    tech_indicator_list = np.load('./data/feature_list/single_features.npy', allow_pickle=True).tolist()
    tech_indicator_list_trend = np.load('./data/feature_list/trend_features.npy', allow_pickle=True).tolist()

    transcation_cost = args.transcation_cost
    back_time_length = args.back_time_length
    n_action = 2
    n_state_1 = len(tech_indicator_list)
    n_state_2 = len(tech_indicator_list_trend)
    max_holding_number=0.2
    label = 2
    val_data_path = os.path.join(ROOT, "MacroHFT", "data", "ETHUSDT", "val")
    with open(os.path.join(val_data_path, 'slope_labels.pkl'), 'rb') as file:
        val_index = pickle.load(file)
    var_df_list = val_index[label]
    
    result_path = os.path.join("./result/low_level", '{}'.format(args.dataset), args.exp, '{}'.format(args.clf), args.label)
    epoch_path = os.path.join(result_path)
    val_path = os.path.join(epoch_path, "val")
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    
    dqn_eval = DQN_EVAL(n_state_1,n_state_2,n_action,"cpu",
                        val_data_path,
                        tech_indicator_list,
                        tech_indicator_list_trend,
                        transcation_cost,
                        back_time_length,
                        max_holding_number)
    return_rate = dqn_eval.val_cluster(epoch_path, val_path, 0, var_df_list)
    log.info("Return rate: {}".format(return_rate))