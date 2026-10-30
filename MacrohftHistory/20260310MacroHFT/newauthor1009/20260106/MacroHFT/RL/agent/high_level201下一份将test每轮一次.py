import pathlib
import sys
import random
import argparse
import queue
import threading
import shutil
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
import joblib
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

ROOT = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
sys.path.insert(0, ".")

from model.net import *
from env.high_level_env import Testing_Env, Training_Env
from RL.util.utili import get_ada, get_epsilon, LinearDecaySchedule
from RL.util.replay_buffer import ReplayBuffer_High
from RL.util.memory import episodicmemory
from RL.util.eval_tools import calculate_trading_metrics
from RL.util.graph_utils import plot_money_curve
from env.actions import get_actions
import multiprocessing

# Set multiprocessing start method to 'spawn' to avoid CUDA re-initialization issues
multiprocessing.set_start_method('spawn', force=True)

from logging import StreamHandler, FileHandler, Formatter
import logging as log
import os
import time
def config_log(logs_dir,pfx=''):
    today = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    file_name = f'{today}.log'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)  # 确保目录存在
    file_path = os.path.join(logs_dir, pfx+file_name)

    # 创建一个日志格式化器
    formatter = Formatter('%(asctime)s %(levelname)s: %(message)s')
    # formatter = Formatter('%(message)s')
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

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--buffer_size",type=int,default=1000000)  # 经验缓冲区大小 / Replay buffer capacity
parser.add_argument("--dataset",type=str,default="ETHUSDT")  # 数据集名称 / Dataset name
parser.add_argument("--q_value_memorize_freq",type=int, default=10)  # Q值记忆频率 / Q-value logging frequency
parser.add_argument("--batch_size",type=int,default=8192)  # 批次大小 / Mini-batch size
parser.add_argument("--eval_update_freq",type=int,default=256)  # 网络更新频率 / Network update frequency
parser.add_argument("--lr", type=float, default=1e-4)  # 学习率 / Learning rate
parser.add_argument("--epsilon_start",type=float,default=0.7)  # 初始探索率 / Initial exploration rate
parser.add_argument("--epsilon_end",type=float,default=0.3)  # 最小探索率 / Minimum exploration rate
parser.add_argument("--decay_length",type=int,default=5)  # 探索衰减周期 / Exploration decay length
parser.add_argument("--update_times",type=int,default=20)  # 单步更新次数 / Update times per step
parser.add_argument("--gamma", type=float, default=0.99)  # 折扣因子 / Discount factor
parser.add_argument("--tau", type=float, default=0.005)  # 软更新系数 / Soft update coefficient
parser.add_argument("--transcation_cost",type=float,default=5.0 / 10000)  # 交易成本（注意拼写） / Transaction cost (typo preserved)
parser.add_argument("--back_time_length",type=int,default=1)  # 历史窗口长度 / Historical window length
parser.add_argument("--seed",type=int,default=12345)  # 随机种子 / Random seed
parser.add_argument("--n_step",type=int,default=1)  # n-step TD目标 / N-step TD target
parser.add_argument("--epoch_number",type=int,default=10)  # 训练轮次数 / Training epochs
parser.add_argument("--alpha",type=float,default=0.5)  # KL损失权重系数 / KL loss weight coefficient #alpha 代表了记忆的经验权重， beta代表先验q-table权重
parser.add_argument("--device",type=str,default="cuda:0")  # 计算设备 / Computation device cuda:0
parser.add_argument("--beta",type=int,default=5) #alpha 代表了记忆的经验权重， beta代表先验q-table权重
parser.add_argument("--no_risk_return",type=float,default=4.5) #无风险返回率
parser.add_argument("--exp",type=str,default="short_action_3")
parser.add_argument("--num_step",type=int,default=10)
parser.add_argument("--action_mode",type=str,default="short")  # 动作方向
parser.add_argument("--action_size",type=int,default=1)  # 动作数量
parser.add_argument("--reward_no_action",type=bool,default=False)  # 奖励没有动作

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def _validate_worker(_args):
    (epoch_path, val_path,__args) = _args
    
    logs_dir = os.path.join("./logs/high_level", '{}'.format(__args.dataset), __args.exp)
    os.makedirs(logs_dir, exist_ok=True) 
    
    config_log(logs_dir,pfx='eval-')
    
    evalor = HIGH_LEVEL_DQN_EVAL(__args) 
    return evalor.val_cluster(epoch_path, val_path)   

class DQN(object):
    def __init__(self, args):  # 定义DQN的一系列属性
        self.seed = args.seed
        seed_torch(self.seed)
        if torch.cuda.is_available():
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cpu")
            

        self.epsilon_device = torch.device(args.device) 

        log.info(args)    
            
        self.exp = args.exp
        self.result_path = os.path.join("./result/high_level", '{}'.format(args.dataset), args.exp)
        self.model_path = os.path.join(self.result_path, "seed_{}".format(self.seed))
        self.train_data_path = os.path.join(ROOT, "MacroHFT", "data", args.dataset, "whole")
        self.val_data_path = os.path.join(ROOT, "MacroHFT", "data", args.dataset, "whole")
        self.test_data_path = os.path.join(ROOT, "MacroHFT", "data", args.dataset, "whole")
        self.dataset=args.dataset
        self.num_step = args.num_step
        if "BTC" in self.dataset:
            self.max_holding_number=0.01
        elif "ETH" in self.dataset:
            self.max_holding_number=0.2
        elif "DOT" in self.dataset:
            self.max_holding_number=10
        elif "LTC" in self.dataset:
            self.max_holding_number=10
        else:
            raise Exception ("we do not support other dataset yet")
        self.epoch_number = args.epoch_number
        
        self.log_path = os.path.join(self.model_path, "log")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.writer = SummaryWriter(self.log_path)
        self.update_counter = 0
        self.q_value_memorize_freq = args.q_value_memorize_freq

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.tech_indicator_list = np.load('./data/feature_list/single_features.npy', allow_pickle=True).tolist()
        self.tech_indicator_list_trend = np.load('./data/feature_list/trend_features.npy', allow_pickle=True).tolist()
        self.clf_list = ['slope_360', 'vol_360']  # 趋势  波动 分类

        self.action_mode = args.action_mode
        self.actions, self.n_action, self.action_type_desc = get_actions(self.action_mode, args.action_size)
        
        log.info(f"self.actions:{self.actions}")
        log.info(f"self.n_action:{self.n_action}")
        log.info(f"self.action_type_desc:{self.action_type_desc}")
        
        self.transcation_cost = args.transcation_cost
        self.back_time_length = args.back_time_length
 
        self.n_state_1 = len(self.tech_indicator_list)
        self.n_state_2 = len(self.tech_indicator_list_trend)
        
        high_level_hidden_dim = 32*4
        self.epsilon_hyperagent = hyperagent(self.n_state_1, self.n_state_2, self.n_action, high_level_hidden_dim).to(self.epsilon_device)
        self.hyperagent = hyperagent(self.n_state_1, self.n_state_2, self.n_action, high_level_hidden_dim).to(self.device)
        self.hyperagent_target = hyperagent(self.n_state_1, self.n_state_2, self.n_action, high_level_hidden_dim).to(self.device)
        log.debug(f"self.epsilon_hyperagent:{self.epsilon_hyperagent.state_dict()}")
        self.hyperagent_target.load_state_dict(self.hyperagent.state_dict())
        self.epsilon_hyperagent.load_state_dict(self.hyperagent.state_dict())
        self.epsilon_hyperagent.eval()
        log.debug(f"self.epsilon_hyperagent:{self.epsilon_hyperagent.state_dict()}")
        
                       
        low_level_hidden_dim = 64*4  #*8
        self.slope_1 = subagent(self.n_state_1, self.n_state_2, self.n_action, low_level_hidden_dim).to(self.epsilon_device)
        self.slope_2 = subagent(self.n_state_1, self.n_state_2, self.n_action, low_level_hidden_dim).to(self.epsilon_device)
        self.slope_3 = subagent(self.n_state_1, self.n_state_2, self.n_action, low_level_hidden_dim).to(self.epsilon_device)
        self.vol_1 = subagent(self.n_state_1, self.n_state_2, self.n_action, low_level_hidden_dim).to(self.epsilon_device)
        self.vol_2 = subagent(self.n_state_1, self.n_state_2, self.n_action, low_level_hidden_dim).to(self.epsilon_device)
        self.vol_3 = subagent(self.n_state_1, self.n_state_2, self.n_action, low_level_hidden_dim).to(self.epsilon_device)   
        self.slope_epsilon_1 = subagent(self.n_state_1, self.n_state_2, self.n_action, low_level_hidden_dim).to(self.epsilon_device)
        self.slope_epsilon_2 = subagent(self.n_state_1, self.n_state_2, self.n_action, low_level_hidden_dim).to(self.epsilon_device)
        self.slope_epsilon_3 = subagent(self.n_state_1, self.n_state_2, self.n_action, low_level_hidden_dim).to(self.epsilon_device)
        self.vol_epsilon_1 = subagent(self.n_state_1, self.n_state_2, self.n_action, low_level_hidden_dim).to(self.epsilon_device)
        self.vol_epsilon_2 = subagent(self.n_state_1, self.n_state_2, self.n_action, low_level_hidden_dim).to(self.epsilon_device)
        self.vol_epsilon_3 = subagent(self.n_state_1, self.n_state_2, self.n_action, low_level_hidden_dim).to(self.epsilon_device)    
        if self.action_mode == "long":

            model_list_slope = [
                "./result/low_level/ETHUSDT/long_action_base/slope/label_1/1/best_model/best_model.pkl", 
                "./result/low_level/ETHUSDT/long_action_base/slope/label_2/4/best_model/best_model.pkl",
                "./result/low_level/ETHUSDT/long_action_base/slope/label_3/0/best_model/best_model.pkl"
            ]
            model_list_vol = [
                "./result/low_level/ETHUSDT/long_action_base/vol/label_1/4/best_model/best_model.pkl",
                "./result/low_level/ETHUSDT/long_action_base/vol/label_2/1/best_model/best_model.pkl",
                "./result/low_level/ETHUSDT/long_action_base/vol/label_3/1/best_model/best_model.pkl"
            ]
        elif self.action_mode == "short":
            model_list_slope = [
                "./result/low_level/ETHUSDT/short_action/best_model/slope/1/best_model.pkl", 
                "./result/low_level/ETHUSDT/short_action/best_model/slope/2/best_model.pkl",
                "./result/low_level/ETHUSDT/short_action/best_model/slope/3/best_model.pkl"
            ]
            model_list_vol = [
                "./result/low_level/ETHUSDT/short_action/best_model/vol/1/best_model.pkl",
                "./result/low_level/ETHUSDT/short_action/best_model/vol/2/best_model.pkl",
                "./result/low_level/ETHUSDT/short_action/best_model/vol/3/best_model.pkl"
            ]    
        elif self.action_mode == "both":
            model_list_slope = [
                "./result/low_level/ETHUSDT/both_action/best_model/slope/1/best_model.pkl", 
                "./result/low_level/ETHUSDT/both_action/best_model/slope/2/best_model.pkl",
                "./result/low_level/ETHUSDT/both_action/best_model/slope/3/best_model.pkl"
            ]
            model_list_vol = [
                "./result/low_level/ETHUSDT/both_action/best_model/vol/1/best_model.pkl",
                "./result/low_level/ETHUSDT/both_action/best_model/vol/2/best_model.pkl",
                "./result/low_level/ETHUSDT/both_action/best_model/vol/3/best_model.pkl"
            ]
        log.info(f"self.model_list_slope:{model_list_slope}")
        log.info(f"self.model_list_vol:{model_list_vol}")
        self.slope_1.load_state_dict(torch.load(model_list_slope[0], map_location=self.epsilon_device))
        self.slope_2.load_state_dict(torch.load(model_list_slope[1], map_location=self.epsilon_device))
        self.slope_3.load_state_dict(torch.load(model_list_slope[2], map_location=self.epsilon_device))
        self.vol_1.load_state_dict(torch.load(model_list_vol[0], map_location=self.epsilon_device))
        self.vol_2.load_state_dict(torch.load(model_list_vol[1], map_location=self.epsilon_device))
        self.vol_3.load_state_dict(torch.load(model_list_vol[2], map_location=self.epsilon_device))
        self.slope_1.eval()
        self.slope_2.eval()
        self.slope_3.eval()
        self.vol_1.eval()
        self.vol_2.eval()
        self.vol_3.eval()
        self.slope_agents = {
            0: self.slope_1,
            1: self.slope_2,
            2: self.slope_3
        }
        self.vol_agents = {
            0: self.vol_1,
            1: self.vol_2,
            2: self.vol_3
        }
        self.slope_epsilon_1.load_state_dict(torch.load(model_list_slope[0], map_location=self.epsilon_device))
        self.slope_epsilon_2.load_state_dict(torch.load(model_list_slope[1], map_location=self.epsilon_device))
        self.slope_epsilon_3.load_state_dict(torch.load(model_list_slope[2], map_location=self.epsilon_device))
        self.vol_epsilon_1.load_state_dict(torch.load(model_list_vol[0], map_location=self.epsilon_device))
        self.vol_epsilon_2.load_state_dict(torch.load(model_list_vol[1], map_location=self.epsilon_device))
        self.vol_epsilon_3.load_state_dict(torch.load(model_list_vol[2], map_location=self.epsilon_device))
        self.slope_epsilon_1.eval()
        self.slope_epsilon_2.eval()
        self.slope_epsilon_3.eval()
        self.vol_epsilon_1.eval()
        self.vol_epsilon_2.eval()
        self.vol_epsilon_3.eval() 
        self.slope_epsilon_agents = {
            0: self.slope_epsilon_1,
            1: self.slope_epsilon_2,
            2: self.slope_epsilon_3
        }
        self.vol_epsilon_agents = {
            0: self.vol_epsilon_1,
            1: self.vol_epsilon_2,
            2: self.vol_epsilon_3
        }

        self.update_times = args.update_times
        self.optimizer = torch.optim.Adam(self.hyperagent.parameters(), lr=args.lr)
        self.loss_func = nn.MSELoss()
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.n_step = args.n_step
        self.eval_update_freq = args.eval_update_freq
        self.buffer_size = args.buffer_size
        self.epsilon_start = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.decay_length = args.decay_length
        self.epsilon_scheduler = LinearDecaySchedule(start_epsilon=self.epsilon_start, end_epsilon=self.epsilon_end, decay_length=self.decay_length)
        self.epsilon = args.epsilon_start
        episodicmemory_dim = 64*4
        self.memory = episodicmemory(4320, 10, self.n_state_1, self.n_state_2, episodicmemory_dim, self.device)
        self.no_risk_return = args.no_risk_return
        self.best_return_rate = -float('inf')    # 最佳收益率记录 / Best return rate record 
        self.validation_queue = queue.Queue(maxsize=10)  # 验证任务队列，限制大小避免内存溢出
        self.reward_no_action = args.reward_no_action
        

    def calculate_q(self, w, qs):
        q_tensor = torch.stack(qs)# qs将6个代理的2个动作的权重 [6，2]  => [6,1,2]
        q_tensor = q_tensor.permute(1, 0, 2) # 重排维度为 [1,6,2]
        weights_reshaped = w.view(-1, 1, 6) #  超代理评估的6个子代理权重为[1,6], 改变形状为[1，1，6]
        combined_q = torch.bmm(weights_reshaped, q_tensor).squeeze(1) # 执行批量矩阵乘法[1,1,2]并压缩维度得到最终Q值 [1, 2]
        
        return combined_q


    def update(self, replay_buffer):
        """
        更新策略网络参数，包含TD误差、记忆误差和KL散度的联合优化
        
        Args:
            replay_buffer: 经验回放缓冲区，包含(s, a, r, s', demo_action)等数据
            
        Returns:
            tuple: 包含各损失项和Q值统计的元组
                - td_error: TD误差损失
                - memory_error: 记忆误差损失
                - KL_loss: KL散度损失
                - q_current_mean: 当前Q值均值
                - q_target_mean: 目标Q值均值
        """
        # Sample transition from replay buffer & move to device
        # 从经验回放缓冲区采样并移动到指定设备
        cpu_batch, _, _ = replay_buffer.sample()
        # log.debug("sample")
        #log.debug(batch)
        batch = {k: v.to(self.device) for k, v in cpu_batch.items()}
        # Calculate current and target hypernetwork outputs
        # 计算当前和目标超网络pd.read_feather(os.path.join(self.train_data_path, "train.feather") ) .header输出
        w_current = self.hyperagent(batch['state'], batch['state_trend'], batch['state_clf'], batch['previous_action'])
        w_next = self.hyperagent_target(batch['next_state'], batch['next_state_trend'], batch['next_state_clf'], batch['next_previous_action'])
        w_next_ = self.hyperagent(batch['next_state'], batch['next_state_trend'], batch['next_state_clf'], batch['next_previous_action'])

        # Compute Q-values from slope/volatility agents
        # 计算斜率/波动率代理的Q值
        batch_state = cpu_batch['state'].to(self.epsilon_device)
        batch_state_trend= cpu_batch['state_trend'].to(self.epsilon_device)
        batch_previous_action= cpu_batch['previous_action'].to(self.epsilon_device)
        batch_next_state = cpu_batch['next_state'].to(self.epsilon_device)
        batch_next_state_trend= cpu_batch['next_state_trend'].to(self.epsilon_device)
        batch_next_previous_action= cpu_batch['next_previous_action'].to(self.epsilon_device)
        qs_current = [
                    self.slope_agents[0](batch_state, batch_state_trend, batch_previous_action).to(self.device),
                    self.slope_agents[1](batch_state, batch_state_trend, batch_previous_action).to(self.device),
                    self.slope_agents[2](batch_state, batch_state_trend, batch_previous_action).to(self.device),
                    self.vol_agents[0](batch_state, batch_state_trend, batch_previous_action).to(self.device),
                    self.vol_agents[1](batch_state, batch_state_trend, batch_previous_action).to(self.device),
                    self.vol_agents[2](batch_state, batch_state_trend, batch_previous_action).to(self.device)
        ]
        qs_next = [
                    self.slope_agents[0](batch_next_state, batch_next_state_trend, batch_next_previous_action).to(self.device),
                    self.slope_agents[1](batch_next_state, batch_next_state_trend, batch_next_previous_action).to(self.device), 
                    self.slope_agents[2](batch_next_state, batch_next_state_trend, batch_next_previous_action).to(self.device),
                    self.vol_agents[0](batch_next_state, batch_next_state_trend, batch_next_previous_action).to(self.device),
                    self.vol_agents[1](batch_next_state, batch_next_state_trend, batch_next_previous_action).to(self.device),
                    self.vol_agents[2](batch_next_state, batch_next_state_trend, batch_next_previous_action).to(self.device)
        ]
        # Calculate Q distribution and gather selected actions
        # 计算Q分布并收集选定动作
        q_distribution = self.calculate_q(w_current, qs_current)
        q_current = q_distribution.gather(-1, batch['action']).squeeze(-1)
        # Compute target Q-values with double DQN style argmax
        # 使用Double DQN风格计算目标Q值
        a_argmax = self.calculate_q(w_next_, qs_next).argmax(dim=-1, keepdim=True)
        q_nexts = self.calculate_q(w_next, qs_next)
        q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * q_nexts.gather(-1, a_argmax).squeeze(-1)
        # Calculate individual loss components
        # 计算各损失项
        td_error = self.loss_func(q_current, q_target)
        memory_error = self.loss_func(q_current, batch['q_memory'])
        # Demonstration-guided KL divergence loss
        # 示范引导的KL散度损失
        demonstration = batch['demo_action']
        KL_loss = F.kl_div(
            (q_distribution.softmax(dim=-1) + 1e-8).log(),
            (demonstration.softmax(dim=-1) + 1e-8),
            reduction="batchmean",
        )
        # Total loss with weighted components
        # 加权总损失，   alpha 代表了记忆的经验权重， beta代表先验q-table权重
        loss = td_error + args.alpha * memory_error + args.beta * KL_loss
        self.optimizer.zero_grad()
        loss.backward()
        #log.debug(f"loss: {loss.item()} td_error: {td_error.item()} memory_error: {memory_error.item()} KL_loss: {KL_loss.item()}")
        torch.nn.utils.clip_grad_norm_(self.hyperagent.parameters(), 1)
        self.optimizer.step()
        for param, target_param in zip(self.hyperagent.parameters(), self.hyperagent_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        self.update_counter += 1
        
        
        if self.update_counter % self.q_value_memorize_freq == 1:
            print(f"[update] cnt={self.update_counter} loss={loss.item():.6f} td={td_error.item():.6f} mem={memory_error.item():.6f} KL={KL_loss.item():.6f}", flush=True)

            # debug: 更多中间指标（避免直接打印整批 tensor，输出统计量）
            q_memory = batch['q_memory']
            q_dist_sm = q_distribution.softmax(dim=-1)
            demo_sm = demonstration.softmax(dim=-1)
            print(
                f"[update_debug] cnt={self.update_counter} "
                f"q_current(mean={q_current.mean().item():.6f},min={q_current.min().item():.6f},max={q_current.max().item():.6f}) "
                f"q_target(mean={q_target.mean().item():.6f},min={q_target.min().item():.6f},max={q_target.max().item():.6f}) "
                f"q_memory(mean={q_memory.mean().item():.6f},min={q_memory.min().item():.6f},max={q_memory.max().item():.6f}) "
                f"qdist_sm(min={q_dist_sm.min().item():.8f},max={q_dist_sm.max().item():.8f}) "
                f"log_qdist_sm(min={(q_dist_sm + 1e-8).log().min().item():.8f},max={(q_dist_sm + 1e-8).log().max().item():.8f}) "
                f"demo_sm(min={demo_sm.min().item():.8f},max={demo_sm.max().item():.8f})",
                flush=True,
            )
            # debug: gating 权重分布（6 维）
                # w_current: [B,6]（按实现一般已是 softmax 后的概率）
            with torch.no_grad():
                w = w_current
                # 数值安全：避免 log(0)
                w_eps = (w + 1e-12)
                ent = -(w_eps * w_eps.log()).sum(dim=-1)  # [B]
                w_max, w_argmax = w.max(dim=-1)           # [B]

                # top1 频率
                top1_cnt = torch.bincount(w_argmax, minlength=6).float()
                top1_frac = (top1_cnt / top1_cnt.sum()).detach().cpu().numpy().tolist()

                # 均值权重
                w_mean = w.mean(dim=0).detach().cpu().numpy().tolist()

                print(
                    f"[w_debug] cnt={self.update_counter} "
                    # f"ent(mean={ent.mean().item():.6f},min={ent.min().item():.6f},max={ent.max().item():.6f}) "
                    # f"w_max(mean={w_max.mean().item():.6f},p95={torch.quantile(w_max, 0.95).item():.6f}) "
                    # f"w(min={w.min().item():.6f},max={w.max().item():.6f})\n"
                    # f"         w_mean={['{:.3f}'.format(x) for x in w_mean]}\n"
                    f"         top1_frac={['{:.3f}'.format(x) for x in top1_frac]}",
                    flush=True,
                )
        return td_error, memory_error, KL_loss, torch.mean(q_current), torch.mean(q_target)

    def select_action(self, state, state_trend, state_clf, info):
        """
        选择动作（训练时使用）
        
        Args:
            state: 当前状态
            state_trend: 状态趋势
            state_clf: 状态分类特征
            info: 包含历史动作等信息的字典
            
        Returns:
            int: 选择的动作编号 (0或1)
            
        Process:
            1. 以epsilon概率随机探索
            2. 否则通过超网络计算Q值并选择最优动作
        """
        # Convert input data to PyTorch tensors and move to target device
        # 转换输入数据为张量并移动到目标设备
        x1 = torch.FloatTensor(state).to(self.epsilon_device)
        x2 = torch.FloatTensor(state_trend).to(self.epsilon_device)
        x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.epsilon_device)
        previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long().to(self.epsilon_device), 0).to(self.epsilon_device)
        
        # Epsilon-greedy action selection
        # epsilon-greedy策略选择动作
        if np.random.uniform() < (1-self.epsilon):
            # Get Q-values from slope/volatility agents
            # 获取斜率/波动率代理的Q值
            qs = [
                    self.slope_epsilon_agents[0](x1, x2, previous_action),
                    self.slope_epsilon_agents[1](x1, x2, previous_action),
                    self.slope_epsilon_agents[2](x1, x2, previous_action),
                    self.vol_epsilon_agents[0](x1, x2, previous_action),
                    self.vol_epsilon_agents[1](x1, x2, previous_action),
                    self.vol_epsilon_agents[2](x1, x2, previous_action)
            ]
            # Calculate hypernetwork output
            # 计算超网络输出 6个子代理的权重
            w = self.epsilon_hyperagent(x1, x2, x3, previous_action)
            # Combine Q-values using hypernetwork weights
            # 使用超网络权重组合Q值

            actions_value = self.calculate_q(w, qs)
            # Select action with max Q-value
            # 选择最大Q值的动作 (根据动作q值，选择动作， max()[1]选择的数组下标,max()[0] q值)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
            #log.debug(f"select_action epsilon: {action} {previous_action} {w.data.cpu().numpy()}")
            # log.debug(f"  {qs}")
        else:
            action_choice = [0,1]
            action = random.choice(action_choice)
            #log.debug(f"select_action random: {action}")
        return action

    def q_estimate(self, state, state_trend, state_clf, info):
        """
        状态价值估计
        
        Args:
            state: 当前状态
            state_trend: 状态趋势
            state_clf: 状态分类特征
            info: 包含历史动作等信息的字典
            
        Returns:
            float: 当前状态的最大Q值估计
        """
        # Convert input data to PyTorch tensors and move to target device
        # 转换输入数据为张量并移动到目标设备
        x1 = torch.FloatTensor(state).to(self.epsilon_device)
        x2 = torch.FloatTensor(state_trend).to(self.epsilon_device)
        x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.epsilon_device)
        previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long().to(self.epsilon_device),0).to(self.epsilon_device)
        
        # Get Q-values from slope/volatility agents
        # 获取斜率/波动率代理的Q值
        
        qs = [
                self.slope_epsilon_agents[0](x1, x2, previous_action).to(self.device),
                self.slope_epsilon_agents[1](x1, x2, previous_action).to(self.device),
                self.slope_epsilon_agents[2](x1, x2, previous_action).to(self.device),
                self.vol_epsilon_agents[0](x1, x2, previous_action).to(self.device),
                self.vol_epsilon_agents[1](x1, x2, previous_action).to(self.device),
                self.vol_epsilon_agents[2](x1, x2, previous_action).to(self.device)
        ]
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        x3 = x3.to(self.device)
        previous_action = previous_action.to(self.device)
        # Calculate hypernetwork output
        # 计算超网络输出
        w = self.hyperagent(x1, x2, x3, previous_action)
        # 形状[1,6]， 6个 子网络的权重，
        # Combine Q-values using hypernetwork weights
        # 使用超网络权重组合Q值  
        actions_value = self.calculate_q(w, qs)
        # Return max Q-value as state estimate
        # 返回最大Q值作为状态估计
        q = torch.max(actions_value, 1)[0].detach().cpu().numpy()
        
        return q

    def calculate_hidden(self, state, state_trend, info):
        """
        提取隐藏状态表示
        
        Args:
            state: 当前状态
            state_trend: 状态趋势
            info: 包含历史动作等信息的字典
            
        Returns:
            numpy.ndarray: 隐藏状态的numpy数组表示
        """
        # Convert input data to PyTorch tensors and move to target device
        # 转换输入数据为张量并移动到目标设备
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long().to(self.device), 0).to(self.device)
        with torch.no_grad():
            # Encode to get hidden state representation
            # 编码获取隐藏状态表示
            hs = self.hyperagent.encode(x1, x2, previous_action).cpu().numpy()
        return hs
    
    
    def _train_data_file(self, 
                        episode_counter, 
                        step_counter, 
                        epoch_return_rate_train_list,
                        epoch_final_balance_train_list,
                        epoch_required_money_train_list,
                        epoch_reward_sum_train_list  
                    ):
        """
        训练数据处理方法
        Process training data and execute reinforcement learning loop
        
        功能说明:
        1. 加载训练数据并初始化环境
        2. 执行完整的训练周期
        3. 收集经验数据并更新模型
        4. 记录训练指标
        
        参数说明:
            episode_counter: 总episode计数器
            step_counter: 总步数计数器
            epoch_*_train_list: 各类训练指标记录列表
            
        返回值:
            更新后的episode计数器
            更新后的step计数器
        """
        
        temp_experience_buffer = []
        
        train_env = Training_Env(
                "full",
                df=self.df,
                q_table_path=self.result_path,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                clf_list=self.clf_list,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                num_action = self.n_action,
                actions = self.actions,   
                action_mode=self.action_mode,  
                initial_action=random.choices(range(self.n_action), k=1)[0],
                reward_no_action=self.reward_no_action,
                alpha = 0)
        single_state, trend_state, clf_state, info = train_env.reset()
        episode_reward_sum = 0
        log.info(f" train_env reset done")
        while True:
            start_time = time.time()
            # 使用ε-greedy策略选择动作
            # Select action using ε-greedy strategy
            action = self.select_action(single_state, trend_state, clf_state, info)
            start_time2 =time.time()
            duration1 = start_time2 - start_time
            # 执行环境步进操作
            # Execute environment step
            next_single_state, next_trend_state, next_clf_state, reward, done, next_info = train_env.step(action)
            start_time3 =time.time()
            duration2 = start_time3 - start_time2
            # 获取历史动作信息
            # Get historical action information
            previous_action = info['previous_action']
            demo_action = info['q_value']
            next_previous_action = next_info['previous_action']
            next_demo_action = next_info['q_value']
            
            
            # 计算隐藏状态
            # Calculate hidden state
            hs = self.calculate_hidden(single_state, trend_state, info)
            start_time4 =time.time()
            duration3 = start_time4 - start_time3
            # 查询记忆库中的Q值
            # Query Q-value from memory
            q_memory = self.memory.query(hs, action)
            start_time5 =time.time()
            duration4 = start_time5 - start_time4
            # log.debug(f"info {info} hs {hs} q_memory={q_memory}")
            # 计算目标Q值
            # Calculate target Q-value
            q = reward + self.gamma * (1 - done) * self.q_estimate(next_single_state, next_trend_state, next_clf_state, next_info)
            start_time6 =time.time()
            duration5 = start_time6 - start_time5
            if np.isnan(q_memory):
                q_memory = q
            # 更新记忆库
            # Update memory            
            self.memory.add(hs, action, q, single_state, trend_state, previous_action)

            # 存储经验到回放缓冲区
            # Store transition in replay buffer
            self.replay_buffer.store_transition(single_state, trend_state, clf_state, previous_action, demo_action, action, reward, next_single_state, next_trend_state, next_clf_state, next_previous_action, next_demo_action, done, q_memory)
            start_time7 =time.time()
            duration6 = start_time7 - start_time6

            episode_reward_sum += reward

            single_state, trend_state, clf_state, info = next_single_state, next_trend_state, next_clf_state, next_info
            step_counter += 1
            # 定期执行模型更新
            # Periodically update model parameters
            if step_counter % self.eval_update_freq == 0 and step_counter > (self.batch_size + self.n_step):
                # log.info(f"update network {step_counter} start_time {start_time} duration1 {duration1:.6f} duration2 {duration2:.6f} duration3 {duration3:.6f} duration4 {duration4:.6f} duration5 {duration5:.6f} duration6 {duration6:.6f}")
 
                # log.info(f"批量存储 {len(temp_experience_buffer)} 条经验到replay buffer")                         
                for i in range(self.update_times):
                    td_error, memory_error, KL_loss, q_eval, q_target = self.update(self.replay_buffer)
                    if self.update_counter % self.q_value_memorize_freq == 1:
                        self.writer.add_scalar(tag="td_error", scalar_value=td_error.cpu(), global_step=self.update_counter, walltime=None)
                        self.writer.add_scalar(tag="memory_error", scalar_value=memory_error.cpu(), global_step=self.update_counter, walltime=None)
                        self.writer.add_scalar(tag="KL_loss", scalar_value=KL_loss.cpu(), global_step=self.update_counter, walltime=None)
                        self.writer.add_scalar(tag="q_eval", scalar_value=q_eval.cpu(), global_step=self.update_counter, walltime=None)
                        self.writer.add_scalar(tag="q_target", scalar_value=q_target.cpu(), global_step=self.update_counter, walltime=None)
                    # _state_dict = self.hyperagent.state_dict()
                    # log.debug("_state_dict {_state_dict}")
                self.epsilon_hyperagent.load_state_dict(self.hyperagent.state_dict())
                self.epsilon_hyperagent.to(self.epsilon_device)
                self.epsilon_hyperagent.eval()
                if step_counter > 4320:
                    # 定期重新编码记忆, 因为超代理的隐藏层训练后发生了变化，
                    # Periodically re-encode memory
                    self.memory.re_encode(self.hyperagent)
                return_margin, final_balance, required_money, commission_fee = train_env.get_final_return_rate()                
                # log.info(f"update network {step_counter} return_margin:{return_margin:.2f},final_balance:{final_balance:.2f},required_money:{required_money:.2f},commission_fee:{commission_fee:.2f}")
            
            if done:
                self.epsilon_hyperagent.load_state_dict(self.hyperagent.state_dict())
                self.epsilon_hyperagent.to(self.epsilon_device)
                self.epsilon_hyperagent.eval()
                log.info(f" train_env step done")
                break
        episode_counter += 1
        # 获取最终收益指标
        # Get final financial metrics
        final_balance, required_money = train_env.final_balance, train_env.required_money
        
        self.writer.add_scalar(tag="return_rate_train", scalar_value=final_balance / (required_money), global_step=episode_counter, walltime=None)
        self.writer.add_scalar(tag="final_balance_train", scalar_value=final_balance, global_step=episode_counter, walltime=None)
        self.writer.add_scalar(tag="required_money_train", scalar_value=required_money, global_step=episode_counter, walltime=None)
        self.writer.add_scalar(tag="reward_sum_train", scalar_value=episode_reward_sum, global_step=episode_counter, walltime=None)
        epoch_return_rate_train_list.append(final_balance / (required_money))
        epoch_final_balance_train_list.append(final_balance)
        epoch_required_money_train_list.append(required_money)
        epoch_reward_sum_train_list.append(episode_reward_sum)
        
        return episode_counter, step_counter
    
    def _save_trained_model(self,
                            epoch_counter,
                            epoch_path,
                            epoch_return_rate_train_list,
                            epoch_final_balance_train_list,
                            epoch_required_money_train_list,
                            epoch_reward_sum_train_list  ):
        """
        保存训练完成的模型及其训练指标
        
        中文说明：
        本函数负责计算当前训练周期的平均性能指标，
        将这些指标写入TensorBoard日志，并保存模型参数到指定路径。
        
        English description:
        This function calculates the average performance metrics for the current training epoch,
        writes these metrics to TensorBoard logs, and saves the model parameters to the specified path.
        
        Parameters:
            epoch_counter (int): 当前训练周期计数器 / Current epoch counter
            epoch_path (str): 模型保存路径 / Path to save the model
            epoch_return_rate_train_list (list): 周期收益率列表 / List of return rates
            epoch_final_balance_train_list (list): 周期最终余额列表 / List of final balances
            epoch_required_money_train_list (list): 周期所需资金列表 / List of required money
            epoch_reward_sum_train_list (list): 周期总奖励列表 / List of total rewards
            
        Returns:
            None
        """
        mean_return_rate_train = np.mean(epoch_return_rate_train_list)
        mean_final_balance_train = np.mean(epoch_final_balance_train_list)
        mean_required_money_train = np.mean(epoch_required_money_train_list)
        mean_reward_sum_train = np.mean(epoch_reward_sum_train_list)
        self.writer.add_scalar(tag="epoch_return_rate_train", scalar_value=mean_return_rate_train, global_step=epoch_counter, walltime=None)
        self.writer.add_scalar(tag="epoch_final_balance_train", scalar_value=mean_final_balance_train, global_step=epoch_counter, walltime=None)
        self.writer.add_scalar(tag="epoch_required_money_train", scalar_value=mean_required_money_train, global_step=epoch_counter, walltime=None)
        self.writer.add_scalar(tag="epoch_reward_sum_train", scalar_value=mean_reward_sum_train, global_step=epoch_counter, walltime=None)
        torch.save(self.hyperagent.state_dict(), os.path.join(epoch_path, "trained_model.pkl"))  
        
    def train(self):
        """
        训练主方法
        Main training loop for the reinforcement learning agent
        
        功能说明:
        1. 初始化训练参数和缓冲区
        2. 执行多轮训练周期(epochs)
        3. 定期保存模型检查点
        4. 执行验证和测试评估
        5. 保存最佳模型
        
        训练流程:
        - 每个epoch包含完整的训练周期
        - 动态调整探索率(epsilon)
        - 模型验证与持久化
        - 最终测试评估
        
        返回值:
            None (模型保存到磁盘文件)
        """
        print(f"[train] pid={os.getpid()}")
        step_counter = 0
        episode_counter = 0
        epoch_counter = 0
        best_return_rate = -float('inf')
        best_model = None
        self._start_validation_consumer()
        self.df = pd.read_feather(os.path.join(self.train_data_path, "train.feather") )
        #.head(10000)
        log.info(f"train data length: {len(self.df)}")
        # 初始化经验回放缓冲区
        # Initialize replay buffer for experience storage
        self.replay_buffer = ReplayBuffer_High(args, self.n_state_1, self.n_state_2, self.n_action) 
        for sample in range(self.epoch_number):
            epoch_return_rate_train_list = []
            epoch_final_balance_train_list = []
            epoch_required_money_train_list = []
            epoch_reward_sum_train_list = []
            
            log.info(f'epoch {epoch_counter + 1}')
            episode_counter, step_counter = self._train_data_file(episode_counter, step_counter, 
                                                                        epoch_return_rate_train_list,
                                                                        epoch_final_balance_train_list,
                                                                        epoch_required_money_train_list,
                                                                        epoch_reward_sum_train_list)    
            epoch_counter += 1
            # 更新探索率(epsilon)
            # Update exploration rate (epsilon)
            self.epsilon = self.epsilon_scheduler.get_epsilon(epoch_counter)
            
            epoch_path = os.path.join(self.model_path, "epoch_{}".format(epoch_counter))
            if not os.path.exists(epoch_path):
                os.makedirs(epoch_path)
            self._save_trained_model(epoch_counter,epoch_path,
                                        epoch_return_rate_train_list,
                                        epoch_final_balance_train_list,
                                        epoch_required_money_train_list,
                                        epoch_reward_sum_train_list  )
            # 执行验证评估
            # Execute validation evaluation
            val_path = os.path.join(epoch_path, "val")
            if not os.path.exists(val_path):
                os.makedirs(val_path)
                
            validation_task = (epoch_counter, epoch_path, val_path,args)
            self.validation_queue.put(validation_task, timeout=5.0)
            log.info(f"验证任务已加入队列: epoch {epoch_counter}")
            
            # evalor = HIGH_LEVEL_DQN_EVAL(args) 
            
            # return_rate_eval = evalor.val_cluster(epoch_path, val_path)
            # # 更新最佳模型
            # # Update best model if improved
            # if return_rate_eval > best_return_rate:
            #     best_return_rate = return_rate_eval
            #     best_model = self.hyperagent.state_dict()
            #     best_model_path = os.path.join(self.result_path, 'best_model.pkl')
            #     torch.save(best_model, best_model_path)
            #     log.info(f"best train eval return_rate_eval:{return_rate_eval} epoch_path:{epoch_path}")
            #     # 保存最佳模型到文件
            #     # Save best model to disk

        log.info(f"train done")
        # 执行最终测试评估
        # Execute final test evaluation
        final_result_path = self.result_path
        #self.test_cluster(best_model_path, final_result_path)

    def _start_validation_consumer(self):
        def validation_consumer():
            log.info("验证消费者线程启动")
            while True:
                try:
                    # 从队列获取验证任务，设置超时避免无限等待
                    validation_task = self.validation_queue.get_nowait()
                    if validation_task is None:  # 停止信号
                        break
                    
                    epoch_counter, epoch_path, val_path,_args = validation_task
                    log.info(f"验证消费者处理任务: epoch {epoch_counter}")                    
                    self._async_validate_and_save(epoch_counter, epoch_path, val_path,_args)
                except queue.Empty:
                    time.sleep(60)
                    continue  # 队列为空，继续等待
                except Exception as e:
                    log.error(f"验证消费者线程异常: {e}")
                    continue
            
            log.info("验证消费者线程停止")
        
        # 启动验证消费者线程 
        self.validation_thread = threading.Thread(target=validation_consumer, daemon=True)
        self.validation_thread.start()
        log.info("验证消费者线程已启动")
    def _async_validate_and_save(self, epoch_counter, epoch_path, val_path,_args ):
        """
        异步执行验证和保存最佳模型的逻辑
        
        中文说明：
        本函数在单独的线程中执行验证和最佳模型保存操作，
        避免阻塞主训练进程。
        
        English description:
        This function performs validation and best model saving in a separate thread,
        preventing blocking of the main training process.
        
        Parameters:
            epoch_counter (int): 当前训练周期 / Current epoch counter
            epoch_path (str): 模型保存路径 / Model saving path
            val_path (str): 验证结果保存路径 / Validation result saving path
            var_df_list (list): 验证数据索引列表 / Validation data index list
            
        Returns:
            tuple: return_rate_eval 
        """
        log.info(f"异步验证开始: epoch {epoch_counter}")
        return_rates = []
        args_list = [
            (epoch_path, val_path,_args)          
        ]
        with multiprocessing.Pool(processes=1) as pool:
            results = pool.imap_unordered(_validate_worker, args_list)
            # 创建验证实例 / Create validation instance
            for result in results:
                return_rates.append(result)   
             
        # 执行验证 / Perform validation     
        # 计算平均验证收益率 / Calculate average validation return rate
        return_rate_eval = np.mean(return_rates)
        log.info(f"异步验证结束: epoch {epoch_counter}, 验证收益率: {return_rate_eval} best:{self.best_return_rate}")
        if return_rate_eval > self.best_return_rate:
            self.best_return_rate = return_rate_eval 
            model_file = os.path.join(epoch_path, "trained_model.pkl")
            log.info(f"best model updated to epoch {epoch_counter}.best_return_rate:{self.best_return_rate}")
            best_model_folder_path = os.path.join(self.result_path, 'best_model')
            if not os.path.exists(best_model_folder_path):
                os.makedirs(best_model_folder_path)
            best_model_path = os.path.join(best_model_folder_path, 'best_model.pkl')
            #将模型保存到指定路径 / Save model to specified path    
            shutil.copy2(model_file, best_model_path)
            
            
class HIGH_LEVEL_DQN_EVAL(DQN):
    def __init__(self, args):  # 定义DQN的一系列属性
        super(HIGH_LEVEL_DQN_EVAL,
              self).__init__(args)
        
    def act_test(self, state, state_trend, state_clf, info):
        """
        测试模式下的动作选择（无随机性）
        
        Args:
            state: 当前状态
            state_trend: 状态趋势
            state_clf: 状态分类特征
            info: 包含历史动作等信息的字典
            
        Returns:
            int: 确定性选择的最优动作
        """
        with torch.no_grad():
            x1 = torch.FloatTensor(state).to(self.epsilon_device)
            x2 = torch.FloatTensor(state_trend).to(self.epsilon_device)
            x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.epsilon_device)
            previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long().to(self.epsilon_device), 0).to(self.epsilon_device)
            qs = [
                    self.slope_epsilon_agents[0](x1, x2, previous_action),
                    self.slope_epsilon_agents[1](x1, x2, previous_action),
                    self.slope_epsilon_agents[2](x1, x2, previous_action),
                    self.vol_epsilon_agents[0](x1, x2, previous_action),
                    self.vol_epsilon_agents[1](x1, x2, previous_action),
                    self.vol_epsilon_agents[2](x1, x2, previous_action)
            ]
            w = self.epsilon_hyperagent(x1, x2, x3, previous_action)
            actions_value = self.calculate_q(w, qs)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
            return action
        
    def val_cluster(self, epoch_path, save_path):
        log.info(f"val_cluster epoch_path:{epoch_path}")
        self.epsilon_hyperagent.load_state_dict(
            torch.load(os.path.join(epoch_path, "trained_model.pkl")))
        self.epsilon_hyperagent.eval()
        counter = False
        action_list = []
        reward_list = []
        final_balance_list = []
        required_money_list = []
        commission_fee_list = []
        self.df = pd.read_feather(os.path.join(self.val_data_path, "val.feather"))
        log.info(f"val data length: {len(self.df)}")
        
        val_env = Testing_Env(
                df=self.df,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                clf_list=self.clf_list,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                initial_action=0,
                actions = self.actions,   
                action_mode=self.action_mode 
                )
        s, s2, s3, info = val_env.reset()
        done = False
        action_list_episode = []
        reward_list_episode = []
        while not done:
            a = self.act_test(s, s2, s3, info)
            s_, s2_, s3_, r, done, info_ = val_env.step(a)
            reward_list_episode.append(r)
            s, s2, s3, info = s_, s2_, s3_, info_
            action_list_episode.append(a)
        return_margin, final_balance, required_money, commission_fee = val_env.get_final_return_rate(slient=True)
        trade_records_len = len(val_env.trade_records)
        # plot_money_curve(val_env.trade_records, val_env.value_history, self.df, save_path)
        log.info(f"val return_margin:{return_margin:.2f},final_balance:{final_balance:.2f},required_money:{required_money:.2f},commission_fee:{commission_fee:.2f},trade_records count:{trade_records_len}")
     
        
        final_balance = val_env.final_balance
        action_list.append(action_list_episode)
        reward_list.append(reward_list_episode)
        final_balance_list.append(final_balance)
        required_money_list.append(required_money)
        commission_fee_list.append(commission_fee)
        action_list = np.array(action_list)
        reward_list = np.array(reward_list)
        final_balance_list = np.array(final_balance_list)
        required_money_list = np.array(required_money_list)
        commission_fee_list = np.array(commission_fee_list)
        np.save(os.path.join(save_path, "action_val.npy"), action_list)
        np.save(os.path.join(save_path, "reward_val.npy"), reward_list)
        np.save(os.path.join(save_path, "final_balance_val.npy"), final_balance_list)
        np.save(os.path.join(save_path, "require_money_val.npy"), required_money_list)
        np.save(os.path.join(save_path, "commission_fee_history_val.npy"), commission_fee_list)
        return_rate = final_balance / required_money
        return return_rate
 
class HIGH_LEVEL_DQN_TEST(DQN):
    def __init__(self, args):  # 定义DQN的一系列属性
        super(HIGH_LEVEL_DQN_TEST,
              self).__init__(args)
        
    def act_test(self, state, state_trend, state_clf, info):
        """
        测试模式下的动作选择（无随机性）
        
        Args:
            state: 当前状态
            state_trend: 状态趋势
            state_clf: 状态分类特征
            info: 包含历史动作等信息的字典
            
        Returns:
            int: 确定性选择的最优动作
        """
        with torch.no_grad():
            x1 = torch.FloatTensor(state).to(self.device)
            x2 = torch.FloatTensor(state_trend).to(self.device)
            x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device)
            previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long().to(self.device), 0).to(self.device)
            qs = [
                    self.slope_agents[0](x1, x2, previous_action),
                    self.slope_agents[1](x1, x2, previous_action),
                    self.slope_agents[2](x1, x2, previous_action),
                    self.vol_agents[0](x1, x2, previous_action),
                    self.vol_agents[1](x1, x2, previous_action),
                    self.vol_agents[2](x1, x2, previous_action)
            ]
            w = self.hyperagent(x1, x2, x3, previous_action)
            actions_value = self.calculate_q(w, qs)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
            return action
        
    def test_cluster(self, epoch_path, save_path):     
        log.info(f"开始测试: epoch {epoch_path}")
        self.hyperagent.load_state_dict(torch.load(epoch_path))
        self.hyperagent.to(self.device)
        self.hyperagent.eval()
        self.slope_1.to(self.device)
        self.slope_2.to(self.device)
        self.slope_3.to(self.device)
        self.vol_1.to(self.device)
        self.vol_2.to(self.device)
        self.vol_3.to(self.device)
        counter = False
        action_list = []
        reward_list = []
        final_balance_list = []
        required_money_list = []
        commission_fee_list = []
        self.df = pd.read_feather(os.path.join(self.test_data_path, "test.feather"))  
        log.info(self.df.head(10))
        log.info(self.df.tail(10))
        log.info(len(self.df))
        test_env = Testing_Env(
                df=self.df,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                clf_list=self.clf_list,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                initial_action=0,
                actions = self.actions,   
                action_mode=self.action_mode ,
                reward_no_action=self.reward_no_action)
        s, s2, s3, info = test_env.reset()
        done = False
        action_list_episode = []
        reward_list_episode = []
        step_times = 0
        while not done:
            a = self.act_test(s, s2, s3, info)
            s_, s2_, s3_, r, done, info_ = test_env.step(a)
            reward_list_episode.append(r)
            s, s2, s3, info = s_, s2_, s3_, info_
            action_list_episode.append(a)
            step_times += 1
            if step_times%10000==0:
                log.info(f"step_times:{step_times},a:{a},r:{r}")
        return_margin, final_balance, required_money, commission_fee = test_env.get_final_return_rate(slient=True)    
        metrics = calculate_trading_metrics(self.df,test_env.trade_records, test_env.value_history, self.no_risk_return) 
    
        total_amount= metrics['total_amount'] #总交易金额
        annualized_volatility = metrics['annualized_volatility'] #年化波动率
        win_rate = metrics['win_rate'] #胜率
        profit_loss_ratio = metrics['profit_loss_ratio'] #盈亏比
        total_trades = metrics['total_trades'] #总交易次数
        trade_frequency = metrics['trade_frequency'] #交易频率
        downside_deviation = metrics['downside_deviation'] #下行标准差
        alpha = metrics['alpha'] #alpha
        beta = metrics['beta'] #beta
        
        log_metrics_string = f"""
        初始化金额:{test_env.initial_money:.2f},
        累计收益率:{return_margin:.2f},
        总交易金额:{total_amount:.2f},
        累积净收益:{final_balance:.2f},
        年化波动率:{annualized_volatility:.4f},
        最大回撤:{required_money:.2f},
        胜率:{win_rate:.2f},
        盈亏比:{profit_loss_ratio:.2f},
        交易总次数:{total_trades:.2f},
        交易频率:{trade_frequency:.2f},
        下行标准差:{downside_deviation:.4f},
        alpha:{alpha:.4f},
        beta:{beta:.4f},
        手续费:{commission_fee:.2f}
        """
        log.info(log_metrics_string)
        plot_money_curve(test_env.trade_records, test_env.value_history, self.df, save_path)
        
        final_balance = test_env.final_balance
        action_list.append(action_list_episode)
        reward_list.append(reward_list_episode)
        final_balance_list.append(final_balance)
        required_money_list.append(required_money)
        commission_fee_list.append(commission_fee)

        action_list = np.array(action_list)
        reward_list = np.array(reward_list)
        final_balance_list = np.array(final_balance_list)
        required_money_list = np.array(required_money_list)
        commission_fee_list = np.array(commission_fee_list)         
        
        np.save(os.path.join(save_path, "action.npy"), action_list)
        np.save(os.path.join(save_path, "reward.npy"), reward_list)
        np.save(os.path.join(save_path, "final_balance.npy"), final_balance_list)
        np.save(os.path.join(save_path, "require_money.npy"), required_money_list)
        np.save(os.path.join(save_path, "commission_fee_history.npy"), commission_fee_list)
        
        np.save(os.path.join(save_path, "trade_records.npy"), np.array(test_env.trade_records))
        np.save(os.path.join(save_path, "value_history.npy"), np.array(test_env.value_history))
            
 
 
def _validate_test_worker(args):
    final_result_path = os.path.join("./result/high_level", '{}'.format(args.dataset), args.exp)
    best_model_path = os.path.join("./result/high_level", '{}'.format(args.dataset), args.exp, "seed_{}".format(args.seed),
                                       'epoch_{}'.format(args.i) ,'trained_model.pkl')
    log.info(f"start  test_cluster best_model_path:{best_model_path},self.result_path:{final_result_path}")
    logs_dir = os.path.join("./logs/high_level", '{}'.format(args.dataset), args.exp)
    os.makedirs(logs_dir, exist_ok=True) 
    pfx = f"TEST-{args.i}-"
    config_log(logs_dir,pfx=pfx)
    test_agent = HIGH_LEVEL_DQN_TEST(args)
    test_agent.test_cluster(best_model_path, final_result_path)
    return 1


    

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    logs_dir = os.path.join("./logs/high_level", '{}'.format(args.dataset), args.exp)
    os.makedirs(logs_dir, exist_ok=True) 
    pfx = f""
    config_log(logs_dir,pfx=pfx)
    agent = DQN(args)
    agent.train() 
 
    num_processes = 3
    args_list = [
            
    ]
    for i in range(args.epoch_number):
        args_copy = copy.deepcopy(args)
        args_copy.i = i+1
        args_list.append(args_copy)
    
    with multiprocessing.Pool(processes=num_processes) as pool:
         
            results = pool.imap_unordered(_validate_test_worker, args_list)
            for result in results:
                print(result)
                pass
            
    
         
