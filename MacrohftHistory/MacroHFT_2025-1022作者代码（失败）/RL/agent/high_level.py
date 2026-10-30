# 添加运行时路径修正，确保可以找到同级项目内的 model、env 等目录
import sys as _sys_fix
import pathlib as _pl_fix
_PROJECT_ROOT = _pl_fix.Path(__file__).resolve().parents[2]  # RL/agent/ -> RL -> 项目根目录
_project_root_str = str(_PROJECT_ROOT)
if _project_root_str not in _sys_fix.path:
    _sys_fix.path.insert(0, _project_root_str)
# 兼容旧代码使用的 ROOT 变量
ROOT = _project_root_str

from model.net import subagent, hyperagent
from env.high_level_env import Testing_Env, Training_Env
from RL.util.utili import LinearDecaySchedule
from RL.util.replay_buffer import ReplayBuffer_High
from RL.util.memory import episodicmemory
from RL.util.eval_tools import calculate_trading_metrics
from RL.util.graph_utils import plot_money_curve
import pathlib
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from logging import StreamHandler, FileHandler, Formatter
import logging as log
import time
import pandas as pd  # 读取 feather 使用

# ROOT = str(pathlib.Path(__file__).resolve().parents[3])



os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--buffer_size",type=int,default=1000000)  # 经验缓冲区大小 / Replay buffer capacity
parser.add_argument("--dataset",type=str,default="ETHUSDT")  # 数据集名称 / Dataset name
parser.add_argument("--q_value_memorize_freq",type=int, default=20)  # Q值记忆频率 / Q-value logging frequency
parser.add_argument("--batch_size",type=int,default=512)  # 批次大小 / Mini-batch size
parser.add_argument("--eval_update_freq",type=int,default=512)  # 网络更新频率 / Network update frequency
parser.add_argument("--lr", type=float, default=1e-4)  # 学习率 / Learning rate
parser.add_argument("--epsilon_start",type=float,default=0.7)  # 初始探索率 / Initial exploration rate
parser.add_argument("--epsilon_end",type=float,default=0.3)  # 最小探索率 / Minimum exploration rate
parser.add_argument("--decay_length",type=int,default=5)  # 探索衰减周期 / Exploration decay length
parser.add_argument("--update_times",type=int,default=10)  # 单步更新次数 / Update times per step
parser.add_argument("--gamma", type=float, default=0.99)  # 折扣因子 / Discount factor
parser.add_argument("--tau", type=float, default=0.005)  # 软更新系数 / Soft update coefficient
parser.add_argument("--transcation_cost",type=float,default=2.0 / 100000)  # 交易成本（注意拼写） / Transaction cost (typo preserved)
parser.add_argument("--back_time_length",type=int,default=1)  # 历史窗口长度 / Historical window length
parser.add_argument("--seed",type=int,default=12345)  # 随机种子 / Random seed
parser.add_argument("--n_step",type=int,default=1)  # n-step TD目标 / N-step TD target
parser.add_argument("--epoch_number",type=int,default=20)  # 训练轮次数 / Training epochs
parser.add_argument("--alpha",type=float,default=0.5)  # KL损失权重系数 / KL loss weight coefficient #alpha 代表了记忆的经验权重， beta代表先验q-table权重
parser.add_argument("--device",type=str,default="cpu")  # 计算设备 / Computation device cuda:0
parser.add_argument("--beta",type=int,default=5) #alpha 代表了记忆的经验权重， beta代表先验q-table权重
parser.add_argument("--no_risk_return",type=float,default=4.5) #无风险返回率
parser.add_argument("--exp",type=str,default="exp1")
parser.add_argument("--num_step",type=int,default=10)


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class DQN(object):
    def __init__(self, args):  # 定义DQN的一系列属性
        self.seed = args.seed
        seed_torch(self.seed)
        if torch.cuda.is_available():
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cpu")
            

        self.epsilon_device = torch.device("cpu") 
        self.logs_dir = os.path.join("./logs/high_level", '{}'.format(args.dataset), args.exp)
        os.makedirs(self.logs_dir, exist_ok=True) 
        
        config_log(self.logs_dir,pfx='')
        log.info(args)    
            
            
        self.result_path = os.path.join("./result/high_level", '{}'.format(args.dataset), args.exp)
        self.model_path = os.path.join(self.result_path, "seed_{}".format(self.seed))
        self.train_data_path = os.path.join(ROOT, "data", args.dataset, "whole")
        self.val_data_path = os.path.join(ROOT, "data", args.dataset, "whole")
        self.test_data_path = os.path.join(ROOT, "data", args.dataset, "whole")
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

        self.transcation_cost = args.transcation_cost
        self.back_time_length = args.back_time_length
        self.n_action = 2
        self.n_state_1 = len(self.tech_indicator_list)
        self.n_state_2 = len(self.tech_indicator_list_trend)
        # 初始化低层子代理到与训练一致的设备，避免推理阶段设备不一致
        self.slope_1 = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.slope_2 = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.slope_3 = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.vol_1 = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.vol_2 = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.vol_3 = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        model_list_slope = [
            "./result/low_level/ETHUSDT/best_modelepp30/slope/1/best_model.pkl", 
            "./result/low_level/ETHUSDT/best_modelepp30/slope/2/best_model.pkl",
            "./result/low_level/ETHUSDT/best_modelepp30/slope/3/best_model.pkl"
        ]
        model_list_vol = [
            "./result/low_level/ETHUSDT/best_modelepp30/vol/1/best_model.pkl",
            "./result/low_level/ETHUSDT/best_modelepp30/vol/2/best_model.pkl",
            "./result/low_level/ETHUSDT/best_modelepp30/vol/3/best_model.pkl"
        ]
        
        print("[INFO] model_list_slope:")
        for p in model_list_slope:
            print(p)
        print("[INFO] model_list_vol:")
        for p in model_list_vol:
            print(p)
        
        self.slope_1.load_state_dict(torch.load(model_list_slope[0], map_location=self.device))
        self.slope_2.load_state_dict(torch.load(model_list_slope[1], map_location=self.device))
        self.slope_3.load_state_dict(torch.load(model_list_slope[2], map_location=self.device))
        self.vol_1.load_state_dict(torch.load(model_list_vol[0], map_location=self.device))
        self.vol_2.load_state_dict(torch.load(model_list_vol[1], map_location=self.device))
        self.vol_3.load_state_dict(torch.load(model_list_vol[2], map_location=self.device))
        # 分行写 eval() 避免多语句同行
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
        self.epsilon_hyperagent = hyperagent(self.n_state_1, self.n_state_2, self.n_action, 32).to(self.epsilon_device)
        self.hyperagent = hyperagent(self.n_state_1, self.n_state_2, self.n_action, 32).to(self.device)
        self.hyperagent_target = hyperagent(self.n_state_1, self.n_state_2, self.n_action, 32).to(self.device)
        self.hyperagent_target.load_state_dict(self.hyperagent.state_dict())
        self.epsilon_hyperagent.load_state_dict(self.hyperagent.state_dict())
        self.epsilon_hyperagent.eval()
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
        self.memory = episodicmemory(4320, 5, self.n_state_1, self.n_state_2, 64, self.epsilon_device)
        self.no_risk_return = args.no_risk_return


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
        batch, _, _ = replay_buffer.sample()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        # Calculate current and target hypernetwork outputs
        # 计算当前和目标超网络输出
        w_current = self.hyperagent(batch['state'], batch['state_trend'], batch['state_clf'], batch['previous_action'])
        w_next = self.hyperagent_target(batch['next_state'], batch['next_state_trend'], batch['next_state_clf'], batch['next_previous_action'])
        w_next_ = self.hyperagent(batch['next_state'], batch['next_state_trend'], batch['next_state_clf'], batch['next_previous_action'])

        # Compute Q-values from slope/volatility agents
        # 计算斜率/波动率代理的Q值
        qs_current = [
                    self.slope_agents[0](batch['state'], batch['state_trend'], batch['previous_action']),
                    self.slope_agents[1](batch['state'], batch['state_trend'], batch['previous_action']),
                    self.slope_agents[2](batch['state'], batch['state_trend'], batch['previous_action']),
                    self.vol_agents[0](batch['state'], batch['state_trend'], batch['previous_action']),
                    self.vol_agents[1](batch['state'], batch['state_trend'], batch['previous_action']),
                    self.vol_agents[2](batch['state'], batch['state_trend'], batch['previous_action'])
        ]
        qs_next = [
                    self.slope_agents[0](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                    self.slope_agents[1](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                    self.slope_agents[2](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                    self.vol_agents[0](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                    self.vol_agents[1](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                    self.vol_agents[2](batch['next_state'], batch['next_state_trend'], batch['next_previous_action'])
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

        torch.nn.utils.clip_grad_norm_(self.hyperagent.parameters(), 1)
        self.optimizer.step()
        for param, target_param in zip(self.hyperagent.parameters(), self.hyperagent_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        self.update_counter += 1
        return td_error.cpu(), memory_error.cpu(), KL_loss.cpu(), torch.mean(q_current.cpu()), torch.mean(q_target.cpu())

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
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device)
        previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long(), 0).to(self.device)
        
        # Epsilon-greedy action selection
        # epsilon-greedy策略选择动作
        if np.random.uniform() < (1 - self.epsilon):
            # Get Q-values from slope/volatility agents
            # 获取斜率/波动率代理的Q值
            qs = [
                    self.slope_agents[0](x1, x2, previous_action),
                    self.slope_agents[1](x1, x2, previous_action),
                    self.slope_agents[2](x1, x2, previous_action),
                    self.vol_agents[0](x1, x2, previous_action),
                    self.vol_agents[1](x1, x2, previous_action),
                    self.vol_agents[2](x1, x2, previous_action)
            ]
            # Calculate hypernetwork output
            # 计算超网络输出 6个子代理的权重
            w = self.hyperagent(x1, x2, x3, previous_action)
            # Combine Q-values using hypernetwork weights
            # 使用超网络权重组合Q值
            actions_value = self.calculate_q(w, qs)
            # Select action with max Q-value
            # 选择最大Q值的动作 (根据动作q值，选择动作， max()[1]选择的数组下标,max()[0] q值)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
        else:
            action = random.choice([0, 1])
        return action

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
            previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long(), 0).to(self.device)
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
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
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
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device)
        previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long(), 0).to(self.device)
        
        # Get Q-values from slope/volatility agents
        # 获取斜率/波动率代理的Q值
        
        qs = [
                self.slope_agents[0](x1, x2, previous_action),
                self.slope_agents[1](x1, x2, previous_action),
                self.slope_agents[2](x1, x2, previous_action),
                self.vol_agents[0](x1, x2, previous_action),
                self.vol_agents[1](x1, x2, previous_action),
                self.vol_agents[2](x1, x2, previous_action)
        ]
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
        previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long(), 0).to(self.device)
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
        
        
        train_env = Training_Env(
                "full",
                df=self.df,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                clf_list=self.clf_list,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                # n_action = self.n_action,
                initial_action=random.choices(range(self.n_action), k=1)[0],
                alpha = 0)
        single_state, trend_state, clf_state, info = train_env.reset()
        episode_reward_sum = 0
        log.info("train_env reset done")
        while True:
            # 使用ε-greedy策略选择动作
            # Select action using ε-greedy strategy
            action = self.select_action(single_state, trend_state, clf_state, info)
            # 执行环境步进操作
            # Execute environment step
            next_single_state, next_trend_state, next_clf_state, reward, done, next_info = train_env.step(action)
            # 获取历史动作信息
            # Get historical action information
            previous_action = info['previous_action']
            demo_action = info['q_value']
            next_previous_action = next_info['previous_action']
            next_demo_action = next_info['q_value']
            
            
            # 计算隐藏状态
            # Calculate hidden state
            hs = self.calculate_hidden(single_state, trend_state, info)
            # 查询记忆库中的Q值
            # Query Q-value from memory
            q_memory = self.memory.query(hs, action)
            # 计算目标Q值
            # Calculate target Q-value
            q = reward + self.gamma * (1 - done) * self.q_estimate(next_single_state, next_trend_state, next_clf_state, next_info)
            
            if np.isnan(q_memory):
                q_memory = q
            # 更新记忆库
            # Update memory            
            self.memory.add(hs, action, q, single_state, trend_state, previous_action)

            # 存储经验到回放缓冲区
            # Store transition in replay buffer
            self.replay_buffer.store_transition(single_state, trend_state, clf_state, previous_action, demo_action, action, reward, next_single_state, next_trend_state, next_clf_state, next_previous_action, next_demo_action, done, q_memory)


            episode_reward_sum += reward

            single_state, trend_state, clf_state, info = next_single_state, next_trend_state, next_clf_state, next_info
            step_counter += 1
            # 定期执行模型更新
            # Periodically update model parameters
            if step_counter % self.eval_update_freq == 0 and step_counter > (self.batch_size + self.n_step):
                # log.info(f"update network {step_counter}")
                for i in range(self.update_times):
                    td_error, memory_error, KL_loss, q_eval, q_target = self.update(self.replay_buffer)
                    if self.update_counter % self.q_value_memorize_freq == 1:
                        self.writer.add_scalar(tag="td_error", scalar_value=td_error, global_step=self.update_counter, walltime=None)
                        self.writer.add_scalar(tag="memory_error", scalar_value=memory_error, global_step=self.update_counter, walltime=None)
                        self.writer.add_scalar(tag="KL_loss", scalar_value=KL_loss, global_step=self.update_counter, walltime=None)
                        self.writer.add_scalar(tag="q_eval", scalar_value=q_eval, global_step=self.update_counter, walltime=None)
                        self.writer.add_scalar(tag="q_target", scalar_value=q_target, global_step=self.update_counter, walltime=None)
                self.epsilon_hyperagent.load_state_dict(self.hyperagent.state_dict())
                self.epsilon_hyperagent.to(self.epsilon_device)
                if step_counter > 4320:
                    # 定期重新编码记忆, 因为超代理的隐藏层训练后发生了变化，
                    # Periodically re-encode memory
                    self.memory.re_encode(self.epsilon_hyperagent)
                return_margin, final_balance, required_money, commission_fee = train_env.get_final_return_rate()                
                log.info(f"update network {step_counter} return_margin:{return_margin:.2f},final_balance:{final_balance:.2f},required_money:{required_money:.2f},commission_fee:{commission_fee:.2f}".replace("\n", " "))
            
            if done:
                self.epsilon_hyperagent.load_state_dict(self.hyperagent.state_dict())
                self.epsilon_hyperagent.to(self.epsilon_device)
                log.info("train_env step done")
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
        step_counter = 0
        episode_counter = 0
        epoch_counter = 0
        best_return_rate = -float('inf')
        best_model = None
        
        self.df = pd.read_feather(os.path.join(self.train_data_path, "train.feather") ) 
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
            return_rate_eval = self.val_cluster(epoch_path, val_path)
            # 更新最佳模型
            # Update best model if improved
            if return_rate_eval > best_return_rate:
                best_return_rate = return_rate_eval
                best_model = self.hyperagent.state_dict()
                best_model_path = os.path.join(self.result_path, 'best_model.pkl')
                torch.save(best_model, best_model_path)
                # 保存最佳模型到文件
                # Save best model to disk

        
        # 执行最终测试评估
        # Execute final test evaluation
        final_result_path = self.result_path
        # self.test_cluster('/home/ustc1958/lxy/1/MacroHFT_2025-20251022/result/high_level/ETHUSDT/exp2epoch30/best_model.pkl', final_result_path)
        self.test_cluster(best_model_path, final_result_path)
        

    def val_cluster(self, epoch_path, save_path):
        self.hyperagent.load_state_dict(
            torch.load(os.path.join(epoch_path, "trained_model.pkl")))
        self.hyperagent.eval()
        action_list = []
        reward_list = []
        final_balance_list = []
        required_money_list = []
        commission_fee_list = []
        self.df = pd.read_feather(os.path.join(self.val_data_path, "val.feather"))
        
        val_env = Testing_Env(
                df=self.df,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                clf_list=self.clf_list,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                initial_action=0,
                n_action = self.n_action)
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
          
        log.info(f"val return_margin:{return_margin:.2f},final_balance:{final_balance:.2f},required_money:{required_money:.2f},commission_fee:{commission_fee:.2f}")
     
        
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

    def test_cluster(self, epoch_path, save_path):
        self.hyperagent.load_state_dict(torch.load(epoch_path))
        self.hyperagent.eval()
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
                n_action = self.n_action)
        s, s2, s3, info = test_env.reset()
        done = False
        action_list_episode = []
        reward_list_episode = []
        while not done:
            a = self.act_test(s, s2, s3, info)
            s_, s2_, s3_, r, done, info_ = test_env.step(a)
            reward_list_episode.append(r)
            s, s2, s3, info = s_, s2_, s3_, info_
            action_list_episode.append(a)
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
    print(args)
    agent = DQN(args)
    agent.train()
    final_result_path = os.path.join("./result/high_level", '{}'.format(agent.dataset),'{}'.format(args.exp))
    best_model_path = os.path.join("./result/high_level", '{}'.format(agent.dataset), 'best_model.pkl')
    agent.test_cluster(best_model_path, final_result_path)
    # agent.test_cluster('/home/ustc1958/lxy/1/MacroHFT_2025-20251022/result/high_level/ETHUSDT/exp2epoch30/best_model.pkl', final_result_path)
