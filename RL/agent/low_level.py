# 1.核心组件初始化
# * 超参数配置：通过argparse定义交易成本(0.0002)、学习率(1e-4)、探索率(0.5→0.1)、批处理大小(512)等
# * 网络架构：使用subagent作为Q网络，包含状态特征提取和Q值预测
# * 经验回放缓冲区：存储(s, a, r, s')转移样本
# * 优化器：Adam优化器进行参数更新
# * 数据加载：根据交易对(如ETHUSDT)加载预处理的特征数据和标签
# 2.训练流程 (train方法)
# python
# for epoch in epochs:
#     打乱训练数据顺序
#     生成随机初始持仓
#     for 每个训练数据文件:
#         与环境交互收集经验:
#             while not done:
#                 ε-greedy选择动作
#                 执行动作获得转移样本
#                 存储到经验回放缓冲区
#                 累计奖励
#                 定期更新网络参数
#     保存训练模型
#     在验证集上评估:
#         计算平均收益率
#         更新最佳模型
# 3.网络更新机制 (update方法)
# 双网络结构：使用eval_net和target_net分离Q值估计与目标计算
# 复合损失函数：TD误差 + KL散度损失 (α=0控制权重)
# 软更新(target_net): τ=0.005的指数平滑更新
# 梯度处理：梯度裁剪(阈值1)防止爆炸
# 4.动作选择策略
# 训练时：ε-greedy策略 (初始ε=0.5，线性衰减至0.1)
# 测试时：确定性策略 (直接选择最大Q值动作)
# 5.特征处理
# 状态表示：技术指标(single_state) + 趋势特征(trend_state)
# 历史动作：作为网络输入的一部分
# 6.验证流程 (val_cluster方法)
# 加载训练模型
# 在验证集上执行完整交易测试
# 收集动作序列、奖励、账户余额等指标
# 保存结果用于分析
# 7.结果保存
# 模型保存：每个epoch保存一次
# 最佳模型：根据验证集收益率保存最优参数
# 验证结果：保存动作序列、奖励、收益等指标为npy文件
# 8.关键超参数
# buffer_size=1M    # 经验回放缓冲区容量
# batch_size=512    # 训练批大小
# gamma=0.99        # 折扣因子
# tau=0.005         # target网络更新系数
# alpha=0           # KL损失权重(未启用)
# lr=1e-4           # 学习率
# 该实现结合了深度强化学习与高频交易场景，通过双网络架构(DQN)和经验回放机制来学习最优交易策略，
# 使用KL散度作为正则化项(可通过α参数控制)。训练过程中通过TensorBoard记录关键指标，并保存最佳模型用于实际交易。
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

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--buffer_size",type=int,default=1000000)  # 经验缓冲区大小 / Replay buffer capacity
parser.add_argument("--dataset",type=str,default="ETHUSDT")  # 数据集名称 / Dataset name
parser.add_argument("--q_value_memorize_freq",type=int, default=100)  # Q值记忆频率 / Q-value logging frequency
parser.add_argument("--batch_size",type=int,default=1024)  # 批次大小 / Mini-batch size
parser.add_argument("--eval_update_freq",type=int,default=50)  # 网络更新频率 / Network update frequency
parser.add_argument("--lr", type=float, default=1e-6)  # 学习率 / Learning rate
parser.add_argument("--epsilon_start",type=float,default=0.7)  # 初始探索率 / Initial exploration rate
parser.add_argument("--epsilon_end",type=float,default=0.2)  # 最小探索率 / Minimum exploration rate
parser.add_argument("--decay_length",type=int,default=5)  # 探索衰减周期 / Exploration decay length
parser.add_argument("--update_times",type=int,default=30)  # 单步更新次数 / Update times per step
parser.add_argument("--gamma", type=float, default=0.999)  # 折扣因子 / Discount factor
parser.add_argument("--tau", type=float, default=0.005)  # 软更新系数 / Soft update coefficient
parser.add_argument("--transcation_cost",type=float,default=2.0 / 10000)  # 交易成本（注意拼写） / Transaction cost (typo preserved)
parser.add_argument("--back_time_length",type=int,default=1)  # 历史窗口长度 / Historical window length
parser.add_argument("--seed",type=int,default=12345)  # 随机种子 / Random seed
parser.add_argument("--n_step",type=int,default=1)  # n-step TD目标 / N-step TD target
parser.add_argument("--epoch_number",type=int,default=20)  # 训练轮次数 / Training epochs
parser.add_argument("--label",type=str,default="label_1")  # 标签列名称 / Label column name
parser.add_argument("--clf",type=str,default="slope")  # 分类器类型 / Classifier type
parser.add_argument("--alpha",type=float,default=4)  # KL损失权重系数 / KL loss weight coefficient
parser.add_argument("--exp",type=str,default="exp4")
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

def calculate_alpha(diff, k):
    alpha = 16 * (1 - torch.exp(-k * diff))
    return torch.clip(alpha, 0, 16)


class DQN(object):
    def __init__(self, args):  # 定义DQN的一系列属性
        log.info('==slope==')
        self.seed = args.seed
        seed_torch(self.seed)
        if torch.cuda.is_available():
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cpu") 
        self.epsilon_device = torch.device("cpu") 
        log.info(self.device)
        self.result_path = os.path.join("./result/low_level", '{}'.format(args.dataset), args.exp, '{}'.format(args.clf), args.label, str(int(args.alpha)))
        self.label = int(args.label.split('_')[1])
        
        
        self.logs_dir = os.path.join("./logs/low_level", '{}'.format(args.dataset), args.exp, '{}'.format(args.clf), args.label, str(int(args.alpha)))
        os.makedirs(self.logs_dir, exist_ok=True) 
        
        config_log(self.logs_dir,pfx='train-')
        log.info(args)
        
        
        self.model_path = os.path.join(self.result_path, "seed_{}".format(self.seed))
        self.train_data_path = os.path.join(ROOT, "MacroHFT", "data", args.dataset, "train")
        log.info(f'train_data_path:{self.train_data_path}')
        self.val_data_path = os.path.join(ROOT, "MacroHFT", "data", args.dataset, "val")
        self.test_data_path = os.path.join(ROOT, "MacroHFT", "data", args.dataset, "test")
        if args.clf == 'slope':
            log.info('==slope==')
            with open(os.path.join(self.train_data_path, 'slope_labels.pkl'), 'rb') as file:
                self.train_index = pickle.load(file)
                log.info(f"self.train_index:{self.train_index}")
            with open(os.path.join(self.val_data_path, 'slope_labels.pkl'), 'rb') as file:
                self.val_index = pickle.load(file)
            with open(os.path.join(self.test_data_path, 'slope_labels.pkl'), 'rb') as file:
                self.test_index = pickle.load(file)
        elif args.clf == 'vol':
            with open(os.path.join(self.train_data_path, 'vol_labels.pkl'), 'rb') as file:
                self.train_index = pickle.load(file)
            with open(os.path.join(self.val_data_path, 'vol_labels.pkl'), 'rb') as file:
                self.val_index = pickle.load(file)
            with open(os.path.join(self.test_data_path, 'vol_labels.pkl'), 'rb') as file:
                self.test_index = pickle.load(file)


        self.dataset=args.dataset
        self.clf = args.clf
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

        self.transcation_cost = args.transcation_cost
        self.back_time_length = args.back_time_length
        self.n_action = 2
        self.n_state_1 = len(self.tech_indicator_list)
        self.n_state_2 = len(self.tech_indicator_list_trend)
        self.epsilon_net = subagent(self.n_state_1, self.n_state_2, self.n_action, 128).to(self.epsilon_device)
        self.eval_net = subagent(self.n_state_1, self.n_state_2, self.n_action, 128).to(self.device)
        self.target_net =subagent(self.n_state_1, self.n_state_2, self.n_action, 128).to(self.device)
        self.hardupdate()
        self.update_times = args.update_times
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=args.lr)
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
        self.alpha = args.alpha
        
    def update(self, replay_buffer):
        """
        更新评估网络参数，包含TD误差和KL散度的联合优化
        
        中文说明：
        本函数执行网络参数更新，包含以下步骤：
        1. 从经验回放缓冲区采样数据
        2. 计算目标Q值和当前Q值
        3. 计算TD误差损失和KL散度损失
        4. 执行梯度下降和参数更新
        5. 软更新目标网络参数
        
        English description:
        This function performs network parameter update with the following steps:
        1. Sample data from replay buffer
        2. Calculate target Q-values and current Q-values
        3. Compute TD error loss and KL divergence loss
        4. Execute gradient descent and parameter update
        5. Soft update target network parameters
        
        Parameters:
            replay_buffer (ReplayBuffer): 经验回放缓冲区 / Replay buffer containing transitions
            
        Returns:
            tuple: (td_error, KL_loss, q_current_mean, q_target_mean)
                  td_error (float): TD误差损失值
                  KL_loss (float): KL散度损失值
                  q_current_mean (float): 当前Q值均值
                  q_target_mean (float): 目标Q值均值
        """
        # start_time = time.time()  # Start timing            
        self.eval_net.train()
        batch, _, _ = replay_buffer.sample()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # 使用评估网络选择最优动作 / Select optimal action using eval net
        a_argmax = self.eval_net(batch['next_state'], batch['next_state_trend'], batch['next_previous_action']).argmax(dim=-1, keepdim=True)
        # 使用目标网络计算Q值 / Calculate Q-values with target net
        q_next = self.target_net(batch['next_state'], batch['next_state_trend'], batch['next_previous_action']).gather(-1, a_argmax).squeeze(-1)
        # 计算目标Q值 / Compute target Q-values
        q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * q_next
        
        # 获取当前状态的Q分布 / Get Q-distribution for current state
        q_distribution = self.eval_net(batch['state'], batch['state_trend'], batch['previous_action'])
        # 获取当前动作的Q值 / Get Q-values for selected actions
        q_current = q_distribution.gather(-1, batch['action']).squeeze(-1)
        # 计算TD误差损失 / Compute TD error loss
        td_error = self.loss_func(q_current, q_target)
        # 获取专家示范动作 / Get expert demonstration actions
        demonstration = batch['demo_action']
        # 计算KL散度损失 / Compute KL divergence loss
        KL_loss = F.kl_div(
            (q_distribution.softmax(dim=-1) + 1e-8).log(),
            (demonstration.softmax(dim=-1) + 1e-8),
            reduction="batchmean",
        )
        # 结合两种损失 / Combine two losses
        alpha = args.alpha # 权重系数 / Weight coefficient
        loss = td_error + alpha * KL_loss
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), 1)
        self.optimizer.step()
        for param, target_param in zip(self.eval_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.update_counter += 1
        self.eval_net.eval() 
        
        # end_time = time.time()  # End timing
        # duration = end_time - start_time
        # log.info(f"end training with df update {end_time}  {duration:.4f} seconds" )
        
        
        # 返回损失值供记录 / Return loss values for logging
        return loss, td_error, KL_loss, torch.mean(q_current), torch.mean(q_target)

    def hardupdate(self):
        self.epsilon_net.eval() 
        self.epsilon_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.epsilon_net.to(self.epsilon_device)

    def select_action(self, state, state_trend, info):
        """
        基于epsilon-greedy策略选择动作
        
        中文说明：
        使用epsilon-greedy策略选择动作：大部分时间选择最优动作，
        以epsilon概率随机探索。适用于训练阶段。
        
        English description:
        Selects action using epsilon-greedy strategy: selects optimal action most of the time,
        with epsilon probability for random exploration. Suitable for training phase.
        
        Parameters:
            state (np.array): 当前状态 / Current state
            state_trend (np.array): 当前趋势状态 / Current trend state
            info (dict): 包含历史信息的字典 / Dictionary containing historical information
            
        Returns:
            int: 选择的动作编号 / Selected action number
        """
        x1 = torch.FloatTensor(state).to(self.epsilon_device)
        x2 = torch.FloatTensor(state_trend).to(self.epsilon_device)
        previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long().to(self.epsilon_device), 0).to(self.epsilon_device)
        if np.random.uniform() < (1-self.epsilon):
            actions_value = self.epsilon_net(x1, x2, previous_action)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
        else:
            action_choice = list(range(int(self.n_action))) 
            action = random.choice(action_choice)
        return action


    def _train_data_file(self,
                        df_path,
                        df_dataset, 
                        i,
                        episode_counter, 
                        step_counter, 
                        df_index, 
                        random_position_list,
                        epoch_return_rate_train_list,
                        epoch_final_balance_train_list,
                        epoch_required_money_train_list,
                        epoch_reward_sum_train_list  
                    ):
        """
        使用指定数据文件进行训练，执行完整的训练周期
        
        中文说明：
        本函数负责加载指定索引的数据文件，初始化训练环境，
        执行强化学习的交互循环，并更新神经网络参数。
        同时记录训练过程中的关键指标用于可视化分析。
        
        English description:
        This function loads the data file with specified index, initializes training environment,
        executes reinforcement learning interaction loop, and updates neural network parameters.
        It also records key metrics for visualization analysis during training.
        
        Parameters:
            episode_counter (int): 当前训练周期计数器 / Current episode counter
            step_counter (int): 全局步数计数器 / Global step counter
            df_index (int): 数据文件索引 / Data file index
            random_position_list (list): 随机初始持仓列表 / Random initial positions list
            epoch_return_rate_train_list (list): 存储周期收益率 / Storage for epoch return rates
            epoch_final_balance_train_list (list): 存储周期最终余额 / Storage for final balances
            epoch_required_money_train_list (list): 存储周期所需资金 / Storage for required money
            epoch_reward_sum_train_list (list): 存储周期总奖励 / Storage for total rewards
            
        Returns:
            tuple: 更新后的episode计数器和step计数器
        """
                # 加载数据文件 / Load data file
        # self.df = pd.read_feather(os.path.join(self.train_data_path, "df_{}.feather".format(df_index)))
        self.df = df_dataset
        start_time = time.time()  # Start timing       
        data_file_size = len(self.df)          
        log.info(f"training with df {df_index} {data_file_size} {start_time}" )

        # 设置评估网络为验证模式 / Set evaluation network to validation mode
        self.eval_net.eval()
                        
        # 初始化训练环境 / Initialize training environment
        train_env = Training_Env(
                df_path=df_path,
                df=self.df,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                num_action=self.n_action,                
                initial_action=0 )
        # 重置环境获取初始状态 / Reset environment to get initial state
        single_state, trend_state, info = train_env.reset()
        episode_reward_sum = 0
        # 开始交互循环 / Start interaction loop
        while True:
            action = self.select_action(single_state, trend_state, info)
            next_single_state, next_trend_state, reward, done, next_info = train_env.step(action)

            previous_action = info['previous_action']
            demo_action = info['q_value']
            next_previous_action = next_info['previous_action']
            next_demo_action = next_info['q_value']
            # 存储经验回放 / Store transition in replay buffer
            self.replay_buffer.store_transition(single_state, trend_state, previous_action, demo_action, action, reward, 
                                                next_single_state, next_trend_state, next_previous_action, next_demo_action,
                                                done)
            # 累计奖励 / Accumulate reward
            episode_reward_sum += reward
            # 更新状态 / Update states
            single_state, trend_state, info = next_single_state, next_trend_state, next_info
            step_counter += 1
            
            # 定期更新网络参数 / Periodically update network parameters
            if step_counter % self.eval_update_freq == 0 and step_counter > (self.batch_size + self.n_step):
                for i in range(self.update_times):
                    loss, td_error, KL_loss, q_eval, q_target = self.update(self.replay_buffer)
                    
                    # 定期记录日志 / Periodically log metrics
                    if self.update_counter % self.q_value_memorize_freq == 1:
                        self.writer.add_scalar(tag="loss", scalar_value=loss.cpu(), global_step=self.update_counter, walltime=None)
                        self.writer.add_scalar(tag="td_error", scalar_value=td_error.cpu(), global_step=self.update_counter, walltime=None)
                        self.writer.add_scalar(tag="KL_loss", scalar_value=KL_loss.cpu(), global_step=self.update_counter, walltime=None)
                        self.writer.add_scalar(tag="q_eval", scalar_value=q_eval.cpu(), global_step=self.update_counter, walltime=None)
                        self.writer.add_scalar(tag="q_target", scalar_value=q_target.cpu(), global_step=self.update_counter, walltime=None)
                self.epsilon_net.load_state_dict(self.eval_net.state_dict())
                self.epsilon_net.to(self.epsilon_device)
            # 判断回合结束 / Check if episode is done
            if done:
                
                break
        self.epsilon_net.load_state_dict(self.eval_net.state_dict())
        self.epsilon_net.to(self.epsilon_device)
        # 更新周期计数器 / Update episode counter
        episode_counter += 1
        # 获取最终账户信息 / Get final account information
        final_balance, required_money = train_env.final_balance, train_env.required_money
        # 记录训练指标 / Record training metrics
        self.writer.add_scalar(tag="return_rate_train", scalar_value=final_balance / (required_money), global_step=episode_counter, walltime=None)
        self.writer.add_scalar(tag="final_balance_train", scalar_value=final_balance, global_step=episode_counter, walltime=None)
        self.writer.add_scalar(tag="required_money_train", scalar_value=required_money, global_step=episode_counter, walltime=None)
        self.writer.add_scalar(tag="reward_sum_train", scalar_value=episode_reward_sum, global_step=episode_counter, walltime=None)
        # 存储周期统计信息 / Store epoch statistics
        epoch_return_rate_train_list.append(final_balance / (required_money))
        epoch_final_balance_train_list.append(final_balance)
        epoch_required_money_train_list.append(required_money)
        epoch_reward_sum_train_list.append(episode_reward_sum)
         
        end_time = time.time()  # End timing
        duration = end_time - start_time
        log.info(f"end training with df {df_index} {end_time}  {duration:.2f} seconds" )
        
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
        # 计算平均训练指标 / Calculate average training metrics
        mean_return_rate_train = np.mean(epoch_return_rate_train_list)
        mean_final_balance_train = np.mean(epoch_final_balance_train_list)
        mean_required_money_train = np.mean(epoch_required_money_train_list)
        mean_reward_sum_train = np.mean(epoch_reward_sum_train_list)
        self.writer.add_scalar(tag="epoch_return_rate_train", scalar_value=mean_return_rate_train, global_step=epoch_counter, walltime=None)
        self.writer.add_scalar(tag="epoch_final_balance_train", scalar_value=mean_final_balance_train, global_step=epoch_counter, walltime=None)
        self.writer.add_scalar(tag="epoch_required_money_train", scalar_value=mean_required_money_train, global_step=epoch_counter, walltime=None)
        self.writer.add_scalar(tag="epoch_reward_sum_train", scalar_value=mean_reward_sum_train, global_step=epoch_counter, walltime=None)
        torch.save(self.eval_net.state_dict(), os.path.join(epoch_path, "trained_model.pkl"))
                        
    def train(self):        
        """
        执行完整的训练流程，包含多周期训练、模型保存和验证
        
        中文说明：
        本函数负责初始化训练参数，执行多个训练周期，
        在每个周期中打乱数据并进行训练，保存训练模型，
        通过验证集评估模型性能并保留最佳模型。
        
        English description:
        This function initializes training parameters, executes multiple training epochs,
        shuffles data and performs training in each epoch, saves the trained models,
        evaluates model performance on validation set and keeps the best model.
        
        Parameters:
            None (所有参数通过self对象属性获取 / All parameters are obtained via self object attributes)
            
        Returns:
            None
        """
        # 获取训练数据索引列表 / Get training data index list
        
        df_list = self.train_index[self.label]
        df_number=int(len(df_list))       
        step_counter = 0  # 全局步数计数器 / Global step counter
        episode_counter = 0  # 训练周期计数器 / Episode counter
        epoch_counter = 0  # 总训练轮次计数器 / Epoch counter    
         
        # 初始化经验回放缓冲区 / Initialize replay buffer 
        self.replay_buffer = ReplayBuffer(args, self.n_state_1, self.n_state_2, self.n_action)   
        best_return_rate = -float('inf')    # 最佳收益率记录 / Best return rate record
        best_model = None                   # 最佳模型参数存储 / Best model parameters storage
        
        #缓存训练数据，防止重复加载
        df_datasets_dict = {}
        # 开始训练周期循环 / Start training epochs loop
        for sample in range(self.epoch_number):
            # 初始化周期统计列表 / Initialize epoch statistics lists
            epoch_return_rate_train_list = []
            epoch_final_balance_train_list = []
            epoch_required_money_train_list = []
            epoch_reward_sum_train_list = []
            log.info(f'epoch {epoch_counter + 1}')
            
            # 复制并打乱数据索引 / Copy and shuffle data indexes
            random_list = self.train_index[self.label]
            random.shuffle(random_list)
            
            # 生成随机初始持仓 / Generate random initial positions
            random_position_list = random.choices(range(self.n_action), k=df_number)
            log.info(f"random_list:{random_list}")
            
            
            for i in range(df_number):
                df_index = random_list[i]
                if df_datasets_dict.get(df_index,None) is None:
                    df_data = pd.read_feather(os.path.join(self.train_data_path, "df_{}.feather".format(df_index)))
                    # .head(100)
                    df_datasets_dict[df_index] = df_data
            # 遍历所有数据文件进行训练 / Train with all data files
            log.info(f"epoch {epoch_counter + 1} start files")
            for i in range(df_number):
                df_index = random_list[i]
                # 单个数据文件训练过程 / Training with single data file
                df_dataset= df_datasets_dict[df_index]
                df_dataset_path = os.path.join(self.train_data_path, "df_{}.feather".format(df_index))
                episode_counter, step_counter = self._train_data_file(
                                                                        df_dataset_path, df_dataset, 
                                                                        i, episode_counter, step_counter, df_index, random_position_list, 
                                                                        epoch_return_rate_train_list,
                                                                        epoch_final_balance_train_list,
                                                                        epoch_required_money_train_list,
                                                                        epoch_reward_sum_train_list)
                    
               
                
            # 更新训练轮次计数器 / Update epoch counter
            epoch_counter += 1            
            # 更新探索率(epsilon) / Update exploration rate (epsilon)
            self.epsilon = self.epsilon_scheduler.get_epsilon(epoch_counter)  
            # 创建模型保存路径 / Create model saving path          
            epoch_path = os.path.join(self.model_path, "epoch_{}".format(epoch_counter))
            if not os.path.exists(epoch_path):
                os.makedirs(epoch_path)
            # 保存训练模型及指标 / Save trained model and metrics
            self._save_trained_model(epoch_counter,epoch_path,
                                        epoch_return_rate_train_list,
                                        epoch_final_balance_train_list,
                                        epoch_required_money_train_list,
                                        epoch_reward_sum_train_list  )
            # 创建验证路径 / Create validation path
            val_path = os.path.join(epoch_path, "val")
            if not os.path.exists(val_path):
                os.makedirs(val_path)
            log.info(f"start val epoch {epoch_counter}")
            return_rates = []
            var_df_list = self.val_index[self.label]
            if len(var_df_list) >0:
                for initial_action in range(0,self.n_action):
                    dqn_eval = DQN_EVAL(self.n_state_1,self.n_state_2,self.n_action,"cpu",
                        self.val_data_path,
                        self.tech_indicator_list,
                        self.tech_indicator_list_trend,
                        self.transcation_cost,
                        self.back_time_length,
                        self.max_holding_number)
                    return_rate = dqn_eval.val_cluster(epoch_path, val_path, int(initial_action), var_df_list)
                    return_rates.append(return_rate)
                # 计算平均验证收益率 / Calculate average validation return rate
                return_rate_eval = np.mean(return_rates)
                log.info(f"end val epoch {epoch_counter}.return_rate_eval:{return_rate_eval}")
                            # 更新最佳模型 / Update best model if improved
                if return_rate_eval > best_return_rate:
                    best_return_rate = return_rate_eval
                    best_model = self.eval_net.state_dict()
                    log.info(f"best model updated to epoch {epoch_counter}.best_return_rate:{best_return_rate}")
            else:
                log.info(f"end val epoch {epoch_counter}.self.label {self.label} var_df_list 没有，使用最后的模型")
                best_model = self.eval_net.state_dict()

        # 保存最佳模型到指定路径 / Save best model to specified path
        if best_model is not None:
            best_model_folder_path = os.path.join(self.result_path, 'best_model')
            if not os.path.exists(best_model_folder_path):
                os.makedirs(best_model_folder_path)
            best_model_path = os.path.join(best_model_folder_path, 'best_model.pkl')
            torch.save(best_model, best_model_path)

    
    
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
   
    
    # Create log directory
 

    agent = DQN(args)

    
    start_time = time.time()  # Start timing
    log.info(f"start training {start_time}")
    agent.train()
    end_time = time.time()  # End timing
    duration = end_time - start_time
    log.info(f"done, {end_time} total time: {duration:.2f} seconds")