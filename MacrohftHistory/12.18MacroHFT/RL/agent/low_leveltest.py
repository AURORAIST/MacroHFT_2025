import pathlib
import sys
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pickle
import os
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

ROOT = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
sys.path.insert(0, ".")

from MacroHFT.model.net import *
from MacroHFT.env.low_level_env import Testing_Env, Training_Env
from MacroHFT.RL.util.utili import get_ada, get_epsilon, LinearDecaySchedule
from MacroHFT.RL.util.replay_buffer import ReplayBuffer

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--buffer_size",type=int,default=1000000,)
parser.add_argument("--dataset",type=str,default="ETHUSDT")
parser.add_argument("--q_value_memorize_freq",type=int, default=10,)
parser.add_argument("--batch_size",type=int,default=512)
parser.add_argument("--eval_update_freq",type=int,default=100)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--epsilon_start",type=float,default=0.5)
parser.add_argument("--epsilon_end",type=float,default=0.1)
parser.add_argument("--decay_length",type=int,default=5)
parser.add_argument("--update_times",type=int,default=10)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--transcation_cost",type=float,default=2.0 / 10000)
parser.add_argument("--back_time_length",type=int,default=1)
parser.add_argument("--seed",type=int,default=12345)
parser.add_argument("--n_step",type=int,default=1)
parser.add_argument("--epoch_number",type=int,default=15)
parser.add_argument("--label",type=str,default="label_1")
parser.add_argument("--clf",type=str,default="slope")
parser.add_argument("--alpha",type=float,default="0")
parser.add_argument("--device",type=str,default="cuda:0")
# 仅保留 --n_action 参数
parser.add_argument("--n_action",type=int,default=2, help="Number of position levels (e.g., 2 means 0%,100%; 3 means 0%,50%,100%)")
parser.add_argument("--exp",type=str,default="exp1")
parser.add_argument("--reward_scale",type=float,default=1000, help="Reward scale for Q-table demonstration learning (default: 1000)")
# 新增：检查点路径，直接用于验证
parser.add_argument("--checkpoint", type=str, default="", help="Path to low-level checkpoint .pkl to validate (state_dict)")

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
        self.seed = args.seed
        seed_torch(self.seed)
        if torch.cuda.is_available():
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cpu")
        # 统一使用 --n_action
        self.n_action = args.n_action
        # 结果路径结构: ./result/low_level/{dataset}/{exp}/{clf}/pos_{n_action}/{alpha}/{label}
        self.result_path = os.path.join(
            "./result/low_level",
            f"{args.dataset}",
            f"{args.exp}",
            f"{args.clf}",
            f"pos_{self.n_action}",
            str(int(float(args.alpha))),
            f"{args.label}",
        )
        self.label = int(args.label.split('_')[1])
        self.model_path = os.path.join(self.result_path, f"seed_{self.seed}")
        self.train_data_path = os.path.join(ROOT, "MacroHFT", "data", args.dataset, "train")
        self.val_data_path = os.path.join(ROOT, "MacroHFT", "data", args.dataset, "val")
        self.test_data_path = os.path.join(ROOT, "MacroHFT", "data", args.dataset, "test")
        if args.clf == 'slope':
            with open(os.path.join(self.train_data_path, 'slope_labels.pkl'), 'rb') as file:
                self.train_index = pickle.load(file)
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
        self.n_state_1 = len(self.tech_indicator_list)
        self.n_state_2 = len(self.tech_indicator_list_trend)
        self.eval_net, self.target_net = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device), subagent(
                self.n_state_1, self.n_state_2, self.n_action,
                64).to(self.device)
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
        # 记录检查点路径
        self.checkpoint_path = args.checkpoint

    def update(self, replay_buffer):
        self.eval_net.train()
        batch, _, _ = replay_buffer.sample()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        a_argmax = self.eval_net(batch['next_state'], batch['next_state_trend'], batch['next_previous_action']).argmax(dim=-1, keepdim=True)
        q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * self.target_net(batch['next_state'], batch['next_state_trend'], 
                                                    batch['next_previous_action']).gather(-1, a_argmax).squeeze(-1)

        q_distribution = self.eval_net(batch['state'], batch['state_trend'], batch['previous_action'])
        q_current = q_distribution.gather(-1, batch['action']).squeeze(-1)

        td_error = self.loss_func(q_current, q_target)

        demonstration = batch['demo_action']
        KL_loss = F.kl_div(
            (q_distribution.softmax(dim=-1) + 1e-8).log(),
            (demonstration.softmax(dim=-1) + 1e-8),
            reduction="batchmean",
        )

        alpha = args.alpha
        loss = td_error + alpha * KL_loss
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), 1)
        self.optimizer.step()
        for param, target_param in zip(self.eval_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.update_counter += 1
        self.eval_net.eval()
        return td_error.cpu(), KL_loss.cpu(), torch.mean(
            q_current.cpu()), torch.mean(q_target.cpu())

    def hardupdate(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def act(self, state, state_trend, info):
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long().to(self.device),
            0).to(self.device)
        if np.random.uniform() < (1-self.epsilon):
            actions_value = self.eval_net(x1, x2, previous_action)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
        else:
            # epsilon-greedy 随机选择动作，范围从0到n_action-1
            action = random.choice(range(self.n_action))
        return action

    def act_test(self, state, state_trend, info):
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long(), 0).to(self.device)
        actions_value = self.eval_net(x1, x2, previous_action)
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        action = action[0]
        return action

    def _check_checkpoint_actions(self, checkpoint_path):
        """从检查点中推断动作数，并与当前 self.n_action 对齐校验。"""
        try:
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and 'embedding.weight' in state:
                n_actions_ckpt = state['embedding.weight'].shape[0]
            elif isinstance(state, dict) and 'state_dict' in state and 'embedding.weight' in state['state_dict']:
                n_actions_ckpt = state['state_dict']['embedding.weight'].shape[0]
            else:
                n_actions_ckpt = None
        except Exception:
            n_actions_ckpt = None
        if n_actions_ckpt is not None and n_actions_ckpt != self.n_action:
            raise ValueError(f"n_action mismatch: checkpoint has {n_actions_ckpt}, but args specify {self.n_action}. 请使用匹配的 --n_action 或切换相应检查点。")

    def train(self):
        # 打印维度信息（仅打印一次）
        print("\n" + "="*60)
        print("多仓位配置信息:")
        print("="*60)
        print(f"动作空间大小 (n_action): {self.n_action}")
        print(f"动作范围: [0, {self.n_action-1}]")
        print("仓位映射: ", end="")
        scale_factor = self.n_action - 1
        for i in range(self.n_action):
            ratio = (i / scale_factor if scale_factor > 0 else 0) * 100
            print(f"{i}→{ratio:.1f}%", end="  " if i < self.n_action-1 else "\n")
        print("\n状态空间维度:")
        print(f"  - 单特征状态 (state): {self.n_state_1} 个特征")
        print(f"  - 趋势特征状态 (state_trend): {self.n_state_2} 个特征")
        print(f"  - 前一动作 (previous_action): 标量 ∈ [0, {self.n_action-1}]")
        print(f"\n网络输出维度: (batch_size, {self.n_action})")
        print(f"ReplayBuffer demo_action 维度: (batch_size, {self.n_action})")
        print("="*60 + "\n")

        if not self.checkpoint_path or not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found or not provided: '{self.checkpoint_path}'")
        print(f"Using checkpoint for validation: {self.checkpoint_path}")
        self._check_checkpoint_actions(self.checkpoint_path)

        save_root = os.path.join(self.model_path, "val")
        os.makedirs(save_root, exist_ok=True)

        val_return_rates = []
        for init_action in range(self.n_action):
            return_rate = self.val_cluster(self.checkpoint_path, save_root, init_action)
            val_return_rates.append(return_rate)
        return_rate_eval = np.mean(val_return_rates)
        print(f"Validation return rates for all initial positions: {val_return_rates}")
        print(f"Average validation return rate: {return_rate_eval}")

    def val_cluster(self, checkpoint_path, save_path, initial_action):
        # 直接从检查点加载（已校验动作数）
        self.eval_net.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.eval_net.eval()
        df_list = self.val_index[self.label]
        df_number=int(len(df_list)) 
        action_list = []
        reward_list = []
        final_balance_list = []
        required_money_list = []
        commission_fee_list = []
        for i in range(df_number):
            print("validating on df", df_list[i])
            self.df = pd.read_feather(
                os.path.join(self.val_data_path, "df_{}.feather".format(df_list[i])))
            
            val_env = Testing_Env(
                    df=self.df,
                    tech_indicator_list=self.tech_indicator_list,
                    tech_indicator_list_trend=self.tech_indicator_list_trend,
                    transcation_cost=self.transcation_cost,
                    back_time_length=self.back_time_length,
                    max_holding_number=self.max_holding_number,
                    initial_action=initial_action,
                    num_actions=self.n_action)
            s, s2, info = val_env.reset()
            done = False
            action_list_episode = []
            reward_list_episode = []
            while not done:
                a = self.act_test(s, s2, info)
                s_, s2_, r, done, info_ = val_env.step(a)
                reward_list_episode.append(r)
                s, s2, info = s_, s2_, info_
                action_list_episode.append(a)
            portfit_magine, final_balance, required_money, commission_fee = val_env.get_final_return_rate(
                slient=True)
            final_balance = val_env.final_balance
            required_money = val_env.required_money
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
        
        # 计算平均收益率
        return_rate_mean = np.nan_to_num(final_balance_list / required_money_list).mean()
        
        # 统计仓位使用情况
        action_counts = np.zeros(self.n_action)
        total_actions = 0
        for episode_actions in action_list:
            for a in episode_actions:
                action_counts[int(a)] += 1
                total_actions += 1
        
        action_distribution = action_counts / total_actions if total_actions > 0 else action_counts
        
        # 打印统计信息
        print(f"\n--- Validation Statistics (Initial Position: {initial_action}) ---")
        print(f"Total steps: {total_actions}")
        print("Action distribution:")
        scale_factor = self.n_action - 1
        for i in range(self.n_action):
            position_pct = (i / scale_factor if scale_factor > 0 else 0) * 100
            count = int(action_counts[i])
            pct = action_distribution[i] * 100
            print(f"  Action {i} ({position_pct:5.1f}% position): {count:6d} times ({pct:5.2f}%)")
        print(f"Mean return rate: {return_rate_mean:.4f}")
        print("-" * 60 + "\n")
        
        # 保存所有结果（按初始仓位分文件名）
        np.save(os.path.join(save_path, "action_val_{}.npy".format(initial_action)), action_list)
        np.save(os.path.join(save_path, "reward_val_{}.npy".format(initial_action)), reward_list)
        np.save(os.path.join(save_path, "final_balance_val_{}.npy".format(initial_action)),
            final_balance_list)
        np.save(os.path.join(save_path, "require_money_val_{}.npy".format(initial_action)),
                required_money_list)
        np.save(os.path.join(save_path, "commission_fee_history_val_{}.npy".format(initial_action)),
                commission_fee_list)
        np.save(os.path.join(save_path, "action_distribution_val_{}.npy".format(initial_action)),
                action_distribution)
        np.save(os.path.join(save_path, "return_rate_mean_val_{}.npy".format(initial_action)),
                return_rate_mean)
        
        return return_rate_mean
        

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    agent = DQN(args)
    agent.train()
    print('happy end')