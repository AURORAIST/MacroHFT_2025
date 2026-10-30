import pathlib
import sys
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
import joblib
import numpy as np
import pandas as pd
import time
import traceback
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")
import json
import datetime
# 尝试开启行缓冲，确保立即输出
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
sys.path.insert(0, ".")

from MacroHFT.model.net import *
from MacroHFT.env.high_level_env import Testing_Env, Training_Env
from MacroHFT.RL.util.utili import get_ada, get_epsilon, LinearDecaySchedule
from MacroHFT.RL.util.replay_buffer import ReplayBuffer_High
from MacroHFT.RL.util.memory import episodicmemory

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--buffer_size",type=int,default=1000000,)
parser.add_argument("--dataset",type=str,default="ETHUSDT")
parser.add_argument("--q_value_memorize_freq",type=int, default=10,)
parser.add_argument("--batch_size",type=int,default=512)
parser.add_argument("--eval_update_freq",type=int,default=512)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epsilon_start",type=float,default=0.4)
parser.add_argument("--epsilon_end",type=float,default=0.3)
parser.add_argument("--decay_length",type=int,default=3)
parser.add_argument("--update_times",type=int,default=10)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--transcation_cost",type=float,default=3 / 10000)
parser.add_argument("--back_time_length",type=int,default=1)
parser.add_argument("--seed",type=int,default=12345)
parser.add_argument("--n_step",type=int,default=1)
parser.add_argument("--epoch_number",type=int,default=5)
parser.add_argument("--device",type=str,default="cuda:0")
parser.add_argument("--alpha",type=float,default=0.5)
parser.add_argument("--beta",type=int,default=1)
parser.add_argument("--exp",type=str,default="exp1")
parser.add_argument("--num_step",type=int,default=10)
parser.add_argument("--k",type=int,default=2)
parser.add_argument("--max_steps", type=int, default=3000, help="每个 episode 最大步数")
parser.add_argument("--turnover_penalty_coef", type=float, default=0, help="换手率惩罚系数，建议取手续费的1.0~2.0倍")# 新增：换手率惩罚系数（按 |Δposition|*price*coef 从reward扣减，只在训练环境生效）
parser.add_argument("--early_stop_patience", type=int, default=3, help="Early stopping patience (epochs)")


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
        # 路径全部加上k，便于区分不同动作空间
        k_str = f"k_{args.k}"
        self.result_path = os.path.join("./result/high_level", '{}'.format(args.dataset), k_str, args.exp)
        self.model_path = os.path.join(self.result_path, "seed_{}".format(self.seed))
        self.log_path = os.path.join(self.model_path, "log")
        self.train_data_path = os.path.join(ROOT, "MacroHFT",
                                        "data", args.dataset, "whole")
        self.val_data_path = os.path.join(ROOT, "MacroHFT",
                                        "data", args.dataset, "whole")
        self.test_data_path = os.path.join(ROOT, "MacroHFT",
                                        "data", args.dataset, "whole")
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
        
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.writer = SummaryWriter(self.log_path)
        self.update_counter = 0
        self.q_value_memorize_freq = args.q_value_memorize_freq

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.tech_indicator_list = np.load('./data/feature_list/single_features.npy', allow_pickle=True).tolist()
        self.tech_indicator_list_trend = np.load('./data/feature_list/trend_features.npy', allow_pickle=True).tolist()
        self.clf_list = ['slope_360', 'vol_360']

        self.transcation_cost = args.transcation_cost
        self.back_time_length = args.back_time_length
        self.n_action = args.k + 1
        self.n_state_1 = len(self.tech_indicator_list)
        self.n_state_2 = len(self.tech_indicator_list_trend)
        self.slope_1 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.slope_2 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.slope_3 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.vol_1 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.vol_2 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.vol_3 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)        
        model_list_slope = [
            "./result/low_level/ETHUSDT/best_model/slope/1/best_model.pkl", 
            "./result/low_level/ETHUSDT/best_model/slope/2/best_model.pkl",
            "./result/low_level/ETHUSDT/best_model/slope/3/best_model.pkl"
        ]
        model_list_vol = [
            "./result/low_level/ETHUSDT/best_model/vol/1/best_model.pkl",
            "./result/low_level/ETHUSDT/best_model/vol/2/best_model.pkl",
            "./result/low_level/ETHUSDT/best_model/vol/3/best_model.pkl"
        ]
        self.slope_1.load_state_dict(
            torch.load(model_list_slope[0], map_location=self.device))
        self.slope_2.load_state_dict(
            torch.load(model_list_slope[1], map_location=self.device))
        self.slope_3.load_state_dict(
            torch.load(model_list_slope[2], map_location=self.device))
        self.vol_1.load_state_dict(
            torch.load(model_list_vol[0], map_location=self.device))
        self.vol_2.load_state_dict(
            torch.load(model_list_vol[1], map_location=self.device))
        self.vol_3.load_state_dict(
            torch.load(model_list_vol[2], map_location=self.device))
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
        self.hyperagent = hyperagent(self.n_state_1, self.n_state_2, self.n_action, 32).to(self.device)
        self.hyperagent_target = hyperagent(self.n_state_1, self.n_state_2, self.n_action, 32).to(self.device)
        self.hyperagent_target.load_state_dict(self.hyperagent.state_dict())
        self.update_times = args.update_times
        self.optimizer = torch.optim.Adam(self.hyperagent.parameters(),
                                          lr=args.lr)
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
        self.memory = episodicmemory(4320, 5, self.n_state_1, self.n_state_2, 64, self.device)
        # 保存换手惩罚系数
        self.turnover_penalty_coef = getattr(args, 'turnover_penalty_coef', 0.0)

    def calculate_q(self, w, qs):
        q_tensor = torch.stack(qs)
        q_tensor = q_tensor.permute(1, 0, 2)
        # 动态匹配子代理数量，避免硬编码6
        num_subagents = q_tensor.shape[1]
        # 断言权重维度与子代理数量一致，便于尽早发现不匹配问题
        if w.dim() == 1:
            w = w.unsqueeze(0)
        assert w.shape[-1] == num_subagents, f"hyperagent weight dim ({w.shape[-1]}) != num_subagents ({num_subagents})"
        weights_reshaped = w.view(-1, 1, num_subagents)
        combined_q = torch.bmm(weights_reshaped, q_tensor).squeeze(1)
        
        return combined_q


    def update(self, replay_buffer):
        batch, _, _ = replay_buffer.sample()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        w_current = self.hyperagent(batch['state'], batch['state_trend'], batch['state_clf'], batch['previous_action'])
        w_next = self.hyperagent_target(batch['next_state'], batch['next_state_trend'], batch['next_state_clf'], batch['next_previous_action'])
        w_next_ = self.hyperagent(batch['next_state'], batch['next_state_trend'], batch['next_state_clf'], batch['next_previous_action'])


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
        # 用 hyperagent 权重融合六个子代理
        q_distribution = self.calculate_q(w_current, qs_current)
        q_current = q_distribution.gather(-1, batch['action']).squeeze(-1)
        a_argmax = self.calculate_q(w_next_, qs_next).argmax(dim=-1, keepdim=True)
        q_nexts = self.calculate_q(w_next, qs_next)
        q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * q_nexts.gather(-1, a_argmax).squeeze(-1)

        td_error = self.loss_func(q_current, q_target)
        memory_error = self.loss_func(q_current, batch['q_memory'])

        demonstration = batch['demo_action']
        KL_loss = F.kl_div(
            (q_distribution.softmax(dim=-1) + 1e-8).log(),
            (demonstration.softmax(dim=-1) + 1e-8),
            reduction="batchmean",
        )

        loss = td_error + args.alpha * memory_error + args.beta * KL_loss
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.hyperagent.parameters(), 1)
        self.optimizer.step()
        for param, target_param in zip(self.hyperagent.parameters(), self.hyperagent_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        self.update_counter += 1
        # 适度地输出一次训练指标，帮助判断是否卡死
        # if self.update_counter % self.q_value_memorize_freq == 1:
        #     print(f"[update] cnt={self.update_counter} td={td_error.item():.6f} mem={memory_error.item():.6f} KL={KL_loss.item():.6f}", flush=True)
        return td_error.cpu(), memory_error.cpu(), KL_loss.cpu(), torch.mean(q_current.cpu()), torch.mean(q_target.cpu())

    def act(self, state, state_trend, state_clf, info):
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long().to(self.device),
            0).to(self.device)
        if np.random.uniform() < (1-self.epsilon):
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
        else:
            action_choice = list(range(self.n_action))
            action = random.choice(action_choice)
        return action

    def act_test(self, state, state_trend, state_clf, info):
        with torch.no_grad():
            x1 = torch.FloatTensor(state).to(self.device)
            x2 = torch.FloatTensor(state_trend).to(self.device)
            x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device)
            previous_action = torch.unsqueeze(
                torch.tensor(info["previous_action"]).long().to(self.device),
                0).to(self.device)
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

    def q_estimate(self, state, state_trend, state_clf, info):
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long().to(self.device),
            0).to(self.device)
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
        q = torch.max(actions_value, 1)[0].detach().cpu().numpy()
        
        return q

    def calculate_hidden(self, state, state_trend, info):
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long().to(self.device),
            0).to(self.device)
        with torch.no_grad():
            hs = self.hyperagent.encode(x1, x2, previous_action).cpu().numpy()
        return hs

    def act_single_subagent(self, state, state_trend, state_clf, info, agent_idx=0, agent_type='slope'):
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long().to(self.device), 0).to(self.device)
        if agent_type == 'slope':
            agent = self.slope_agents[agent_idx]
        else:
            agent = self.vol_agents[agent_idx]
        qs = agent(x1, x2, previous_action)
        action = torch.max(qs, 1)[1].data.cpu().numpy()[0]
        return action

    def calculate_q_single_weight(self, qs, main_idx=5):
        # qs: list of subagent outputs
        num_subagents = len(qs)
        batch_size = qs[0].shape[0]
        w = torch.zeros((batch_size, num_subagents), device=qs[0].device)
        w[:, main_idx] = 1.0
        q_tensor = torch.stack(qs).permute(1, 0, 2)  # [batch, subagent, n_action]
        combined_q = torch.bmm(w.view(batch_size, 1, num_subagents), q_tensor).squeeze(1)
        return combined_q

    def train(self):
        epoch_return_rate_train_list = []
        epoch_final_balance_train_list = []
        epoch_required_money_train_list = []
        epoch_reward_sum_train_list = []
        epoch_loss_list = []  # 新增：收集每个epoch的loss
        val_return_list = []  # 新增：收集每个epoch的验证集收益率
        step_counter = 0
        episode_counter = 0
        epoch_counter = 0
        best_return_rate = -float('inf')
        best_model = None
        self.replay_buffer = ReplayBuffer_High(args, self.n_state_1, self.n_state_2, self.n_action) 
        last_heartbeat = time.time()
        max_steps = getattr(args, 'max_steps')
        # 早停机制相关变量
        early_stop_patience = getattr(args, 'early_stop_patience', 10)
        no_improve_count = 0
        best_val_return = -float('inf')
        for sample in range(self.epoch_number):
            print('epoch ', epoch_counter + 1, flush=True)
            self.df = pd.read_feather(
                os.path.join(self.train_data_path, "train.feather"))
            
            
            train_env = Training_Env(
                    df=self.df,
                    tech_indicator_list=self.tech_indicator_list,
                    tech_indicator_list_trend=self.tech_indicator_list_trend,
                    clf_list=self.clf_list,
                    transcation_cost=self.transcation_cost,
                    back_time_length=self.back_time_length,
                    max_holding_number=self.max_holding_number,
                    initial_action=random.choices(range(self.n_action), k=1)[0],
                    alpha = 0,
                    k=self.n_action - 1,
                    turnover_penalty_coef=self.turnover_penalty_coef)
            s, s2, s3, info = train_env.reset()
            episode_reward_sum = 0
            local_step = 0
            epoch_loss = []  # 每个epoch内累计loss
            while True:
                a = self.act(s, s2, s3, info)
                # a = self.act_single_subagent(s, s2, s3, info, agent_idx=0, agent_type='slope')  # 只用slope_1
                s_, s2_, s3_, r, done, info_ = train_env.step(a)
                hs = self.calculate_hidden(s, s2, info)
                q = r + self.gamma * (1 - done) * self.q_estimate(s_, s2_, s3_, info_)
                q_memory = self.memory.query(hs, a)
                if np.isnan(q_memory):
                    q_memory = q
                self.replay_buffer.store_transition(s, s2, s3, info['previous_action'], info['q_value'], a, r, s_, s2_, s3_, info_['previous_action'],
                                info_['q_value'], done, q_memory)
                self.memory.add(hs, a, q, s, s2, info['previous_action'])
                episode_reward_sum += r

                s, s2, s3, info = s_, s2_, s3_, info_
                step_counter += 1
                local_step += 1
                # 周期性进度输出
                if step_counter % 1000 == 0:
                    print(f"[train] steps={step_counter} ep_reward={episode_reward_sum:.4f} eps={self.epsilon:.3f}", flush=True)
                # 心跳输出，避免长时间无日志误判卡死
                if time.time() - last_heartbeat > 30:
                    print(f"[heartbeat] steps={step_counter} ep_reward={episode_reward_sum:.4f}", flush=True)
                    last_heartbeat = time.time()
                if step_counter % self.eval_update_freq == 0 and step_counter > (
                        self.batch_size + self.n_step):
                    for i in range(self.update_times):
                        td_error, memory_error, KL_loss, q_eval, q_target = self.update(self.replay_buffer)
                        epoch_loss.append(td_error.item())  # 收集loss
                        if self.update_counter % self.q_value_memorize_freq == 1:
                            self.writer.add_scalar(
                                tag="td_error",
                                scalar_value=td_error,
                                global_step=self.update_counter,
                                walltime=None)
                            self.writer.add_scalar(
                                tag="memory_error",
                                scalar_value=memory_error,
                                global_step=self.update_counter,
                                walltime=None)
                            self.writer.add_scalar(
                                tag="KL_loss",
                                scalar_value=KL_loss,
                                global_step=self.update_counter,
                                walltime=None)
                            self.writer.add_scalar(
                                tag="q_eval",
                                scalar_value=q_eval,
                                global_step=self.update_counter,
                                walltime=None)
                            self.writer.add_scalar(
                                tag="q_target",
                                scalar_value=q_target,
                                global_step=self.update_counter,
                                walltime=None)
                    if step_counter > 4320:
                        self.memory.re_encode(self.hyperagent)
                if done:
                # if done or local_step >= max_steps:
                    # 只要done或max_steps，确保final_balance/required_money赋值
                    if not hasattr(train_env, 'final_balance') or not hasattr(train_env, 'required_money'):
                        _, pure_balance, required_money, _ = train_env.get_final_return_rate(slient=True)
                        train_env.pured_balance = pure_balance
                        train_env.final_balance = train_env.pured_balance + train_env.calculate_value(train_env.data.iloc[-1], train_env.position)
                        train_env.required_money = required_money
                    print(f"[episode] done. steps={step_counter} final_balance={train_env.final_balance:.4f} required_money={train_env.required_money:.4f}", flush=True)
                    break
            # print("111")
            episode_counter += 1
            final_balance, required_money = train_env.final_balance, train_env.required_money
            self.writer.add_scalar(tag="return_rate_train",
                                scalar_value=final_balance / (required_money),
                                global_step=episode_counter,
                                walltime=None)
            self.writer.add_scalar(tag="final_balance_train",
                                scalar_value=final_balance,
                                global_step=episode_counter,
                                walltime=None)
            self.writer.add_scalar(tag="required_money_train",
                                scalar_value=required_money,
                                global_step=episode_counter,
                                walltime=None)
            self.writer.add_scalar(tag="reward_sum_train",
                                scalar_value=episode_reward_sum,
                                global_step=episode_counter,
                                walltime=None)
            epoch_return_rate_train_list.append(final_balance / (required_money))
            epoch_final_balance_train_list.append(final_balance)
            epoch_required_money_train_list.append(required_money)
            epoch_reward_sum_train_list.append(episode_reward_sum)
                

            epoch_counter += 1
            self.epsilon = self.epsilon_scheduler.get_epsilon(epoch_counter)
            mean_return_rate_train = np.mean(epoch_return_rate_train_list)
            mean_final_balance_train = np.mean(epoch_final_balance_train_list)
            mean_required_money_train = np.mean(epoch_required_money_train_list)
            mean_reward_sum_train = np.mean(epoch_reward_sum_train_list)
            # 记录本epoch的loss均值
            if len(epoch_loss) > 0:
                epoch_loss_list.append(np.mean(epoch_loss))
            # 记录本epoch的验证集收益率
            # 修复：调用 val_cluster 获取验证集收益率
            epoch_model_path = self.model_path  # 当前模型路径
            # 保存当前模型参数
            torch.save(self.hyperagent.state_dict(), os.path.join(epoch_model_path, "trained_model.pkl"))
            val_ret = self.val_cluster(epoch_model_path, epoch_model_path)
            val_return_list.append(val_ret)
            # --- 早停机制 ---
            if val_ret > best_val_return:
                best_val_return = val_ret
                no_improve_count = 0
            else:
                no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                print(f"[EarlyStopping] No improvement in val_return for {early_stop_patience} epochs. Stopping training.", flush=True)
                break
            # --- 可视化部分 ---
            # loss曲线
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 4))
            plt.plot(range(1, len(epoch_loss_list)+1), epoch_loss_list, marker='o', label='Train Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Train Loss Curve')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_path, 'loss_curve.png'))
            plt.close()
            # 验证集收益率曲线
            plt.figure(figsize=(8, 4))
            plt.plot(range(1, len(val_return_list)+1), val_return_list, marker='o', color='g', label='Val Return Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Validation Return Rate')
            plt.title('Validation Return Rate Curve')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_path, 'val_return_curve.png'))
            plt.close()
            # --- 可视化部分结束 ---
            k_str = f"k_{self.n_action - 1}"
            best_model_path = os.path.join("./result/high_level", '{}'.format(self.dataset), k_str, 'best_model.pkl')
            # 自动创建保存目录，避免RuntimeError
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(best_model, best_model_path)
            final_result_path = os.path.join("./result/high_level", '{}'.format(self.dataset), k_str)
            self.test_cluster(best_model_path, final_result_path)

        # 训练结束后保存 loss 曲线
        np.save(os.path.join(self.model_path, 'loss.npy'), np.array(epoch_loss_list, dtype=float))
        print(f"[Info] Loss curve saved to {os.path.join(self.model_path, 'loss.npy')}", flush=True)

    def val_cluster(self, epoch_path, save_path):
        self.hyperagent.load_state_dict(
            torch.load(os.path.join(epoch_path, "trained_model.pkl")))
        self.hyperagent.eval()
        counter = False
        action_list = []
        reward_list = []
        final_balance_list = []
        required_money_list = []
        commission_fee_list = []
        self.df = pd.read_feather(
            os.path.join(self.val_data_path, "val.feather"))
        
        val_env = Testing_Env(
                df=self.df,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                clf_list=self.clf_list,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                initial_action=0,
                k=self.n_action - 1)
        s, s2, s3, info = val_env.reset()
        done = False
        action_list_episode = []
        reward_list_episode = []
        while not done:
            a = self.act(s, s2, s3, info)
            # a = self.act_single_subagent(s, s2, s3, info, agent_idx=0, agent_type='slope')  # 只用slope_1
            s_, s2_, s3_, r, done, info_ = val_env.step(a)
            reward_list_episode.append(r)
            s, s2, s3, info = s_, s2_, s3_, info_
            action_list_episode.append(a)
        portfit_magine, final_balance, required_money, commission_fee = val_env.get_final_return_rate(
            slient=True)
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
        np.save(os.path.join(save_path, "final_balance_val.npy"),
            final_balance_list)
        np.save(os.path.join(save_path, "require_money_val.npy"),
                required_money_list)
        np.save(os.path.join(save_path, "commission_fee_history_val.npy"),
                commission_fee_list)
        return_rate = final_balance / required_money
        # 计算并保存验证集指标
        try:
            V_1 = np.load(os.path.join(save_path, 'require_money_val.npy'))
            reward = np.load(os.path.join(save_path, 'reward_val.npy'))
            if reward.ndim > 1:
                reward = reward[0]
            if V_1.ndim > 0:
                V_1 = V_1[0]
            m = 525600.0
            r = np.divide(reward, V_1, out=np.zeros_like(reward, dtype=float), where=(V_1!=0))
            nav = 1.0 + np.cumsum(r)
            if nav.size == 0:
                TR = AVOL = ASR = MDD = ACR = ASoR = 0.0
            else:
                TR = float(nav[-1] - 1.0)
                std_r = float(np.std(r))
                mean_r = float(np.mean(r))
                AVOL = float(std_r * np.sqrt(m)) if std_r > 0 else 0.0
                ASR = float((mean_r / std_r) * np.sqrt(m)) if std_r > 0 else 0.0
                roll_max = np.maximum.accumulate(nav)
                dd = (roll_max - nav) / np.where(roll_max==0, 1.0, roll_max)
                MDD = float(np.max(dd)) if dd.size > 0 else 0.0
                annual_ret = float(mean_r * m)
                ACR = float(annual_ret / MDD) if MDD > 0 else 0.0
                downside = r[r < 0]
                dd_std = float(np.std(downside)) if downside.size > 0 else 0.0
                ASoR = float((mean_r / dd_std) * np.sqrt(m)) if dd_std > 0 else 0.0
            metrics = {
                "TR": TR,
                "AVOL": AVOL,
                "MDD": MDD,
                "ASR": ASR,
                "ACR": ACR,
                "ASoR": ASoR,
            }
            with open(os.path.join(save_path, 'metrics_val.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"[metrics][val] TR={TR:.6f} AVOL={AVOL:.6f} MDD={MDD:.6f} ASR={ASR:.6f} ACR={ACR:.6f} ASoR={ASoR:.6f}", flush=True)
        except Exception as e:
            print(f"[metrics][val] failed: {repr(e)}", flush=True)
        return return_rate

    def test_cluster(self, epoch_path, save_path):
        self.hyperagent.load_state_dict(
            torch.load(os.path.join(epoch_path)))
        self.hyperagent.eval()
        counter = False
        action_list = []
        reward_list = []
        final_balance_list = []
        required_money_list = []
        commission_fee_list = []
        self.df = pd.read_feather(
            os.path.join(self.test_data_path, "test.feather"))
        
        test_env = Testing_Env(
                df=self.df,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                clf_list=self.clf_list,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                initial_action=0,
                k=self.n_action - 1)
        s, s2, s3, info = test_env.reset()
        done = False
        action_list_episode = []
        reward_list_episode = []
        while not done:
            a = self.act_test(s, s2, s3, info)
            # a = self.act_single_subagent(s, s2, s3, info, agent_idx=0, agent_type='slope')  # 只用slope_1
            s_, s2_, s3_, r, done, info_ = test_env.step(a)
            reward_list_episode.append(r)
            s, s2, s3, info = s_, s2_, s3_, info_
            action_list_episode.append(a)
        portfit_magine, final_balance, required_money, commission_fee = test_env.get_final_return_rate(
            slient=True)
        final_balance = test_env.final_balance
        action_list.append(action_list_episode)
        reward_list.append(reward_list_episode)
        final_balance_list.append(final_balance)
        required_money_list.append(required_money)
        commission_fee_list.append(commission_fee)

        # 自动创建保存目录，避免FileNotFoundError
        os.makedirs(save_path, exist_ok=True)
        action_list = np.array(action_list)
        reward_list = np.array(reward_list)
        final_balance_list = np.array(final_balance_list)
        required_money_list = np.array(required_money_list)
        commission_fee_list = np.array(commission_fee_list)
        np.save(os.path.join(save_path, "action.npy"), action_list)
        np.save(os.path.join(save_path, "reward.npy"), reward_list)
        np.save(os.path.join(save_path, "final_balance.npy"),
            final_balance_list)
        np.save(os.path.join(save_path, "require_money.npy"),
                required_money_list)
        np.save(os.path.join(save_path, "commission_fee_history.npy"),
                commission_fee_list)
        # 累计收益率可视化
        import matplotlib.pyplot as plt
        V_1 = np.load(os.path.join(save_path, 'require_money.npy'))
        reward = np.load(os.path.join(save_path, 'reward.npy'))
        # reward 可能是二维，取第一个episode
        if reward.ndim > 1:
            reward = reward[0]
        if V_1.ndim > 0:
            V_1 = V_1[0]
        total_reward = np.cumsum(reward)
        cum_return = np.divide(total_reward, V_1, out=np.zeros_like(total_reward, dtype=float), where=(V_1!=0)) * 100
        timestamps = range(len(cum_return))
        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, cum_return, label='Cumulative Return (%)')
        plt.xlabel('Timestamp')
        plt.ylabel('Cumulative Return (%)')
        plt.title('Cumulative Return Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'cum_return.png'))
        plt.close()
        # 计算六项指标并保存
        m = 525600.0
        # 每步收益率近似：reward / V1（与上面累计收益一致口径）
        r = np.divide(reward, V_1, out=np.zeros_like(reward, dtype=float), where=(V_1!=0))
        nav = 1.0 + np.cumsum(r)
        if nav.size == 0:
            TR = 0.0
            AVOL = 0.0
            ASR = 0.0
            MDD = 0.0
            ACR = 0.0
            ASoR = 0.0
        else:
            TR = float(nav[-1] - 1.0)
            std_r = float(np.std(r))
            mean_r = float(np.mean(r))
            AVOL = float(std_r * np.sqrt(m)) if std_r > 0 else 0.0
            ASR = float((mean_r / std_r) * np.sqrt(m)) if std_r > 0 else 0.0
            roll_max = np.maximum.accumulate(nav)
            dd = (roll_max - nav) / np.where(roll_max==0, 1.0, roll_max)
            MDD = float(np.max(dd)) if dd.size > 0 else 0.0
            annual_ret = float(mean_r * m)
            ACR = float(annual_ret / MDD) if MDD > 0 else 0.0
            downside = r[r < 0]
            dd_std = float(np.std(downside)) if downside.size > 0 else 0.0
            ASoR = float((mean_r / dd_std) * np.sqrt(m)) if dd_std > 0 else 0.0
        metrics = {
            "TR": TR,
            "AVOL": AVOL,
            "MDD": MDD,
            "ASR": ASR,
            "ACR": ACR,
            "ASoR": ASoR,
        }
        with open(os.path.join(save_path, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"[metrics][test] TR={TR:.6f} AVOL={AVOL:.6f} MDD={MDD:.6f} ASR={ASR:.6f} ACR={ACR:.6f} ASoR={ASoR:.6f}", flush=True)
        # action 可视化
        action = np.load(os.path.join(save_path, 'action.npy'))
        if action.ndim > 1:
            action = action[0]
        plt.figure(figsize=(10, 3))
        plt.plot(range(len(action)), action, drawstyle='steps-post', label='Action')
        plt.xlabel('Timestamp')
        plt.ylabel('Action')
        plt.title('Action Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'action.png'))
        plt.close()    


if __name__ == "__main__":
    print(datetime.datetime.now())
    args = parser.parse_args()
    # print(args)
    agent = DQN(args)
    # print("Training agent...", flush=True)
    try:
        agent.train()
    except Exception as e:
        print(f"[FATAL] Training crashed: {repr(e)}", flush=True)
        traceback.print_exc()
        raise
    print("Training is over!", flush=True)
    print(datetime.datetime.now())
