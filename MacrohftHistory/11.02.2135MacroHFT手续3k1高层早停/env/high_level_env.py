from logging import raiseExceptions
import numpy as np
# import gymnasium as gym
from gym.utils import seeding
import gym
from gym import spaces
import pandas as pd
import argparse
import os
import torch
import sys
import pathlib
import pdb

ROOT = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(ROOT)
sys.path.insert(0, ".")

from MacroHFT.tools.demonstration import make_q_table_reward

tech_indicator_list = np.load('./data/feature_list/single_features.npy', allow_pickle=True).tolist()
tech_indicator_list_trend = np.load('./data/feature_list/trend_features.npy', allow_pickle=True).tolist()
clf_list = ['slope_360', 'vol_360']


transcation_cost = 0.0003
back_time_length = 1
max_holding_number = 0.01
alpha = 0


class Testing_Env(gym.Env):
    def __init__(
        self,
        df: pd.DataFrame,
        tech_indicator_list=tech_indicator_list,
        tech_indicator_list_trend=tech_indicator_list_trend,
        clf_list=clf_list,
        transcation_cost=transcation_cost,
        back_time_length=back_time_length,
        max_holding_number=max_holding_number,
        initial_action=0,
        k=1,
        # 新增：换手率惩罚系数（默认0）
        turnover_penalty_coef: float = 0.0,
    ):
        self.k = k
        self.tech_indicator_list = tech_indicator_list
        self.tech_indicator_list_trend = tech_indicator_list_trend
        self.clf_list=clf_list
        self.df = df
        self.initial_action = initial_action
        self.action_space = spaces.Discrete(self.k + 1)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=+np.inf,
            shape=(back_time_length * len(self.tech_indicator_list), ))
        self.terminal = False
        self.stack_length = back_time_length
        self.m = back_time_length
        self.data = self.df.iloc[self.m - self.stack_length:self.m]
        self.single_state = self.data[self.tech_indicator_list].values
        self.trend_state = self.data[self.tech_indicator_list_trend].values
        self.initial_reward = 0
        self.reward_history = [self.initial_reward]
        self.previous_action = 0
        self.comission_fee = transcation_cost
        self.max_holding_number = max_holding_number
        self.needed_money_memory = []
        self.sell_money_memory = []
        self.comission_fee_history = []
        # 新增：换手惩罚
        self.turnover_penalty_coef = float(turnover_penalty_coef)
        self.turnover_penalty_history = []
        self.previous_position = 0
        self.position = 0



    def calculate_value(self, price_information, position):
        return price_information["close"] * position

    def reset(self):
        self.terminal = False
        self.m = back_time_length
        self.data = self.df.iloc[self.m - self.stack_length:self.m]
        self.single_state = self.data[self.tech_indicator_list].values
        self.trend_state = self.data[self.tech_indicator_list_trend].values
        self.clf_state = self.data[self.clf_list].values
        self.initial_reward = 0
        self.reward_history = [self.initial_reward]
        self.previous_action = 0
        price_information = self.data.iloc[-1]
        self.position = 0
        self.needed_money_memory = []
        self.sell_money_memory = []
        self.comission_fee_history = []
        # 重置换手惩罚记录
        self.turnover_penalty_history = []
        self.previous_position = (self.initial_action / self.k) * self.max_holding_number
        self.position = (self.initial_action / self.k) * self.max_holding_number
        # 记录初始建仓所需现金流，保持与低层环境一致，避免 required_money 为 0
        init_cash_needed = self.position * price_information["close"]
        self.needed_money_memory.append(init_cash_needed)
        self.sell_money_memory.append(0)
        return self.single_state, self.trend_state, self.clf_state.reshape(-1), {
            "previous_action": self.initial_action,
        }

    def step(self, action):
        normlized_action = action / self.k
        position = self.max_holding_number * normlized_action
        self.terminal = (self.m >= len(self.df.index.unique()) - 1)
        previous_position = self.previous_position
        previous_price_information = self.data.iloc[-1]
        self.m += 1
        self.data = self.df.iloc[self.m - self.stack_length:self.m]
        current_price_information = self.data.iloc[-1]
        self.single_state = self.data[self.tech_indicator_list].values
        self.trend_state = self.data[self.tech_indicator_list_trend].values
        self.clf_state = self.data[self.clf_list].values
        self.previous_position = previous_position
        self.position = position
        self.changing = (self.position != self.previous_position)
        # 计算换手量（用前一根bar的价格评估）并得到惩罚
        trade_amount = abs(position - previous_position) * float(previous_price_information['close'])
        turnover_penalty = self.turnover_penalty_coef * trade_amount
        if previous_position >= position:
            self.sell_size = previous_position - position

            cash = self.sell_size * previous_price_information['close'] * (1 - self.comission_fee)
            self.comission_fee_history.append(self.comission_fee * self.sell_size * previous_price_information['close'])

            self.sell_money_memory.append(cash)
            self.needed_money_memory.append(0)
            self.position = position
            previous_value = self.calculate_value(previous_price_information,
                                                  self.previous_position)
            current_value = self.calculate_value(current_price_information,
                                                 self.position)
            self.reward = current_value + cash - previous_value
            # 应用换手惩罚
            if turnover_penalty > 0:
                self.reward -= turnover_penalty
                self.turnover_penalty_history.append(turnover_penalty)
            else:
                self.turnover_penalty_history.append(0.0)
            if previous_value == 0:
                return_rate = 0
            else:
                return_rate = (current_value + cash -
                               previous_value) / previous_value
            self.return_rate = return_rate
            self.reward_history.append(self.reward)

        if previous_position < position:
            self.buy_size = position - previous_position
            needed_cash = self.buy_size * previous_price_information['close'] * (1 + self.comission_fee)
            self.comission_fee_history.append(self.comission_fee * self.buy_size * previous_price_information['close'])

            self.needed_money_memory.append(needed_cash)
            self.sell_money_memory.append(0)

            self.position = position
            previous_value = self.calculate_value(previous_price_information,
                                                  self.previous_position)
            current_value = self.calculate_value(current_price_information,
                                                 self.position)
            self.reward = current_value - needed_cash - previous_value
            # 应用换手惩罚
            if turnover_penalty > 0:
                self.reward -= turnover_penalty
                self.turnover_penalty_history.append(turnover_penalty)
            else:
                self.turnover_penalty_history.append(0.0)
            return_rate = (current_value - needed_cash -
                           previous_value) / (previous_value + needed_cash)

            self.reward_history.append(self.reward)
            self.return_rate = return_rate
        self.previous_position = self.position

        if self.terminal:
            return_margin, pure_balance, required_money, commission_fee = self.get_final_return_rate(
            )
            self.pured_balance = pure_balance
            self.final_balance = self.pured_balance + self.calculate_value(
                current_price_information, self.position)
            self.required_money = required_money
            print("the portfit margine is ",
                  self.final_balance / self.required_money)

        return self.single_state, self.trend_state, self.clf_state.reshape(-1), self.reward, self.terminal, {
            "previous_action": action,
        }

    def get_final_return_rate(self, slient=False):
        sell_money_memory = np.array(self.sell_money_memory)
        needed_money_memory = np.array(self.needed_money_memory)
        true_money = sell_money_memory - needed_money_memory
        final_balance = np.sum(true_money)
        balance_list = []
        for i in range(len(true_money)):
            balance_list.append(np.sum(true_money[:i + 1]))
        required_money = -np.min(balance_list) if len(balance_list) > 0 else 0.0
        commission_fee = np.sum(self.comission_fee_history)
        # 安全除法：当 required_money 为 0 时返回率置 0，并提示
        if required_money == 0 or not np.isfinite(required_money):
            if not slient:
                print("[警告] required_money=0, 分片无有效资金流，建议剔除该样本")
            # return 0.0, final_balance, 0.0, commission_fee
        # 计算六个指标（仅打印，不改变返回值签名）
        if not slient:
            m = 525600.0
            rewards = np.array(self.reward_history[1:]) if len(self.reward_history) > 1 else np.array([])
            r = np.divide(rewards, required_money, out=np.zeros_like(rewards, dtype=float), where=(required_money!=0))
            nav = 1.0 + np.cumsum(r) if r.size > 0 else np.array([1.0])
            TR = float(nav[-1] - 1.0)
            std_r = float(np.std(r)) if r.size > 0 else 0.0
            mean_r = float(np.mean(r)) if r.size > 0 else 0.0
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
            print(f"[metrics] TR={TR:.6f} AVOL={AVOL:.6f} MDD={MDD:.6f} ASR={ASR:.6f} ACR={ACR:.6f} ASoR={ASoR:.6f}")
        return final_balance / required_money, final_balance, required_money, commission_fee


class Training_Env(Testing_Env):
    def __init__(
        self,
        df: pd.DataFrame,
        tech_indicator_list=tech_indicator_list,
        tech_indicator_list_trend=tech_indicator_list_trend,
        clf_list=clf_list,
        transcation_cost=transcation_cost,
        back_time_length=back_time_length,
        max_holding_number=max_holding_number,
        initial_action = 0,
        alpha=alpha,
        k=1,
        # 训练时可开启换手惩罚
        turnover_penalty_coef: float = 0.0,
    ):
        super(Training_Env,
              self).__init__(df, tech_indicator_list, tech_indicator_list_trend, clf_list, transcation_cost,
                             back_time_length, max_holding_number, initial_action, k, turnover_penalty_coef)
        self.q_table = make_q_table_reward(df,
                                           num_action=k+1,
                                           max_holding=max_holding_number,
                                           commission_fee=transcation_cost,
                                           reward_scale=1,
                                           gamma=0.99,
                                           max_punish=1e12)
        self.initial_action = initial_action

    def reset(self):
        single_state, trend_state, clf_state, info = super(Training_Env, self).reset()
        self.previous_action = self.initial_action
        self.previous_position = (self.initial_action / self.k) * self.max_holding_number
        self.position = (self.initial_action / self.k) * self.max_holding_number
        info['q_value'] = self.q_table[self.m - 1][self.previous_action][:]
        return single_state, trend_state, clf_state.reshape(-1), info

    def step(self, action):
        single_state, trend_state, clf_state, reward, done, info = super(Training_Env, self).step(action)
        info['q_value'] = self.q_table[self.m - 1][action][:]
        # 保证无论done还是max_steps break都能赋值final_balance/required_money
        if done and (not hasattr(self, 'final_balance') or not hasattr(self, 'required_money')):
            _, pure_balance, required_money, _ = self.get_final_return_rate(slient=True)
            self.pured_balance = pure_balance
            self.final_balance = self.pured_balance + self.calculate_value(self.data.iloc[-1], self.position)
            self.required_money = required_money
        return single_state, trend_state, clf_state.reshape(-1), reward, done, info


