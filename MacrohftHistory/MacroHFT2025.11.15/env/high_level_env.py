from logging import raiseExceptions
import numpy as np
import pandas as pd
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    # 如果 gymnasium 未安装，回退到 gym（但会有警告）
    import gym
    from gym import spaces
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


transcation_cost = 0.0002
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
        num_actions=2,  # 📌 支持多仓位，默认为2（与原始版本一致）
    ):
        self.tech_indicator_list = tech_indicator_list
        self.tech_indicator_list_trend = tech_indicator_list_trend
        self.clf_list=clf_list
        self.df = df
        self.initial_action = initial_action
        self.num_actions = num_actions
        # 📌 支持多仓位，但当num_actions=2时与原始版本一致
        self.action_space = spaces.Discrete(num_actions)
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
        # 📌 支持多仓位，但当num_actions=2时与原始版本一致
        if self.num_actions == 2:
            # 当num_actions=2时，与原始版本完全一致
            self.previous_position = self.initial_action * self.max_holding_number
            self.position = self.initial_action * self.max_holding_number
        else:
            # 多仓位支持：动作范围[0, num_actions-1]映射到[0, max_holding]
            scale_factor = self.num_actions - 1 if self.num_actions > 1 else 1
            self.previous_position = (self.initial_action / scale_factor) * self.max_holding_number
            self.position = (self.initial_action / scale_factor) * self.max_holding_number
        return self.single_state, self.trend_state, self.clf_state.reshape(-1), {
            "previous_action": self.initial_action,
        }

    def step(self, action):
        # 📌 支持多仓位的动作到仓位映射
        if self.num_actions == 2:
            # 当num_actions=2时，与原始版本完全一致
            normlized_action = action
            position = self.max_holding_number * normlized_action
        else:
            # 多仓位支持：动作范围[0, num_actions-1]映射到[0, max_holding]
            scale_factor = self.num_actions - 1 if self.num_actions > 1 else 1
            normlized_action = action / scale_factor
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
        
        # 计算持仓价值
        previous_value = self.calculate_value(previous_price_information, self.previous_position)
        current_value = self.calculate_value(current_price_information, self.position)
        
        if previous_position >= position:
            # 卖出逻辑 (原始版本: >= 而不是 >)
            self.sell_size = previous_position - position
            cash = self.sell_size * previous_price_information['close'] * (1 - self.comission_fee)
            self.comission_fee_history.append(self.comission_fee * self.sell_size * previous_price_information['close'])
            self.sell_money_memory.append(cash)
            self.needed_money_memory.append(0)
            self.position = position
            
            # 📌 原始奖励函数：使用绝对收益而非收益率
            self.reward = current_value + cash - previous_value
            if previous_value == 0:
                return_rate = 0
            else:
                return_rate = (current_value + cash - previous_value) / previous_value
            self.return_rate = return_rate
            self.reward_history.append(self.reward)
            
        if previous_position < position:
            # 买入逻辑
            self.buy_size = position - previous_position
            needed_cash = self.buy_size * previous_price_information['close'] * (1 + self.comission_fee)
            self.comission_fee_history.append(self.comission_fee * self.buy_size * previous_price_information['close'])
            self.needed_money_memory.append(needed_cash)
            self.sell_money_memory.append(0)
            self.position = position
            
            # 📌 原始奖励函数：使用绝对收益而非收益率
            self.reward = current_value - needed_cash - previous_value
            return_rate = (current_value - needed_cash - previous_value) / (previous_value + needed_cash)
            self.reward_history.append(self.reward)
            self.return_rate = return_rate
            
        # 📌 更新前一个位置：原始版本在所有情况后都设置
        self.previous_position = self.position

        if self.terminal:
            return_margin, pure_balance, required_money, commission_fee = self.get_final_return_rate()
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
        if len(balance_list) > 0:
            required_money = -np.min(balance_list)
            if required_money <= 0:
                required_money = 1.0  # 避免除零
        else:
            required_money = 1.0  # 避免除零
        commission_fee = np.sum(self.comission_fee_history)
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
        num_actions=2,  # 📌 支持多仓位，默认为2（与原始版本一致）
    ):
        super(Training_Env,
              self).__init__(df, tech_indicator_list, tech_indicator_list_trend, clf_list, transcation_cost,
                             back_time_length, max_holding_number, initial_action, num_actions)
        
        # 📌 Q表配置：当num_actions=2时使用原始配置，否则适配多仓位
        if num_actions == 2:
            # 与原始版本完全一致的Q表配置，但使用1500的reward_scale
            self.q_table = make_q_table_reward(df,
                                               num_action=2,
                                               max_holding=max_holding_number,
                                               commission_fee=0.001,
                                               reward_scale=1500,
                                               gamma=0.99,
                                               max_punish=1e12)
        else:
            # 多仓位的Q表配置
            self.q_table = make_q_table_reward(df,
                                               num_action=num_actions,
                                               max_holding=max_holding_number,
                                               commission_fee=0.001,
                                               reward_scale=1500,
                                               gamma=0.99,
                                               max_punish=1e12)
        self.initial_action = initial_action


    def reset(self):
        single_state, trend_state, clf_state, info = super(Training_Env, self).reset()
        self.previous_action = self.initial_action
        # 📌 支持多仓位，但当num_actions=2时与原始版本一致
        if self.num_actions == 2:
            # 当num_actions=2时，与原始版本完全一致
            self.previous_position = self.initial_action * self.max_holding_number
            self.position = self.initial_action * self.max_holding_number
        else:
            # 多仓位支持
            scale_factor = self.num_actions - 1 if self.num_actions > 1 else 1
            self.previous_position = (self.initial_action / scale_factor) * self.max_holding_number
            self.position = (self.initial_action / scale_factor) * self.max_holding_number
        # 📌 Q表索引（原始版本）
        info['q_value'] = self.q_table[self.m - 1][self.previous_action][:]
        return single_state, trend_state, clf_state.reshape(-1), info

    def step(self, action):
        single_state, trend_state, clf_state, reward, done, info = super(Training_Env, self).step(action)
        # 📌 Q表索引（原始版本）
        info['q_value'] = self.q_table[self.m - 1][action][:]
        return single_state, trend_state, clf_state.reshape(-1), reward, done, info


