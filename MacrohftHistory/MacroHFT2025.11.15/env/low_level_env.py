
from logging import raiseExceptions
import numpy as np
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



transcation_cost = 0.0002
back_time_length = 1
max_holding_number = 0.01
alpha = 0
num_actions = 5  # 默认支持5个仓位级别：0%, 25%, 50%, 75%, 100%


class Testing_Env(gym.Env):
    def __init__(
        self,
        df: pd.DataFrame,
        tech_indicator_list=tech_indicator_list,
        tech_indicator_list_trend=tech_indicator_list_trend,
        transcation_cost=transcation_cost,
        back_time_length=back_time_length,
        max_holding_number=max_holding_number,
        initial_action=0,
        num_actions=num_actions,
    ):
        self.tech_indicator_list = tech_indicator_list
        self.tech_indicator_list_trend = tech_indicator_list_trend
        self.df = df
        self.num_actions = num_actions
        # 验证 initial_action 在有效范围内
        assert 0 <= initial_action < num_actions, \
            f"initial_action ({initial_action}) must be in range [0, {num_actions-1}]"
        self.initial_action = initial_action
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
        self.previous_action = initial_action
        self.comission_fee = transcation_cost
        self.max_holding_number = max_holding_number
        self.needed_money_memory = []
        self.sell_money_memory = []
        self.comission_fee_history = []
        self.previous_position = initial_action * self.max_holding_number
        self.position = initial_action * self.max_holding_number
        self.initial_action = initial_action



    def calculate_value(self, price_information, position):
        return price_information["close"] * position

    def reset(self):
        self.terminal = False
        self.m = back_time_length
        self.data = self.df.iloc[self.m - self.stack_length:self.m]
        self.single_state = self.data[self.tech_indicator_list].values
        self.trend_state = self.data[self.tech_indicator_list_trend].values
        self.initial_reward = 0
        self.reward_history = [self.initial_reward]
        self.previous_action = self.initial_action
        # 修复：使用与step()一致的仓位计算方式
        scale_factor = self.num_actions - 1
        normalized_initial_action = self.initial_action / scale_factor if scale_factor > 0 else 0
        initial_position = self.max_holding_number * normalized_initial_action
        self.previous_position = initial_position
        self.position = initial_position
        self.needed_money_memory = []
        self.sell_money_memory = []
        self.comission_fee_history = []
        self.needed_money_memory.append(self.position *
                                        self.data.iloc[-1]["close"])
        self.sell_money_memory.append(0)
        return self.single_state, self.trend_state, {
            "previous_action": self.initial_action,
        }

    def step(self, action):
        # 将离散动作转换为仓位比例 (0 到 num_actions-1 -> 0.0 到 1.0)
        scale_factor = self.num_actions - 1
        normlized_action = action / scale_factor if scale_factor > 0 else 0
        position = self.max_holding_number * normlized_action
        self.terminal = (self.m >= len(self.df.index.unique()) - 1)
        previous_position = self.previous_position
        previous_price_information = self.data.iloc[-1]
        self.m += 1
        self.data = self.df.iloc[self.m - self.stack_length:self.m]
        current_price_information = self.data.iloc[-1]
        self.single_state = self.data[self.tech_indicator_list].values
        self.trend_state = self.data[self.tech_indicator_list_trend].values
        self.previous_position = previous_position
        self.position = position
        self.changing = (self.position != self.previous_position)
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

        return self.single_state, self.trend_state, self.reward, self.terminal, {
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
        required_money = -np.min(balance_list)
        commission_fee = np.sum(self.comission_fee_history)
        return final_balance / required_money, final_balance, required_money, commission_fee


class Training_Env(Testing_Env):
    def __init__(
        self,
        df: pd.DataFrame,
        tech_indicator_list=tech_indicator_list,
        tech_indicator_list_trend=tech_indicator_list_trend,
        transcation_cost=transcation_cost,
        back_time_length=back_time_length,
        max_holding_number=max_holding_number,
        initial_action = 0,
        alpha=alpha,
        num_actions=num_actions,
        reward_scale=1000, 
    ):
        super(Training_Env,
              self).__init__(df, tech_indicator_list, tech_indicator_list_trend, transcation_cost,
                             back_time_length, max_holding_number, initial_action, num_actions)
        self.q_table = make_q_table_reward(df,
                                           num_action=num_actions,
                                           max_holding=max_holding_number,
                                           commission_fee=0.001,
                                           reward_scale=reward_scale,
                                           gamma=0.99,
                                           max_punish=1e12)
        self.initial_action = initial_action


    def reset(self):
        single_state, trend_state, info = super(Training_Env, self).reset()
        # previous_position 和 position 已经在父类的 reset() 中正确设置
        # 这里只需要设置 previous_action
        self.previous_action = self.initial_action
        # 安全地访问Q表，避免越界
        if self.m - 1 < len(self.q_table):
            info['q_value'] = self.q_table[self.m - 1][self.previous_action][:]
        else:
            info['q_value'] = np.zeros(self.num_actions)
        return single_state, trend_state, info

    def step(self, action):
        single_state, trend_state, reward, done, info = super(Training_Env, self).step(action)
        # 安全地访问Q表，避免越界
        if self.m - 1 < len(self.q_table):
            info['q_value'] = self.q_table[self.m - 1][action][:]
        else:
            info['q_value'] = np.zeros(self.num_actions)
        return single_state, trend_state, reward, done, info
