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
reward_scale = 1
commission_fee = 0.001

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
    ):
        self.tech_indicator_list = tech_indicator_list
        self.tech_indicator_list_trend = tech_indicator_list_trend
        self.df = df
        # For plotting
        self.trade_records = []
        self.value_history = []
        self.initial_action = initial_action
        self.action_space = spaces.Discrete(2)
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

        # For plotting (does not affect training logic)
        self.initial_money = 0.0
        self.cash_balance = 0.0



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
        price_information = self.data.iloc[-1]
        self.previous_position = self.initial_action * self.max_holding_number
        self.position = self.initial_action * self.max_holding_number
        self.needed_money_memory = []
        self.sell_money_memory = []
        self.comission_fee_history = []
        self.previous_position = self.initial_action * self.max_holding_number
        self.position = self.initial_action * self.max_holding_number
        self.needed_money_memory.append(self.position *
                                        self.data.iloc[-1]["close"])
        self.sell_money_memory.append(0)

        # For plotting: reset histories and record initial point
        self.trade_records = []
        self.value_history = []
        try:
            price_information = self.data.iloc[-1]
            # margin-style curve: start from 0 return
            self.value_history.append([price_information.get('timestamp', self.m - 1), 0.0])
        except Exception:
            pass

        return self.single_state, self.trend_state, {
            "previous_action": self.initial_action,
        }

    def step(self, action):
        normlized_action = action
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

            # For plotting: update cash balance on sell
            try:
                self.cash_balance += float(cash)
            except Exception:
                pass

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

            # For plotting: update cash balance on buy
            try:
                self.cash_balance -= float(needed_cash)
            except Exception:
                pass
        self.previous_position = self.position

        # create trade record when position changes (for plotting)
        try:
            if previous_position != position:
                trade_type = 'buy' if position > previous_position else 'sell'
                self.trade_records.append({
                    'type': trade_type,
                    'datetime': previous_price_information.get('timestamp', self.m - 1),
                    'price': float(previous_price_information['close']),
                    'position_before': float(previous_position),
                    'position_after': float(position),
                })
        except Exception:
            pass

        # record value history from margin(required_money) perspective
        try:
            sell_money_memory = np.array(self.sell_money_memory, dtype=float)
            needed_money_memory = np.array(self.needed_money_memory, dtype=float)
            true_money = sell_money_memory - needed_money_memory  # cashflow per step
            balance_list = np.cumsum(true_money)
            required_money = -float(np.min(balance_list)) if len(balance_list) > 0 else 0.0
            final_balance_so_far = float(balance_list[-1]) if len(balance_list) > 0 else 0.0

            # normalized performance on margin: pnl / required_money
            value = 0.0 if required_money == 0 else (final_balance_so_far / required_money)
            self.value_history.append([current_price_information.get('timestamp', self.m - 1), value])
        except Exception:
            pass

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
        commission_fee=commission_fee,
        reward_scale=reward_scale,
    ):
        super(Training_Env,
              self).__init__(df, tech_indicator_list, tech_indicator_list_trend, transcation_cost,
                             back_time_length, max_holding_number, initial_action=initial_action)
        self.q_table = make_q_table_reward(df,
                                           num_action=2,
                                           max_holding=max_holding_number,
                                           commission_fee=commission_fee,
                                           reward_scale=reward_scale,
                                           gamma=0.99,
                                           max_punish=1e12)
        self.initial_action = initial_action


    def reset(self):
        single_state, trend_state, info = super(Training_Env, self).reset()
        self.previous_action = self.initial_action
        self.previous_position = self.initial_action * self.max_holding_number
        self.position = self.initial_action * self.max_holding_number
        info['q_value'] = self.q_table[self.m - 1][self.previous_action][:]
        return single_state, trend_state, info

    def step(self, action):
        single_state, trend_state, reward, done, info = super(Training_Env, self).step(action)
        info['q_value'] = self.q_table[self.m - 1][action][:]
        return single_state, trend_state, reward, done, info