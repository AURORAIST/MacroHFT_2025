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
import logging as log
ROOT = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(ROOT)
sys.path.insert(0, ".")
 
from tools.demonstration import make_q_table_reward

tech_indicator_list = np.load('./data/feature_list/single_features.npy', allow_pickle=True).tolist()
tech_indicator_list_trend = np.load('./data/feature_list/trend_features.npy', allow_pickle=True).tolist()



transcation_cost = 0.0002
back_time_length = 1
max_holding_number = 0.01  #最大持仓量
alpha = 0

# 功能特性：

# 动作空间：Discrete(2) 表示买入/卖出两个基础动作
# 观测空间：Box类型，包含：
        # 技术指标序列数据
        # 趋势特征数据
    # 交易机制：
        # 支持限价单逻辑
        # 包含手续费计算
        # 持仓管理（最大持仓限制）
# 奖励机制：基于资产价值变化和交易成本计算即时收益

# 关键设计模式
# 状态空间设计：
    # 使用时间序列窗口（back_time_length）维护历史数据
    # 分离技术指标和趋势特征，支持多维状态表示
# 交易成本处理：
    # 精确计算每笔交易的手续费
    # 区分买入和卖出的资金流向
# Q-learning集成：
    # 使用预计算的Q表提供动作价值预测
    # 在训练环境中动态注入Q值信息
# 风险调整评估：
    # 通过资金曲线计算最大回撤
    # 使用风险调整收益率作为核心评估指

class Testing_Env(gym.Env):
    """
    高阶强化学习交易环境
    
    功能说明：
    1. 基于历史金融数据实现交易策略模拟环境
    2. 支持基于技术指标和趋势分析的强化学习训练
    3. 提供完整的交易生命周期管理（建仓/平仓/持仓管理）
    4. 包含多维度状态观测系统（技术指标+趋势特征+分类特征）
    
    主要架构：
    - 动作空间：Discrete(2) 代表买入/卖出两个基础动作
    - 观察空间：Box类型，包含三个状态组件：
        * single_state: 技术指标序列数据
        * trend_state: 趋势特征数据
        * clf_state: 分类特征数据
    - 状态管理：通过stack_length维护时间序列窗口
    - 奖励机制：基于资产价值变化和交易成本计算即时收益
    - 交易系统：支持限价单逻辑，包含手续费计算和资金管理
    """
    def __init__(
        self,
        df: pd.DataFrame,
        tech_indicator_list=tech_indicator_list,
        tech_indicator_list_trend=tech_indicator_list_trend,
        transcation_cost=transcation_cost,  #交易手续费率
        back_time_length=back_time_length,  #状态回溯时间步长
        max_holding_number=max_holding_number,  #最大持仓量
        actions=[],
        action_mode="long",
        initial_action=0,
        reward_no_action=False
    ):
        # 初始化交易环境参数
        # df: 原始金融数据DataFrame
        # tech_indicator_list: 基础技术指标列表
        # tech_indicator_list_trend: 趋势相关指标列表
        # transcation_cost: 交易手续费率
        # back_time_length: 状态回溯时间长度
        # max_holding_number: 最大持仓数量
        # initial_action: 初始动作
        
        initial_action_detail = actions[initial_action]
        initial_long_action = initial_action_detail[0]
        initial_short_action = initial_action_detail[1]
        self.reward_no_action = reward_no_action
        # 定义动作空间和观测空间
        self.actions = actions
        self.tech_indicator_list = tech_indicator_list
        self.tech_indicator_list_trend = tech_indicator_list_trend
        self.df = df
        self.initial_long_action = initial_long_action
        self.initial_short_action = initial_short_action
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=+np.inf,
            shape=(back_time_length * len(self.tech_indicator_list), ))
        self.terminal = False
        self.stack_length = back_time_length
        self.m = back_time_length
        self.data = self.df.iloc[self.m - self.stack_length:self.m]
        self.single_state = self.data[self.tech_indicator_list].values  # 技术指标状态
        self.trend_state = self.data[self.tech_indicator_list_trend].values # 趋势特征状态
        self.initial_reward = 0
        self.reward_history = [self.initial_reward]
        self.previous_action = initial_action
        self.initial_action = initial_action
        self.comission_fee = transcation_cost
        self.max_holding_number = max_holding_number
        self.long_needed_money_memory = []
        self.long_sell_money_memory = []
        self.short_needed_money_memory = []
        self.short_sell_money_memory = []
        self.comission_fee_history = []
        # 初始化多头和空头仓位
        self.previous_long_position = initial_long_action * self.max_holding_number
        self.long_position = initial_long_action * self.max_holding_number
        self.previous_short_position = initial_short_action * self.max_holding_number
        self.short_position = initial_short_action * self.max_holding_number
        self.initial_long_action = initial_long_action
        self.initial_short_action = initial_short_action
        self.action_mode = action_mode
        self.step_times = 0
        self.trade_id_counter = 0  # 重置交易ID计数器
        self.trade_records = []  # 清空交易记录

        # === 资金曲线记录（用于验证/测试阶段画 money curve）===
        # 与 high_level_env.Testing_Env 的字段保持一致，供 graph_utils.plot_money_curve 使用。
        # initial_money/current_money 用于资金管理；current_value/value_history 用于绘制资产曲线。
        self.initial_money = self.data["open"].iloc[0] * self.max_holding_number
        self.current_money = self.initial_money
        self.current_value = self.current_money
        self.value_history = []


    def calculate_value(self, price_information, position):
        """计算当前持仓价值"""
        return price_information["close"] * position

    def reset(self):
        """
        环境重置方法
        重置交易环境到初始状态，初始化所有状态变量和交易记录
        
        返回:
            observation: 初始状态观测值，包含三个状态组件：
                - single_state: 技术指标序列数据 (形状: [back_time_length * len(tech_indicator_list)])
                - trend_state: 趋势特征数据 (形状: [back_time_length * len(tech_indicator_list_trend)])
            info: 附加信息字典，包含：
                - previous_action: 初始执行动作 (0/1)
        处理流程:
            1. 重置时间步和数据窗口
            2. 初始化交易状态（持仓、资金记录）
            3. 构建初始观测状态
            4. 返回初始观测值和动作信息
        """
        # 重置终止标志
        self.terminal = False
        # 重置时间步计数器到初始窗口长度
        self.m = back_time_length
        # 初始化市场数据窗口（从起始位置到当前步）
        self.data = self.df.iloc[self.m - self.stack_length:self.m]
        # 构建初始状态向量
        self.single_state = self.data[self.tech_indicator_list].values # 技术指标状态
        self.trend_state = self.data[self.tech_indicator_list_trend].values # 趋势特征状态
        # 重置奖励记录
        self.initial_reward = 0
        self.reward_history = [self.initial_reward]
        # 重置动作记录
        self.previous_action = self.initial_action
        # 获取当前价格信息（最新时间步）
        price_information = self.data.iloc[-1]
        # 重置交易状态
        # 添加交易记录成员变量
        self.trade_records = []  # 记录所有买卖记录
        self.trade_id_counter = 0  # 交易ID计数器
        # self.needed_money_memory = [] 
        # self.sell_money_memory = [] 
        
        self.long_needed_money_memory = [] # 清空买入资金记录
        self.long_sell_money_memory = [] # 清空卖出资金记录
        self.short_needed_money_memory = []
        self.short_sell_money_memory = []
        
        self.comission_fee_history = []  # 清空手续费记录
        # 设置初始持仓（根据初始动作参数）
        self.previous_long_position = self.initial_long_action * self.max_holding_number
        self.long_position = self.initial_long_action * self.max_holding_number
        self.previous_short_position = self.initial_short_action * self.max_holding_number
        self.short_position = self.initial_short_action * self.max_holding_number
        # 特殊处理：记录初始持仓所需资金
        if self.long_position>0:
            self.long_needed_money_memory.append(self.long_position * self.data.iloc[-1]["close"])
            self.long_sell_money_memory.append(0)
        
        if self.short_position>0:
            self.short_needed_money_memory.append(self.short_position * self.data.iloc[-1]["close"])
            self.short_sell_money_memory.append(0)
        
        
        self.step_times = 0
        self.trade_id_counter = 0  # 重置交易ID计数器
        self.trade_records = []  # 清空交易记录

        # === 重置资金曲线记录 ===
        self.initial_money = self.data["open"].iloc[0] * self.max_holding_number
        self.current_money = self.initial_money
        self.current_value = self.current_money
        self.value_history = []

        # 返回初始观测值和动作信息
        return self.single_state, self.trend_state, {
            "previous_action": self.initial_action,
        }

    def step(self, action):
        """
        执行一个交易步骤并更新环境状态
        
        参数:
            action: 离散动作 (0: 卖出, 1: 买入)
                - 0 表示卖出操作，将持仓量减少到目标值
                - 1 表示买入操作，将持仓量增加到目标值
        
        返回:
            observation: 新的状态观测值，包含三个状态组件：
                - single_state: 技术指标序列数据 (形状: [back_time_length * len(tech_indicator_list)])
                - trend_state: 趋势特征数据 (形状: [back_time_length * len(tech_indicator_list_trend)])
            reward: 即时奖励值（当前步骤的交易收益）
            done: 布尔值，表示当前回合是否结束
            info: 附加信息字典，包含：
                - previous_action: 当前执行的动作 (0/1)
                - q_value: 与当前状态相关的Q值（仅训练环境）
        
        处理流程:
            1. 根据动作计算目标持仓量
            2. 更新时间步和市场数据窗口
            3. 计算持仓变化带来的现金流动
            4. 更新交易历史记录（手续费、买卖资金等）
            5. 计算即时奖励（考虑交易成本）
            6. 检查是否达到终止条件（数据末尾）
        """
        
        # 将动作转换为实际持仓量
        action_detail = self.actions[action]
        long_action = action_detail[0]
        short_action = action_detail[1]
        normlized_long_action = long_action
        normlized_short_action = short_action
        long_position = self.max_holding_number * normlized_long_action  # 将离散动作(0/1)转换为实际持仓量（0或max_holding_number）
        short_position = self.max_holding_number * normlized_short_action  # 空头持仓量
        self.terminal = (self.m >= len(self.df.index.unique()) - 1) # 检查是否达到数据末尾（终止条件）
        # 保存当前状态信息
        previous_long_position = self.previous_long_position
        previous_short_position = self.previous_short_position
        previous_price_information = self.data.iloc[-1]
        # 更新时间步和数据窗口
        self.m += 1
        # 更新市场数据窗口（保持固定长度的时间序列）
        self.data = self.df.iloc[self.m - self.stack_length:self.m]
        current_price_information = self.data.iloc[-1]
        # 更新状态向量
        self.single_state = self.data[self.tech_indicator_list].values
        self.trend_state = self.data[self.tech_indicator_list_trend].values
        # 保存状态转移信息
        self.previous_long_position = previous_long_position
        self.long_position = long_position
        self.previous_short_position = previous_short_position
        self.short_position = short_position
        # self.changing = (self.long_position != self.previous_long_position)
        
        # 计算多头仓位变化带来的收益
        long_reward = 0
        short_reward = 0
        scale_factor = 1
        self.step_times += 1
        
        # === 资金曲线：本步的现金流（买入为负、卖出为正；空头开仓为负、平仓为正）===
        step_cash_flow = 0.0
        
        if previous_long_position < long_position:
            # 处理买入操作
            self.buy_size = long_position - previous_long_position # 计算买入数量
            # 计算买入所需资金（包含手续费）
            needed_cash = self.buy_size * previous_price_information['close'] * (1 + self.comission_fee)
            step_cash_flow -= float(needed_cash)
            self.comission_fee_history.append(self.comission_fee * self.buy_size * previous_price_information['close']) # 记录交易成本
            # 更新资金记录
            self.long_needed_money_memory.append(needed_cash)  # 买入支出
            self.long_sell_money_memory.append(0) # 卖出收入

            self.long_position = long_position
            # 计算持仓价值变化
            previous_long_value = self.calculate_value(previous_price_information, self.previous_long_position)
            current_long_value = self.calculate_value(current_price_information, self.long_position)
            # 买入奖励计算：当前价值 - 所需现金 - 上一时刻价值
            long_reward = current_long_value - (needed_cash + previous_long_value)
            # return_rate = (current_long_value - needed_cash - previous_long_value) / (previous_long_value + needed_cash)
            # # 保存指标
            # self.reward_history.append(self.reward)
            # self.return_rate = return_rate
            # 记录交易信息
            if self.buy_size > 0:  # 只有实际发生交易时才记录
                trade_record = {
                    'id': self.trade_id_counter,
                    'datetime': previous_price_information['timestamp'],  # 使用date作为交易时间
                    'amount': needed_cash,
                    'quantity': self.buy_size,
                    'direction': 'long',
                    'type': 'buy',                     
                    'price': previous_price_information['close']
                }
                self.trade_records.append(trade_record)
                self.trade_id_counter += 1
        elif long_position == 0 and self.previous_long_position ==0 :
            #什么都不干，并且没有仓位， 就需要惩罚下一天可能的收益             
            long_reward=0 
        # 处理卖出操作
   
        else:
            # previous_long_position >= long_position:
            self.sell_size = previous_long_position - long_position
            # 计算卖出收入（扣除手续费）
            cash = self.sell_size * previous_price_information['close'] * (1 - self.comission_fee)
            step_cash_flow += float(cash)
            self.comission_fee_history.append(self.comission_fee * self.sell_size * previous_price_information['close']) # 记录交易成本
            # 更新资金记录
            self.long_sell_money_memory.append(cash) # 卖出收入
            self.long_needed_money_memory.append(0) # 买入支出
            self.long_position = long_position
            # 记录交易信息
            if self.sell_size > 0:  # 只有实际发生交易时才记录
                trade_record = {
                    'id': self.trade_id_counter,
                    'datetime': previous_price_information['timestamp'],  # 使用date作为交易时间
                    'amount': cash,
                    'quantity': self.sell_size,
                    'direction': 'long',
                    'type': 'sell',                     
                    'price': previous_price_information['close']
                }
                self.trade_records.append(trade_record)
                self.trade_id_counter += 1
            
            # 计算持仓价值变化
            previous_long_value = self.calculate_value(previous_price_information, self.previous_long_position)
            current_long_value = self.calculate_value(current_price_information, self.long_position)       
            # if self.reward_no_action: 
            #     if self.long_position ==0:
            #     # 需要惩罚， 清仓后下一天可能的收益
            #     # 惩罚下一天可能的收益 9-10 跌1 奖励+1    11-10 涨1  惩罚-1
            #         # long_reward = cash - previous_long_value  - ((current_price_information['close']-previous_price_information['close'])*self.previous_long_position)
            #         long_reward = 0
            #     else:
            #         long_reward = (current_long_value + cash) - previous_long_value
            # else:
            if self.action_mode == 'both' and previous_long_position > long_position and not (self.short_position == 0 and self.previous_short_position ==0):
                # 多转空                
                long_reward=-self.comission_fee * self.sell_size * previous_price_information['close']        
            else:
                #多方持有或者多方平仓
                long_reward = (current_long_value + cash) - previous_long_value
                                    
            # # 卖出奖励计算：当前价值 + 现金流入 - 上一时刻价值
            # self.reward = current_long_value + cash - previous_long_value
            # # 牛市有效（猜测，原论文没有这个部分）
            # # if previous_position==0 and position==0:
            # #     self.reward = -previous_price_information['close']*0.01*self.max_holding_number
                
            # if previous_long_value == 0:
            #     return_rate = 0
            # else:
            #     return_rate = (current_long_value + cash - previous_long_value) / previous_long_value
            # # 保存指标
            # self.return_rate = return_rate
            # self.reward_history.append(self.reward)
 
            
        # 处理空头仓位变化
        if previous_short_position < short_position:
            # 空头加仓（开仓）
            open_size = short_position - previous_short_position        
            # 开空头仓位：卖出获得现金，但需要支付保证金和手续费
            cash_value = open_size * previous_price_information['close']
            cash_out = cash_value * (1 + self.comission_fee)
            step_cash_flow -= float(cash_out)
            commission_fee_amount = self.comission_fee * open_size * previous_price_information['close']
            self.comission_fee_history.append(commission_fee_amount)
            self.short_sell_money_memory.append(0)
            self.short_needed_money_memory.append(cash_out)
            # 记录交易信息
            if open_size > 0:  # 只有实际发生交易时才记录
                trade_record = {
                    'id': self.trade_id_counter,
                    'datetime': previous_price_information['timestamp'],  # 使用date作为交易时间
                    'amount': cash_out,
                    'quantity': open_size,
                    'direction': 'short',
                    'type': 'sell',                     
                    'price': previous_price_information['close']
                }
                self.trade_records.append(trade_record)
                self.trade_id_counter += 1
            # 空头开仓时获得现金，但需要承担未来平仓的风险
            # 计算上一刻和这一刻的仓位价值
            previous_short_value = self.calculate_value(previous_price_information, self.previous_short_position)
            current_short_value = self.calculate_value(current_price_information, self.short_position)
            # 收益 = 上一刻仓位价值 + 开仓的价值 - 手续费- 这下一刻仓位价值       
            short_reward = previous_short_value + (cash_value - commission_fee_amount) - current_short_value         
        
        elif self.short_position == 0 and self.previous_short_position ==0 : 
            short_reward=0 
        
        else:
            # elif previous_short_position >= short_position:
            # 空头减仓（平仓）
            close_size = previous_short_position - short_position            
            # 平空头仓位：支付平仓价格，获得开仓时的收益
            cash_value = close_size * previous_price_information['close']
            cash_in = cash_value * (1 - self.comission_fee)
            step_cash_flow += float(cash_in)
            commission_fee_amount = self.comission_fee * close_size * previous_price_information['close']
            self.comission_fee_history.append(commission_fee_amount)
            self.short_sell_money_memory.append(cash_in)
            self.short_needed_money_memory.append(0)   
            # 记录交易信息
            if close_size > 0:  # 只有实际发生交易时才记录
                trade_record = {
                    'id': self.trade_id_counter,
                    'datetime': previous_price_information['timestamp'],  # 使用date作为交易时间
                    'amount': cash_in,
                    'quantity': close_size,
                    'direction': 'short',
                    'type': 'buy',                     
                    'price': previous_price_information['close']
                }
                self.trade_records.append(trade_record)
                self.trade_id_counter += 1         
            # 空头收益计算：开仓时卖出，平仓时买入
            # 计算上一刻和这一刻的仓位价值
            previous_short_value = self.calculate_value(previous_price_information, self.previous_short_position)
            current_short_value = self.calculate_value(current_price_information, self.short_position)
            #if self.short_position == 0:
                # 没有未来没有仓位
                # 收益 = 上一刻仓位价值 - 现金流入 - 费用 + 未来的（假设）价差
                # 惩罚下一天可能的收益 10-9 跌1 奖励1    10-11 涨1  惩罚-1
            #    short_reward =  previous_short_value - (cash_value+commission_fee_amount) + ((previous_price_information['close']-current_price_information['close'])*self.previous_short_position)
            #else:
                # 收益 = 上一刻仓位价值 - 现金流入 - 费用 - 下一刻仓位价值 
            if self.action_mode == 'both' and previous_short_position > short_position and not (self.long_position == 0 and self.previous_long_position ==0):
                # 空转多             
                short_reward=-self.comission_fee * close_size * previous_price_information['close']
            else:
                short_reward =  previous_short_value - (cash_value+commission_fee_amount) - current_short_value 

        
        if long_position == 0 and self.previous_long_position ==0 and short_position == 0 and self.previous_short_position ==0:
            # self.reward = (abs(previous_price_information['close']-current_price_information['close'])*self.max_holding_number)*-0.5/scale_factor
            self.reward = 0
        else:   
            # 计算总收益
            self.reward = (long_reward + short_reward)/scale_factor
        self.reward_history.append(self.reward)

        # === 更新资金/资产曲线（用于画 money curve）===
        # 说明：不试图完全复现 get_final_return_rate 的资金占用逻辑，只提供一致可画的资产价值曲线。
        # current_money: 现金余额（按交易现金流更新）
        # current_value: 现金 + 当前持仓按 close 计价的市值
        try:
            self.current_money = float(self.current_money) + float(step_cash_flow)
            long_value_mark = float(self.calculate_value(current_price_information, self.long_position))
            short_value_mark = float(self.calculate_value(current_price_information, self.short_position))
            self.current_value = float(self.current_money) + long_value_mark + short_value_mark
            ts = current_price_information.get('timestamp', self.m)
            self.value_history.append([ts, float(self.current_value)])
        except Exception:
            # 绘图不应影响训练/验证主流程
            pass
        
        # 更新持仓记录
        self.previous_long_position = self.long_position
        self.previous_short_position = self.short_position       


        if self.terminal:
            # 终止时平掉所有仓位
            if self.long_position > 0:
                self.sell_size = self.long_position
                cash = self.sell_size * previous_price_information['close'] * (1 - self.comission_fee)
                # 终止清仓也计入资金曲线
                try:
                    self.current_money = float(self.current_money) + float(cash)
                except Exception:
                    pass
                commission_fee_amount = self.comission_fee * self.sell_size * previous_price_information['close']
                self.comission_fee_history.append(commission_fee_amount)
                self.long_sell_money_memory.append(cash)
                self.long_needed_money_memory.append(0)
                self.long_position = 0
            if self.short_position > 0:
                close_size = self.short_position
                cash_in = close_size * previous_price_information['close'] * (1 - self.comission_fee)
                # 终止清仓也计入资金曲线
                try:
                    self.current_money = float(self.current_money) + float(cash_in)
                except Exception:
                    pass
                commission_fee_amount = self.comission_fee * close_size * previous_price_information['close']
                self.comission_fee_history.append(commission_fee_amount)
                self.short_sell_money_memory.append(cash_in)
                self.short_needed_money_memory.append(0)
                self.short_position = 0
            # 终止时再追加一个最终资产点（timestamp 用上一个 price_information，保持与交易记录一致）
            try:
                self.current_value = float(self.current_money)
                ts_final = previous_price_information.get('timestamp', self.m)
                self.value_history.append([ts_final, float(self.current_value)])
            except Exception:
                pass
            # 终止时计算最终收益
            return_margin, final_balance, required_money, commission_fee = self.get_final_return_rate()
            self.pured_balance = final_balance
            self.final_balance = final_balance
            self.required_money = required_money
            portfit_margine = self.final_balance / self.required_money
            # - return_margin: 风险调整收益率（核心评估指标）
            # - final_balance: 绝对收益值
            # - required_money: 最大资金需求（资金曲线最低点绝对值）
            # - commission_fee: 累计交易手续费
            # log.info(f"the portfit return_margin:{return_margin},final_balance:{final_balance},required_money:{required_money},commission_fee:{commission_fee},final_balance:{self.final_balance},portfit_margine:{portfit_margine}")
            
        # 返回观测值和环境状态
        return self.single_state, self.trend_state, self.reward, self.terminal, {
            "previous_action": action,
        }

    def get_final_return_rate(self, slient=False):
        """
        计算交易策略的最终收益指标（包含风险调整后的收益率）
        
        参数:
            slient: 静默模式标志（当前未使用）
            
        返回:
            tuple: 包含4个指标的元组
                - relative_return: 风险调整收益率（核心评估指标）
                - final_balance: 绝对收益值
                - required_money: 最大资金需求（资金曲线最低点绝对值）
                - commission_fee: 累计交易手续费
        """
        # 将卖出记录和买入记录转换为numpy数组
        long_sell_money_memory = np.array(self.long_sell_money_memory)
        long_needed_money_memory = np.array(self.long_needed_money_memory)   
        # 计算每笔交易的真实收益（卖出收入 - 买入支出）
        long_true_money = long_sell_money_memory - long_needed_money_memory
         
        short_sell_money_memory = np.array(self.short_sell_money_memory)
        short_needed_money_memory = np.array(self.short_needed_money_memory)   
        # 计算每笔交易的真实收益（卖出收入 - 买入支出）
        short_true_money =  short_needed_money_memory -short_sell_money_memory
 
        # 计算总净收益
        
        final_balance = np.sum(long_true_money) + (np.sum(short_true_money))
        
        true_money = np.concatenate((long_true_money, short_true_money))
        balance_list = np.cumsum(true_money)
        # balance_list = []
        # 创建资金曲线（余额变化序列），用于分析资金波动情况
        # for i in range(len(true_money)):
        #     # 累计计算每个时间点的余额
        #     balance_list.append(np.sum(true_money[:i + 1]))
        # 计算最大资金需求（历史最低余额的绝对值） （风险度量指标）
        required_money = 0.01  # 默认最小值，避免除零
        if len(balance_list) > 0:
            min_balance = np.min(balance_list)
            # 修复bug：当最小值为正数时，required_money会变成负数
            # 使用max确保required_money始终为正
            required_money = max(0.01, -min_balance)
        # 计算总手续费（注意：字段名存在拼写错误 comission -> commission）
        commission_fee = np.sum(self.comission_fee_history)
        # 返回相对收益率、净收益、最大资金需求、总手续费
        # 确保required_money > 0以避免除零错误
        if required_money > 0:
            return final_balance / required_money, final_balance, required_money, commission_fee
        else:
            return 0.0, final_balance, 0.01, commission_fee

q_table_dict = {}

class Training_Env(Testing_Env):
    """
    训练专用交易环境类，继承自测试环境 Testing_Env
    
    功能扩展：
    1. 集成 Q 表支持强化学习训练
    2. 在状态信息中附加 Q 值数据
    3. 支持基于 Q 学习的策略优化
    
    核心特性：
    - 动作空间：Discrete(2) 买入/卖出
    - 状态观测：技术指标 + 趋势特征 + Q 值
    - 训练机制：基于 Q 表的奖励预测
    """
    def __init__(
        self,
        df_path,
        df: pd.DataFrame,
        tech_indicator_list=tech_indicator_list,
        tech_indicator_list_trend=tech_indicator_list_trend,
        transcation_cost=transcation_cost,
        back_time_length=back_time_length,
        max_holding_number=max_holding_number,
        num_action=2,  
        actions = [],   
        action_mode="long",   
        initial_action =0,
        reward_no_action=False,
       
    ):
        """
        训练环境初始化
        
        参数：
            df: 输入的历史金融数据 DataFrame
            tech_indicator_list: 技术指标列名列表
            tech_indicator_list_trend: 趋势特征列名列表
            transcation_cost: 交易手续费率
            back_time_length: 状态回溯时间步长
            max_holding_number: 最大持仓量
            initial_long_action: 初始动作（0-空仓，1-满仓）
            alpha: Q 表奖励衰减系数（未使用）
        """
        super(Training_Env,
              self).__init__(df, tech_indicator_list, tech_indicator_list_trend, transcation_cost,
                             back_time_length, max_holding_number, actions, action_mode, initial_action,reward_no_action)
        if q_table_dict.get(df_path,None) is None:            
            # 构建 Q 表（用于强化学习策略优化）
            q_table_dict[df_path] = make_q_table_reward(df,
                                            actions=actions,
                                            num_action=num_action,
                                            max_holding=max_holding_number,
                                            commission_fee=0.001,
                                            action_mode=action_mode,
                                            reward_scale=1,
                                            gamma=0.99,
                                            max_punish=1e12)
        self.q_table = q_table_dict[df_path]
        # 记录初始动作
        self.initial_action = initial_action


    def reset(self):
        """
        环境重置（带 Q 值初始化）
        
        返回：
            observation: 初始状态观测值
            info: 附加信息（包含 Q 值）
        """
        single_state, trend_state, info = super(Training_Env, self).reset()
        # 初始化交易状态
        
        self.previous_long_position = self.initial_long_action * self.max_holding_number
        self.long_position = self.initial_long_action * self.max_holding_number
        self.previous_short_position = self.initial_short_action * self.max_holding_number
        self.short_position = self.initial_short_action * self.max_holding_number         
        
        # 从 Q 表获取初始 Q 值
        info['q_value'] = self.q_table[self.m - 1][self.initial_action][:]
        return single_state, trend_state, info

    def step(self, action):
        """
        执行训练步骤（带 Q 值更新）
        
        参数：
            action: 当前动作（0-卖出，1-买入）
            
        返回：
            observation: 新的状态观测值
            reward: 即时奖励
            done: 是否终止
            info: 附加信息（包含当前 Q 值）
        """
        # 执行基础环境步骤
        single_state, trend_state, reward, done, info = super(Training_Env, self).step(action)
        # 更新 Q 值信息（基于当前时间步和动作）
        info['q_value'] = self.q_table[self.m - 1][action][:]
        return single_state, trend_state, reward, done, info


