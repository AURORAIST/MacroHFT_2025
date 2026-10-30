import numpy as np
import pandas as pd
import logging as log
# 该文件实现了基于强化学习的Q-table奖励值生成模块，主要用于金融交易场景中的决策优化。以下是详细架构分析：

# 1. 核心功能：
# 构建三维Q-table奖励矩阵，用于强化学习智能体的动作价值评估
# 支持双向交易（买入/卖出）的收益计算
# 集成交易手续费、时间折扣因子等金融场景要素
# 2. 主要组件架构：
# [1] 输入参数层：

# df: 带时间序列的金融数据（含open/high/low/close价格）
# num_action: 离散动作空间大小（0-N的持仓等级）
# max_holding: 最大持仓上限（物理持仓限制）
# reward_scale: 奖励放大系数（解决数值稳定性问题）
# gamma: 时序差分折扣因子（强化学习核心参数）
# commission_fee: 交易手续费率（真实交易环境模拟）
# [2] 内部计算模块： ① 价值计算器(calculate_value)： 输入：价格信息 + 持仓量 → 输出：当前持仓市值 公式：市值 = 当前收盘价 × 持仓量

# ② 双向交易奖励计算器：

# 买入场景(current_action > previous_action) 计算持仓增加带来的收益变化 包含：买入成本(含手续费)、持仓增值收益
# 卖出场景(current_action ≤ previous_action) 计算持仓减少带来的收益变化 包含：卖出收益(扣除手续费)、持仓减值损失
# [3] 时序传播机制： 采用动态规划思想进行逆序计算： q_table[t] = 即时奖励 + γ × max(q_table[t+1])

# 3. 输出结构： 返回三维numpy数组： dim[时间步][前一动作][当前动作] → 奖励值

# 4. 应用场景： 该模块可作为强化学习环境的一部分，为智能体提供：

# 动作选择的即时反馈
# 未来收益的预期评估
# 交易成本的真实模拟
# 5. 扩展方向：
# 可扩展滑点计算模块
# 可增加风险控制因子

def make_q_table_reward(df: pd.DataFrame,
                        actions,
                        num_action,
                        max_holding,
                        reward_scale=1000,
                        gamma=0.999,
                        commission_fee=0.001,
                        action_mode="long",
                        reward_no_action=False,
                        max_punish=1e12):
    """
    Generate Q-table rewards with bilingual comments
    生成Q表格奖励值（双语注释）
    
    Parameters 参数说明:
    df - Price data DataFrame (价格数据DataFrame)
    num_action - Number of possible actions (可选动作数量)
    max_holding - Maximum holding position (最大持仓量)
    reward_scale - Reward scaling factor (奖励缩放系数)
    gamma - Discount factor (折扣因子)
    commission_fee - Transaction cost rate (交易手续费率)
    max_punish - Maximum penalty value (最大惩罚值)
    """
    q_table = np.zeros((len(df), num_action, num_action))

     # Calculate value based on price information and position
    # 根据价格信息和持仓计算价值
    def calculate_value(price_information, position):
        return price_information["close"] * position

    # # Calculate short position value (negative value for short positions)
    # # 计算空头持仓价值（空头持仓为负值）
    # def calculate_short_value(price_information, position):
    #     return -price_information["close"] * position

    # scale_factor = num_action - 1
    # if action_mode == "both":
    #     scale_factor = (num_action - 1)/2
    # else:
    #     scale_factor = num_action - 1
    scale_factor = 1

    for t in range(2, len(df) + 1):
        current_price_information = df.iloc[-t]
        future_price_information = df.iloc[-t + 1]
        for previous_action_index in range(num_action):
            previous_action = actions[previous_action_index]
            for current_action_index in range(num_action):
                current_action = actions[current_action_index]
                
                # 提取多头和空头持仓
                previous_long_action = previous_action[0]
                previous_short_action = previous_action[1]
                current_long_action = current_action[0]
                current_short_action = current_action[1]
                
                # 计算多头持仓奖励
                long_reward = 0
                if current_long_action > previous_long_action:
                    # 多头买入操作计算
                    previous_long_position = previous_long_action / scale_factor * max_holding
                    current_long_position = current_long_action / scale_factor * max_holding
                    long_position_change = (current_long_action - previous_long_action) / scale_factor * max_holding
                    long_buy_money = long_position_change * current_price_information['close'] * (1 + commission_fee)
                    current_long_value = calculate_value(current_price_information, previous_long_position)
                    future_long_value = calculate_value(future_price_information, current_long_position)
                    long_reward = future_long_value - (current_long_value + long_buy_money)
                elif current_long_action ==0 and previous_long_action ==0:
                    # if reward_no_action:
                    #     long_reward = (future_price_information['close']-current_price_information['close'])*max_holding*-1 
                    # else:                        
                    long_reward = 0
                
                else:
                    # 多头卖出操作计算
                    previous_long_position = previous_long_action / scale_factor * max_holding
                    current_long_position = current_long_action / scale_factor * max_holding
                    long_position_change = (previous_long_action - current_long_action) / scale_factor * max_holding
                    long_sell_money = long_position_change * current_price_information['close'] * (1 - commission_fee)
                    current_long_value = calculate_value(current_price_information, previous_long_position)
                    future_long_value = calculate_value(future_price_information, current_long_position)
                    # if reward_no_action:
                    #     if current_long_position ==0:
                    # #     # 需要惩罚， 清仓后下一天可能的收益
                    # #     # 惩罚下一天可能的收益 明天-今天价格  9-10 跌1 奖励+1 (应该清仓)   11-10 涨1  惩罚-1 ， (不应该清仓)
                    #         # long_reward = long_sell_money - current_long_value - ((future_price_information['close']-current_price_information['close'])*previous_long_position)
                    #         long_reward = 0
                    #     else:
                    #         long_reward = future_long_value + long_sell_money - current_long_value
                    # else:
                    if action_mode == 'both' and previous_long_action > current_long_action and not (current_short_action ==0 and previous_short_action ==0):
                        long_reward = long_position_change * current_price_information['close'] * ( -commission_fee) 
                    else:
                        long_reward = future_long_value + long_sell_money - current_long_value
                
                # 计算空头持仓奖励
                short_reward = 0
                if current_short_action > previous_short_action:
                    # 空头开仓操作计算（相当于卖出）
                    previous_short_position = previous_short_action / scale_factor * max_holding
                    current_short_position = current_short_action / scale_factor * max_holding
                    short_position_change = (current_short_action - previous_short_action) / scale_factor * max_holding
                    short_open_money = short_position_change * current_price_information['close'] * (1 - commission_fee)  # 开仓收钱
                    commission_fee_momery = short_position_change * current_price_information['close'] * commission_fee
                    current_short_value = calculate_value(current_price_information, previous_short_position)
                    future_short_value = calculate_value(future_price_information, current_short_position)
                    # short_reward = future_short_value + short_open_money - current_short_value
                    short_reward = (current_short_value + short_open_money) - future_short_value
                    # 0 +9.8 -10*1 ping = -0.2
                    # 0 +9.8 -11*1 zhang = -1.2
                    # 0 +9.8 -9*1 die = 0.8
                elif current_short_action ==0 and previous_short_action ==0 :                         
                    short_reward = 0

                else:
                    # 空头平仓操作计算（相当于买入）
                    previous_short_position = previous_short_action / scale_factor * max_holding
                    current_short_position = current_short_action / scale_factor * max_holding
                    short_position_change = (previous_short_action - current_short_action) / scale_factor * max_holding
                    short_close_money = short_position_change * current_price_information['close'] * (1 + commission_fee)  # 平仓付钱
                    commission_fee_momery = short_position_change * current_price_information['close'] * commission_fee
                    current_short_value = calculate_value(current_price_information, previous_short_position)
                    future_short_value = calculate_value(future_price_information, current_short_position)
                    # if current_short_position == 0 :
                    #     # 特殊设计：当空头平仓时，计算的是当前价格与未来价格的差值，而不是当前价格与当前价格的差值
                    #     # 收益 = 上一刻仓位价值 - 现金流出 - 费用 + 未来的（假设）价差
                    #     # 惩罚下一天可能的收益 今天-明天价格 10-9 跌1 奖励+1 (不应该清仓)  10-11 涨1  惩罚-1  (应该清仓) 
                    #     short_reward =  current_short_value - short_close_money + ((current_price_information['close']-future_price_information['close'])*previous_short_position)
                    # else:  
                    #     # 收益 = 上一刻仓位价值 - 现金流入 - 费用 - 下一刻仓位价值 
                    if action_mode == 'both' and previous_short_action > current_short_action and not (current_long_action ==0 and previous_long_action ==0):                    
                        short_reward=short_position_change * current_price_information['close'] * (-commission_fee) 
                    else:
                        short_reward =  current_short_value - short_close_money - future_short_value
                    
                    #previous_short_position = 1  and current_short_position=0
                    # 10*1 -10.2 +(10-10)*1 ping = -0.2
                    # 10*1 -10.2 +(10-11)*1 zhang = -1.2
                    # 10*1 -10.2 +(10-9)*1 die = 0.8
                    #previous_short_position = 2  and current_short_position=0
                    # 10*2 -20.2 +(10-10)*2 ping = -0.2
                    # 10*2 -20.2 +(10-11)*2 zhang = -2.2
                    # 10*2 -20.2 +(10-9)*2 die = 1.8
                    #previous_short_position = 2  and current_short_position=1
                    # 10*2 -10.2 -10*1 ping = -0.2
                    # 10*2 -10.2 -11*1 zhang = -1.2
                    # 10*2 -10.2 -9*1 die = 0.8
                if current_short_action ==0 and previous_short_action ==0 and current_long_action ==0 and previous_long_action ==0 and action_mode == "both":
                    total_reward = 0
                    # abs(current_price_information['close']-future_price_information['close'])*max_holding *-0.5 / scale_factor
                    # total_reward = current_price_information['close']*-0.001
                else:   
                    # 计算总收益             
                    total_reward = long_reward + short_reward
                total_reward = reward_scale * total_reward
                _q_value = total_reward + gamma * np.max(q_table[len(df) - t + 1][current_action_index][:])
                q_table[len(df) - t][previous_action_index][current_action_index] = _q_value
                #log.info(f"t={t},current_price={current_price_information['close']}, future_price={future_price_information['close']}, previous_action={previous_action_index}, current_action={current_action_index}, long_reward={long_reward:.4f},short_reward={short_reward:.4f}, total_reward={total_reward:.4f}, _q_value={_q_value:.4f}")
    return q_table