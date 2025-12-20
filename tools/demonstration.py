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
                        num_action,
                        max_holding,
                        reward_scale=1000,
                        gamma=0.999,
                        commission_fee=0.001,
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

    scale_factor = num_action - 1

    for t in range(2, len(df) + 1):
        current_price_information = df.iloc[-t]
        future_price_information = df.iloc[-t + 1]
        for previous_action in range(num_action):
            for current_action in range(num_action):
                if current_action > previous_action:
                    # Buy operation calculation
                    # 买入操作计算
                    previous_position = previous_action / (scale_factor) * max_holding
                    current_position = current_action / (scale_factor) * max_holding
                    position_change = (current_action-previous_action) / scale_factor*max_holding
                    buy_money = position_change * current_price_information['close'] * (1 + commission_fee)
                    current_value = calculate_value(current_price_information, previous_position)
                    future_value = calculate_value(future_price_information, current_position)
                    reward = future_value - (current_value + buy_money)
                    reward = reward_scale * reward
                    _q_value= reward + gamma * np.max(q_table[len(df) - t + 1][current_action][:])
                    q_table[len(df) - t][previous_action][current_action] = _q_value
                    # log.info(f"t={t}, previous_action_index={previous_action}, current_action_index={current_action}, total_reward={reward}, _q_value={_q_value}")
                else:
                     # Sell operation calculation
                    # 卖出操作计算
                    previous_position = previous_action / (scale_factor) * max_holding
                    current_position = current_action / (scale_factor) * max_holding
                    position_change = (previous_action-current_action) / scale_factor*max_holding
                    sell_money = position_change * current_price_information['close'] * (1 - commission_fee)
                    current_value = calculate_value(current_price_information, previous_position)
                    future_value = calculate_value(future_price_information, current_position)
                    reward = future_value + sell_money - current_value
                    reward = reward_scale * reward
                    _q_value = reward + gamma * np.max(q_table[len(df) - t + 1][current_action][:])
                    q_table[len(df) - t][previous_action][current_action] = _q_value
                    # log.info(f"t={t}, previous_action_index={previous_action}, current_action_index={current_action}, total_reward={reward}, _q_value={_q_value}")
    return q_table