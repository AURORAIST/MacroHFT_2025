import numpy as np
import pandas as pd


def make_low_q_table_reward(df: pd.DataFrame,
                        num_action,
                        max_holding,
                        reward_scale=1000,
                        gamma=0.999,
                        commission_fee=0.002,
                        max_punish=1e12):
    """
    使用动态规划构建Q表（倒序遍历，允许看未来一步）

    设计要点：
    - 即时回报使用 t->t+1 的未来价格，体现"专家"能看到未来一步的优势；
    - 买/卖/持有三种情况统一为"权益（持仓价值+现金流）"的相对变化；
    - Q[t, prev, curr] = r(t, prev->curr) + gamma * max_a Q[t+1, curr, a]；
    - t 遍历范围为 [0..T-2]，T-1 无未来价，保持为 0；
    - 数值稳定：分母加 eps，scale_factor 保护；
    - 奖励缩放 reward_scale 默认 100，与环境 reward 百分比*100 对齐。

    Args:
        df: 必须包含收盘价列 'close' 或 'Close'
        num_action: 动作数量（仓位级别数）
        max_holding: 最大持仓量（单位）
        reward_scale: 奖励缩放因子（默认 100）
        gamma: 折扣因子（默认 0.999）
        commission_fee: 手续费率（双边费）
        max_punish: 保留参数（未直接使用）

    Returns:
        q_table: 形状为 (T, num_action, num_action) 的Q表，
                 q_table[t][prev_action][curr_action] 表示在时刻 t 从 prev_action 转到 curr_action 的价值。
    """
    if len(df) < 2:
        raise ValueError("DF length must be >= 2 for future-aware reward")

    # 价格列兼容
    if 'close' in df.columns:
        prices = df['close'].to_numpy(dtype=float)
    elif 'Close' in df.columns:
        prices = df['Close'].to_numpy(dtype=float)
    else:
        raise KeyError("DataFrame must contain 'close' or 'Close' column")

    T = len(prices)
    q_table = np.zeros((T, num_action, num_action), dtype=np.float32)

    # 位置与数值稳定项
    scale_factor = max(1, num_action - 1)
    eps = 1e-12

    def pos_from_action(a: int) -> float:
        return (a / scale_factor) * max_holding

    # 即时回报（t -> t+1）的统一计算：基于权益（持仓价值 + 当期现金流）变化
    def immediate_return(prev_a: int, curr_a: int, p_t: float, p_tp1: float) -> float:
        prev_pos = pos_from_action(prev_a)
        curr_pos = pos_from_action(curr_a)
        equity_t = prev_pos * p_t
        equity_tp1 = curr_pos * p_tp1
        if curr_a > prev_a:
            # 买入 delta_pos
            delta = curr_pos - prev_pos
            buy_cost = delta * p_t * (1 + commission_fee)  # 现金流出
            denom = max(eps, equity_t + buy_cost)
            ret = (equity_tp1 - buy_cost - equity_t) / denom
        elif curr_a < prev_a:
            # 卖出 delta_pos
            delta = prev_pos - curr_pos
            sell_cash = delta * p_t * (1 - commission_fee)  # 现金流入
            denom = max(eps, equity_t)
            ret = (equity_tp1 + sell_cash - equity_t) / denom
        else:
            # 持仓不变
            denom = max(eps, equity_t)
            ret = (equity_tp1 - equity_t) / denom
        # 奖励缩放到和环境一致的量级
        return float(ret) * reward_scale

    # 反向动态规划：t = T-2 .. 0
    for t in range(T - 2, -1, -1):
        p_t = prices[t]
        p_tp1 = prices[t + 1]
        for prev_a in range(num_action):
            for curr_a in range(num_action):
                r = immediate_return(prev_a, curr_a, p_t, p_tp1)
                # 未来项
                future = np.max(q_table[t + 1, curr_a, :]) if t + 1 < T else 0.0
                q_table[t, prev_a, curr_a] = r + gamma * future

    return q_table
   
   
def make_high_q_table_reward(df: pd.DataFrame,
                        num_action,
                        max_holding,
                        reward_scale=1000,
                        gamma=0.999,
                        commission_fee=0.002,
                        max_punish=1e12):
    q_table = np.zeros((len(df), num_action, num_action))

    def calculate_value(price_information, position):
        return price_information["close"] * position

    scale_factor = num_action - 1

    for t in range(2, len(df) + 1):
        current_price_information = df.iloc[-t]
        future_price_information = df.iloc[-t + 1]
        for previous_action in range(num_action):
            for current_action in range(num_action):
                if current_action > previous_action:
                    previous_position = previous_action / (
                        scale_factor) * max_holding
                    current_position = current_action / (
                        scale_factor) * max_holding
                    position_change = (current_action-previous_action) / \
                        scale_factor*max_holding
                    buy_money = position_change * current_price_information['close'] * (1 + commission_fee)
                    current_value = calculate_value(current_price_information,
                                                    previous_position)
                    future_value = calculate_value(future_price_information,
                                                   current_position)
                    reward = future_value - (current_value + buy_money)
                    reward = reward_scale * reward
                    q_table[len(df) - t][previous_action][
                        current_action] = reward + gamma * np.max(
                            q_table[len(df) - t + 1][current_action][:])
                else:
                    previous_position = previous_action / (
                        scale_factor) * max_holding
                    current_position = current_action / (
                        scale_factor) * max_holding
                    position_change = (previous_action-current_action) / \
                        scale_factor*max_holding
                    sell_money = position_change * current_price_information['close'] * (1 - commission_fee)
                    current_value = calculate_value(current_price_information,
                                                    previous_position)
                    future_value = calculate_value(future_price_information,
                                                   current_position)
                    reward = future_value + sell_money - current_value
                    reward = reward_scale * reward
                    q_table[len(df) - t][previous_action][
                        current_action] = reward + gamma * np.max(
                            q_table[len(df) - t + 1][current_action][:])
    return q_table