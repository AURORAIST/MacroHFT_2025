import numpy as np
import pandas as pd


def make_q_table_reward(df: pd.DataFrame,
                        num_action,
                        max_holding,
                        reward_scale=1500,
                        gamma=0.999,
                        commission_fee=0.001,
                        max_punish=1e12):
    """
    使用动态规划构建Q表（倒序遍历，允许看未来一步）

    设计要点：
    - 即时回报使用 t->t+1 的未来价格，体现"专家"能看到未来一步的优势；
    - 买/卖/持有三种情况统一为"权益（持仓价值+现金流）"的相对变化；
    - Q[t, prev, curr] = r(t, prev->curr) + gamma * max_a Q[t+1, curr, a]；
    - t 遍历范围为 [0..T-2]，T-1 无未来价，保持为 0；
    - 数值稳定：分母加 eps，scale_factor 保护；
    - 奖励缩放 reward_scale 默认 1500，与环境 reward 百分比*1500 对齐。

    Args:
        df: 必须包含收盘价列 'close' 或 'Close'
        num_action: 动作数量（仓位级别数）
        max_holding: 最大持仓量（单位）
        reward_scale: 奖励缩放因子（默认 1500）
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
            position_change = curr_pos - prev_pos
            buy_money = position_change * p_t * (1 + commission_fee)
            reward = equity_tp1 - (equity_t + buy_money)
        else:
            position_change = prev_pos - curr_pos
            sell_money = position_change * p_t * (1 - commission_fee)
            reward = equity_tp1 + sell_money - equity_t
        return float(reward) * reward_scale

    # 反向动态规划：t = T-2 .. 0
    for t in range(T - 2, -1, -1):
        p_t = prices[t]
        p_tp1 = prices[t + 1]
        for prev_a in range(num_action):
            for curr_a in range(num_action):
                r = immediate_return(prev_a, curr_a, p_t, p_tp1)
                future = np.max(q_table[t + 1, curr_a, :]) if t + 1 < T else 0.0
                q_table[t, prev_a, curr_a] = r + gamma * future

    return q_table