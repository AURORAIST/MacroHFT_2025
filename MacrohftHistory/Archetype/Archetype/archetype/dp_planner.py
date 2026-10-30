import numpy as np


# Actions are fixed to 3 classes: 0=short, 1=flat, 2=long
A_SHORT, A_FLAT, A_LONG = 0, 1, 2


def _position_from_action(action: int, max_holding: float = 1.0, action_mode: str = "both"):
    """Map discrete action to (long_position, short_position).

    与 MacroHFT 环境一致：
    - action_mode='both'：short/flat/long 同时支持多空，short 表示开空，long 表示开多
    - action_mode='long'：short 会被视为 flat（不允许开空）
    - action_mode='short'：long 会被视为 flat（不允许开多）

    Returns:
        (long_position, short_position)
    """
    if action_mode not in ("both", "long", "short"):
        raise ValueError(f"Unsupported action_mode: {action_mode}")

    if action == A_LONG:
        if action_mode == "short":
            return 0.0, 0.0
        return float(max_holding), 0.0

    if action == A_SHORT:
        if action_mode == "long":
            return 0.0, 0.0
        return 0.0, float(max_holding)

    return 0.0, 0.0


def _step_reward(
    price_t: float,
    price_tp1: float,
    action_i: int,
    action_j: int,
    commission_fee: float,
    max_holding: float,
    reward_scale: float,
    action_mode: str,
    reward_no_action: bool,
):
    """Immediate reward for switching from action_i at t to action_j at t+1.

    目标：尽量对齐 MacroHFT 的 reward/手续费/多空现金流设定（参考 tools/demonstration.py），
    但动作空间固定为三类 {short, flat, long}。

    Notes:
        - 这里的 reward 是“刻度化前”的真钱收益，最后乘以 reward_scale。
        - reward_no_action 参数仅做兼容；MacroHFT 默认在 no-position 且 no-action 时 reward=0。
    """
    long_i, short_i = _position_from_action(action_i, max_holding=max_holding, action_mode=action_mode)
    long_j, short_j = _position_from_action(action_j, max_holding=max_holding, action_mode=action_mode)

    # Long leg cashflow (buy -> pay, sell -> receive)
    long_reward = 0.0
    if long_j > long_i:  # buy/open long
        delta = long_j - long_i
        buy_cost = delta * price_t * (1.0 + commission_fee)
        long_reward = (long_j * price_tp1) - (long_i * price_t + buy_cost)
    elif long_j == 0.0 and long_i == 0.0:
        long_reward = 0.0
    else:  # sell/close long
        delta = long_i - long_j
        sell_cash = delta * price_t * (1.0 - commission_fee)
        long_reward = (long_j * price_tp1) + sell_cash - (long_i * price_t)

    # Short leg (open short -> receive cash, close -> pay cash)
    short_reward = 0.0
    if short_j > short_i:  # open short (sell)
        delta = short_j - short_i
        open_cash = delta * price_t * (1.0 - commission_fee)
        short_reward = (short_i * price_t + open_cash) - (short_j * price_tp1)
    elif short_j == 0.0 and short_i == 0.0:
        short_reward = 0.0
    else:  # close short (buy to cover)
        delta = short_i - short_j
        close_cost = delta * price_t * (1.0 + commission_fee)
        short_reward = (short_i * price_t) - close_cost - (short_j * price_tp1)

    # MacroHFT 在 both 模式下，如果完全不持仓且不交易，返回 0（避免鼓励频繁空仓）
    if (
        action_mode == "both"
        and long_i == 0.0
        and short_i == 0.0
        and long_j == 0.0
        and short_j == 0.0
    ):
        total = 0.0
    else:
        total = long_reward + short_reward

    return float(reward_scale) * total


def single_trade_dp_plan(
    prices: np.ndarray,
    gamma: float = 0.999,
    commission_fee: float = 0.0005,
    start_action: int = A_FLAT,
    *,
    # 对齐 MacroHFT 环境常用默认值
    max_holding: float = 1.0,
    reward_scale: float = 1000.0,
    action_mode: str = "both",
    reward_no_action: bool = False,
):
    """Algorithm 1: Single-trade DP planner.

    按论文约束：每个 horizon 最多一次 action change（single trade）。

    Args:
        prices: 1D array length N (typically close price).
        gamma: discount factor.
        commission_fee: transaction cost.
        start_action: initial discrete action at t=0.
        max_holding: 持仓规模（与 MacroHFT 环境 max_holding_number 含义一致）。
        reward_scale: 奖励缩放（与 MacroHFT 一致，默认 1000）。
        action_mode: 'both'|'long'|'short'。
        reward_no_action: 兼容参数。

    Returns:
        actions: np.ndarray shape (N,), values in {0,1,2}.
    """
    prices = np.asarray(prices, dtype=np.float64)
    N = int(prices.shape[0])
    if N < 2:
        return np.full((N,), int(start_action), dtype=np.int64)

    A = 3
    # V[t, i, c] where c in {0,1} indicates whether a trade has already occurred
    V = np.zeros((N + 1, A, 2), dtype=np.float64)
    Pi = -np.ones((N, A, 2), dtype=np.int64)

    for t in range(N - 2, -1, -1):
        p_t = float(prices[t])
        p_tp1 = float(prices[t + 1])
        for i in range(A):
            for c in (0, 1):
                best_val = -1e100
                best_j = 0
                for j in range(A):
                    trade = 1 if (i != j) else 0
                    if c + trade > 1:
                        continue
                    r = _step_reward(
                        p_t,
                        p_tp1,
                        i,
                        j,
                        commission_fee=commission_fee,
                        max_holding=max_holding,
                        reward_scale=reward_scale,
                        action_mode=action_mode,
                        reward_no_action=reward_no_action,
                    )
                    val = r + gamma * V[t + 1, j, c + trade]
                    if val > best_val:
                        best_val = val
                        best_j = j
                Pi[t, i, c] = best_j
                V[t, i, c] = best_val

    # Rollout policy
    actions = np.empty((N,), dtype=np.int64)
    i = int(start_action)
    c = 0
    actions[0] = i
    for t in range(0, N - 1):
        j = int(Pi[t, i, c])
        actions[t + 1] = j
        c = min(1, c + (1 if i != j else 0))
        i = j

    # Align with paper: a_{N-1} = a_{N-2}
    actions[-1] = actions[-2]
    return actions
