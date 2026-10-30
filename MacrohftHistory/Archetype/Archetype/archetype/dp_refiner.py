"""Stage 3 helper: hindsight DP for step-level single override within a horizon.

We compute top-k hindsight-optimal adaptations where you are allowed to override the
base (decoded) action at most once within the horizon.

Given base action sequence a_base[0:h-1], choose (tau, a_override) where:
- a_override in {-1, +1} refinement signal (mapped to final action by Eq.(6))
- only a single tau is allowed per horizon; other steps follow base actions.
- return is cumulative sum of step rewards over horizon.

We return the top-5 (tau_opt^n, a_opt^n, R_opt^n) for regret-aware reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from archetype.dp_planner import _step_reward  # type: ignore


def _apply_refinement(*, a_base: np.ndarray, tau: int, a_ref: int):
    """Apply refinement signal at time tau (Eq.(6) strict).

    a_ref:
      -1 => final action becomes SHORT (0)
      +1 => final action becomes LONG  (2)

    Constraint: never override original trades, i.e. if base changes at tau
    (a_base[tau] != a_base[tau-1]) then keep base.
    """
    a_base = np.asarray(a_base, dtype=np.int64)
    h = int(a_base.shape[0])
    a_final = a_base.copy()

    if tau <= 0 or tau >= h:
        return a_final

    # if base already trades at tau, no override
    if a_base[tau] != a_base[tau - 1]:
        return a_final

    if a_ref not in (-1, 1):
        return a_final

    a_final[tau] = 0 if a_ref == -1 else 2

    # keep last action consistent a_{h-1}=a_{h-2}
    if h >= 2:
        a_final[-1] = a_final[-2]
    return a_final


def horizon_return(*,
                   prices: np.ndarray,
                   actions: np.ndarray,
                   commission_fee: float,
                   max_holding: float,
                   reward_scale: float,
                   action_mode: str):
    prices = np.asarray(prices, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.int64)
    h = int(actions.shape[0])

    total = 0.0
    for t in range(h - 1):
        total += float(_step_reward(
            float(prices[t]),
            float(prices[t + 1]),
            int(actions[t]),
            int(actions[t + 1]),
            commission_fee=commission_fee,
            max_holding=max_holding,
            reward_scale=reward_scale,
            action_mode=action_mode,
            reward_no_action=False,
        ))
    return float(total)


def dp_topk_single_override(*,
                            prices: np.ndarray,
                            a_base: np.ndarray,
                            topk: int = 5,
                            commission_fee: float,
                            max_holding: float,
                            reward_scale: float,
                            action_mode: str):
    """Return list of (tau_opt, a_opt in {-1,1}, R_opt)."""
    prices = np.asarray(prices, dtype=np.float64)
    a_base = np.asarray(a_base, dtype=np.int64)
    h = int(a_base.shape[0])

    base_R = horizon_return(
        prices=prices,
        actions=a_base,
        commission_fee=commission_fee,
        max_holding=max_holding,
        reward_scale=reward_scale,
        action_mode=action_mode,
    )

    cand: List[Tuple[int, int, float]] = []
    for tau in range(1, h - 1):
        for a_ref in (-1, 1):
            a_final = _apply_refinement(a_base=a_base, tau=tau, a_ref=a_ref)
            # if no change, skip (means override not allowed)
            if np.array_equal(a_final, a_base):
                continue
            R = horizon_return(
                prices=prices,
                actions=a_final,
                commission_fee=commission_fee,
                max_holding=max_holding,
                reward_scale=reward_scale,
                action_mode=action_mode,
            )
            cand.append((int(tau), int(a_ref), float(R)))

    cand.sort(key=lambda x: x[2], reverse=True)
    return {
        "R_base": float(base_R),
        "top": cand[: int(topk)],
    }
