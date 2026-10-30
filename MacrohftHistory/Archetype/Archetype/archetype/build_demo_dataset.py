"""Build DP demonstration dataset for ArchetypeTrader (Stage 1).

Paper-aligned defaults:
- horizon h = 72
- n = 30,000 DP trajectories

This script samples fixed-length horizons from `data/<dataset>/whole/train.feather`,
then uses Algorithm 1 (single-trade DP planner) to generate demonstration actions.

Outputs:
- .npz file containing:
  - s_demo: float32 array [n, h, state_dim] (tech+trend features)
  - close: float32 array [n, h] (close prices used by DP)
  - a_demo: int64 array [n, h] actions in {0,1,2}
  - r_demo: float32 array [n, h-1] one-step rewards (scaled) aligned with (t->t+1)

Note:
- We intentionally keep features identical to MacroHFT: `single_features.npy` + `trend_features.npy`.
- The DP is computed on `close` prices, matching existing environment/value calculations.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Ensure project root is on sys.path when running as a script.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from archetype.dp_planner import single_trade_dp_plan


@dataclass
class DemoConfig:
    dataset: str = "ETHUSDT"
    horizon: int = 72
    n_traj: int = 30_000
    seed: int = 12345
    commission_fee: float = 0.0005
    gamma: float = 0.999
    reward_scale: float = 1000.0
    action_mode: str = "both"  # 'both'|'long'|'short'
    max_holding: float = 0.2  # default ETH-like; override in CLI


def _load_feature_lists():
    # repo may store feature lists under `data/feature_list/` (original) or `data/data/feature_list/` (this workspace)
    candidates = [
        "./data/feature_list",
        "./data/data/feature_list",
    ]
    base = None
    for c in candidates:
        if os.path.exists(os.path.join(c, "single_features.npy")) and os.path.exists(os.path.join(c, "trend_features.npy")):
            base = c
            break
    if base is None:
        raise FileNotFoundError(
            "Cannot find feature_list. Tried: " + ", ".join(candidates)
        )

    single = np.load(os.path.join(base, "single_features.npy"), allow_pickle=True).tolist()
    trend = np.load(os.path.join(base, "trend_features.npy"), allow_pickle=True).tolist()
    return single, trend


def _compute_step_rewards(prices: np.ndarray, actions: np.ndarray, *,
                          commission_fee: float, max_holding: float, reward_scale: float,
                          action_mode: str, reward_no_action: bool):
    # Import internal helper without exporting it as public API.
    from archetype.dp_planner import _step_reward  # type: ignore

    h = int(prices.shape[0])
    r = np.empty((h - 1,), dtype=np.float32)
    for t in range(h - 1):
        r[t] = _step_reward(
            float(prices[t]),
            float(prices[t + 1]),
            int(actions[t]),
            int(actions[t + 1]),
            commission_fee=float(commission_fee),
            max_holding=float(max_holding),
            reward_scale=float(reward_scale),
            action_mode=str(action_mode),
            reward_no_action=bool(reward_no_action),
        )
    return r


def build_demo_dataset(cfg: DemoConfig, out_path: str):
    single_feats, trend_feats = _load_feature_lists()
    feat_cols = list(single_feats) + list(trend_feats)

    whole_path = f"./data/{cfg.dataset}/whole/train.feather"
    if not os.path.exists(whole_path):
        raise FileNotFoundError(f"Missing file: {whole_path}. Run preprocess/decomposition.py first.")

    df = pd.read_feather(whole_path)
    for col in ["close"] + feat_cols:
        if col not in df.columns:
            raise ValueError(f"Column `{col}` not found in {whole_path}")

    # Ensure enough length
    if len(df) < cfg.horizon + 1:
        raise ValueError(f"Data too short: len={len(df)} horizon={cfg.horizon}")

    rng = np.random.default_rng(cfg.seed)
    max_start = len(df) - cfg.horizon

    starts = rng.integers(0, max_start, size=cfg.n_traj, endpoint=False)

    state_dim = len(feat_cols)
    s_demo = np.empty((cfg.n_traj, cfg.horizon, state_dim), dtype=np.float32)
    close = np.empty((cfg.n_traj, cfg.horizon), dtype=np.float32)
    a_demo = np.empty((cfg.n_traj, cfg.horizon), dtype=np.int64)
    r_demo = np.empty((cfg.n_traj, cfg.horizon - 1), dtype=np.float32)

    for idx, st in enumerate(starts):
        chunk = df.iloc[int(st): int(st) + cfg.horizon]
        prices = chunk["close"].to_numpy(dtype=np.float64)

        # states
        s_demo[idx] = chunk[feat_cols].to_numpy(dtype=np.float32)
        close[idx] = prices.astype(np.float32)

        # DP actions
        actions = single_trade_dp_plan(
            prices,
            gamma=cfg.gamma,
            commission_fee=cfg.commission_fee,
            start_action=1,  # flat
            max_holding=cfg.max_holding,
            reward_scale=cfg.reward_scale,
            action_mode=cfg.action_mode,
            reward_no_action=False,
        )
        a_demo[idx] = actions
        r_demo[idx] = _compute_step_rewards(
            prices,
            actions,
            commission_fee=cfg.commission_fee,
            max_holding=cfg.max_holding,
            reward_scale=cfg.reward_scale,
            action_mode=cfg.action_mode,
            reward_no_action=False,
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        s_demo=s_demo,
        close=close,
        a_demo=a_demo,
        r_demo=r_demo,
        horizon=np.array([cfg.horizon], dtype=np.int64),
        n_traj=np.array([cfg.n_traj], dtype=np.int64),
        feat_cols=np.array(feat_cols, dtype=object),
        dataset=np.array([cfg.dataset], dtype=object),
        commission_fee=np.array([cfg.commission_fee], dtype=np.float32),
        gamma=np.array([cfg.gamma], dtype=np.float32),
        reward_scale=np.array([cfg.reward_scale], dtype=np.float32),
        action_mode=np.array([cfg.action_mode], dtype=object),
        max_holding=np.array([cfg.max_holding], dtype=np.float32),
    )


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ETHUSDT")
    p.add_argument("--horizon", type=int, default=72)
    p.add_argument("--n_traj", type=int, default=30000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--commission_fee", type=float, default=0.0005)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--reward_scale", type=float, default=1000.0)
    p.add_argument("--action_mode", type=str, default="both", choices=["both", "long", "short"])
    p.add_argument("--max_holding", type=float, default=0.2)
    p.add_argument(
        "--out",
        type=str,
        default="./data/ETHUSDT/archetype/demo_dp_h72_n30000.npz",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    cfg = DemoConfig(
        dataset=args.dataset,
        horizon=args.horizon,
        n_traj=args.n_traj,
        seed=args.seed,
        commission_fee=args.commission_fee,
        gamma=args.gamma,
        reward_scale=args.reward_scale,
        action_mode=args.action_mode,
        max_holding=args.max_holding,
    )
    build_demo_dataset(cfg, args.out)


if __name__ == "__main__":
    main()
