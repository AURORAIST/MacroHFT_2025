"""Stage 2 environment: macro-level selector chooses one archetype code per horizon.

This is a lightweight Gym-style environment that:
- samples a horizon of length h from whole/train.feather (same as Stage 1)
- observation: aggregated features of the horizon start (default: last row features at t=0)
- action: archetype code k in [0..K-1]
- reward: decode actions with frozen VQ decoder and execute them on close prices (single horizon)

To keep alignment with MacroHFT and minimize coupling, we compute PnL-based reward using
`archetype.dp_planner._step_reward` over the decoded actions.

NOTE: This env is designed for DQN-like algorithms.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

# Ensure project root on sys.path.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gym
import numpy as np
import pandas as pd
from gym import spaces

from model.vq_archetype import VQArchetype


@dataclass
class MacroEnvConfig:
    dataset: str = "ETHUSDT"
    horizon: int = 72
    action_mode: str = "both"  # 'both'|'long'|'short'
    commission_fee: float = 0.0005
    gamma: float = 0.999
    reward_scale: float = 1000.0
    max_holding: float = 0.2


class MacroSelectorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, df_whole: pd.DataFrame, feat_cols: List[str], vq_model: VQArchetype, *, cfg: MacroEnvConfig):
        super().__init__()
        self.df = df_whole.reset_index(drop=True)
        self.feat_cols = list(feat_cols)
        self.vq = vq_model
        self.cfg = cfg

        self.h = int(cfg.horizon)
        self.K = int(self.vq.cfg.codebook_size)

        self.action_space = spaces.Discrete(self.K)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.feat_cols),), dtype=np.float32)

        # sampling
        self._rng = np.random.default_rng(12345)
        self._start = 0
        self._chunk = None

    def seed(self, seed: int | None = None):
        if seed is None:
            seed = 12345
        self._rng = np.random.default_rng(int(seed))

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.seed(seed)

        max_start = len(self.df) - self.h
        if max_start <= 1:
            raise ValueError("Data too short for horizon")
        self._start = int(self._rng.integers(0, max_start, endpoint=False))
        self._chunk = self.df.iloc[self._start:self._start + self.h]

        obs = self._chunk[self.feat_cols].iloc[0].to_numpy(dtype=np.float32)
        info = {"start": self._start}
        return obs, info

    def step(self, action: int):
        if self._chunk is None:
            raise RuntimeError("Call reset() before step().")

        k = int(action)
        s = self._chunk[self.feat_cols].to_numpy(dtype=np.float32)  # [h, D]
        prices = self._chunk["close"].to_numpy(dtype=np.float64)    # [h]

        # decode actions using chosen code k
        # decoder expects z_q vector; we take the codebook embedding directly as z_q
        with np.errstate(all="ignore"):
            pass
        import torch
        self.vq.eval()
        device = next(self.vq.parameters()).device
        with torch.no_grad():
            z_q = self.vq.vq.codebook.weight[k:k+1]  # [1, z_dim]
            s_t = torch.from_numpy(s).to(device).unsqueeze(0)  # [1,h,D]
            logits = self.vq.decoder(s_t, z_q)  # [1,h,3]
            a_hat = torch.argmax(logits, dim=-1).squeeze(0).detach().cpu().numpy().astype(np.int64)  # [h]

        # compute horizon reward by executing actions
        from archetype.dp_planner import single_trade_dp_plan  # noqa: F401
        from archetype.dp_planner import _step_reward  # type: ignore
        total = 0.0
        for t in range(self.h - 1):
            total += float(_step_reward(
                float(prices[t]),
                float(prices[t + 1]),
                int(a_hat[t]),
                int(a_hat[t + 1]),
                commission_fee=self.cfg.commission_fee,
                max_holding=self.cfg.max_holding,
                reward_scale=self.cfg.reward_scale,
                action_mode=self.cfg.action_mode,
                reward_no_action=False,
            ))

        # one-step episode: choose once per horizon
        obs = self._chunk[self.feat_cols].iloc[0].to_numpy(dtype=np.float32)
        terminated = True
        truncated = False
        info = {"start": self._start, "code": k, "a_hat": a_hat}
        return obs, float(total), terminated, truncated, info


def load_whole_and_features(dataset: str):
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
        raise FileNotFoundError("Cannot find feature_list under ./data. Tried: " + ", ".join(candidates))

    single = np.load(os.path.join(base, "single_features.npy"), allow_pickle=True).tolist()
    trend = np.load(os.path.join(base, "trend_features.npy"), allow_pickle=True).tolist()
    feat_cols = list(single) + list(trend)

    whole_path = f"./data/{dataset}/whole/train.feather"
    if not os.path.exists(whole_path):
        raise FileNotFoundError(f"Missing: {whole_path}")
    df = pd.read_feather(whole_path)
    return df, feat_cols
