"""Stage 3 environment: step-level archetype refinement (policy adapter).

Paper summary:
- Freeze horizon-level selector pi_sel.
- For each horizon, selector chooses an archetype code, decoder produces base micro-action sequence a_base.
- Adapter acts at step-level with refinement signal a_ref in {-1,0,1}.
- Constraint: override at most once per horizon; episode ends when adapter chooses non-zero.
- Observation s_ref = [ market_state s_tau , archetype_context (e_sel, a_base_tau, R_arche_so_far, tau_remain) ]

Reward:
- regret-aware terminal reward (Eq.(8)) when a_ref != 0; otherwise 0.
  r_ref = (R - R_base) + beta1 * (R - R_opt1)
  where R is horizon return with the chosen adaptation; R_opt1 is best return among DP top-k candidates.

This env is designed for simple policy-gradient / actor-critic or DQN over {-1,0,1}.
We do not implement the trainer here; `train_stage3_adapter.py` will use it.
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

import torch

from archetype.dp_planner import _step_reward  # type: ignore
from archetype.dp_refiner import dp_topk_single_override
from model.vq_archetype import VQArchetype


@dataclass
class RefineEnvConfig:
    dataset: str = "ETHUSDT"
    horizon: int = 72
    action_mode: str = "both"
    commission_fee: float = 0.0005
    reward_scale: float = 1000.0
    max_holding: float = 0.2

    # regret reward
    beta1: float = 1.0


def _apply_ref_eq6(*, a_base: np.ndarray, tau: int, a_ref: int):
    """Eq.(6) strict reproduction.

    Paper:
      a_final_tau = a_base_tau, if (a_base_tau != a_base_{tau-1}) or (a_ref_tau = 0)
                  = 0, if a_ref_tau = -1
                  = 2, if a_ref_tau = 1

    Key constraint: the original trade-change step (a_base_tau != a_base_{tau-1}) is never overridden.
    """
    a_base = np.asarray(a_base, dtype=np.int64)
    h = int(a_base.shape[0])
    a_final = a_base.copy()

    if tau <= 0 or tau >= h:
        return a_final

    # never override original trades (trade-change step)
    if a_base[tau] != a_base[tau - 1]:
        return a_final

    if a_ref == 0:
        return a_final
    if a_ref == -1:
        a_final[tau] = 0
    elif a_ref == 1:
        a_final[tau] = 2

    if h >= 2:
        a_final[-1] = a_final[-2]
    return a_final


class ArchetypeRefineEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, df_whole: pd.DataFrame, feat_cols: List[str], vq: VQArchetype, selector_q: torch.nn.Module, *, cfg: RefineEnvConfig):
        super().__init__()
        self.df = df_whole.reset_index(drop=True)
        self.feat_cols = list(feat_cols)
        self.vq = vq
        self.selector_q = selector_q
        self.cfg = cfg

        self.h = int(cfg.horizon)
        self.K = int(self.vq.cfg.codebook_size)

        # obs = market_state + archetype_context
        # context: [code_embedding(z_dim), a_base_tau(onehot3), R_arche_so_far(1), tau_remain(1)]
        self.z_dim = int(self.vq.cfg.z_dim)
        self.ctx_dim = self.z_dim + 3 + 1 + 1
        self.obs_dim = len(self.feat_cols) + self.ctx_dim

        self.action_space = spaces.Discrete(3)  # 0:-1, 1:0, 2:+1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        self._rng = np.random.default_rng(12345)
        self._start = 0
        self._chunk = None

        # per-episode state
        self._tau = 0
        self._code = 0
        self._z = None
        self._a_base = None
        self._prices = None
        self._R_base = 0.0
        self._R_opt1 = 0.0
        self._cum_arche = 0.0
        self._used_override = False

        # Eq.(9) supervision targets from DP top-5
        self._dp_top: List[Tuple[int, int, float]] | None = None
        self._tau_opt1: int = -1
        self._a_opt1: int = 0  # in {-1, +1}

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

        # selector picks code from first obs
        obs0 = self._chunk[self.feat_cols].iloc[0].to_numpy(dtype=np.float32)
        dev = next(self.vq.parameters()).device
        with torch.no_grad():
            o_t = torch.from_numpy(obs0).to(dev).unsqueeze(0)
            logits_sel, _v = self.selector_q(o_t)  # actor-critic selector
            self._code = int(torch.argmax(logits_sel, dim=1).item())
            self._z = self.vq.vq.codebook.weight[self._code].detach().cpu().numpy().astype(np.float32)

            # decode base actions
            s = self._chunk[self.feat_cols].to_numpy(dtype=np.float32)
            s_t = torch.from_numpy(s).to(dev).unsqueeze(0)
            z_q = self.vq.vq.codebook.weight[self._code:self._code + 1]
            logits = self.vq.decoder(s_t, z_q)
            self._a_base = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy().astype(np.int64)

        self._prices = self._chunk["close"].to_numpy(dtype=np.float64)

        # precompute regret targets (DP top-k under single override)
        dp = dp_topk_single_override(
            prices=self._prices,
            a_base=self._a_base,
            topk=5,
            commission_fee=self.cfg.commission_fee,
            max_holding=self.cfg.max_holding,
            reward_scale=self.cfg.reward_scale,
            action_mode=self.cfg.action_mode,
        )
        self._R_base = float(dp["R_base"])
        top = dp["top"]
        self._dp_top = list(top)
        if len(top):
            self._tau_opt1 = int(top[0][0])
            self._a_opt1 = int(top[0][1])
            self._R_opt1 = float(top[0][2])
        else:
            self._tau_opt1 = -1
            self._a_opt1 = 0
            self._R_opt1 = float(self._R_base)

        self._tau = 0
        self._cum_arche = 0.0
        self._used_override = False

        return self._make_obs(), {"start": self._start, "code": self._code}

    def get_supervision_label(self, tau: int) -> int:
        """Eq.(9) step-level label \hat{a}_ref^\tau.

        Only the best DP candidate (tau_opt^1, a_opt^1) gets non-zero label.
        All other steps are labeled as 0 (no-op).

        Returns discrete label in {0,1,2} corresponding to {-1,0,+1}.
        """
        if self._dp_top is None or self._tau_opt1 < 0:
            return 1  # no-op
        if int(tau) != int(self._tau_opt1):
            return 1
        # map a_opt1 in {-1,+1} to discrete
        return 0 if int(self._a_opt1) == -1 else 2

    def get_dp_top5(self) -> List[Tuple[int, int, float]]:
        """Return DP top-5 list of (tau, a_ref in {-1,1}, R)."""
        return list(self._dp_top or [])

    def _make_obs(self):
        assert self._chunk is not None
        assert self._a_base is not None
        assert self._z is not None

        s_tau = self._chunk[self.feat_cols].iloc[self._tau].to_numpy(dtype=np.float32)

        a = int(self._a_base[self._tau])
        a_oh = np.zeros((3,), dtype=np.float32)
        a_oh[a] = 1.0

        tau_remain = float(self.h - self._tau)
        ctx = np.concatenate([
            self._z.astype(np.float32),
            a_oh,
            np.array([float(self._cum_arche)], dtype=np.float32),
            np.array([tau_remain], dtype=np.float32),
        ], axis=0)

        return np.concatenate([s_tau, ctx], axis=0).astype(np.float32)

    def step(self, action: int):
        if self._chunk is None or self._a_base is None or self._prices is None:
            raise RuntimeError("Call reset() before step().")

        # map discrete to refinement signal
        a_ref = {-1: -1, 0: 0, 1: 1}[int(action) - 1]  # action 0->-1, 1->0, 2->+1

        # advance base archetype reward for this step (for context)
        # use base actions only for cum_arche tracking
        if self._tau < self.h - 1:
            self._cum_arche += float(_step_reward(
                float(self._prices[self._tau]),
                float(self._prices[self._tau + 1]),
                int(self._a_base[self._tau]),
                int(self._a_base[self._tau + 1]),
                commission_fee=self.cfg.commission_fee,
                max_holding=self.cfg.max_holding,
                reward_scale=self.cfg.reward_scale,
                action_mode=self.cfg.action_mode,
                reward_no_action=False,
            ))

        terminated = False
        truncated = False
        reward = 0.0

        # allow at most once
        if a_ref != 0 and (not self._used_override):
            self._used_override = True

            a_final = _apply_ref_eq6(a_base=self._a_base, tau=self._tau, a_ref=a_ref)
            # if override not applied (constraint), treat as no-op
            if np.array_equal(a_final, self._a_base):
                reward = 0.0
            else:
                # compute horizon return for final actions
                R = 0.0
                for t in range(self.h - 1):
                    R += float(_step_reward(
                        float(self._prices[t]),
                        float(self._prices[t + 1]),
                        int(a_final[t]),
                        int(a_final[t + 1]),
                        commission_fee=self.cfg.commission_fee,
                        max_holding=self.cfg.max_holding,
                        reward_scale=self.cfg.reward_scale,
                        action_mode=self.cfg.action_mode,
                        reward_no_action=False,
                    ))

                # regret-aware reward Eq.(8)
                reward = (R - self._R_base) + float(self.cfg.beta1) * (R - self._R_opt1)

            # episode ends on non-zero action
            terminated = True

        # if no override yet, move to next step; if reach end => terminate with 0 reward
        if not terminated:
            self._tau += 1
            if self._tau >= self.h - 1:
                terminated = True

        obs = self._make_obs() if not terminated else self._make_obs()  # stable output
        info = {
            "tau": int(self._tau),
            "code": int(self._code),
            "R_base": float(self._R_base),
            "R_opt1": float(self._R_opt1),
            "cum_arche": float(self._cum_arche),
            "used_override": bool(self._used_override),
        }
        return obs, float(reward), terminated, truncated, info
