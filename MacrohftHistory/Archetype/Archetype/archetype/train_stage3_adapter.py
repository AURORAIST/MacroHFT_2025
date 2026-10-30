"""Stage 3: Train step-level archetype refinement adapter.

We train an actor-critic (A2C style) on `ArchetypeRefineEnv`.

Objective:
- maximize expected regret-aware terminal reward (Eq.(8) implemented in env)
- add step-level cross-entropy supervision to DP hindsight label (Eq.(9))

Eq.(9) strict: build full label sequence \hat{a}_ref^\tau from DP top-1, apply
cross-entropy at *each step* on the taken action distribution.

Notes:
- env terminates when adapter takes non-zero action OR horizon ends.
- reward is 0 for no-op steps, and regret-aware terminal reward for override.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root on sys.path.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd

from archetype.env_stage2 import load_whole_and_features
from archetype.env_stage3 import ArchetypeRefineEnv, RefineEnvConfig
from archetype.eval_stage2 import _load_selector  # reuse
from model.vq_archetype import VQArchetype, VQConfig
from model.adapter_policy import AdapterActorCritic, AdapterConfig


@dataclass
class Cfg:
    dataset: str = "ETHUSDT"
    horizon: int = 72

    vq_ckpt: str = "./result/archetype/ETHUSDT/vq/vq_archetype_final.pt"
    selector_ckpt: str = "./result/archetype/ETHUSDT/stage2/selector.pt"
    out_dir: str = "./result/archetype/ETHUSDT/stage3"

    device: str = "cuda:0"

    # env
    action_mode: str = "both"
    commission_fee: float = 0.0005
    reward_scale: float = 1000.0
    max_holding: float = 0.2
    beta1: float = 1.0

    # training
    steps: int = 200_000
    lr: float = 3e-4
    gamma: float = 0.99
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0

    # imitation (Eq.(9))
    beta2: float = 0.1  # coefficient for step-level CE guidance
    ce_nonzero_weight: float = 3.0  # emphasize non-zero (override) labels to prevent no-op collapse

    log_every: int = 2000
    seed: int = 12345


def load_vq(ckpt_path: str, device: torch.device) -> VQArchetype:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = VQConfig(**ckpt["cfg"])
    model = VQArchetype(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def train(cfg: Cfg):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    df, feat_cols = load_whole_and_features(cfg.dataset)

    vq = load_vq(cfg.vq_ckpt, dev)
    selector, _ = _load_selector(cfg.selector_ckpt, dev)

    env_cfg = RefineEnvConfig(
        dataset=cfg.dataset,
        horizon=cfg.horizon,
        action_mode=cfg.action_mode,
        commission_fee=cfg.commission_fee,
        reward_scale=cfg.reward_scale,
        max_holding=cfg.max_holding,
        beta1=cfg.beta1,
    )
    env = ArchetypeRefineEnv(df, feat_cols, vq, selector, cfg=env_cfg)

    market_dim = len(feat_cols)
    ctx_dim = env.ctx_dim

    policy = AdapterActorCritic(AdapterConfig(market_dim=market_dim, ctx_dim=ctx_dim, hidden=256)).to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    obs, info = env.reset(seed=cfg.seed)
    obs_t = torch.from_numpy(obs).to(dev).unsqueeze(0)

    # logging
    recent_R = []
    recent_loss = []
    recent_ce = []
    recent_override = []

    for step in range(1, cfg.steps + 1):
        logits, value = policy(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        act = dist.sample()

        # Eq.(9) step label from env (DP top-1 sequence)
        loss_ce = torch.tensor(0.0, device=dev)
        if cfg.beta2 > 0.0:
            # label for current tau
            y = int(env.get_supervision_label(int(env._tau)))  # pylint: disable=protected-access
            y_t = torch.tensor([y], device=dev)
            # weighted CE to avoid trivial all-zero solution when most rewards are delayed
            w = torch.tensor([cfg.ce_nonzero_weight if y != 0 else 1.0], device=dev)
            loss_ce = (F.cross_entropy(logits, y_t, reduction="none") * w).mean()

        next_obs, rew, done, _, info = env.step(int(act.item()))

        # A2C one-step update
        rew_t = torch.tensor([float(rew)], device=dev)
        done_t = torch.tensor([float(done)], device=dev)

        with torch.no_grad():
            next_obs_t = torch.from_numpy(next_obs).to(dev).unsqueeze(0)
            _next_logits, next_value = policy(next_obs_t)
            target_v = rew_t + cfg.gamma * (1.0 - done_t) * next_value
            adv = target_v - value

        logp = dist.log_prob(act)
        entropy = dist.entropy()

        loss_pi = -(logp * adv.detach()).mean()
        loss_v = F.mse_loss(value, target_v.detach())

        loss = loss_pi + cfg.vf_coef * loss_v - cfg.ent_coef * entropy.mean() + cfg.beta2 * loss_ce

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
        opt.step()

        recent_loss.append(float(loss.detach().cpu().item()))
        recent_R.append(float(rew))
        recent_ce.append(float(loss_ce.detach().cpu().item()))
        recent_override.append(1.0 if int(act.item()) != 0 else 0.0)
        if len(recent_loss) > 1000:
            recent_loss.pop(0)
            recent_R.pop(0)
            recent_ce.pop(0)
            recent_override.pop(0)

        obs_t = next_obs_t
        if done:
            obs, info = env.reset()
            obs_t = torch.from_numpy(obs).to(dev).unsqueeze(0)

        if step % cfg.log_every == 0:
            print(
                f"step={step} avgR(1k)={float(np.mean(recent_R)):.4f} "
                f"avgLoss(1k)={float(np.mean(recent_loss)):.6f} avgCE(1k)={float(np.mean(recent_ce)):.6f} "
                f"overrideRate(1k)={float(np.mean(recent_override)):.3f}"
            )

    os.makedirs(cfg.out_dir, exist_ok=True)
    out = os.path.join(cfg.out_dir, "adapter.pt")
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "market_dim": market_dim,
            "ctx_dim": ctx_dim,
            "cfg": cfg.__dict__,
        },
        out,
    )
    print("saved:", out)


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ETHUSDT")
    p.add_argument("--horizon", type=int, default=72)
    p.add_argument("--vq_ckpt", type=str, default="./result/archetype/ETHUSDT/vq/vq_archetype_final.pt")
    p.add_argument("--selector_ckpt", type=str, default="./result/archetype/ETHUSDT/stage2/selector.pt")
    p.add_argument("--out_dir", type=str, default="./result/archetype/ETHUSDT/stage3")
    p.add_argument("--device", type=str, default="cuda:0")

    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--beta1", type=float, default=1.0)
    p.add_argument("--beta2", type=float, default=0.1)
    p.add_argument("--ce_nonzero_weight", type=float, default=3.0)
    p.add_argument("--ent_coef", type=float, default=0.01)

    p.add_argument("--commission_fee", type=float, default=0.0005)
    p.add_argument("--reward_scale", type=float, default=1000.0)
    p.add_argument("--max_holding", type=float, default=0.2)
    p.add_argument("--action_mode", type=str, default="both", choices=["both", "long", "short"])

    return p.parse_args()


def main():
    a = _parse()
    cfg = Cfg(
        dataset=a.dataset,
        horizon=a.horizon,
        vq_ckpt=a.vq_ckpt,
        selector_ckpt=a.selector_ckpt,
        out_dir=a.out_dir,
        device=a.device,
        steps=a.steps,
        lr=a.lr,
        gamma=a.gamma,
        beta1=a.beta1,
        beta2=a.beta2,
        ce_nonzero_weight=a.ce_nonzero_weight,
        ent_coef=a.ent_coef,
        commission_fee=a.commission_fee,
        reward_scale=a.reward_scale,
        max_holding=a.max_holding,
        action_mode=a.action_mode,
    )
    train(cfg)


if __name__ == "__main__":
    main()
