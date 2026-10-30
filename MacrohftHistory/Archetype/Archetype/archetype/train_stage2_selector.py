"""Stage 2: Train macro-level selector (policy-based) to pick archetype code once per horizon.

Strict paper reproduction notes:
- We model an explicit policy π_φ^sel(a|s) (actor) + value baseline V(s) (critic).
- Objective matches Eq.(5): maximize E[ r_sel ] with a demonstration consistency
  regularizer: -α KL( â_sel || π_φ^sel(·|s) ).
  Since â_sel is a one-hot label, KL reduces to cross-entropy CE(π, â_sel).

Implementation: A2C-style (single-step episode per horizon).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root on sys.path.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from archetype.env_stage2 import MacroEnvConfig, MacroSelectorEnv, load_whole_and_features
from model.vq_archetype import VQArchetype, VQConfig


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_action: int, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, n_action)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        logits = self.pi(h)
        value = self.v(h).squeeze(-1)
        return logits, value


@dataclass
class Cfg:
    dataset: str = "ETHUSDT"
    horizon: int = 72
    vq_ckpt: str = "./result/archetype/ETHUSDT/vq/vq_archetype_final.pt"
    out_dir: str = "./result/archetype/ETHUSDT/stage2"
    device: str = "cuda:0"
    # env reward parameters
    action_mode: str = "both"
    commission_fee: float = 0.0005
    reward_scale: float = 1000.0
    max_holding: float = 0.2

    # Eq.(5) demonstration consistency regularizer
    alpha_kl: float = 0.1

    # A2C
    steps: int = 200_000
    lr: float = 3e-4
    gamma: float = 1.0  # horizon-level one-step, keep as 1.0
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0

    hidden: int = 256
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


def _vq_demo_label(vq: VQArchetype, s_h: np.ndarray, a_h: np.ndarray, r_h: np.ndarray) -> int:
    """Compute demo archetype label â_sel via frozen VQ encoder (paper Eq.(5))."""
    device = next(vq.parameters()).device
    with torch.no_grad():
        s_t = torch.from_numpy(s_h.astype(np.float32)).to(device).unsqueeze(0)
        a_t = torch.from_numpy(a_h.astype(np.int64)).to(device).unsqueeze(0)
        r_t = torch.from_numpy(r_h.astype(np.float32)).to(device).unsqueeze(0)
        z_e = vq.encoder(s_t, a_t, r_t)
        z_e2 = (z_e ** 2).sum(dim=1, keepdim=True)
        e2 = (vq.vq.codebook.weight ** 2).sum(dim=1).unsqueeze(0)
        dist = z_e2 + e2 - 2.0 * z_e @ vq.vq.codebook.weight.t()
        k = int(torch.argmin(dist, dim=1).item())
    return k


def train(cfg: Cfg):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    df, feat_cols = load_whole_and_features(cfg.dataset)
    vq = load_vq(cfg.vq_ckpt, device)

    env_cfg = MacroEnvConfig(
        dataset=cfg.dataset,
        horizon=cfg.horizon,
        action_mode=cfg.action_mode,
        commission_fee=cfg.commission_fee,
        gamma=0.999,
        reward_scale=cfg.reward_scale,
        max_holding=cfg.max_holding,
    )
    env = MacroSelectorEnv(df, feat_cols, vq, cfg=env_cfg)

    obs_dim = len(feat_cols)
    n_action = int(vq.cfg.codebook_size)

    ac = ActorCritic(obs_dim, n_action, hidden=cfg.hidden).to(device)
    opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr)

    # helpers
    def _compute_r_from_prices_actions(prices: np.ndarray, actions: np.ndarray):
        from archetype.dp_planner import _step_reward  # type: ignore

        h = int(len(actions))
        r = np.empty((h - 1,), dtype=np.float32)
        for t in range(h - 1):
            r[t] = _step_reward(
                float(prices[t]),
                float(prices[t + 1]),
                int(actions[t]),
                int(actions[t + 1]),
                commission_fee=cfg.commission_fee,
                max_holding=cfg.max_holding,
                reward_scale=cfg.reward_scale,
                action_mode=cfg.action_mode,
                reward_no_action=False,
            )
        return r

    # logging
    recent_r = []
    recent_loss = []
    recent_ce = []

    demo_fail = 0
    demo_ok = 0
    demo_hist = np.zeros((n_action,), dtype=np.int64)
    act_hist = np.zeros((n_action,), dtype=np.int64)

    obs, _ = env.reset(seed=cfg.seed)

    for step in range(1, cfg.steps + 1):
        # compute â_sel for current sampled horizon
        demo_k = 0
        try:
            chunk = env._chunk  # type: ignore[attr-defined]
            if chunk is not None:
                s_h = chunk[feat_cols].to_numpy(np.float32)
                prices = chunk["close"].to_numpy(np.float64)
                from archetype.dp_planner import single_trade_dp_plan

                a_dp = single_trade_dp_plan(
                    prices,
                    gamma=0.999,
                    commission_fee=cfg.commission_fee,
                    start_action=1,
                    max_holding=cfg.max_holding,
                    reward_scale=cfg.reward_scale,
                    action_mode=cfg.action_mode,
                    reward_no_action=False,
                ).astype(np.int64)
                r_dp = _compute_r_from_prices_actions(prices, a_dp)
                demo_k = _vq_demo_label(vq, s_h, a_dp, r_dp)
                demo_ok += 1
                if 0 <= int(demo_k) < n_action:
                    demo_hist[int(demo_k)] += 1
        except Exception as e:
            # IMPORTANT: silent fallback biases selector toward code 0.
            demo_fail += 1
            demo_k = 0
            if demo_fail <= 5 or (demo_fail % 500 == 0):
                print(f"[warn] demo_k computation failed (count={demo_fail}) err={type(e).__name__}: {e}")

        obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)
        logits, value = ac(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        act = dist.sample()  # [1]
        a_int = int(act.item())
        if 0 <= a_int < n_action:
            act_hist[a_int] += 1

        next_obs, r, done, _, info = env.step(a_int)

        # one-step A2C target
        r_t = torch.tensor([float(r)], device=device)
        with torch.no_grad():
            target_v = r_t  # terminal after one macro decision
            adv = target_v - value

        logp = dist.log_prob(act)
        entropy = dist.entropy()

        # Eq.(5) KL(onehot(demo_k) || π) == CE(π, demo_k)
        demo_t = torch.tensor([int(demo_k)], device=device)
        loss_ce = F.cross_entropy(logits, demo_t)

        loss_pi = -(logp * adv.detach()).mean()
        loss_v = F.mse_loss(value, target_v.detach())

        loss = loss_pi + cfg.vf_coef * loss_v - cfg.ent_coef * entropy.mean() + float(cfg.alpha_kl) * loss_ce

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
        opt.step()

        recent_r.append(float(r))
        recent_loss.append(float(loss.detach().cpu().item()))
        recent_ce.append(float(loss_ce.detach().cpu().item()))
        if len(recent_r) > 1000:
            recent_r.pop(0)
            recent_loss.pop(0)
            recent_ce.pop(0)

        # env is one-step; always reset
        obs, _ = env.reset()

        if step % 5000 == 0:
            ok = max(1, demo_ok)
            fail_rate = float(demo_fail) / float(demo_fail + demo_ok) if (demo_fail + demo_ok) > 0 else 0.0
            topk = np.argsort(-demo_hist)[:5]
            topk_str = ",".join([f"{int(i)}:{int(demo_hist[int(i)])}" for i in topk])
            topa = np.argsort(-act_hist)[:5]
            topa_str = ",".join([f"{int(i)}:{int(act_hist[int(i)])}" for i in topa])
            print(
                f"step={step} avgR(1k)={float(np.mean(recent_r)):.4f} "
                f"avgLoss(1k)={float(np.mean(recent_loss)):.6f} avgCE(1k)={float(np.mean(recent_ce)):.6f} "
                f"demoFail={demo_fail} failRate={fail_rate:.3f} demoTop5=[{topk_str}] actTop5=[{topa_str}]"
            )

    os.makedirs(cfg.out_dir, exist_ok=True)
    out = os.path.join(cfg.out_dir, "selector.pt")
    torch.save(
        {
            "state_dict": ac.state_dict(),
            "obs_dim": obs_dim,
            "n_action": n_action,
            "feat_cols": feat_cols,
            "cfg": cfg.__dict__,
            "arch": "actor_critic",
        },
        out,
    )
    print(f"saved: {out}")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ETHUSDT")
    p.add_argument("--horizon", type=int, default=72)
    p.add_argument("--vq_ckpt", type=str, default="./result/archetype/ETHUSDT/vq/vq_archetype_final.pt")
    p.add_argument("--out_dir", type=str, default="./result/archetype/ETHUSDT/stage2")
    p.add_argument("--device", type=str, default="cuda:0")

    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--alpha_kl", type=float, default=0.1)

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
        out_dir=a.out_dir,
        device=a.device,
        steps=a.steps,
        lr=a.lr,
        ent_coef=a.ent_coef,
        vf_coef=a.vf_coef,
        alpha_kl=a.alpha_kl,
        commission_fee=a.commission_fee,
        reward_scale=a.reward_scale,
        max_holding=a.max_holding,
        action_mode=a.action_mode,
    )
    train(cfg)


if __name__ == "__main__":
    main()
