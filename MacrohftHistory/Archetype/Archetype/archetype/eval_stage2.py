"""Evaluate Stage 2 selector on whole/{val,test}.feather.

This script:
- loads VQ model (frozen)
- loads trained selector
- rolls on a dataset by sliding non-overlapping horizons of length h
- for each horizon: selector picks one code, decoder generates actions, we compute horizon PnL reward

Outputs:
- summary printed to stdout
- npz saved under result/archetype/<dataset>/stage2_eval/...

Extended evaluation:
- baselines: random code, most-common code (from selector rollout)
- richer trading metrics (equity curve, max drawdown, win rate, avg win/loss)
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

# Ensure project root on sys.path.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from archetype.env_stage2 import load_whole_and_features, MacroEnvConfig
from model.vq_archetype import VQArchetype, VQConfig


def load_vq(ckpt_path: str, device: torch.device) -> VQArchetype:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = VQConfig(**ckpt["cfg"])
    model = VQArchetype(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


class MLPQ(torch.nn.Module):
    def __init__(self, obs_dim: int, n_action: int, hidden: int = 256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_action),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class ActorCritic(torch.nn.Module):
    def __init__(self, obs_dim: int, n_action: int, hidden: int = 256):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
        )
        self.pi = torch.nn.Linear(hidden, n_action)
        self.v = torch.nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        logits = self.pi(h)
        value = self.v(h).squeeze(-1)
        return logits, value


def _load_selector(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    obs_dim = int(ckpt["obs_dim"])
    n_action = int(ckpt["n_action"])

    arch = ckpt.get("arch", "dqn")
    if arch == "actor_critic":
        model = ActorCritic(obs_dim, n_action)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()
        return model.to(device), ckpt

    # default: DQN
    model = MLPQ(obs_dim, n_action)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model.to(device), ckpt


def _step_reward_and_pnl(prices: np.ndarray, actions: np.ndarray, cfg: MacroEnvConfig):
    """Return (reward_sum, pnl_step_array).

    pnl_step_array is in *unscaled* PnL units (i.e. before reward_scale).
    """
    from archetype.dp_planner import _step_reward  # type: ignore

    T = len(actions)
    pnl = np.zeros((max(T - 1, 0),), dtype=np.float64)
    total = 0.0

    # _step_reward returns scaled reward; convert back to pnl by dividing reward_scale.
    for t in range(T - 1):
        r = float(_step_reward(
            float(prices[t]),
            float(prices[t + 1]),
            int(actions[t]),
            int(actions[t + 1]),
            commission_fee=cfg.commission_fee,
            max_holding=cfg.max_holding,
            reward_scale=cfg.reward_scale,
            action_mode=cfg.action_mode,
            reward_no_action=False,
        ))
        total += r
        pnl[t] = r / float(cfg.reward_scale)

    return total, pnl


def _equity_from_pnl(pnl_steps: np.ndarray, init_equity: float = 1.0):
    eq = init_equity + np.cumsum(pnl_steps)
    eq = np.concatenate([[init_equity], eq])  # include t0
    return eq


def _max_drawdown(equity: np.ndarray):
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak)
    return float(dd.min())


def _trade_metrics(pnl_steps: np.ndarray):
    """Compute simple metrics from step PnL series."""
    pnl_steps = np.asarray(pnl_steps, dtype=np.float64)
    eq = _equity_from_pnl(pnl_steps)

    wins = pnl_steps[pnl_steps > 0]
    losses = pnl_steps[pnl_steps < 0]

    win_rate = float(len(wins) / max(len(pnl_steps), 1))
    avg_win = float(np.mean(wins)) if len(wins) else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) else 0.0

    return {
        "equity_final": float(eq[-1]),
        "equity_mdd": _max_drawdown(eq),
        "pnl_sum": float(np.sum(pnl_steps)),
        "pnl_mean": float(np.mean(pnl_steps)) if len(pnl_steps) else 0.0,
        "pnl_std": float(np.std(pnl_steps)) if len(pnl_steps) else 0.0,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "n_steps": int(len(pnl_steps)),
    }


def _rollout_codes_and_pnl(*,
                          df: pd.DataFrame,
                          feat_cols,
                          horizon: int,
                          vq: VQArchetype,
                          selector: MLPQ | None,
                          cfg: MacroEnvConfig,
                          policy: str,
                          rng: np.random.Generator,
                          most_common_code: int | None = None):
    """Rollout a policy over non-overlapping horizons.

    policy: 'selector' | 'random' | 'most_common'
    Returns: (codes[horizons], rewards[horizons], pnl_steps_concat)
    """
    h = int(horizon)
    n_h = (len(df) // h)

    rewards = np.zeros((n_h,), dtype=np.float64)
    codes = np.zeros((n_h,), dtype=np.int64)
    pnl_all = []

    dev = next(vq.parameters()).device

    with torch.no_grad():
        for i in range(n_h):
            chunk = df.iloc[i * h:(i + 1) * h]
            obs = chunk[feat_cols].iloc[0].to_numpy(np.float32)
            prices = chunk["close"].to_numpy(np.float64)

            o_t = torch.from_numpy(obs).to(dev).unsqueeze(0)
            out = selector(o_t)
            if isinstance(out, tuple):
                logits, _v = out
                code = int(torch.argmax(logits, dim=1).item())
            else:
                qv = out
                code = int(torch.argmax(qv, dim=1).item())

            s = chunk[feat_cols].to_numpy(np.float32)
            s_t = torch.from_numpy(s).to(dev).unsqueeze(0)
            z_q = vq.vq.codebook.weight[code:code + 1]
            logits = vq.decoder(s_t, z_q)
            a_hat = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy().astype(np.int64)

            r, pnl_steps = _step_reward_and_pnl(prices, a_hat, cfg)
            rewards[i] = r
            codes[i] = code
            pnl_all.append(pnl_steps)

    pnl_all = np.concatenate(pnl_all, axis=0) if len(pnl_all) else np.zeros((0,), dtype=np.float64)
    return codes, rewards, pnl_all


def evaluate(*,
             dataset: str,
             split: str,
             horizon: int,
             vq_ckpt: str,
             selector_ckpt: str,
             device: str,
             action_mode: str,
             commission_fee: float,
             reward_scale: float,
             max_holding: float,
             out_dir: str,
             baselines: bool = True,
             random_trials: int = 5,
             seed: int = 12345):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # features (train path only returns feat_cols; we will load split file ourselves)
    _, feat_cols = load_whole_and_features(dataset)

    whole_path = f"./data/{dataset}/whole/{split}.feather"
    df = pd.read_feather(whole_path).reset_index(drop=True)

    vq = load_vq(vq_ckpt, dev)
    selector, _sel_meta = _load_selector(selector_ckpt, dev)

    cfg = MacroEnvConfig(
        dataset=dataset,
        horizon=horizon,
        action_mode=action_mode,
        commission_fee=commission_fee,
        gamma=0.999,
        reward_scale=reward_scale,
        max_holding=max_holding,
    )

    rng = np.random.default_rng(seed)

    # selector rollout
    codes_sel, rewards_sel, pnl_sel = _rollout_codes_and_pnl(
        df=df,
        feat_cols=feat_cols,
        horizon=horizon,
        vq=vq,
        selector=selector,
        cfg=cfg,
        policy="selector",
        rng=rng,
    )

    # summary (reward in scaled units)
    summary_sel = {
        "policy": "selector",
        "dataset": dataset,
        "split": split,
        "horizon": int(horizon),
        "n_horizons": int(len(rewards_sel)),
        "reward_mean": float(np.mean(rewards_sel)),
        "reward_std": float(np.std(rewards_sel)),
        "reward_sum": float(np.sum(rewards_sel)),
        "trade": _trade_metrics(pnl_sel),
    }

    uniq, cnt = np.unique(codes_sel, return_counts=True)
    top = sorted(zip(uniq.tolist(), cnt.tolist()), key=lambda x: x[1], reverse=True)[:10]
    most_common_code = int(top[0][0]) if len(top) else 0

    all_results = {
        "selector": {
            "codes": codes_sel,
            "rewards": rewards_sel,
            "pnl_steps": pnl_sel,
            "equity": _equity_from_pnl(pnl_sel),
            "summary": summary_sel,
        }
    }

    # baselines
    if baselines:
        # most-common code baseline
        codes_mc, rewards_mc, pnl_mc = _rollout_codes_and_pnl(
            df=df,
            feat_cols=feat_cols,
            horizon=horizon,
            vq=vq,
            selector=None,
            cfg=cfg,
            policy="most_common",
            rng=rng,
            most_common_code=most_common_code,
        )
        all_results["most_common"] = {
            "codes": codes_mc,
            "rewards": rewards_mc,
            "pnl_steps": pnl_mc,
            "equity": _equity_from_pnl(pnl_mc),
            "summary": {
                "policy": "most_common",
                "most_common_code": most_common_code,
                "reward_mean": float(np.mean(rewards_mc)),
                "reward_std": float(np.std(rewards_mc)),
                "reward_sum": float(np.sum(rewards_mc)),
                "trade": _trade_metrics(pnl_mc),
            },
        }

        # random code baseline (multiple trials)
        rand_summaries = []
        for t in range(int(random_trials)):
            rng_t = np.random.default_rng(seed + 1000 + t)
            codes_r, rewards_r, pnl_r = _rollout_codes_and_pnl(
                df=df,
                feat_cols=feat_cols,
                horizon=horizon,
                vq=vq,
                selector=None,
                cfg=cfg,
                policy="random",
                rng=rng_t,
            )
            rand_summaries.append({
                "trial": int(t),
                "reward_mean": float(np.mean(rewards_r)),
                "reward_std": float(np.std(rewards_r)),
                "reward_sum": float(np.sum(rewards_r)),
                "trade": _trade_metrics(pnl_r),
            })

        all_results["random"] = {
            "random_trials": int(random_trials),
            "summaries": rand_summaries,
        }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"stage2_eval_{split}_h{int(horizon)}.npz")

    # store as npz: use allow_pickle structures via object arrays
    np.savez_compressed(
        out_path,
        results=np.array([all_results], dtype=object),
    )

    print("[Stage2 Eval]", summary_sel)
    print("[Stage2 Eval] top codes:", ", ".join([f"{k}:{v}" for k, v in top]))
    if baselines:
        print("[Stage2 Eval] baseline most_common_code=", most_common_code)
        print("[Stage2 Eval] baseline most_common:", all_results["most_common"]["summary"])  # type: ignore[index]
        rm = [s["reward_mean"] for s in all_results["random"]["summaries"]]  # type: ignore[index]
        print(f"[Stage2 Eval] baseline random trials={random_trials} reward_mean: mean={float(np.mean(rm)):.4f} std={float(np.std(rm)):.4f}")

    print("[Stage2 Eval] saved:", out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ETHUSDT")
    p.add_argument("--split", type=str, default="test", choices=["val", "test", "train"])
    p.add_argument("--horizon", type=int, default=72)
    p.add_argument("--vq_ckpt", type=str, default="./result/archetype/ETHUSDT/vq/vq_archetype_final.pt")
    p.add_argument("--selector_ckpt", type=str, default="./result/archetype/ETHUSDT/stage2/selector.pt")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--action_mode", type=str, default="both", choices=["both", "long", "short"])
    p.add_argument("--commission_fee", type=float, default=0.0005)
    p.add_argument("--reward_scale", type=float, default=1000.0)
    p.add_argument("--max_holding", type=float, default=0.2)
    p.add_argument("--out_dir", type=str, default="./result/archetype/ETHUSDT/stage2_eval")
    p.add_argument("--no_baselines", action="store_true")
    p.add_argument("--random_trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()

    evaluate(
        dataset=args.dataset,
        split=args.split,
        horizon=args.horizon,
        vq_ckpt=args.vq_ckpt,
        selector_ckpt=args.selector_ckpt,
        device=args.device,
        action_mode=args.action_mode,
        commission_fee=args.commission_fee,
        reward_scale=args.reward_scale,
        max_holding=args.max_holding,
        out_dir=args.out_dir,
        baselines=(not args.no_baselines),
        random_trials=args.random_trials,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
