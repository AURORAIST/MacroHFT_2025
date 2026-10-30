"""Evaluate Stage 3 adapter (with frozen selector+VQ) on whole/{val,test}.feather.

We compare:
- selector only (Stage2)
- selector + adapter (Stage3)

We report reward (scaled) and trade metrics on step PnL (unscaled).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

# Ensure project root on sys.path.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from archetype.env_stage2 import load_whole_and_features, MacroEnvConfig
from archetype.eval_stage2 import _equity_from_pnl, _trade_metrics, _step_reward_and_pnl, _load_selector
from archetype.env_stage3 import _apply_ref_eq6
from model.vq_archetype import VQArchetype, VQConfig
from model.adapter_policy import AdapterActorCritic, AdapterConfig


def load_vq(ckpt_path: str, device: torch.device) -> VQArchetype:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = VQConfig(**ckpt["cfg"])
    model = VQArchetype(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def load_adapter(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    policy = AdapterActorCritic(AdapterConfig(market_dim=int(ckpt["market_dim"]), ctx_dim=int(ckpt["ctx_dim"])))
    policy.load_state_dict(ckpt["state_dict"], strict=True)
    policy.eval()
    return policy.to(device)


def eval_adapter(*,
                 dataset: str,
                 split: str,
                 horizon: int,
                 vq_ckpt: str,
                 selector_ckpt: str,
                 adapter_ckpt: str,
                 device: str,
                 action_mode: str,
                 commission_fee: float,
                 reward_scale: float,
                 max_holding: float,
                 out_dir: str,
                 seed: int = 12345):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    _, feat_cols = load_whole_and_features(dataset)

    df = pd.read_feather(f"./data/{dataset}/whole/{split}.feather").reset_index(drop=True)
    vq = load_vq(vq_ckpt, dev)
    selector, _ = _load_selector(selector_ckpt, dev)
    adapter = load_adapter(adapter_ckpt, dev)

    cfg = MacroEnvConfig(
        dataset=dataset,
        horizon=horizon,
        action_mode=action_mode,
        commission_fee=commission_fee,
        gamma=0.999,
        reward_scale=reward_scale,
        max_holding=max_holding,
    )

    h = int(horizon)
    n_h = len(df) // h

    rewards_base = []
    pnl_base = []
    rewards_ref = []
    pnl_ref = []

    # debug stats
    n_override_pred = 0
    n_override_applied = 0
    n_action_changed = 0
    first_logs = []

    rng = np.random.default_rng(seed)

    with torch.no_grad():
        for i in range(n_h):
            chunk = df.iloc[i * h:(i + 1) * h]
            obs0 = chunk[feat_cols].iloc[0].to_numpy(np.float32)
            prices = chunk["close"].to_numpy(np.float64)

            # selector code + decode
            o_t = torch.from_numpy(obs0).to(dev).unsqueeze(0)
            sel_out = selector(o_t)
            if isinstance(sel_out, tuple):
                sel_logits, _sel_v = sel_out
                code = int(torch.argmax(sel_logits, dim=1).item())
            else:
                code = int(torch.argmax(sel_out, dim=1).item())
            s = chunk[feat_cols].to_numpy(np.float32)
            s_t = torch.from_numpy(s).to(dev).unsqueeze(0)
            z_q = vq.vq.codebook.weight[code:code + 1]
            logits = vq.decoder(s_t, z_q)
            a_base = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy().astype(np.int64)

            r_sum, pnl_steps = _step_reward_and_pnl(prices, a_base, cfg)
            rewards_base.append(r_sum)
            pnl_base.append(pnl_steps)

            # run adapter: allow one non-zero at most; greedy argmax
            z = vq.vq.codebook.weight[code].detach().cpu().numpy().astype(np.float32)
            cum_arche = 0.0
            used = False
            a_final = a_base.copy()

            chosen_tau = -1
            chosen_aref = 0

            for tau in range(h - 1):
                # context
                a = int(a_base[tau])
                a_oh = np.zeros((3,), dtype=np.float32)
                a_oh[a] = 1.0
                tau_remain = float(h - tau)
                ctx = np.concatenate([
                    z,
                    a_oh,
                    np.array([cum_arche], dtype=np.float32),
                    np.array([tau_remain], dtype=np.float32),
                ], axis=0)
                obs = np.concatenate([s[tau].astype(np.float32), ctx], axis=0)
                obs_t = torch.from_numpy(obs).to(dev).unsqueeze(0)
                ad_logits, _v = adapter(obs_t)
                act = int(torch.argmax(ad_logits, dim=1).item())  # 0,1,2
                a_ref = act - 1

                if not used and a_ref != 0:
                    n_override_pred += 1
                    used = True
                    chosen_tau = int(tau)
                    chosen_aref = int(a_ref)

                    a_try = _apply_ref_eq6(a_base=a_base, tau=tau, a_ref=a_ref)
                    # if override does not change anything (Eq.(6) constraint), it is not applied
                    if not np.array_equal(a_try, a_base):
                        n_override_applied += 1
                        a_final = a_try
                    break

                # update cum arche
                cum_arche += float(_step_reward_and_pnl(prices[tau:tau+2], a_base[tau:tau+2], cfg)[0])

            if not np.array_equal(a_final, a_base):
                n_action_changed += 1

            if len(first_logs) < 10:
                first_logs.append({
                    "i": int(i),
                    "code": int(code),
                    "tau": int(chosen_tau),
                    "a_ref": int(chosen_aref),
                    "changed": bool(not np.array_equal(a_final, a_base)),
                })

            r_sum2, pnl_steps2 = _step_reward_and_pnl(prices, a_final, cfg)
            rewards_ref.append(r_sum2)
            pnl_ref.append(pnl_steps2)

    # quick debug print
    print(
        f"[Stage3 Eval][debug] horizons={n_h} "
        f"override_pred={n_override_pred}({(n_override_pred/max(n_h,1)):.3f}) "
        f"override_applied={n_override_applied}({(n_override_applied/max(n_h,1)):.3f}) "
        f"action_changed={n_action_changed}({(n_action_changed/max(n_h,1)):.3f})"
    )
    print(f"[Stage3 Eval][debug] first10={first_logs}")

    rewards_base = np.asarray(rewards_base, dtype=np.float64)
    pnl_base = np.concatenate(pnl_base, axis=0) if len(pnl_base) else np.zeros((0,), dtype=np.float64)
    rewards_ref = np.asarray(rewards_ref, dtype=np.float64)
    pnl_ref = np.concatenate(pnl_ref, axis=0) if len(pnl_ref) else np.zeros((0,), dtype=np.float64)

    res = {
        "base": {
            "reward_mean": float(np.mean(rewards_base)),
            "reward_sum": float(np.sum(rewards_base)),
            "trade": _trade_metrics(pnl_base),
            "equity": _equity_from_pnl(pnl_base),
        },
        "refined": {
            "reward_mean": float(np.mean(rewards_ref)),
            "reward_sum": float(np.sum(rewards_ref)),
            "trade": _trade_metrics(pnl_ref),
            "equity": _equity_from_pnl(pnl_ref),
        },
        "meta": {
            "dataset": dataset,
            "split": split,
            "horizon": h,
            "n_horizons": int(n_h),
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"stage3_eval_{split}_h{h}.npz")
    np.savez_compressed(out_path, results=np.array([res], dtype=object))

    print("[Stage3 Eval]", res["meta"])
    print("[Stage3 Eval] base:", res["base"])
    print("[Stage3 Eval] refined:", res["refined"])
    print("[Stage3 Eval] saved:", out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ETHUSDT")
    p.add_argument("--split", type=str, default="test", choices=["val", "test", "train"])
    p.add_argument("--horizon", type=int, default=72)
    p.add_argument("--vq_ckpt", type=str, default="./result/archetype/ETHUSDT/vq/vq_archetype_final.pt")
    p.add_argument("--selector_ckpt", type=str, default="./result/archetype/ETHUSDT/stage2/selector.pt")
    p.add_argument("--adapter_ckpt", type=str, default="./result/archetype/ETHUSDT/stage3/adapter.pt")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--action_mode", type=str, default="both", choices=["both", "long", "short"])
    p.add_argument("--commission_fee", type=float, default=0.0005)
    p.add_argument("--reward_scale", type=float, default=1000.0)
    p.add_argument("--max_holding", type=float, default=0.2)
    p.add_argument("--out_dir", type=str, default="./result/archetype/ETHUSDT/stage3_eval")
    args = p.parse_args()

    eval_adapter(
        dataset=args.dataset,
        split=args.split,
        horizon=args.horizon,
        vq_ckpt=args.vq_ckpt,
        selector_ckpt=args.selector_ckpt,
        adapter_ckpt=args.adapter_ckpt,
        device=args.device,
        action_mode=args.action_mode,
        commission_fee=args.commission_fee,
        reward_scale=args.reward_scale,
        max_holding=args.max_holding,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
