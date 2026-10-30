"""Train VQ encoder-decoder to discover discrete archetypes (Stage 1).

Input: DP demonstration dataset built by `archetype/build_demo_dataset.py`.
Output:
- model checkpoint: result/archetype/<dataset>/vq/vq_archetype.pt
- code assignments for each trajectory: data/<dataset>/archetype/code_idx_h72_n30000.npy

This is a minimal trainer (single-GPU or CPU) to complete Stage 1 end-to-end.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Ensure project root on sys.path when running as a script.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.vq_archetype import VQArchetype, VQConfig


class DemoNPZDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.s = data["s_demo"].astype(np.float32)
        self.a = data["a_demo"].astype(np.int64)
        self.r = data["r_demo"].astype(np.float32)

    def __len__(self):
        return self.s.shape[0]

    def __getitem__(self, idx):
        return self.s[idx], self.a[idx], self.r[idx]


@dataclass
class TrainCfg:
    dataset: str = "ETHUSDT"
    npz: str = "./data/ETHUSDT/archetype/demo_dp_h72_n30000.npz"
    out_dir: str = "./result/archetype/ETHUSDT/vq"
    device: str = "cuda:0"
    epochs: int = 20
    batch_size: int = 256
    lr: float = 3e-4
    codebook_size: int = 10
    hidden_dim: int = 128
    z_dim: int = 64
    beta: float = 0.25
    num_layers: int = 1
    dropout: float = 0.0
    seed: int = 12345
    num_workers: int = 2


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(cfg: TrainCfg):
    set_seed(cfg.seed)

    ds = DemoNPZDataset(cfg.npz)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)

    state_dim = ds.s.shape[-1]
    model_cfg = VQConfig(
        state_dim=state_dim,
        action_dim=3,
        hidden_dim=cfg.hidden_dim,
        z_dim=cfg.z_dim,
        codebook_size=cfg.codebook_size,
        beta=cfg.beta,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = VQArchetype(model_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    os.makedirs(cfg.out_dir, exist_ok=True)

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        loss_sum = 0.0
        rec_sum = 0.0
        vq_sum = 0.0
        n = 0
        for s, a, r in dl:
            s = torch.from_numpy(s) if isinstance(s, np.ndarray) else s
            a = torch.from_numpy(a) if isinstance(a, np.ndarray) else a
            r = torch.from_numpy(r) if isinstance(r, np.ndarray) else r

            s = s.to(device)
            a = a.to(device)
            r = r.to(device)

            opt.zero_grad(set_to_none=True)
            loss, logs, _ = model.loss(s, a, r)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = s.shape[0]
            loss_sum += float(logs["loss"]) * bs
            rec_sum += float(logs["rec_loss"]) * bs
            vq_sum += float(logs["vq_loss"]) * bs
            n += bs

        ckpt_path = os.path.join(cfg.out_dir, f"vq_archetype_epoch{epoch}.pt")
        torch.save({"cfg": model_cfg.__dict__, "state_dict": model.state_dict()}, ckpt_path)

        print(
            f"epoch={epoch} loss={(loss_sum/n):.6f} rec={(rec_sum/n):.6f} vq={(vq_sum/n):.6f} saved={ckpt_path}")

    # Final code assignment
    model.eval()
    code_idx_all = np.empty((len(ds),), dtype=np.int64)
    assign_dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    offset = 0
    with torch.no_grad():
        for s, a, r in assign_dl:
            s = torch.from_numpy(s) if isinstance(s, np.ndarray) else s
            a = torch.from_numpy(a) if isinstance(a, np.ndarray) else a
            r = torch.from_numpy(r) if isinstance(r, np.ndarray) else r
            s = s.to(device)
            a = a.to(device)
            r = r.to(device)
            _, codes, _ = model.forward(s, a, r)
            codes = codes.detach().cpu().numpy()
            code_idx_all[offset: offset + len(codes)] = codes
            offset += len(codes)

    code_out = f"./data/{cfg.dataset}/archetype/code_idx_h72_n30000.npy"
    os.makedirs(os.path.dirname(code_out), exist_ok=True)
    np.save(code_out, code_idx_all)

    final_ckpt = os.path.join(cfg.out_dir, "vq_archetype_final.pt")
    torch.save({"cfg": model_cfg.__dict__, "state_dict": model.state_dict()}, final_ckpt)
    print(f"saved final checkpoint: {final_ckpt}")
    print(f"saved code assignments: {code_out}")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ETHUSDT")
    p.add_argument("--npz", type=str, default="./data/ETHUSDT/archetype/demo_dp_h72_n30000.npz")
    p.add_argument("--out_dir", type=str, default="./result/archetype/ETHUSDT/vq")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--codebook_size", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--z_dim", type=int, default=64)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def main():
    args = _parse_args()
    cfg = TrainCfg(
        dataset=args.dataset,
        npz=args.npz,
        out_dir=args.out_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        codebook_size=args.codebook_size,
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
        beta=args.beta,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    train(cfg)


if __name__ == "__main__":
    main()
