"""VQ encoder-decoder for Archetype discovery (Stage 1).

This implements a minimal VQ-VAE style model aligned with the paper description:
- LSTM encoder produces z_e
- Vector quantizer maps z_e to nearest codebook entry z_q
- Decoder reconstructs the demonstration action sequence a_demo given s_demo and z_q

We keep the interface simple and independent from MacroHFT DQN code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQConfig:
    state_dim: int
    action_dim: int = 3
    hidden_dim: int = 128
    z_dim: int = 64
    codebook_size: int = 10
    beta: float = 0.25
    num_layers: int = 1
    dropout: float = 0.0


class LSTMEncoder(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, z_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.action_emb = nn.Embedding(action_dim, hidden_dim)
        self.in_proj = nn.Linear(state_dim + hidden_dim + 1, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.to_ze = nn.Linear(hidden_dim, z_dim)

    def forward(self, s: torch.Tensor, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        # s: [B,T,D], a: [B,T] int64, r: [B,T-1] or [B,T]
        B, T, _ = s.shape
        if r.dim() == 2 and r.shape[1] == T - 1:
            r = torch.cat([r, r[:, -1:].clone()], dim=1)
        if r.dim() == 2:
            r = r.unsqueeze(-1)
        a_emb = self.action_emb(a)
        x = torch.cat([s, a_emb, r], dim=-1)
        x = self.in_proj(x)
        out, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]  # [B,H]
        z_e = self.to_ze(h_last)
        return z_e


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, z_dim: int, beta: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.z_dim = z_dim
        self.beta = beta
        self.codebook = nn.Embedding(codebook_size, z_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z_e: [B, z_dim]
        # Compute nearest code
        z_e2 = (z_e ** 2).sum(dim=1, keepdim=True)  # [B,1]
        e2 = (self.codebook.weight ** 2).sum(dim=1).unsqueeze(0)  # [1,K]
        # dist = ||z_e - e||^2 = z_e^2 + e^2 - 2 z_e·e
        dist = z_e2 + e2 - 2.0 * z_e @ self.codebook.weight.t()  # [B,K]
        code_idx = torch.argmin(dist, dim=1)  # [B]
        z_q = self.codebook(code_idx)  # [B,z_dim]

        # VQ losses
        # commitment: ||sg[z_e] - z_q||^2
        # codebook:   ||z_e - sg[z_q]||^2
        commitment = F.mse_loss(z_q, z_e.detach())
        codebook = F.mse_loss(z_e, z_q.detach())
        vq_loss = commitment + self.beta * codebook

        # straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, code_idx, vq_loss


class ActionDecoder(nn.Module):
    def __init__(self, state_dim: int, z_dim: int, hidden_dim: int, action_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(state_dim + z_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, s: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
        # s: [B,T,D], z_q: [B,z_dim]
        B, T, _ = s.shape
        z = z_q.unsqueeze(1).expand(B, T, z_q.shape[-1])
        x = torch.cat([s, z], dim=-1)
        x = self.in_proj(x)
        out, _ = self.lstm(x)
        logits = self.head(out)  # [B,T,A]
        return logits


class VQArchetype(nn.Module):
    def __init__(self, cfg: VQConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = LSTMEncoder(cfg.state_dim, cfg.action_dim, cfg.hidden_dim, cfg.z_dim, cfg.num_layers, cfg.dropout)
        self.vq = VectorQuantizer(cfg.codebook_size, cfg.z_dim, cfg.beta)
        self.decoder = ActionDecoder(cfg.state_dim, cfg.z_dim, cfg.hidden_dim, cfg.action_dim, cfg.num_layers, cfg.dropout)

    def forward(self, s: torch.Tensor, a: torch.Tensor, r: torch.Tensor):
        z_e = self.encoder(s, a, r)
        z_q, code_idx, vq_loss = self.vq(z_e)
        logits = self.decoder(s, z_q)
        return logits, code_idx, vq_loss

    def loss(self, s: torch.Tensor, a: torch.Tensor, r: torch.Tensor):
        logits, code_idx, vq_loss = self.forward(s, a, r)
        rec_loss = F.cross_entropy(logits.reshape(-1, self.cfg.action_dim), a.reshape(-1))
        total = rec_loss + vq_loss
        return total, {"loss": total.detach(), "rec_loss": rec_loss.detach(), "vq_loss": vq_loss.detach()}, code_idx
