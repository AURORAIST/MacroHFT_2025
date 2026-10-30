"""Stage 3 policy adapter network.

We implement a small actor-critic that uses AdaLN to condition market features on
archetype context.

- input split: x_market (D), x_ctx (C)
- apply AdaLN on market features
- MLP trunk
- actor logits over 3 actions (-1,0,+1)
- critic value
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from model.adaln import AdaLayerNorm


@dataclass
class AdapterConfig:
    market_dim: int
    ctx_dim: int
    hidden: int = 256


class AdapterActorCritic(nn.Module):
    def __init__(self, cfg: AdapterConfig):
        super().__init__()
        self.cfg = cfg
        self.adaln = AdaLayerNorm(cfg.market_dim, cfg.ctx_dim, hidden=128)
        self.trunk = nn.Sequential(
            nn.Linear(cfg.market_dim, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.ReLU(),
        )
        self.pi = nn.Linear(cfg.hidden, 3)
        self.v = nn.Linear(cfg.hidden, 1)

    def forward(self, obs: torch.Tensor):
        # obs: [B, market_dim+ctx_dim]
        m = obs[:, : self.cfg.market_dim]
        c = obs[:, self.cfg.market_dim :]
        m = self.adaln(m, c)
        h = self.trunk(m)
        logits = self.pi(h)
        value = self.v(h).squeeze(-1)
        return logits, value
