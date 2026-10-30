"""Adaptive LayerNorm (AdaLN) utilities.

We follow a simple conditioning design:
- normalize x with LayerNorm (no affine)
- produce per-feature (gamma, beta) from conditioning vector c
- output: (1 + gamma) * norm(x) + beta

This is enough for Stage 3 policy adapter conditioning on archetype context.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, cond_dim: int, hidden: int = 128):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * normalized_shape),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: [..., D], cond: [..., C] (broadcastable in prefix dims)."""
        x_n = self.ln(x)
        gb = self.mlp(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        return (1.0 + gamma) * x_n + beta
