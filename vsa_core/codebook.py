"""Codebooks of random bipolar HVs + Fractional Power Encoding for positions."""

from __future__ import annotations

import math

import torch
from torch import Tensor


class Codebook:
    """Fixed random bipolar codebook of N items at D dims.

    Quasi-orthogonal by construction: for any two random {-1,+1}^D vectors,
    E[cos sim] = 0 and Var[cos sim] = 1/D, so for D=10K the off-diagonal
    similarities concentrate within ~+/-3/sqrt(D) ~= +/-0.03.
    """

    def __init__(
        self,
        n_items: int,
        d: int,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ) -> None:
        self.n = n_items
        self.d = d
        self.device = torch.device(device)
        g = None
        if seed is not None:
            g = torch.Generator(device="cpu").manual_seed(seed)
        # Sample on CPU then move; bipolar via sign(rand-0.5).
        raw = torch.randint(0, 2, (n_items, d), generator=g, dtype=torch.int8)
        self.hvs: Tensor = (raw * 2 - 1).to(dtype=torch.float32, device=self.device)

    def __getitem__(self, idx: int | Tensor) -> Tensor:
        return self.hvs[idx]

    def all(self) -> Tensor:
        return self.hvs

    def __len__(self) -> int:
        return self.n


def fpe_positions(n: int, d: int, base_seed: int = 0) -> Tensor:
    """Fractional Power Encoding for positions (Komer et al. 2019).

    For each position p in [0..n-1], returns a complex unit vector
    a_base^p in C^D, returned as a real tensor of shape (n, 2D)
    where columns 0..D-1 are the real part and D..2D-1 the imaginary part.

    FPE produces similarity-preserving positional HVs: adjacent positions
    have high cosine similarity, distant positions are quasi-orthogonal.
    Use when sequence length > ~200 tokens and the permute+bundle scheme
    would otherwise saturate.
    """
    g = torch.Generator(device="cpu").manual_seed(base_seed)
    # Random phases per dimension, drawn uniform in [-pi, pi].
    phases = (torch.rand(d, generator=g) * 2.0 - 1.0) * math.pi  # (D,)
    pos = torch.arange(n, dtype=torch.float32).unsqueeze(-1)  # (n,1)
    arg = pos * phases.unsqueeze(0)  # (n, D)
    real = torch.cos(arg)
    imag = torch.sin(arg)
    return torch.cat([real, imag], dim=-1)  # (n, 2D)
