"""Cleanup memory: nearest-neighbor retrieval against a codebook."""

from __future__ import annotations

from torch import Tensor


def similarity(query: Tensor, codebook: Tensor) -> Tensor:
    """Cosine similarity. query: (..., D). codebook: (N, D). -> (..., N)."""
    q_norm = query / (query.norm(dim=-1, keepdim=True) + 1e-8)
    c_norm = codebook / (codebook.norm(dim=-1, keepdim=True) + 1e-8)
    return q_norm @ c_norm.T


def cleanup(query: Tensor, codebook: Tensor) -> tuple[Tensor, Tensor]:
    """Return (best_index, best_similarity). Argmax cosine sim."""
    sims = similarity(query, codebook)
    best_sim, best_idx = sims.max(dim=-1)
    return best_idx, best_sim
