"""Outer-product associative memory.

Stores (key, value) pairs as a single bundled hypervector:
    M += bind(key, value)
Retrieves the value for a query key:
    value_approx = unbind(M, key)
    value_clean = cleanup(value_approx, codebook)

Per Plate (1995) capacity bound: ~D / (2 ln N) reliable retrievals at codebook
size N. At D=10000, N=64 vocab: ~1200 pairs. At N=1000: ~720.
"""

from __future__ import annotations

import torch
from torch import Tensor

from vsa_core import bind, unbind
from vsa_core.cleanup import cleanup, similarity


class AssociativeMemory:
    """A bundled-bind memory of (key, value) pairs.

    Internal state is real-valued (the unbinarized bundle). Retrieval
    optionally sign-binarizes before unbinding, but keeping it real
    improves SNR for small training sets.
    """

    def __init__(self, d: int, device: torch.device | str = "cpu") -> None:
        self.d = d
        self.device = torch.device(device)
        self.state: Tensor = torch.zeros(d, device=self.device)
        self.n_stored = 0

    def store(self, key: Tensor, value: Tensor) -> None:
        """Add a (key, value) pair to memory."""
        self.state = self.state + bind(key, value)
        self.n_stored += 1

    def store_batch(self, keys: Tensor, values: Tensor) -> None:
        """Add a batch of pairs. keys/values: (N, D)."""
        bound = bind(keys, values)  # (N, D)
        self.state = self.state + bound.sum(dim=0)
        self.n_stored += keys.shape[0]

    def retrieve_raw(self, key: Tensor) -> Tensor:
        """Return the unbound (still real-valued) value HV."""
        return unbind(self.state, key)

    def retrieve(self, key: Tensor, codebook: Tensor) -> tuple[Tensor, Tensor]:
        """Return (best_codebook_idx, best_cos_sim) after cleanup."""
        raw = self.retrieve_raw(key)
        return cleanup(raw, codebook)

    def retrieve_topk(
        self, key: Tensor, codebook: Tensor, k: int = 5
    ) -> tuple[Tensor, Tensor]:
        """Return (top_k_indices, top_k_similarities)."""
        raw = self.retrieve_raw(key)
        sims = similarity(raw, codebook)
        return sims.topk(k)

    def clear(self) -> None:
        self.state = torch.zeros(self.d, device=self.device)
        self.n_stored = 0

    def __len__(self) -> int:
        return self.n_stored
