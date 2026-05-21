"""Core VSA operations on bipolar hypervectors.

All ops accept torch.Tensor with last dim = D and arbitrary leading batch dims.
Bipolar HVs live in {-1,+1}^D as float32.
"""

from __future__ import annotations

import torch
from torch import Tensor


def bind(a: Tensor, b: Tensor) -> Tensor:
    """Circular convolution via FFT, normalized so norm is preserved in expectation.

    For bipolar a, b in {-1,+1}^D with ||a||=||b||=sqrt(D), the unnormalized
    result has norm ~D. Dividing by sqrt(D) restores ||bind(a,b)|| ~ sqrt(D),
    matching Plate's HRR convention and keeping intermediate values O(1) for
    cleaner gradient flow.
    """
    d = a.shape[-1]
    A = torch.fft.fft(a, dim=-1)
    B = torch.fft.fft(b, dim=-1)
    return torch.fft.ifft(A * B, dim=-1).real / (d ** 0.5)


def unbind(c: Tensor, b: Tensor) -> Tensor:
    """Approximate inverse of bind: recovers a from bind(a, b) given b.

    Uses circular correlation with matching sqrt(D) normalization.
    For random bipolar b, |fft(b)|^2 ~ D per component, so the round trip
    bind/unbind recovers a within additive noise from the non-flat fft
    magnitude spectrum of b.
    """
    d = c.shape[-1]
    C = torch.fft.fft(c, dim=-1)
    B = torch.fft.fft(b, dim=-1)
    return torch.fft.ifft(C * torch.conj(B), dim=-1).real / (d ** 0.5)


def bundle(vectors: Tensor) -> Tensor:
    """Sum + sign over the second-to-last dim. (..., N, D) -> (..., D).

    Hard bundle for inference. For training, use bundle_soft which keeps
    gradients alive through a tanh(beta*x) projection.
    """
    summed = vectors.sum(dim=-2)
    return torch.sign(summed)


def bundle_soft(vectors: Tensor, beta: float = 1.0) -> Tensor:
    """tanh-projected bundle. Differentiable everywhere.

    beta controls the sharpness: beta=1 is soft, beta=10 approaches sign().
    """
    summed = vectors.sum(dim=-2)
    return torch.tanh(beta * summed)


def permute(v: Tensor, shift: int = 1) -> Tensor:
    """Cyclic rotation by `shift` positions along the last dim."""
    return torch.roll(v, shifts=shift, dims=-1)


def recursive_bind(hvs: Tensor, beta: float | None = None) -> Tensor:
    """Recursive-bind sequence encoding: S_n = HV_n bind permute(S_{n-1}).

    Provably distinguishes all sequences of length up to ~sqrt(D) at D-dim
    (Frady et al., arXiv:2201.11691). Use this instead of permute+bundle
    for sequences longer than ~200 tokens at D=10K.

    Input: (..., N, D). Output: (..., D) -- the final cumulative state.
    If beta is given, applies tanh(beta*x) after each bind for soft binarization.
    """
    n = hvs.shape[-2]
    s = hvs[..., 0, :]
    for i in range(1, n):
        s_rolled = torch.roll(s, shifts=1, dims=-1)
        s = bind(hvs[..., i, :], s_rolled)
        if beta is not None:
            s = torch.tanh(beta * s)
        else:
            s = torch.sign(s)
    return s
