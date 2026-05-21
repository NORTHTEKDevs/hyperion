"""Straight-through estimator for sign() and soft binarization helpers."""

from __future__ import annotations

import torch
from torch import Tensor


class _SignSTE(torch.autograd.Function):
    """sign(x) forward, identity backward with optional clipping at |x|<=1.

    Per BNN literature (Bengio 2013, Courbariaux 2016), this STE is biased
    but trainable. Clipping the backward pass at |x|<=1 prevents gradient
    explosion when the MLP output saturates.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        # Identity gradient where |x|<=1, zero otherwise. Standard STE.
        mask = (x.abs() <= 1.0).to(grad_output.dtype)
        return grad_output * mask


def sign_ste(x: Tensor) -> Tensor:
    """Differentiable sign() via straight-through estimator."""
    return _SignSTE.apply(x)


def soft_binarize(x: Tensor, beta: float = 1.0) -> Tensor:
    """tanh(beta*x). Annealing beta from 1 to 10 over training approaches sign()
    while keeping gradients alive. Use this during training; switch to sign()
    or sign_ste at inference.
    """
    return torch.tanh(beta * x)
