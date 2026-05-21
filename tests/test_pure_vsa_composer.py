"""Tests for the RuleComposer."""

from __future__ import annotations

import pytest
import torch

from pure_vsa.composer import RuleComposer


D = 1024


def test_rule_extract_and_apply():
    """If base + delta == modified, extract_rule(base, modified) gives delta back."""
    torch.manual_seed(0)
    composer = RuleComposer()
    base = torch.randn(D)
    delta = torch.randn(D)
    modified = base + delta
    composer.extract_rule("twice", [base], [modified])
    applied = composer.apply_rule("twice", base)
    assert torch.allclose(applied, modified, atol=1e-5)


def test_rule_averages_over_examples():
    """Rule extraction averages the delta across multiple (base, modified) pairs."""
    torch.manual_seed(0)
    composer = RuleComposer()
    bases = [torch.randn(D) for _ in range(5)]
    delta = torch.randn(D)
    modifieds = [b + delta + torch.randn(D) * 0.01 for b in bases]
    composer.extract_rule("twice", bases, modifieds)
    # extracted rule should be close to delta (averaged noise should cancel)
    recovered_delta = composer.rules["twice"]
    cos_sim = torch.nn.functional.cosine_similarity(
        recovered_delta.unsqueeze(0), delta.unsqueeze(0)
    ).item()
    assert cos_sim > 0.95


def test_rule_unknown_raises():
    composer = RuleComposer()
    with pytest.raises(KeyError):
        composer.apply_rule("does_not_exist", torch.zeros(D))


def test_rule_mismatched_lengths_raises():
    composer = RuleComposer()
    with pytest.raises(ValueError):
        composer.extract_rule(
            "x", [torch.zeros(D), torch.zeros(D)], [torch.zeros(D)]
        )
