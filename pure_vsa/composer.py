"""Rule extraction and application via VSA algebra.

A "rule" is a hypervector that represents the *difference* between a base
input/output mapping and a modified one. Given examples like:
    walk -> [walk]
    walk twice -> [walk, walk]
    jump -> [jump]
    jump twice -> [jump, jump]
We extract:
    rule_twice = mean( out(walk twice) - out(walk),
                       out(jump twice) - out(jump),
                       ... )
At test time:
    out(swim twice) = out(swim) + rule_twice
The bundle of deltas averages out the verb-specific noise and isolates the
function of "twice". This is the VSA-algebraic analog of in-context learning.
"""

from __future__ import annotations

import torch
from torch import Tensor


class RuleComposer:
    """Extracts and applies VSA-algebraic rules."""

    def __init__(self) -> None:
        self.rules: dict[str, Tensor] = {}

    def extract_rule(
        self, name: str, base_outputs: list[Tensor], modified_outputs: list[Tensor]
    ) -> None:
        """Compute a rule HV by averaging deltas (modified - base).

        Inputs must be aligned: modified_outputs[i] is the output of applying
        the rule to base_outputs[i]'s input.
        """
        if len(base_outputs) != len(modified_outputs):
            raise ValueError("base_outputs and modified_outputs must be same length")
        deltas = torch.stack(
            [m - b for b, m in zip(base_outputs, modified_outputs)]
        )
        # bundle = average (NOT sign) — we want the real-valued direction in HV space
        self.rules[name] = deltas.mean(dim=0)

    def apply_rule(self, name: str, base_output: Tensor) -> Tensor:
        """Return base_output + extracted rule HV."""
        if name not in self.rules:
            raise KeyError(f"unknown rule: {name}")
        return base_output + self.rules[name]

    def has_rule(self, name: str) -> bool:
        return name in self.rules
