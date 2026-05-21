"""Autonomous rule discovery: group training examples by modifier value
and extract a (pattern, residual) HV pair per group, without being told
which examples belong to which modifier.

This is the "learning" step the v0.1 reasoner was missing. Input is a flat
list of (input_slots, output_slots) pairs. Output is a dict of discovered
modifier rules keyed by the modifier symbol index.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable




def group_by_modifier(
    examples: Iterable[tuple[dict[int, int], dict[int, int]]],
    modifier_role: int,
    verb_role: int,
) -> tuple[dict[int, list[tuple[dict[int, int], dict[int, int]]]], dict[int, dict[int, int]]]:
    """Partition examples into:
        - modified: dict[modifier_idx -> list of (input_slots, output_slots)]
        - bare:     dict[verb_idx -> bare output_slots]

    A "bare" example is one whose input_slots lacks modifier_role.
    A "modified" example is one whose input_slots has modifier_role set.

    The bare lookup is later used to recover each verb's output symbol.
    """
    modified: dict[int, list] = defaultdict(list)
    bare: dict[int, dict[int, int]] = {}
    for input_slots, output_slots in examples:
        if modifier_role in input_slots:
            modifier_idx = input_slots[modifier_role]
            modified[modifier_idx].append((input_slots, output_slots))
        else:
            verb_idx = input_slots[verb_role]
            bare[verb_idx] = output_slots
    return dict(modified), bare


def discover_verb_output_symbol(
    bare_output_slots: dict[int, int],
    primary_out_role: int,
) -> int:
    """Given a bare verb's output slot dict, return the verb's output symbol.

    For TinySCAN, bare outputs are single-slot {ROLE_OUT1: SYM_INDEX}. The
    verb's output symbol is whatever is in that slot. This is the
    information-theoretic minimum needed to apply modifier rules at test time.
    """
    if primary_out_role not in bare_output_slots:
        raise ValueError(
            f"bare output missing primary slot {primary_out_role}: {bare_output_slots}"
        )
    return bare_output_slots[primary_out_role]
