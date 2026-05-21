"""PureVSAReasoner: associative memory + rule composer, no neural networks.

The "model" is just two state objects:
  * AssociativeMemory holds (input_encoding -> output_encoding) pairs as
    a bundled bind sum.
  * RuleComposer holds extracted compositional rules as real-valued HVs.

Inputs and outputs are encoded as VSA expressions via role-filler binding.
Output decoding is done by repeatedly unbinding role HVs and cleaning up
against a token codebook.
"""

from __future__ import annotations

import torch
from torch import Tensor

from vsa_core import bind, unbind
from vsa_core.cleanup import cleanup, similarity
from vsa_core.codebook import Codebook

from pure_vsa.composer import RuleComposer
from pure_vsa.memory import AssociativeMemory


class PureVSAReasoner:
    """A reasoner that learns by memory accumulation and composes by VSA algebra."""

    def __init__(
        self,
        d: int,
        symbol_codebook: Codebook,
        role_codebook: Codebook,
        device: torch.device | str = "cpu",
    ) -> None:
        """
        symbol_codebook: codebook for content symbols (words, output tokens).
        role_codebook:   codebook for role HVs (e.g. verb_role, mod_role, out1_role, ...)
                         used to label slots in compositional expressions.
        """
        self.d = d
        self.symbols = symbol_codebook
        self.roles = role_codebook
        self.device = torch.device(device)
        self.memory = AssociativeMemory(d=d, device=self.device)
        self.composer = RuleComposer()
        # Modifier rules: name -> (pattern, residual) tuple.
        # Apply via bind(pattern, verb_symbol) + residual. Two components so the
        # mechanism handles both verb-multiplier rules (residual=0) and
        # constant-prepend rules (residual carries the constant term).
        self.modifier_patterns: dict[str, Tensor] = {}
        self.modifier_residuals: dict[str, Tensor] = {}

    # ------------------------------------------------------------------
    # encoding helpers
    # ------------------------------------------------------------------

    def encode_struct(self, slots: dict[int, int]) -> Tensor:
        """Encode a structured input/output as a bundle of role-filler bindings.

        slots: dict from role_index -> symbol_index.
        Returns the bundle (real-valued, not sign-binarized).
        """
        terms = []
        for role_idx, sym_idx in slots.items():
            terms.append(bind(self.roles[role_idx], self.symbols[sym_idx]))
        return torch.stack(terms).sum(dim=0)

    # ------------------------------------------------------------------
    # learning (memory accumulation only — no gradients)
    # ------------------------------------------------------------------

    def remember_example(
        self, input_slots: dict[int, int], output_slots: dict[int, int]
    ) -> None:
        """Store (input, output) as a single (key, value) pair in memory."""
        key = self.encode_struct(input_slots)
        value = self.encode_struct(output_slots)
        # Re-binarize the key for storage so unbind round-trip stays well-conditioned.
        # Value stays real-valued so we don't lose information at storage time.
        key_hard = torch.sign(key)
        self.memory.store(key_hard, value)

    def extract_rule_from_pairs(
        self,
        rule_name: str,
        base_examples: list[dict[int, int]],
        modified_examples: list[dict[int, int]],
    ) -> None:
        """Given lists of (base_input, modified_input) slot dicts that share
        a single transformation, extract that transformation as a VSA delta.

        Internally: query memory for each base and modified input, then
        average their output deltas.
        """
        base_outs = []
        mod_outs = []
        for b, m in zip(base_examples, modified_examples):
            base_outs.append(self.memory.retrieve_raw(torch.sign(self.encode_struct(b))))
            mod_outs.append(self.memory.retrieve_raw(torch.sign(self.encode_struct(m))))
        self.composer.extract_rule(rule_name, base_outs, mod_outs)

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def recall(self, input_slots: dict[int, int]) -> Tensor:
        """Return raw retrieved output HV (real-valued)."""
        key = torch.sign(self.encode_struct(input_slots))
        return self.memory.retrieve_raw(key)

    def recall_with_rule(
        self,
        base_input_slots: dict[int, int],
        rule_name: str,
    ) -> Tensor:
        """Apply an extracted (additive) rule to a base-query output."""
        base_out = self.recall(base_input_slots)
        return self.composer.apply_rule(rule_name, base_out)

    # ------------------------------------------------------------------
    # Modifier patterns: role-templates bound with a verb's output symbol.
    # The "right" rule mechanism for SCAN-style structural transformations.
    # ------------------------------------------------------------------

    def extract_modifier_pattern(
        self,
        modifier_name: str,
        examples: list[tuple[dict[int, int], int]],
    ) -> None:
        """Extract a modifier's (pattern, residual) HV pair from training examples.

        examples: list of (modified_input_slots, verb_output_symbol_index).

        Two-step extraction:
          1. pattern = mean_V [ unbind(recall(V, M), V_out_symbol) ]
             Captures role-templates that depend on the verb (e.g. out1+out2 for "twice").
          2. residual = mean_V [ recall(V, M) - bind(pattern, V_out_symbol) ]
             Captures verb-independent constant terms (e.g. bind(out1, LEFT) for "left").

        For "twice"/"thrice", residual ~= 0. For "left"/"right", residual carries
        the constant prefix. Apply via bind(pattern, verb_sym) + residual.
        """
        patterns = []
        for input_slots, verb_sym_idx in examples:
            output_hv = self.recall(input_slots)
            verb_sym_hv = self.symbols[verb_sym_idx]
            patterns.append(unbind(output_hv, verb_sym_hv))
        pattern = torch.stack(patterns).mean(dim=0)
        self.modifier_patterns[modifier_name] = pattern

        # Now extract the residual that the pattern-only model cannot account for.
        residuals = []
        for input_slots, verb_sym_idx in examples:
            output_hv = self.recall(input_slots)
            verb_sym_hv = self.symbols[verb_sym_idx]
            predicted_by_pattern = bind(pattern, verb_sym_hv)
            residuals.append(output_hv - predicted_by_pattern)
        self.modifier_residuals[modifier_name] = torch.stack(residuals).mean(dim=0)

    def apply_modifier_pattern(
        self,
        modifier_name: str,
        verb_output_symbol_idx: int,
    ) -> Tensor:
        """Construct predicted output: bind(pattern, verb_sym) + residual."""
        if modifier_name not in self.modifier_patterns:
            raise KeyError(f"unknown modifier pattern: {modifier_name}")
        pattern = self.modifier_patterns[modifier_name]
        residual = self.modifier_residuals[modifier_name]
        verb_sym_hv = self.symbols[verb_output_symbol_idx]
        return bind(pattern, verb_sym_hv) + residual

    def discover_and_extract_rules(
        self,
        examples: list[tuple[dict[int, int], dict[int, int]]],
        modifier_role: int,
        verb_role: int,
        primary_out_role: int,
    ) -> dict[int, str]:
        """End-to-end autonomous rule discovery.

        Given a flat training set of (input_slots, output_slots) pairs:
          1. Group examples into bare-verb cases and modified cases.
          2. For each discovered modifier value, recover each training verb's
             output symbol from its bare counterpart.
          3. Extract a (pattern, residual) rule for each modifier.

        Stores rules in self.modifier_patterns / self.modifier_residuals keyed
        by the modifier's symbol index (converted to string for dict key).

        Returns a map of {modifier_idx: rule_name} so callers know which keys
        were registered.
        """
        from pure_vsa.discovery import (  # noqa: PLC0415
            discover_verb_output_symbol,
            group_by_modifier,
        )

        # Memorize everything first.
        for input_slots, output_slots in examples:
            self.remember_example(input_slots, output_slots)

        modified, bare = group_by_modifier(examples, modifier_role, verb_role)
        registered: dict[int, str] = {}
        for modifier_idx, group in modified.items():
            extracted_examples = []
            for input_slots, _output_slots in group:
                verb_idx = input_slots[verb_role]
                if verb_idx not in bare:
                    # cannot infer this verb's output symbol; skip this example
                    continue
                verb_out_sym = discover_verb_output_symbol(
                    bare[verb_idx], primary_out_role
                )
                extracted_examples.append((input_slots, verb_out_sym))
            if not extracted_examples:
                continue
            rule_name = str(modifier_idx)
            self.extract_modifier_pattern(rule_name, extracted_examples)
            registered[modifier_idx] = rule_name
        return registered

    # ------------------------------------------------------------------
    # N-ary rules (generalizes modifier_pattern to multiple verb arguments).
    # ------------------------------------------------------------------

    def extract_nary_rule(
        self,
        rule_name: str,
        examples: list[tuple[dict[int, int], list[int]]],
    ) -> None:
        """Extract an N-ary rule: N patterns (one per verb argument) + 1 residual.

        examples: list of (input_slots, list_of_verb_output_symbol_indices).
            For "walk and jump" -> [W, J] we'd pass
            ({verb1: walk, verb2: jump}, [W_idx, J_idx]).

        Math:
          pattern_i = mean_examples [ unbind(recall(input), V_i_sym) ]
          residual  = mean_examples [ recall(input) - sum_i bind(pattern_i, V_i_sym) ]

        Apply via sum_i bind(pattern_i, V_i_sym) + residual.
        """
        if not examples:
            raise ValueError("no examples")
        n_args = len(examples[0][1])
        if not all(len(verbs) == n_args for _, verbs in examples):
            raise ValueError("inconsistent number of verb args across examples")

        # extract per-arg patterns
        patterns_per_arg: list[Tensor] = []
        for arg_i in range(n_args):
            per_example = []
            for input_slots, verb_idxs in examples:
                output_hv = self.recall(input_slots)
                vi_hv = self.symbols[verb_idxs[arg_i]]
                per_example.append(unbind(output_hv, vi_hv))
            patterns_per_arg.append(torch.stack(per_example).mean(dim=0))

        # extract residual
        residuals = []
        for input_slots, verb_idxs in examples:
            output_hv = self.recall(input_slots)
            predicted = torch.zeros_like(output_hv)
            for arg_i, p in enumerate(patterns_per_arg):
                predicted = predicted + bind(p, self.symbols[verb_idxs[arg_i]])
            residuals.append(output_hv - predicted)

        # store
        self.modifier_patterns[rule_name] = torch.stack(patterns_per_arg)
        self.modifier_residuals[rule_name] = torch.stack(residuals).mean(dim=0)

    def apply_nary_rule(
        self,
        rule_name: str,
        verb_output_symbol_idxs: list[int],
    ) -> Tensor:
        """Apply an N-ary rule: sum_i bind(pattern_i, verb_i_sym) + residual."""
        if rule_name not in self.modifier_patterns:
            raise KeyError(f"unknown rule: {rule_name}")
        patterns = self.modifier_patterns[rule_name]
        if patterns.ndim != 2:
            raise ValueError(
                f"rule {rule_name} is 1-ary; use apply_modifier_pattern instead"
            )
        if patterns.shape[0] != len(verb_output_symbol_idxs):
            raise ValueError(
                f"rule {rule_name} expects {patterns.shape[0]} verb args, "
                f"got {len(verb_output_symbol_idxs)}"
            )
        residual = self.modifier_residuals[rule_name]
        out = residual.clone()
        for i, v_idx in enumerate(verb_output_symbol_idxs):
            out = out + bind(patterns[i], self.symbols[v_idx])
        return out

    def lookup_verb_output_symbol(
        self, verb_input_idx: int, verb_role_idx: int, out_role_idx: int
    ) -> int:
        """Decode the verb's output symbol from the bare-verb recall.

        Used at test time when the verb appears in compositional inputs we
        haven't seen, but its bare form is in memory.
        """
        from pure_vsa.tinyscan import N_ACTIONS, N_MODIFIERS  # noqa: PLC0415
        bare_slots = {verb_role_idx: verb_input_idx}
        bare_out = self.recall(bare_slots)
        slot_hv = unbind(bare_out, self.roles[out_role_idx])
        # restrict similarity to output-token symbols (last block of codebook)
        out_offset = N_ACTIONS + N_MODIFIERS
        sims = similarity(slot_hv, self.symbols.all()[out_offset:])
        best_local = sims.argmax(dim=-1).item()
        return out_offset + int(best_local)

    # ------------------------------------------------------------------
    # Role-shifted rule application for nested compositions.
    # Requires output role HVs to be related by VSA permutation:
    #   role_out_i = permute(base, shift=i)
    # Then permute^k(pattern) shifts a rule's output positions by k.
    # ------------------------------------------------------------------

    def apply_modifier_pattern_shifted(
        self,
        modifier_name: str,
        verb_output_symbol_idx: int,
        role_shift: int,
    ) -> Tensor:
        """Like apply_modifier_pattern, but permute the pattern + residual by
        role_shift before binding. Used to place a modifier rule's output
        tokens at output positions [shift, shift+1, ...] instead of [0, 1, ...].
        """
        from vsa_core import permute as vsa_permute  # noqa: PLC0415
        if modifier_name not in self.modifier_patterns:
            raise KeyError(f"unknown modifier pattern: {modifier_name}")
        pattern = self.modifier_patterns[modifier_name]
        residual = self.modifier_residuals[modifier_name]
        pattern_shifted = vsa_permute(pattern, shift=role_shift)
        residual_shifted = vsa_permute(residual, shift=role_shift)
        verb_sym_hv = self.symbols[verb_output_symbol_idx]
        return bind(pattern_shifted, verb_sym_hv) + residual_shifted

    def apply_bare_at_position(
        self,
        verb_output_symbol_idx: int,
        bare_output_role_hv: Tensor,
    ) -> Tensor:
        """Return bind(role_hv, verb_sym). Used to place a bare verb's output
        at a specific output position (no modifier rule needed)."""
        verb_sym_hv = self.symbols[verb_output_symbol_idx]
        return bind(bare_output_role_hv, verb_sym_hv)

    def decode_slots(
        self, output_hv: Tensor, roles_to_decode: list[int]
    ) -> dict[int, int]:
        """For each role in roles_to_decode, unbind and cleanup against the symbol codebook.

        Returns slot_index -> recovered_symbol_index.
        """
        result: dict[int, int] = {}
        for role_idx in roles_to_decode:
            slot_hv = unbind(output_hv, self.roles[role_idx])
            sym_idx, _ = cleanup(slot_hv, self.symbols.all())
            result[role_idx] = int(sym_idx.item())
        return result

    def decode_slots_with_threshold(
        self,
        output_hv: Tensor,
        roles_to_decode: list[int],
        threshold: float,
    ) -> dict[int, int]:
        """Like decode_slots but only includes slots whose cleanup similarity
        exceeds threshold. Useful when output structure is variable-length
        (some slots empty)."""
        result: dict[int, int] = {}
        for role_idx in roles_to_decode:
            slot_hv = unbind(output_hv, self.roles[role_idx])
            sims = similarity(slot_hv, self.symbols.all())
            best_sim, best_idx = sims.max(dim=-1)
            if best_sim.item() >= threshold:
                result[role_idx] = int(best_idx.item())
        return result
