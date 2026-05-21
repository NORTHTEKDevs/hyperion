"""HyperionReasoner: the public-facing high-level API.

Wraps the low-level primitives (facts dict + VSA rules + cleanup) into a single
class with a small, sklearn-style interface:

    reasoner = HyperionReasoner(d=2048, n_symbols=100, n_input_roles=2, max_output_len=4)
    reasoner.fit(training_examples, modifier_role_idx=1, verb_role_idx=0)
    predicted_tokens = reasoner.predict(input_slots, output_length)

Internally:
  - Facts (bare-verb -> output-symbol) live in a Python dict (zero noise).
  - Modifier rules live as (pattern, residual) HV pairs in a dict (pure VSA).
  - Composition at predict() is bind + sum + cleanup on the role-permute chain.

See pure_vsa/README.md for the conceptual story and RESULTS.md for benchmarks.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor

from vsa_core import bind, permute as vsa_permute, unbind
from vsa_core.cleanup import similarity
from vsa_core.codebook import Codebook


@dataclass
class Example:
    """A training example.

    input_slots: dict[role_index -> symbol_index]
        Sparse mapping from input roles (e.g. ROLE_VERB, ROLE_MOD) to symbol
        indices in the symbol codebook. A bare verb has only the verb role
        set; a modified example has both verb and modifier roles.

    output_token_indices: list[int]
        Sequence of output symbol indices, position-aligned. Position i goes
        in output role i.
    """
    input_slots: dict[int, int]
    output_token_indices: list[int]


@dataclass
class HyperionConfig:
    """Configuration for HyperionReasoner.

    d:               hypervector dimension. Trade-off: larger = more capacity
                     and computation. 2048 handles vocab up to 25K verbs;
                     4096-8192 is comfortable for production.
    n_symbols:       size of the symbol codebook (input + output tokens combined).
    n_input_roles:   number of distinct input slot roles (e.g. 2 for verb+modifier).
    max_output_len:  maximum output sequence length. Output role HVs are
                     permute-derived from a single base so rules can be shifted.
    output_vocab_offset: where in the symbol codebook output tokens begin.
                     Cleanup at decode is restricted to symbols[output_vocab_offset:].
    seed:            for reproducible codebook initialization.
    """
    d: int
    n_symbols: int
    n_input_roles: int
    max_output_len: int
    output_vocab_offset: int
    seed: int = 0


class HyperionReasoner:
    """Pure-VSA reasoner with facts-in-dict + rules-as-(pattern, residual)."""

    def __init__(self, cfg: HyperionConfig) -> None:
        self.cfg = cfg
        # symbol codebook: one HV per token in the combined vocabulary.
        self.symbols = Codebook(cfg.n_symbols, cfg.d, seed=cfg.seed)
        # input-role codebook: one HV per distinct input role.
        self.input_roles = Codebook(cfg.n_input_roles, cfg.d, seed=cfg.seed + 1)
        # output roles are permute-derived from a single base, so a rule
        # extracted at output position 0 can be re-applied at any position k
        # via permute(pattern, shift=k).
        base = Codebook(1, cfg.d, seed=cfg.seed + 2)[0]
        self.output_roles: Tensor = torch.stack(
            [vsa_permute(base, shift=i) for i in range(cfg.max_output_len)]
        )
        # facts: dict from frozenset(input_slots.items()) -> output_token_indices.
        # Used for primitive lookups (bare verbs) at zero cross-talk.
        self.facts: dict[frozenset, list[int]] = {}
        # modifier rules: dict from modifier_symbol_idx -> (pattern, residual).
        self.rules: dict[int, tuple[Tensor, Tensor]] = {}

    # ------------------------------------------------------------------
    # Training: store facts + extract rules
    # ------------------------------------------------------------------

    def fit(
        self,
        examples: list[Example],
        modifier_role_idx: int,
        verb_role_idx: int,
    ) -> dict[int, int]:
        """Single-pass training.

        Stores bare examples (no modifier role set) as facts.
        Groups modified examples by modifier symbol and extracts a rule per group.

        Returns: dict mapping modifier_symbol_idx -> number of training examples
        used to extract that rule.
        """
        # 1. Store bare examples as facts.
        for ex in examples:
            if modifier_role_idx not in ex.input_slots:
                key = frozenset(ex.input_slots.items())
                self.facts[key] = list(ex.output_token_indices)

        # 2. Group modified examples by modifier symbol.
        modified_groups: dict[int, list[Example]] = defaultdict(list)
        for ex in examples:
            if modifier_role_idx in ex.input_slots:
                mod_sym = ex.input_slots[modifier_role_idx]
                modified_groups[mod_sym].append(ex)

        # 3. Extract a rule per modifier from clean encoded outputs (no memory).
        rule_sizes: dict[int, int] = {}
        for mod_sym, group in modified_groups.items():
            patterns = []
            for ex in group:
                # Recover the verb's output symbol from the bare-verb fact.
                verb_sym = ex.input_slots[verb_role_idx]
                bare_key = frozenset([(verb_role_idx, verb_sym)])
                if bare_key not in self.facts:
                    # No bare-verb fact available for this verb; skip.
                    continue
                verb_out_sym = self.facts[bare_key][0]
                encoded_out = self._encode_output(ex.output_token_indices)
                verb_sym_hv = self.symbols[verb_out_sym]
                patterns.append(unbind(encoded_out, verb_sym_hv))

            if not patterns:
                continue

            pattern = torch.stack(patterns).mean(dim=0)

            residuals = []
            for ex in group:
                verb_sym = ex.input_slots[verb_role_idx]
                bare_key = frozenset([(verb_role_idx, verb_sym)])
                if bare_key not in self.facts:
                    continue
                verb_out_sym = self.facts[bare_key][0]
                encoded_out = self._encode_output(ex.output_token_indices)
                verb_sym_hv = self.symbols[verb_out_sym]
                residuals.append(encoded_out - bind(pattern, verb_sym_hv))

            residual = torch.stack(residuals).mean(dim=0)
            self.rules[mod_sym] = (pattern, residual)
            rule_sizes[mod_sym] = len(patterns)

        return rule_sizes

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        input_slots: dict[int, int],
        output_length: int,
        modifier_role_idx: int,
        verb_role_idx: int,
    ) -> list[int]:
        """Predict output token sequence for a single input.

        For bare inputs (no modifier role set): direct fact lookup.
        For modified inputs: look up bare verb's output symbol, apply rule.
        """
        if modifier_role_idx not in input_slots:
            # Bare lookup.
            key = frozenset(input_slots.items())
            if key in self.facts:
                return list(self.facts[key])
            # Unknown bare input -- can't predict.
            return []

        # Modified input. Look up the verb's bare output symbol, then apply rule.
        verb_sym = input_slots[verb_role_idx]
        bare_key = frozenset([(verb_role_idx, verb_sym)])
        if bare_key not in self.facts:
            return []  # unknown verb
        verb_out_sym = self.facts[bare_key][0]

        mod_sym = input_slots[modifier_role_idx]
        if mod_sym not in self.rules:
            return []  # unknown modifier
        pattern, residual = self.rules[mod_sym]
        verb_sym_hv = self.symbols[verb_out_sym]
        output_hv = bind(pattern, verb_sym_hv) + residual
        return self._decode_output(output_hv, output_length)

    def predict_clauses(
        self,
        clauses: list[dict[int, int]],
        output_lengths: list[int],
        modifier_role_idx: int,
        verb_role_idx: int,
    ) -> list[int]:
        """Predict for a nested input expressed as a list of clauses.

        Each clause has its own input_slots dict and own expected length.
        Clauses are composed with output-role shifts so clause-2's output
        slots start where clause-1's end. Uses permute on rule HVs for the shift.
        """
        if len(clauses) != len(output_lengths):
            raise ValueError("clauses and output_lengths must align")

        output_hv = torch.zeros(self.cfg.d)
        role_shift = 0
        for clause_slots, clause_len in zip(clauses, output_lengths):
            clause_out = self._construct_clause_output(
                clause_slots,
                role_shift,
                modifier_role_idx,
                verb_role_idx,
            )
            output_hv = output_hv + clause_out
            role_shift += clause_len

        total_len = sum(output_lengths)
        return self._decode_output(output_hv, total_len)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_output(self, token_indices: list[int]) -> Tensor:
        """Encode an output token sequence as a bundle of role-filler bindings."""
        terms = [
            bind(self.output_roles[i], self.symbols[t])
            for i, t in enumerate(token_indices)
        ]
        return torch.stack(terms).sum(dim=0)

    def _decode_output(self, output_hv: Tensor, n_slots: int) -> list[int]:
        """Decode n_slots positions by unbinding each role and cleanup against
        the output-vocabulary portion of the symbol codebook."""
        out_cb = self.symbols.all()[self.cfg.output_vocab_offset:]
        tokens = []
        for i in range(n_slots):
            slot_hv = unbind(output_hv, self.output_roles[i])
            sims = similarity(slot_hv, out_cb)
            best_local = sims.argmax(dim=-1).item()
            tokens.append(self.cfg.output_vocab_offset + int(best_local))
        return tokens

    def _construct_clause_output(
        self,
        clause_slots: dict[int, int],
        role_shift: int,
        modifier_role_idx: int,
        verb_role_idx: int,
    ) -> Tensor:
        """Build the output HV for one clause, placed at role_shift onward."""
        verb_sym = clause_slots[verb_role_idx]
        bare_key = frozenset([(verb_role_idx, verb_sym)])
        if bare_key not in self.facts:
            return torch.zeros(self.cfg.d)
        verb_out_sym = self.facts[bare_key][0]

        if modifier_role_idx not in clause_slots:
            # Bare clause: just bind the verb at the shifted position.
            return bind(self.output_roles[role_shift], self.symbols[verb_out_sym])

        mod_sym = clause_slots[modifier_role_idx]
        if mod_sym not in self.rules:
            return torch.zeros(self.cfg.d)
        pattern, residual = self.rules[mod_sym]
        # Shift by permute -- valid because output_roles[i] = permute(base, i).
        pattern_shifted = vsa_permute(pattern, shift=role_shift)
        residual_shifted = vsa_permute(residual, shift=role_shift)
        return bind(pattern_shifted, self.symbols[verb_out_sym]) + residual_shifted

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def num_facts(self) -> int:
        return len(self.facts)

    def num_rules(self) -> int:
        return len(self.rules)

    def __repr__(self) -> str:
        return (
            f"HyperionReasoner(d={self.cfg.d}, "
            f"n_symbols={self.cfg.n_symbols}, "
            f"facts={self.num_facts()}, rules={self.num_rules()})"
        )
