"""HyperionSCAN: extension of HyperionReasoner that handles the real SCAN grammar.

Key extensions vs the base HyperionReasoner:

1. **Atom-shape rules** (one per syntactic atom form, not per modifier):
     atom_only:           verb              -> I_VERB
     atom_dir_X:          verb dir          -> I_TURN_X I_VERB
     atom_opposite_X:     verb opposite X   -> I_TURN_X I_TURN_X I_VERB
     atom_around_X:       verb around X     -> (I_TURN_X I_VERB) x 4

2. **Procedural twice/thrice** via VSA permutation:
     twice(atom_out_hv, L) = atom_out_hv + permute(atom_out_hv, shift=L)
     thrice(atom_out_hv, L) = atom_out_hv + permute(atom_out_hv, L) + permute(atom_out_hv, 2L)
   No rule extraction needed -- the operation is structural.

3. **Procedural and/after** via VSA permutation:
     X and Y -> X_hv + permute(Y_hv, shift=len(X))
     X after Y -> Y_hv + permute(X_hv, shift=len(Y))

4. **Special handling for `turn` verb** (no verb_action in output, just turn tokens).

5. **Variable output length** up to MAX_LEN positions. Output role HVs are
   permute-derived from a single base so role-shifting via permute is exact.

Architecture: facts (verb -> action symbol) in dict; atom-shape rules in VSA;
composition is procedural VSA. Compositional generalization to held-out verb:
the verb is looked up from training facts (bare jump IS in training); modifier
rules are extracted from non-jump examples; the held-out jump compositions
are constructed by substituting jump's action symbol into the rules.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor

from vsa_core import bind, permute as vsa_permute, unbind
from vsa_core.cleanup import similarity
from vsa_core.codebook import Codebook

from pure_vsa.scan_runner import (
    Atom,
    Clause,
    ParsedSCAN,
    VERBS,
)


# Symbol index layout (chosen to make math obvious):
#   0..4    : verbs (walk, look, run, jump, turn)
#   5..6    : output action tokens for non-turn verbs (just I_WALK..I_JUMP)
#             actually we use ACTIONS = [I_WALK, I_LOOK, I_RUN, I_JUMP] (4 symbols)
#   7       : I_TURN_LEFT
#   8       : I_TURN_RIGHT
# Plus tokens for directions (left/right) -- these are inputs only, not encoded as output.

INPUT_VERBS = VERBS  # walk, look, run, jump, turn
OUTPUT_TOKENS = ["I_WALK", "I_LOOK", "I_RUN", "I_JUMP", "I_TURN_LEFT", "I_TURN_RIGHT"]
N_INPUT_VERBS = len(INPUT_VERBS)
N_OUTPUT_TOKENS = len(OUTPUT_TOKENS)
TOTAL_SYMBOLS = N_INPUT_VERBS + N_OUTPUT_TOKENS
OUTPUT_OFFSET = N_INPUT_VERBS

# atom-shape categories we extract a rule per
ATOM_SHAPES = [
    "atom_only",        # verb -> I_VERB (length 1)
    "atom_dir_left",    # verb left -> I_TURN_LEFT I_VERB (length 2)
    "atom_dir_right",   # verb right -> I_TURN_RIGHT I_VERB
    "atom_opposite_left",   # verb opposite left -> I_TURN_LEFT I_TURN_LEFT I_VERB (length 3)
    "atom_opposite_right",
    "atom_around_left",     # verb around left -> (I_TURN_LEFT I_VERB) x 4 (length 8)
    "atom_around_right",
]

# turn-special shapes have NO verb_action, so they're constants (no pattern).
TURN_SHAPES = [
    "turn_dir_left",        # turn left -> I_TURN_LEFT
    "turn_dir_right",
    "turn_opposite_left",
    "turn_opposite_right",
    "turn_around_left",
    "turn_around_right",
]

# NOTE: VERB_POSITIONS_BY_SHAPE used to live here as a hand-coded grammar lookup.
# It is now DISCOVERED from training data by SCANHyperion._discover_structure():
# positions where the output token varies with the input verb are verb-bound;
# positions where it is constant are turn/residual constants. This makes the
# mechanism more honest about what is encoded vs. what is learned.


def _atom_shape(atom: Atom) -> str:
    """Classify an atom into one of the shape categories above.

    NOTE: this is the fine-grained shape (direction-specific). For
    direction-agnostic structural rules, use _spatial_category().
    """
    if atom.verb == "turn":
        if atom.spatial == "opposite":
            return f"turn_opposite_{atom.direction}"
        if atom.spatial == "around":
            return f"turn_around_{atom.direction}"
        if atom.direction:
            return f"turn_dir_{atom.direction}"
        return "turn_bare"
    if atom.spatial == "opposite":
        return f"atom_opposite_{atom.direction}"
    if atom.spatial == "around":
        return f"atom_around_{atom.direction}"
    if atom.direction:
        return f"atom_dir_{atom.direction}"
    return "atom_only"


def _spatial_category(atom: Atom) -> str:
    """Direction-agnostic spatial category. Same category for left/right;
    direction is supplied separately at apply time via dir_to_turn_token.

    This lets one structural rule cover both `verb opposite left` and
    `verb opposite right` -- so we can generalize to `opposite right` even
    if training had only `opposite left`.
    """
    if atom.verb == "turn":
        if atom.spatial == "opposite":
            return "turn_opposite"
        if atom.spatial == "around":
            return "turn_around"
        if atom.direction:
            return "turn_dir"
        return "turn_bare"
    if atom.spatial == "opposite":
        return "atom_opposite"
    if atom.spatial == "around":
        return "atom_around"
    if atom.direction:
        return "atom_dir"
    return "atom_only"


def _atom_output_length(shape: str) -> int:
    """How many tokens does this atom-shape produce?

    NOTE: this is no longer used by SCANHyperion (which now discovers
    per-shape output lengths from training data). Kept for backward
    compatibility with capacity studies and tests that import it.
    """
    if shape == "atom_only":
        return 1
    if shape.startswith("atom_dir") or shape == "turn_dir_left" or shape == "turn_dir_right":
        return 2 if shape.startswith("atom_dir") else 1
    if shape.startswith("atom_opposite"):
        return 3
    if shape.startswith("atom_around"):
        return 8
    if shape.startswith("turn_opposite"):
        return 2
    if shape.startswith("turn_around"):
        return 4
    raise ValueError(f"unknown shape {shape}")


@dataclass
class SCANConfig:
    d: int = 4096
    max_output_len: int = 56  # SCAN max is 48; pad to be safe
    seed: int = 0


class SCANHyperion:
    """SCAN-specific Hyperion reasoner.

    Public API:
      .fit(parsed_train_examples)         # extract rules + facts from training
      .predict(parsed_scan)               # produce output token list
      .accuracy(parsed_examples)          # exact-match accuracy
    """

    def __init__(self, cfg: SCANConfig) -> None:
        self.cfg = cfg
        self.symbols = Codebook(TOTAL_SYMBOLS, cfg.d, seed=cfg.seed)
        # output role HVs: role_out_i = permute(base, i)
        base = Codebook(1, cfg.d, seed=cfg.seed + 99)[0]
        self.output_roles: Tensor = torch.stack(
            [vsa_permute(base, shift=i) for i in range(cfg.max_output_len)]
        )

        # Facts: verb -> action token index (None for `turn`).
        self.verb_to_action: dict[str, int | None] = {}

        # Atom-shape rules: shape -> (pattern, residual). Residual carries the
        # verb-independent turn-token bindings; pattern carries the verb-symbol
        # binding positions.
        self.atom_rules: dict[str, tuple[Tensor, Tensor]] = {}

        # Turn-shape outputs: shape -> precomputed output_hv (no verb, constant).
        self.turn_outputs: dict[str, Tensor] = {}

        # Discovered structure per atom shape (no longer hardcoded):
        #   atom_length_by_shape:   shape -> length of the atom's output
        #   verb_positions_by_shape: shape -> list of positions where the verb's
        #                            action token appears in the output
        self.atom_length_by_shape: dict[str, int] = {}
        self.verb_positions_by_shape: dict[str, list[int]] = {}

        # Direction-agnostic structural rules: spatial_category ->
        # (verb_pattern, turn_pattern). Apply via
        #   bind(verb_pattern, verb_hv) + bind(turn_pattern, turn_token_hv)
        # The turn_token_hv comes from dir_to_turn_token at apply time, so
        # one structural rule covers both left/right directions.
        self.structural_rules: dict[str, tuple[Tensor, Tensor]] = {}

        # Per-direction turn token (I_TURN_LEFT, I_TURN_RIGHT).
        # Learned from `verb DIR` examples: position 0 of those outputs is
        # the direction's turn token.
        self.dir_to_turn_token: dict[str, int] = {}

        # Per-spatial-category atom length (direction-agnostic).
        self.atom_length_by_category: dict[str, int] = {}

    # ------------------------------------------------------------------

    def _symbol_idx(self, name: str) -> int:
        if name in INPUT_VERBS:
            return INPUT_VERBS.index(name)
        if name in OUTPUT_TOKENS:
            return OUTPUT_OFFSET + OUTPUT_TOKENS.index(name)
        raise ValueError(f"unknown symbol: {name}")

    def _encode_output_tokens(self, tokens: list[str]) -> Tensor:
        terms = [
            bind(self.output_roles[i], self.symbols[self._symbol_idx(tok)])
            for i, tok in enumerate(tokens)
        ]
        return torch.stack(terms).sum(dim=0)

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------

    def _discover_structure(
        self, examples: list[tuple[str, list[str]]]
    ) -> tuple[dict[str, int], dict[str, list[int]]]:
        """Discover per-shape atom_length and verb_positions from training data.

        For each non-turn atom shape with at least 2 distinct verbs in training:
          - atom_length = mode of (output_length / 1) across single-clause-no-modifier
            examples of that shape. (All such examples must have the same length;
            we sanity-check.)
          - verb_positions = positions p where the token at p varies across
            different verbs. Positions where the token is constant across
            verbs are "turn/residual" positions.

        This replaces the previously hand-coded VERB_POSITIONS_BY_SHAPE +
        _atom_output_length oracle lookups with empirical inference.
        """
        from pure_vsa.scan_runner import parse_scan  # noqa: PLC0415

        # Collect (verb, atom_output_tokens) per shape from single-clause examples
        # (with OR without modifier). For modifier examples (twice/thrice), the
        # atom output is the first L tokens where L = output_length / N (N=2 or 3).
        # This catches shapes that only appear in compositions in training.
        per_shape: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
        turn_shapes_seen: set[str] = set()
        modifier_n = {"twice": 2, "thrice": 3}
        for inp_str, exp_out in examples:
            try:
                parsed = parse_scan(inp_str)
            except Exception:
                continue
            if parsed.clause2 is not None:
                continue
            atom = parsed.clause1.atom
            shape = _atom_shape(atom)
            if atom.verb == "turn":
                turn_shapes_seen.add(shape)
            if parsed.clause1.modifier is None:
                atom_out = exp_out
            else:
                n = modifier_n.get(parsed.clause1.modifier)
                if n is None or len(exp_out) % n != 0:
                    continue
                atom_len = len(exp_out) // n
                atom_out = exp_out[:atom_len]
            per_shape[shape].append((atom.verb, atom_out))

        atom_length_by_shape: dict[str, int] = {}
        verb_positions_by_shape: dict[str, list[int]] = {}
        for shape, ex_list in per_shape.items():
            lengths = {len(out) for _, out in ex_list}
            if len(lengths) != 1:
                continue
            atom_length_by_shape[shape] = lengths.pop()
            if shape in turn_shapes_seen:
                # turn shapes have no verb-dependent positions (turn produces
                # only turn-direction tokens, no separate verb action)
                verb_positions_by_shape[shape] = []
                continue
            length = atom_length_by_shape[shape]
            verbs_for_shape = {v for v, _ in ex_list}
            if len(verbs_for_shape) < 2:
                if length == 1:
                    verb_positions_by_shape[shape] = [0]
                else:
                    verb_positions_by_shape[shape] = list(range(length))
                continue
            verb_positions = []
            for p in range(length):
                tokens_at_p = {out[p] for _, out in ex_list}
                if len(tokens_at_p) > 1:
                    verb_positions.append(p)
            verb_positions_by_shape[shape] = verb_positions

        return atom_length_by_shape, verb_positions_by_shape

    def fit(self, examples: list[tuple[str, list[str]]]) -> None:
        """examples: list of (raw_input_string, expected_output_tokens).

        Three passes:
          1. Discover per-shape atom_length and verb_positions from training data
             (no longer hardcoded).
          2. Learn verb -> action_token from any single-clause-no-modifier example
             using the discovered verb_positions.
          3. Extract atom-shape rules (pattern, residual) and turn-shape constants.
        """
        from pure_vsa.scan_runner import parse_scan  # noqa: PLC0415

        # Pass 0: discover structure.
        atom_lens, verb_positions = self._discover_structure(examples)
        self.atom_length_by_shape = atom_lens
        self.verb_positions_by_shape = verb_positions

        # Pass 1: learn verb -> action_symbol using discovered verb positions.
        for inp_str, exp_out in examples:
            try:
                parsed = parse_scan(inp_str)
            except Exception:
                continue
            if parsed.clause2 is not None or parsed.clause1.modifier is not None:
                continue
            atom = parsed.clause1.atom
            if atom.verb == "turn":
                self.verb_to_action[atom.verb] = None
                continue
            if atom.verb in self.verb_to_action:
                continue
            shape = _atom_shape(atom)
            if shape not in self.verb_positions_by_shape:
                continue
            verb_positions_here = self.verb_positions_by_shape[shape]
            if not verb_positions_here or max(verb_positions_here) >= len(exp_out):
                continue
            verb_token = exp_out[verb_positions_here[0]]
            try:
                self.verb_to_action[atom.verb] = self._symbol_idx(verb_token)
            except ValueError:
                pass

        # Pass 2: extract atom-shape rules from training atom outputs.
        # For each atom-shape, find any single-clause training example whose
        # atom matches. If the clause has a modifier (twice/thrice), the atom
        # output is the FIRST atom_length tokens of the expected output (since
        # twice/thrice just repeats the atom output). This lets us extract
        # rules even when no bare atom example exists in training.
        atom_examples_by_shape: dict[str, list[tuple[Atom, list[str]]]] = defaultdict(list)
        turn_examples_by_shape: dict[str, list[list[str]]] = defaultdict(list)
        for inp_str, exp_out in examples:
            try:
                parsed = parse_scan(inp_str)
            except Exception:
                continue
            # Only use single-clause examples (no conjunction).
            if parsed.clause2 is not None:
                continue
            atom = parsed.clause1.atom
            shape = _atom_shape(atom)
            atom_len = self.atom_length_by_shape.get(shape)
            if atom_len is None or atom_len > len(exp_out):
                continue
            # Slice out the atom's portion of the output: it's the first
            # atom_len tokens, regardless of whether the clause has a modifier
            # (since twice/thrice just repeats the atom output).
            atom_out = exp_out[:atom_len]
            if atom.verb == "turn":
                turn_examples_by_shape[shape].append(atom_out)
            else:
                atom_examples_by_shape[shape].append((atom, atom_out))

        # Atom rules: pattern via unbind(encoded_out, verb_action_sym).
        for shape, ex_list in atom_examples_by_shape.items():
            patterns = []
            for atom, exp_out in ex_list:
                verb_sym_idx = self.verb_to_action.get(atom.verb)
                if verb_sym_idx is None:
                    continue
                encoded = self._encode_output_tokens(exp_out)
                verb_sym_hv = self.symbols[verb_sym_idx]
                patterns.append(unbind(encoded, verb_sym_hv))
            if not patterns:
                continue
            pattern = torch.stack(patterns).mean(dim=0)
            residuals = []
            for atom, exp_out in ex_list:
                verb_sym_idx = self.verb_to_action.get(atom.verb)
                if verb_sym_idx is None:
                    continue
                encoded = self._encode_output_tokens(exp_out)
                verb_sym_hv = self.symbols[verb_sym_idx]
                residuals.append(encoded - bind(pattern, verb_sym_hv))
            residual = torch.stack(residuals).mean(dim=0)
            self.atom_rules[shape] = (pattern, residual)

        # Turn rules: constant output (no verb), stored as the encoded HV.
        for shape, ex_list in turn_examples_by_shape.items():
            encoded_list = [self._encode_output_tokens(exp_out) for exp_out in ex_list]
            self.turn_outputs[shape] = torch.stack(encoded_list).mean(dim=0)

        # Pass 3: learn direction-token mapping (left -> I_TURN_LEFT, right -> I_TURN_RIGHT).
        # For any `verb DIR` training example, position 0 of the output is the
        # direction's turn token. This enables direction-agnostic structural rules.
        for inp_str, exp_out in examples:
            try:
                parsed = parse_scan(inp_str)
            except Exception:
                continue
            if parsed.clause2 is not None or parsed.clause1.modifier is not None:
                continue
            atom = parsed.clause1.atom
            if atom.direction is None or atom.spatial is not None:
                continue
            # `verb DIR` shape: position 0 is the turn token for this direction
            if atom.verb == "turn" and len(exp_out) >= 1:
                try:
                    if atom.direction not in self.dir_to_turn_token:
                        self.dir_to_turn_token[atom.direction] = self._symbol_idx(exp_out[0])
                except ValueError:
                    pass
            elif atom.verb != "turn" and len(exp_out) >= 2:
                try:
                    if atom.direction not in self.dir_to_turn_token:
                        self.dir_to_turn_token[atom.direction] = self._symbol_idx(exp_out[0])
                except ValueError:
                    pass

        # Pass 4: extract direction-agnostic structural rules per spatial category.
        # For each spatial category, find any training example involving it
        # (any direction works) with a known verb_action_token AND a known
        # direction_turn_token. Extract:
        #   verb_pattern = unbind(encoded_output, verb_action_hv)
        #   turn_pattern = unbind(encoded_output, direction_turn_hv)
        # Then at apply time with a NEW direction:
        #   output = bind(verb_pattern, verb_hv) + bind(turn_pattern, new_dir_turn_hv)
        cat_examples_by_category: dict[str, list[tuple[Atom, list[str]]]] = defaultdict(list)
        for inp_str, exp_out in examples:
            try:
                parsed = parse_scan(inp_str)
            except Exception:
                continue
            if parsed.clause2 is not None:
                continue
            atom = parsed.clause1.atom
            category = _spatial_category(atom)
            shape = _atom_shape(atom)
            atom_len = self.atom_length_by_shape.get(shape)
            if atom_len is None or atom_len > len(exp_out):
                continue
            atom_out = exp_out[:atom_len]
            cat_examples_by_category[category].append((atom, atom_out))
            if category not in self.atom_length_by_category:
                self.atom_length_by_category[category] = atom_len

        for category, ex_list in cat_examples_by_category.items():
            if category in {"atom_only", "turn_bare"}:
                continue  # no direction, no turn template needed
            verb_patterns = []
            turn_patterns = []
            for atom, atom_out in ex_list:
                if atom.direction not in self.dir_to_turn_token:
                    continue
                turn_sym_idx = self.dir_to_turn_token[atom.direction]
                encoded = self._encode_output_tokens(atom_out)
                # turn pattern: unbind by direction's turn token
                turn_hv = self.symbols[turn_sym_idx]
                turn_patterns.append(unbind(encoded, turn_hv))
                # verb pattern: only meaningful for non-turn verbs
                if atom.verb != "turn":
                    verb_sym_idx = self.verb_to_action.get(atom.verb)
                    if verb_sym_idx is None:
                        continue
                    verb_hv = self.symbols[verb_sym_idx]
                    verb_patterns.append(unbind(encoded, verb_hv))
            if not turn_patterns:
                continue
            turn_pattern = torch.stack(turn_patterns).mean(dim=0)
            if verb_patterns:
                verb_pattern = torch.stack(verb_patterns).mean(dim=0)
            else:
                verb_pattern = torch.zeros(self.cfg.d)  # turn-verb category
            self.structural_rules[category] = (verb_pattern, turn_pattern)

    # ------------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------------

    def _atom_output_hv(self, atom: Atom) -> tuple[Tensor, int]:
        """Build the (output_hv, length) for one atom.

        Resolution order:
          1. Per-shape atom_rule (if this exact shape was in training).
          2. Per-shape turn_output (for turn shapes seen in training).
          3. Direction-agnostic structural_rule + per-direction turn token.
             This is the *compositional generalization* path: handles novel
             (modifier, direction) combinations like `verb opposite right`
             when only `verb opposite left` was in training.
          4. Zero vector + length 1 fallback.
        """
        shape = _atom_shape(atom)
        # Try exact-shape lookup first.
        if shape in self.turn_outputs:
            length = self.atom_length_by_shape.get(shape, 1)
            return self.turn_outputs[shape], length
        if shape in self.atom_rules:
            length = self.atom_length_by_shape.get(shape, 1)
            pattern, residual = self.atom_rules[shape]
            verb_sym_idx = self.verb_to_action.get(atom.verb)
            if verb_sym_idx is None:
                return torch.zeros(self.cfg.d), length
            return bind(pattern, self.symbols[verb_sym_idx]) + residual, length

        # Direction-agnostic fallback: combine structural rule with direction's turn token.
        category = _spatial_category(atom)
        if category in self.structural_rules and atom.direction in self.dir_to_turn_token:
            verb_pattern, turn_pattern = self.structural_rules[category]
            turn_sym_idx = self.dir_to_turn_token[atom.direction]
            turn_hv = self.symbols[turn_sym_idx]
            length = self.atom_length_by_category.get(category, 1)
            if atom.verb == "turn":
                output_hv = bind(turn_pattern, turn_hv)
            else:
                verb_sym_idx = self.verb_to_action.get(atom.verb)
                if verb_sym_idx is None:
                    return torch.zeros(self.cfg.d), length
                verb_hv = self.symbols[verb_sym_idx]
                output_hv = bind(verb_pattern, verb_hv) + bind(turn_pattern, turn_hv)
            return output_hv, length

        # Nothing learned for this shape or category -- return zero.
        return torch.zeros(self.cfg.d), self.atom_length_by_shape.get(shape, 1)

    def _clause_output_hv(self, clause: Clause) -> tuple[Tensor, int]:
        """Apply twice/thrice procedurally via VSA permute."""
        atom_hv, atom_len = self._atom_output_hv(clause.atom)
        if clause.modifier is None:
            return atom_hv, atom_len
        if clause.modifier == "twice":
            shifted = vsa_permute(atom_hv, shift=atom_len)
            return atom_hv + shifted, atom_len * 2
        if clause.modifier == "thrice":
            shifted1 = vsa_permute(atom_hv, shift=atom_len)
            shifted2 = vsa_permute(atom_hv, shift=atom_len * 2)
            return atom_hv + shifted1 + shifted2, atom_len * 3
        raise ValueError(f"unknown modifier {clause.modifier}")

    def predict(self, parsed: ParsedSCAN) -> list[str]:
        """Run the full prediction pipeline and decode to token list."""
        c1_hv, c1_len = self._clause_output_hv(parsed.clause1)
        if parsed.clause2 is None:
            return self._decode(c1_hv, c1_len)

        c2_hv, c2_len = self._clause_output_hv(parsed.clause2)
        total_len = c1_len + c2_len
        if total_len > self.cfg.max_output_len:
            return []  # output too long for our role HV pool
        if parsed.connective == "and":
            combined = c1_hv + vsa_permute(c2_hv, shift=c1_len)
        elif parsed.connective == "after":
            # X after Y -> Y first, then X
            combined = c2_hv + vsa_permute(c1_hv, shift=c2_len)
        else:
            raise ValueError(f"unknown connective {parsed.connective}")
        return self._decode(combined, total_len)

    def _decode(self, output_hv: Tensor, n_slots: int) -> list[str]:
        out_cb = self.symbols.all()[OUTPUT_OFFSET:]
        tokens = []
        for i in range(n_slots):
            slot_hv = unbind(output_hv, self.output_roles[i])
            sims = similarity(slot_hv, out_cb)
            tokens.append(OUTPUT_TOKENS[sims.argmax(dim=-1).item()])
        return tokens

    # ------------------------------------------------------------------

    def accuracy(self, examples: list[tuple[str, list[str]]]) -> dict:
        """Evaluate on examples, return exact-match accuracy + breakdown."""
        from pure_vsa.scan_runner import classify_complexity, parse_scan  # noqa: PLC0415
        correct = 0
        per_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [correct, total]
        failures = []
        for inp_str, expected in examples:
            try:
                parsed = parse_scan(inp_str)
                cls = classify_complexity(parsed)
                predicted = self.predict(parsed)
            except Exception as e:
                predicted = []
                cls = f"PARSE_ERROR: {type(e).__name__}"
            ok = predicted == expected
            per_class[cls][1] += 1
            if ok:
                correct += 1
                per_class[cls][0] += 1
            elif len(failures) < 5:
                failures.append((inp_str, predicted, expected))
        return {
            "correct": correct,
            "total": len(examples),
            "acc": correct / len(examples) if examples else 0.0,
            "per_class": dict(per_class),
            "failures": failures,
        }
