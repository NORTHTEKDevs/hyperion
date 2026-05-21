"""Parser-free SCAN: discover the grammar structure from raw input tokens.

The original SCANHyperion uses a hand-written grammar parser (parse_scan).
This module attempts to *discover* the grammar from input-output pairs alone,
with no prior knowledge of which SCAN tokens are verbs, modifiers, directions,
or conjunctions. The only inputs are the (raw_input_string, output_tokens) pairs.

Discovery strategy:
  1. Identify "content" vs "operator" tokens by frequency: tokens appearing
     in <10% of inputs are content tokens (verbs); tokens appearing in >40%
     are operators.
  2. Identify "verb -> action_token" mapping from single-token inputs
     (input = single content token, output = single action token).
  3. Identify "conjunction" tokens: operators that, when present, split the
     input into two sub-sequences whose outputs are concatenated.
  4. Identify "modifier" tokens (twice/thrice): operators whose presence at
     specific positions causes output length to be a multiple of the un-modified
     output length.
  5. Identify "direction" tokens: operators whose presence adds a specific
     constant token at output position 0.
  6. Identify "spatial" tokens (opposite/around): operators that change the
     output pattern in a direction-coupled way.

This is grammar induction from data. The success rate of (1)-(6) determines
how well the parser-free version can substitute for the hand-written parser.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass
class DiscoveredGrammar:
    """The grammar structure inferred from training data."""
    content_tokens: set[str]                       # likely "verbs"
    operator_tokens: set[str]                      # likely structural
    conjunctions: set[str]                         # split-and-combine operators
    modifiers: set[str]                            # length-multiplier operators
    directions: set[str]                           # turn-token operators
    spatial: set[str]                              # structural operators (opposite/around)
    verb_to_action: dict[str, str]                 # content_token -> output_token
    direction_to_token: dict[str, str]             # direction -> turn output token

    def summary(self) -> str:
        return (
            f"DiscoveredGrammar:\n"
            f"  content tokens: {sorted(self.content_tokens)}\n"
            f"  conjunctions:   {sorted(self.conjunctions)}\n"
            f"  modifiers:      {sorted(self.modifiers)}\n"
            f"  directions:     {sorted(self.directions)}\n"
            f"  spatial:        {sorted(self.spatial)}\n"
            f"  verb_to_action: {dict(sorted(self.verb_to_action.items()))}\n"
            f"  direction_to_token: {dict(sorted(self.direction_to_token.items()))}"
        )


def discover_grammar(
    examples: list[tuple[str, list[str]]],
    *,
    content_freq_max: float | None = None,   # kept for back-compat; unused
    operator_freq_min: float | None = None,  # kept for back-compat; unused
) -> DiscoveredGrammar:
    """Discover SCAN-like grammar structure from (input_string, output_tokens) pairs.

    Strategy (no frequency thresholds, no prior on "function words"):
      - Verbs := the set of tokens that appear as a single-token input in
        training. By the SCAN grammar these are exactly the primitive verbs.
      - Operators := all other input tokens.
      - Operators are then categorized into conjunctions, modifiers, directions,
        spatial by structural behavior (see steps 3-5 below).

    This makes the discovery purely behavioral (which inputs produce which
    structural transformations), with no prior frequency heuristics.
    """
    n = len(examples)
    input_token_counts: Counter = Counter()
    for inp_str, _ in examples:
        for tok in set(inp_str.split()):
            input_token_counts[tok] += 1

    # Step 0a: verbs are tokens that appear as the sole input token in some example.
    content_tokens: set[str] = set()
    for inp_str, _ in examples:
        toks = inp_str.split()
        if len(toks) == 1:
            content_tokens.add(toks[0])
    # Step 0b: bootstrap -- also call a token a verb if it appears as the FIRST
    # token of a 2-token input whose second token is a known operator candidate.
    # We don't know operators yet, so iteratively widen.
    # A simpler proxy: a token is a verb if it appears at the start of inputs
    # whose other tokens overlap with already-known verbs' modifier patterns.
    # Heuristic: tokens at position 0 of any input whose remaining tokens are
    # all already accounted for by known verbs' other inputs.
    for _ in range(3):  # 3 bootstrap passes
        new = set()
        for inp_str, _ in examples:
            toks = inp_str.split()
            if not toks:
                continue
            t0 = toks[0]
            if t0 in content_tokens:
                continue
            # is t0 the only non-operator-looking token? if rest are all in
            # the inputs of known verbs, t0 is likely also a verb
            rest = toks[1:]
            for known_verb in content_tokens:
                # check if rest occurs after `known_verb` in any other input
                for inp2_str, _ in examples[:5000]:  # sample
                    toks2 = inp2_str.split()
                    if toks2 and toks2[0] == known_verb and toks2[1:] == rest:
                        new.add(t0)
                        break
                if t0 in new:
                    break
        content_tokens |= new
        if not new:
            break
    all_input_tokens = set(input_token_counts)
    operator_tokens = all_input_tokens - content_tokens
    intermediate: set[str] = set()

    # Step 1: identify conjunctions. A conjunction is an operator that:
    #   - Always appears in the middle of the input (never at start/end)
    #   - When present, the two sides each contain a content_token (verb)
    conjunctions: set[str] = set()
    for cand in operator_tokens:
        n_with = 0
        n_two_verb_sides = 0
        for inp_str, _ in examples:
            toks = inp_str.split()
            if cand not in toks:
                continue
            n_with += 1
            i = toks.index(cand)
            if 0 < i < len(toks) - 1:
                left_has_verb = any(t in content_tokens for t in toks[:i])
                right_has_verb = any(t in content_tokens for t in toks[i + 1:])
                if left_has_verb and right_has_verb:
                    n_two_verb_sides += 1
        if n_with > 50 and n_two_verb_sides / max(n_with, 1) > 0.95:
            conjunctions.add(cand)

    # Step 2: identify verb -> action_token from single-content-token inputs.
    verb_to_action: dict[str, str] = {}
    for inp_str, out_tokens in examples:
        toks = inp_str.split()
        if len(toks) == 1 and toks[0] in content_tokens and len(out_tokens) == 1:
            verb_to_action[toks[0]] = out_tokens[0]

    # Step 3: identify direction tokens from short "content_tok direction" inputs.
    # When input = [verb, direction], output should have a constant turn-token
    # at position 0 across different verbs.
    direction_candidates = (operator_tokens | intermediate) - conjunctions
    direction_to_token: dict[str, str] = {}
    for cand in direction_candidates:
        # gather examples where input = [verb, cand]
        verb_output_p0: dict[str, str] = {}
        for inp_str, out_tokens in examples:
            toks = inp_str.split()
            if len(toks) == 2 and toks[0] in content_tokens and toks[1] == cand:
                if len(out_tokens) >= 1:
                    verb_output_p0[toks[0]] = out_tokens[0]
        # if all verbs produce the same position-0 token, that token is the
        # direction's turn token
        if len(verb_output_p0) >= 2:
            position_0_tokens = set(verb_output_p0.values())
            if len(position_0_tokens) == 1:
                direction_to_token[cand] = position_0_tokens.pop()
    directions = set(direction_to_token.keys())

    # Step 4: identify modifier tokens (twice/thrice). When input ends with
    # a modifier, output length should be a multiple of the corresponding
    # input-without-modifier output length.
    bare_output_len: dict[tuple[str, ...], int] = {}
    for inp_str, out_tokens in examples:
        toks = inp_str.split()
        if any(t in conjunctions for t in toks):
            continue
        key = tuple(toks)
        bare_output_len.setdefault(key, len(out_tokens))
    modifier_candidates = operator_tokens - conjunctions - directions
    modifiers: set[str] = set()
    for cand in modifier_candidates:
        ratios = Counter()
        for inp_str, out_tokens in examples:
            toks = inp_str.split()
            if len(toks) < 2 or toks[-1] != cand:
                continue
            if any(c in toks for c in conjunctions):
                continue
            bare_key = tuple(toks[:-1])
            if bare_key not in bare_output_len:
                continue
            bare_len = bare_output_len[bare_key]
            if bare_len == 0:
                continue
            ratio = len(out_tokens) / bare_len
            if ratio == int(ratio):
                ratios[int(ratio)] += 1
        if ratios:
            most_common_ratio, count = ratios.most_common(1)[0]
            if most_common_ratio in (2, 3) and count / sum(ratios.values()) > 0.7 and count >= 10:
                modifiers.add(cand)

    # Step 5: identify spatial tokens (opposite/around). These are operators that
    # appear BEFORE a direction in input and change the structure of the output.
    spatial: set[str] = set()
    spatial_candidates = (operator_tokens | intermediate) - conjunctions - directions - modifiers
    for cand in spatial_candidates:
        # spatial tokens appear immediately before a direction
        ok_count = 0
        total = 0
        for inp_str, _ in examples:
            toks = inp_str.split()
            if cand not in toks:
                continue
            total += 1
            idx = toks.index(cand)
            if idx + 1 < len(toks) and toks[idx + 1] in directions:
                ok_count += 1
        if total > 50 and ok_count / total > 0.9:
            spatial.add(cand)

    return DiscoveredGrammar(
        content_tokens=content_tokens,
        operator_tokens=operator_tokens | (intermediate - directions - modifiers - spatial - conjunctions),
        conjunctions=conjunctions,
        modifiers=modifiers,
        directions=directions,
        spatial=spatial,
        verb_to_action=verb_to_action,
        direction_to_token=direction_to_token,
    )
