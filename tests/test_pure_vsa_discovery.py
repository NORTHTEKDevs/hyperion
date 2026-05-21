"""Test: the reasoner discovers rules from a flat training set with NO
labels for which examples belong to which modifier.

Compared to test_tinyscan, this is one step less hand-held: we hand the
reasoner ALL training examples in one shot and expect it to figure out
which modifier each example uses and extract rules accordingly.
"""

from __future__ import annotations


from pure_vsa.reasoner import PureVSAReasoner
from pure_vsa.tinyscan import (
    MODIFIERS,
    N_ACTIONS,
    N_MODIFIERS,
    N_ROLES,
    OUTPUT_ROLES,
    OUTPUT_TOKENS,
    ROLE_MOD,
    ROLE_OUT1,
    ROLE_VERB,
    TOTAL_SYMBOLS,
    make_dataset,
    symbol_index,
)
from vsa_core import unbind
from vsa_core.cleanup import similarity
from vsa_core.codebook import Codebook


D = 10_000


def _build_reasoner(d: int = D, seed: int = 0) -> PureVSAReasoner:
    symbols = Codebook(TOTAL_SYMBOLS, d, seed=seed)
    roles = Codebook(N_ROLES, d, seed=seed + 1)
    return PureVSAReasoner(d=d, symbol_codebook=symbols, role_codebook=roles)


def _decode_output_tokens(
    reasoner: PureVSAReasoner, output_hv, n_slots: int
) -> list[str]:
    out_offset = N_ACTIONS + N_MODIFIERS
    output_codebook = reasoner.symbols.all()[out_offset:]
    tokens = []
    for i in range(n_slots):
        slot_hv = unbind(output_hv, reasoner.roles[OUTPUT_ROLES[i]])
        sims = similarity(slot_hv, output_codebook)
        best_local = sims.argmax(dim=-1).item()
        tokens.append(OUTPUT_TOKENS[best_local])
    return tokens


def test_autonomous_discovery_passes_tinyscan():
    """End-to-end: feed the reasoner a flat list of training pairs with no
    modifier labels. It must group, extract, and generalize on the held-out
    (swim, MODIFIER) compositions.
    """
    reasoner = _build_reasoner()
    train, test = make_dataset()

    # Build the flat training corpus the way a real system would receive it.
    training_corpus = [(ex.input_slots(), ex.output_slots()) for ex in train]

    # ONE call. No per-modifier hand-feeding. No per-verb hand-feeding.
    registered = reasoner.discover_and_extract_rules(
        training_corpus,
        modifier_role=ROLE_MOD,
        verb_role=ROLE_VERB,
        primary_out_role=ROLE_OUT1,
    )

    # 4 modifiers should have been discovered.
    expected_modifier_indices = {symbol_index(m) for m in MODIFIERS}
    assert set(registered.keys()) == expected_modifier_indices, (
        f"discovery found {set(registered.keys())} but expected "
        f"{expected_modifier_indices}"
    )

    # Evaluate held-out.
    correct = 0
    results = []
    for ex in test:
        swim_out_sym_idx = reasoner.lookup_verb_output_symbol(
            verb_input_idx=symbol_index(ex.action),
            verb_role_idx=ROLE_VERB,
            out_role_idx=ROLE_OUT1,
        )
        rule_name = registered[symbol_index(ex.modifier)]
        out_hv = reasoner.apply_modifier_pattern(rule_name, swim_out_sym_idx)
        predicted = _decode_output_tokens(reasoner, out_hv, len(ex.output_tokens))
        results.append((ex.action, ex.modifier, predicted, ex.output_tokens))
        if predicted == ex.output_tokens:
            correct += 1

    acc = correct / len(test)
    print(f"\nAutonomous-discovery held-out accuracy: {correct}/{len(test)} = {acc:.3f}")
    for action, mod, pred, expected in results:
        marker = "OK  " if pred == expected else "FAIL"
        print(f"  {marker} ({action}, {mod}) -> {pred}  expected {expected}")

    assert acc >= 0.95, f"autonomous discovery accuracy {acc:.3f} (need >= 0.95)"
