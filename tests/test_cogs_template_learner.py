"""Tests for the COGS template learner — fully automatic schema induction."""

from __future__ import annotations

from pathlib import Path

import pytest

from pure_vsa.cogs_hyperion import load_cogs_tsv
from pure_vsa.cogs_template_learner import (
    COGSTemplateLearner,
    input_signature,
    normalize_output,
    output_to_template_slots,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
COGS_DIR = REPO_ROOT / "data" / "cogs" / "raw"


def _data_available() -> bool:
    return (
        (COGS_DIR / "train.tsv").exists()
        and (COGS_DIR / "test.tsv").exists()
        and (COGS_DIR / "gen.tsv").exists()
    )


def test_input_signature_basics():
    assert input_signature("Emma rolled .") == ("PROP", "w", ".")
    assert input_signature("A cake was helped .") == ("A", "w", "was", "w", ".")
    assert input_signature("The girl needed to cook .") == ("The", "w", "w", "to", "w", ".")


def test_normalize_output_basics():
    parts = normalize_output("eat . agent ( x _ 1 , Emma )")
    assert parts[0] == "eat"
    assert "(" in parts
    assert "," in parts


def test_output_to_template_slots_simple():
    inp_toks = "Emma rolled .".split()
    out_toks = normalize_output("roll . agent ( x _ 1 , Emma )")
    past_to_inf = {"rolled": "roll"}
    slots = output_to_template_slots(inp_toks, out_toks, past_to_inf)
    assert slots is not None
    # roll = INF[1], . = LIT, agent = LIT, ( = LIT, x = LIT, _ = LIT, 1 = LIT, , = LIT, Emma = COPY[0], ) = LIT
    assert slots[0].kind == "INF"
    assert slots[0].value == "1"
    assert slots[2].kind == "LIT"
    assert slots[2].value == "agent"
    assert slots[8].kind == "COPY"
    assert slots[8].value == "0"


@pytest.mark.skipif(not _data_available(), reason="COGS data not downloaded")
def test_template_learner_train_high_coverage_and_accuracy():
    train = load_cogs_tsv(COGS_DIR / "train.tsv")
    train_pairs = [(t[0], t[1]) for t in train]
    learner = COGSTemplateLearner()
    learner.fit(train_pairs)
    stats = learner.coverage_stats(train)
    assert stats["coverage"] >= 0.99, f"train coverage {stats['coverage']:.4f}"
    assert stats["in_scope_acc"] >= 0.99, f"train in-scope acc {stats['in_scope_acc']:.4f}"


@pytest.mark.skipif(not _data_available(), reason="COGS data not downloaded")
def test_template_learner_test_high_coverage_and_accuracy():
    train = load_cogs_tsv(COGS_DIR / "train.tsv")
    test = load_cogs_tsv(COGS_DIR / "test.tsv")
    train_pairs = [(t[0], t[1]) for t in train]
    learner = COGSTemplateLearner()
    learner.fit(train_pairs)
    stats = learner.coverage_stats(test)
    assert stats["coverage"] >= 0.99, f"test coverage {stats['coverage']:.4f}"
    assert stats["in_scope_acc"] >= 0.99, f"test in-scope acc {stats['in_scope_acc']:.4f}"


@pytest.mark.skipif(not _data_available(), reason="COGS data not downloaded")
def test_template_learner_gen_meaningful_coverage():
    """Compositional generalization split: with recursive fallback + control-verb
    handling we expect >=99% absolute accuracy across all 21 conditions."""
    train = load_cogs_tsv(COGS_DIR / "train.tsv")
    gen = load_cogs_tsv(COGS_DIR / "gen.tsv")
    train_pairs = [(t[0], t[1]) for t in train]
    learner = COGSTemplateLearner()
    learner.fit(train_pairs)
    stats = learner.coverage_stats(gen)
    assert stats["coverage"] >= 0.99, f"gen coverage {stats['coverage']:.4f}"
    abs_acc = stats["correct"] / stats["n"]
    assert abs_acc >= 0.99, f"gen abs acc {abs_acc:.4f}"


@pytest.mark.skipif(not _data_available(), reason="COGS data not downloaded")
def test_template_learner_specific_gen_categories():
    """The previously-hard COGS generalization categories should now be solved."""
    train = load_cogs_tsv(COGS_DIR / "train.tsv")
    gen = load_cogs_tsv(COGS_DIR / "gen.tsv")
    train_pairs = [(t[0], t[1]) for t in train]
    learner = COGSTemplateLearner()
    learner.fit(train_pairs)

    # Per-category accuracy on in-scope items
    from collections import defaultdict
    per_cat = defaultdict(lambda: [0, 0])
    for inp, out, cat in gen:
        sig = input_signature(inp)
        if sig not in learner.templates:
            continue
        per_cat[cat][1] += 1
        pred = learner.predict(inp)
        expected = normalize_output(out)
        if pred == expected:
            per_cat[cat][0] += 1

    # Categories that should be solved at high accuracy
    for cat in [
        "prim_to_subj_proper", "prim_to_obj_proper", "prim_to_subj_common", "prim_to_obj_common",
        "obj_to_subj_proper", "obj_to_subj_common", "subj_to_obj_proper", "subj_to_obj_common",
        "unacc_to_transitive", "obj_omitted_transitive_to_transitive",
        "passive_to_active", "active_to_passive",
        "do_dative_to_pp_dative", "pp_dative_to_do_dative",
    ]:
        correct, total = per_cat[cat]
        if total < 10:
            continue  # skip if too few in-scope
        acc = correct / total
        assert acc >= 0.95, f"gen category {cat}: {correct}/{total} = {acc:.4f}"
