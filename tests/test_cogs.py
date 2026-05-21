"""Tests for the COGS simple-intransitive solver (minimal COGS coverage)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pure_vsa.cogs_hyperion import (
    COGSConfig,
    COGSIntransitiveHyperion,
    evaluate_cogs_intransitive,
    is_simple_intransitive,
    load_cogs_tsv,
    parse_intransitive_output,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
COGS_DIR = REPO_ROOT / "data" / "cogs" / "raw"


def _data_available() -> bool:
    return (
        (COGS_DIR / "train.tsv").exists()
        and (COGS_DIR / "test.tsv").exists()
        and (COGS_DIR / "gen.tsv").exists()
    )


def test_is_simple_intransitive():
    assert is_simple_intransitive("Oliver crumpled .")
    assert is_simple_intransitive("Hazel cried .")
    assert not is_simple_intransitive("Emma rolled a teacher .")
    assert not is_simple_intransitive("a rose was helped by a dog .")


def test_parse_intransitive_output():
    parsed = parse_intransitive_output("crumple . theme ( x _ 1 , Oliver )")
    assert parsed == ("crumple", "theme", "Oliver")
    parsed = parse_intransitive_output("investigate . agent ( x _ 1 , James )")
    assert parsed == ("investigate", "agent", "James")
    assert parse_intransitive_output("not a valid cogs output") is None


@pytest.mark.skipif(not _data_available(), reason="COGS data not downloaded")
def test_cogs_train_test_high_accuracy():
    """Across 3 supported constructions: simple_intrans, intrans_w_det, transitive,
    accuracy on train and test in-distribution must be >= 99%."""
    train = load_cogs_tsv(COGS_DIR / "train.tsv")
    test = load_cogs_tsv(COGS_DIR / "test.tsv")
    r = COGSIntransitiveHyperion(COGSConfig(d=4096, seed=0))
    r.fit(train)
    train_result = evaluate_cogs_intransitive(r, train)
    test_result = evaluate_cogs_intransitive(r, test)
    assert train_result["in_scope_acc"] >= 0.99
    assert test_result["in_scope_acc"] >= 0.99


@pytest.mark.skipif(not _data_available(), reason="COGS data not downloaded")
def test_cogs_gen_partial():
    """Gen split: simple_intrans + intrans_w_det should be 100%; transitive is
    partial due to known-hard `unacc_to_transitive` failures."""
    train = load_cogs_tsv(COGS_DIR / "train.tsv")
    gen = load_cogs_tsv(COGS_DIR / "gen.tsv")
    r = COGSIntransitiveHyperion(COGSConfig(d=4096, seed=0))
    r.fit(train)
    result = evaluate_cogs_intransitive(r, gen)
    # simple_intrans and intrans_w_det should be perfect
    assert result["per_class_correct"].get("simple_intrans", 0) == result["per_class_total"]["simple_intrans"]
    assert result["per_class_correct"].get("intrans_w_det", 0) == result["per_class_total"]["intrans_w_det"]
    # transitive is partial; require at least 50%
    trans_correct = result["per_class_correct"].get("transitive", 0)
    trans_total = result["per_class_total"]["transitive"]
    assert trans_correct / trans_total >= 0.5, (
        f"transitive gen acc {trans_correct}/{trans_total}"
    )
