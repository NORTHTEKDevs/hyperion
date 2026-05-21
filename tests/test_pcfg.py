"""Tests for the pure-VSA PCFG SET solver."""

from __future__ import annotations

from pathlib import Path

import pytest

from pure_vsa.pcfg_hyperion import (
    PCFGConfig,
    PCFGHyperion,
    apply_pcfg_operation,
    is_single_op,
    load_pcfg_split,
    parse_pcfg,
    tokenize_pcfg,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PCFG_DIR = REPO_ROOT / "data" / "pcfg"


def _data_available() -> bool:
    return (PCFG_DIR / "train.src").exists() and (PCFG_DIR / "test.src").exists()


def test_tokenize_pcfg_single_op():
    op, groups = tokenize_pcfg("copy A B C")
    assert op == "copy"
    assert groups == [["A", "B", "C"]]


def test_tokenize_pcfg_two_groups():
    op, groups = tokenize_pcfg("append A B , C D")
    assert op == "append"
    assert groups == [["A", "B"], ["C", "D"]]


def test_apply_pcfg_operations():
    assert apply_pcfg_operation("copy", [["A", "B"]]) == ["A", "B"]
    assert apply_pcfg_operation("reverse", [["A", "B", "C"]]) == ["C", "B", "A"]
    assert apply_pcfg_operation("echo", [["A", "B"]]) == ["A", "B", "B"]
    assert apply_pcfg_operation("swap_first_last", [["A", "B", "C"]]) == ["C", "B", "A"]
    assert apply_pcfg_operation("repeat", [["A", "B"]]) == ["A", "B", "A", "B"]
    assert apply_pcfg_operation("shift", [["A", "B", "C"]]) == ["B", "C", "A"]
    assert apply_pcfg_operation("append", [["A"], ["B", "C"]]) == ["A", "B", "C"]
    assert apply_pcfg_operation("prepend", [["A"], ["B", "C"]]) == ["B", "C", "A"]


def test_parse_pcfg_nested():
    tree = parse_pcfg("reverse copy A B")
    assert tree[0] == "OP"
    assert tree[1] == "reverse"
    assert tree[2][0][0] == "OP"
    assert tree[2][0][1] == "copy"


@pytest.mark.skipif(not _data_available(), reason="PCFG data not downloaded")
def test_pcfg_single_op_accuracy():
    """100% on single-op subset of the test set."""
    train = load_pcfg_split(PCFG_DIR / "train.src", PCFG_DIR / "train.tgt")
    test = load_pcfg_split(PCFG_DIR / "test.src", PCFG_DIR / "test.tgt")
    r = PCFGHyperion(PCFGConfig(d=2048, seed=0, max_output_len=800))
    r.fit(train)
    single_op_test = [(s, t) for s, t in test if is_single_op(s)]
    correct = 0
    for src, tgt in single_op_test:
        try:
            pred = r.predict(src)
        except Exception:
            pred = []
        if pred == tgt:
            correct += 1
    assert correct == len(single_op_test), (
        f"single-op acc {correct}/{len(single_op_test)}, expected 100%"
    )


@pytest.mark.skipif(not _data_available(), reason="PCFG data not downloaded")
def test_pcfg_nested_accuracy_subset():
    """At least 99% on a 500-example random subset of the full test set."""
    train = load_pcfg_split(PCFG_DIR / "train.src", PCFG_DIR / "train.tgt")
    test = load_pcfg_split(PCFG_DIR / "test.src", PCFG_DIR / "test.tgt")
    r = PCFGHyperion(PCFGConfig(d=4096, seed=0, max_output_len=800))
    r.fit(train)
    correct = 0
    n = 500
    for src, tgt in test[:n]:
        try:
            pred = r.predict_nested(src)
        except Exception:
            pred = []
        if pred == tgt:
            correct += 1
    acc = correct / n
    assert acc >= 0.99, f"nested acc {correct}/{n} = {acc:.4f}, expected >=0.99"
