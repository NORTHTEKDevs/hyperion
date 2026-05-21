"""Tests for SCANHyperion on real SCAN.

These tests load the actual SCAN data files (must be downloaded first via
`python data/scan/download_and_prep.py`). If the data isn't present, tests skip.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pure_vsa.scan_hyperion import SCANConfig, SCANHyperion
from pure_vsa.scan_runner import load_scan_split


DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "scan"


def _data_available(split: str) -> bool:
    return (DATA_ROOT / split / "train.txt").exists() and (DATA_ROOT / split / "test.txt").exists()


@pytest.mark.skipif(not _data_available("simple"), reason="SCAN simple split not downloaded")
def test_scan_simple_split_perfect():
    """Random 80/20 split: should be trivial after the fix."""
    train = load_scan_split(DATA_ROOT / "simple" / "train.txt")
    test = load_scan_split(DATA_ROOT / "simple" / "test.txt")
    r = SCANHyperion(SCANConfig(d=4096, seed=0, max_output_len=80))
    r.fit(train)
    result = r.accuracy(test)
    assert result["acc"] == 1.0, f"simple split acc {result['acc']:.4f}, expected 1.0"


@pytest.mark.skipif(not _data_available("addprim_jump"), reason="SCAN addprim_jump not downloaded")
def test_scan_addprim_jump_compositional_generalization():
    """The canonical compositional generalization test.

    Train has bare `jump` only; every test example is a composition involving jump.
    Vanilla transformers/seq2seq get ~1-2% (Lake & Baroni 2018). Pure VSA gets >99%.
    """
    train = load_scan_split(DATA_ROOT / "addprim_jump" / "train.txt")
    test = load_scan_split(DATA_ROOT / "addprim_jump" / "test.txt")
    r = SCANHyperion(SCANConfig(d=8192, seed=0, max_output_len=80))
    r.fit(train)
    result = r.accuracy(test)
    # at D=8192 seed=0 we get 100%; allow slight slack across environments.
    assert result["acc"] >= 0.99, f"addprim_jump acc {result['acc']:.4f}, expected >= 0.99"


@pytest.mark.skipif(not _data_available("length"), reason="SCAN length split not downloaded")
def test_scan_length_extrapolation():
    """length split: test sequences are longer than anything in training."""
    train = load_scan_split(DATA_ROOT / "length" / "train.txt")
    test = load_scan_split(DATA_ROOT / "length" / "test.txt")
    r = SCANHyperion(SCANConfig(d=4096, seed=0, max_output_len=80))
    r.fit(train)
    result = r.accuracy(test)
    assert result["acc"] >= 0.99, f"length acc {result['acc']:.4f}, expected >= 0.99"
