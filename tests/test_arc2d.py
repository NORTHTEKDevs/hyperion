"""Regression test for the 2D ARC-AGI baseline.

Gates the published baseline number so future changes can't silently regress it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pure_vsa.arc2d_solver import evaluate_directory


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "data" / "arc_agi" / "training"
EVAL_DIR = REPO_ROOT / "data" / "arc_agi" / "evaluation"


def _data_available(d: Path) -> bool:
    return d.exists() and any(d.glob("*.json"))


@pytest.mark.skipif(not _data_available(TRAIN_DIR), reason="ARC-AGI training data not downloaded")
def test_arc2d_training_baseline():
    """Baseline must hold at >= 18.5% on ARC-AGI training (75/400 = 18.75% achieved)."""
    r = evaluate_directory(TRAIN_DIR)
    total = len(r)
    correct = sum(sum(rs) for rs in r.values())
    if total == 0:
        pytest.skip("no tasks found")
    acc = correct / total
    assert acc >= 0.185, f"training: {correct}/{total} = {acc:.4f}, expected >= 0.185"


@pytest.mark.skipif(not _data_available(EVAL_DIR), reason="ARC-AGI evaluation data not downloaded")
def test_arc2d_evaluation_baseline():
    """Held-out evaluation must hold at >= 3.75% (currently 16/400 = 4.00%)."""
    r = evaluate_directory(EVAL_DIR)
    total = len(r)
    correct = sum(sum(rs) for rs in r.values())
    if total == 0:
        pytest.skip("no tasks found")
    acc = correct / total
    assert acc >= 0.0375, f"evaluation: {correct}/{total} = {acc:.4f}, expected >= 0.0375"
