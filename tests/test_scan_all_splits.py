"""Tests covering ALL 7 SCAN compositional generalization splits.

The original 3 (simple, addprim_jump, length) are also covered by
test_scan_hyperion.py at >=99%. This file requires perfect 100% at D=8192
on every split, and adds the 4 additional splits that weren't downloaded
by the original prep script:

  - addprim_turn_left:        hold out `turn_left` compositions
  - template_jump_around_right: hold out `jump around right`
  - template_opposite_right:    hold out `verb opposite right` (any verb)
  - template_around_right:      hold out `verb around right` (any verb)

The two template_*_right splits exercise direction-agnostic generalization:
the system must combine `opposite_left`/`around_left` structure (seen) with
`right` direction (seen) to produce `opposite_right`/`around_right` (held out).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pure_vsa.scan_hyperion import SCANConfig, SCANHyperion
from pure_vsa.scan_runner import load_scan_split


REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIR = REPO_ROOT / "data" / "scan"
SCAN_EXTRA_DIR = REPO_ROOT / "data" / "scan_extra"

SPLITS = {
    "simple": SCAN_DIR / "simple",
    "addprim_jump": SCAN_DIR / "addprim_jump",
    "length": SCAN_DIR / "length",
    "addprim_turn_left": SCAN_EXTRA_DIR / "addprim_turn_left",
    "template_jump_around_right": SCAN_EXTRA_DIR / "template_jump_around_right",
    "template_opposite_right": SCAN_EXTRA_DIR / "template_opposite_right",
    "template_around_right": SCAN_EXTRA_DIR / "template_around_right",
}


@pytest.mark.parametrize("name,path", list(SPLITS.items()))
def test_scan_split_100_percent_at_d8192(name: str, path: Path):
    """Every published SCAN compositional split solved at 100% at D=8192."""
    if not (path / "train.txt").exists() or not (path / "test.txt").exists():
        pytest.skip(f"split {name} not downloaded at {path}")
    train = load_scan_split(path / "train.txt")
    test = load_scan_split(path / "test.txt")
    r = SCANHyperion(SCANConfig(d=8192, seed=0, max_output_len=80))
    r.fit(train)
    result = r.accuracy(test)
    assert result["acc"] == 1.0, (
        f"{name}: expected 100% at D=8192 seed=0, got {result['acc']:.4f} "
        f"({result['correct']}/{result['total']})"
    )
