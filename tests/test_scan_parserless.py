"""Tests for the parser-free SCAN reasoner.

The parser-free version discovers the SCAN grammar (verbs, modifiers,
directions, spatial, conjunctions, plus verb -> action token and direction ->
turn token mappings) from training data alone using
pure_vsa.scan_grammar_discovery.discover_grammar, then plugs into the same
SCANHyperion mechanism that achieves 100% with the hand-written parser.

These tests assert that on representative SCAN splits, the parser-free version
achieves the same 100% accuracy as the hand-written-parser version.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pure_vsa.scan_hyperion_parserless import fit_and_eval_parserless


REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIR = REPO_ROOT / "data" / "scan"
SCAN_EXTRA_DIR = REPO_ROOT / "data" / "scan_extra"

SPLITS = {
    "simple": SCAN_DIR / "simple",
    "addprim_jump": SCAN_DIR / "addprim_jump",
    # length test (parserless) is also expected at 100% but takes ~45s; included
    # explicitly for the full guarantee.
    "length": SCAN_DIR / "length",
    "template_opposite_right": SCAN_EXTRA_DIR / "template_opposite_right",
}


@pytest.mark.parametrize("name,path", list(SPLITS.items()))
def test_scan_parserless_100_percent(name: str, path: Path):
    """Parser-free version reaches 100% on representative splits."""
    if not (path / "train.txt").exists() or not (path / "test.txt").exists():
        pytest.skip(f"split {name} not downloaded at {path}")
    r = fit_and_eval_parserless(
        path / "train.txt", path / "test.txt", d=8192, seed=0,
    )
    assert r["acc"] == 1.0, (
        f"parserless {name}: expected 100%, got {r['acc']:.4f} "
        f"({r['correct']}/{r['total']})\n"
        f"Discovered grammar:\n{r['discovered_grammar']}"
    )
