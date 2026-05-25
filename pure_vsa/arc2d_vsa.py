"""VSA-based ARC solver — analogical reasoning over hypervector grids.

EMPIRICAL RESULT: 0 / 400 on training. The analogy mechanism
T = unbind(out_hv, in_hv) followed by bind(test_in_hv, T) doesn't
recover correct outputs for ARC tasks because the position-bound color
encoding loses too much spatial information when going through
unbind→bind. Documented as a null result.

The approach DOES work for purely-algebraic transformations (global
recolors), but those are already handled by explicit recolor primitives
in the main solver, so no net gain.

Kept for the research record. The right novel low-compute approach is
likely small-LLM-as-primitive-proposer (Ollama + Qwen 2.5 Coder small),
not VSA-only analogy.

---



The core idea, in plain English:
  1. Encode each (h, w) grid as a single D-dimensional hypervector by summing
     bind(position_hv[r,c], color_hv[g[r][c]]) for every cell.
  2. Given training pairs (in_i, out_i), compute the 'transformation vector'
     T = bundle over i of unbind(out_i_hv, in_i_hv). T captures 'what changes'.
  3. To predict on a test input, compute predicted = bind(test_in_hv, T).
  4. Decode predicted back to a grid by, for each cell position, finding the
     color hypervector whose bind(pos[r,c], color) has the highest cosine
     similarity to predicted.

This is the classical VSA analogy: A : B :: C : ?, where ? = bind(C, unbind(B, A)).

Why it could work for ARC:
  - Tasks that are 'global recolor' (every color X becomes color Y) decode
    perfectly because the transformation IS an algebraic recolor.
  - Tasks with consistent per-cell rules also fall out.
Why it might not work:
  - Tasks involving 'find this object, do something with it' don't decompose
    holographically — they need symbolic reasoning over objects.
  - Spatial transformations (move, rotate) are not naturally captured by
    cell-position binding.

So this is a complement to the enumerative solver, not a replacement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Self-contained bipolar VSA — circular-convolution bind via FFT
# ---------------------------------------------------------------------------

D = 8192  # hypervector dimensionality
_rng = np.random.default_rng(42)
_HV_CACHE: dict[str, np.ndarray] = {}


def _hv(key: str) -> np.ndarray:
    """Get-or-create a deterministic bipolar hypervector for a string key."""
    if key not in _HV_CACHE:
        # Deterministic seed from key string
        h = abs(hash(key)) % (2**31)
        rng = np.random.default_rng(h)
        _HV_CACHE[key] = (rng.integers(0, 2, D) * 2 - 1).astype(np.float32)
    return _HV_CACHE[key]


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular convolution via FFT (Plate's HRR-style bind)."""
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=D).astype(np.float32)


def unbind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular correlation: unbind a by b -> approximately the other factor."""
    return np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n=D).astype(np.float32)


def bundle(vs: list[np.ndarray]) -> np.ndarray:
    """Sum then sign (binarize) — standard VSA bundling."""
    if not vs:
        return np.zeros(D, dtype=np.float32)
    s = np.zeros(D, dtype=np.float32)
    for v in vs:
        s = s + v
    # Don't binarize here — keep as real-valued for better decoding
    return s


_POS_HVS: dict[tuple[int, int], np.ndarray] = {}
_COLOR_HVS: dict[int, np.ndarray] = {}


def _pos_hv(r: int, c: int) -> np.ndarray:
    if (r, c) not in _POS_HVS:
        _POS_HVS[(r, c)] = bind(_hv(f"row_{r}"), _hv(f"col_{c}"))
    return _POS_HVS[(r, c)]


def _color_hv(v: int) -> np.ndarray:
    if v not in _COLOR_HVS:
        _COLOR_HVS[v] = _hv(f"color_{v}")
    return _COLOR_HVS[v]


def encode_grid(g) -> np.ndarray:
    """Encode a grid as a single hypervector by bundling bind(pos, color)
    for every cell."""
    h = len(g); w = len(g[0]) if g else 0
    parts = [bind(_pos_hv(r, c), _color_hv(g[r][c])) for r in range(h) for c in range(w)]
    return bundle(parts)


def decode_grid_at_shape(hv: np.ndarray, h: int, w: int) -> list[list[int]]:
    """Decode a hypervector back to a (h, w) grid. For each cell position,
    unbind the position from hv, then find the color whose hypervector has
    highest cosine similarity to the result."""
    out = [[0] * w for _ in range(h)]
    color_options = list(range(10))
    color_hvs = np.stack([_color_hv(v) for v in color_options])  # shape (10, D)
    # Normalize for cosine
    color_norms = np.linalg.norm(color_hvs, axis=1, keepdims=True) + 1e-9
    color_normed = color_hvs / color_norms
    for r in range(h):
        for c in range(w):
            unbound = unbind(hv, _pos_hv(r, c))
            unbound_norm = unbound / (np.linalg.norm(unbound) + 1e-9)
            sims = color_normed @ unbound_norm
            best = int(np.argmax(sims))
            out[r][c] = color_options[best]
    return out


# ---------------------------------------------------------------------------
# VSA-based analogical solver
# ---------------------------------------------------------------------------

def solve_task_vsa(task_data: dict) -> tuple[str, list[list[int]]] | None:
    """Try to solve via VSA analogy. Returns (method_name, predicted_grid) or None
    if the test input dimensions don't match anything we can predict."""
    train = task_data.get("train", [])
    test = task_data.get("test", [])
    if not train or not test:
        return None

    train_pairs = [(t["input"], t["output"]) for t in train]
    test_inp = test[0]["input"]

    # Only attempt when output dims equal input dims (this VSA encoding
    # doesn't naturally handle shape change)
    same_shape = all(
        len(i) == len(o) and len(i[0]) == len(o[0])
        for i, o in train_pairs
    )
    if not same_shape:
        return None
    if len(test_inp) != len(train_pairs[0][0]) or len(test_inp[0]) != len(train_pairs[0][0][0]):
        # Test input shape differs from training. Try anyway — we'll decode
        # at the test input's shape. Many tasks vary input dims across examples.
        pass

    # Compute transformation vector T = bundle of unbind(out_hv, in_hv) over train
    transformation_parts = []
    for inp, out in train_pairs:
        in_hv = encode_grid(inp)
        out_hv = encode_grid(out)
        # T_i = unbind(out_hv, in_hv) → what to bind with in to get out
        t_i = unbind(out_hv, in_hv)
        transformation_parts.append(t_i)
    T = bundle(transformation_parts)

    # Predict test output by binding test input with T
    test_in_hv = encode_grid(test_inp)
    pred_hv = bind(test_in_hv, T)

    # Decode at test input shape
    th, tw = len(test_inp), len(test_inp[0])
    pred_grid = decode_grid_at_shape(pred_hv, th, tw)

    return ("vsa_analogy", pred_grid)


def evaluate_directory_vsa(arc_root: Path) -> dict:
    """Run VSA-only solver on training tasks."""
    results: dict[str, list[bool]] = {}
    for f in sorted(arc_root.glob("*.json")):
        data = json.loads(f.read_text())
        sol = solve_task_vsa(data)
        if sol is None:
            results[f.name] = [False]
            continue
        _, pred = sol
        try:
            expected = data["test"][0]["output"]
            results[f.name] = [pred == expected]
        except Exception:
            results[f.name] = [False]
    return results
