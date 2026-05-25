"""Perception-execution split for ARC.

Architecture:
  1. LLM observes training pairs, outputs a STRUCTURED RULE in a constrained
     mini-grammar (not free Python).
  2. Rule compiler converts the structured rule into a deterministic
     primitive composition (drawing from our existing primitive library
     plus a few new ones).
  3. Executor runs the composition. If training matches, apply to test.
  4. On failure, give specific feedback to LLM ("expected color X at (r,c)
     in example N, got Y"), retry up to MAX_RETRIES times.

The breakthrough hypothesis: LLMs are good at PERCEPTION (recognizing
what the transformation is) but bad at EXECUTION (writing correct
imperative code). Symbolic systems are perfect at execution. Splitting
them lets each do what it's good at.
"""

from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "huihui_ai/qwen2.5-coder-abliterate:7b"
LARGER_MODEL = "huihui_ai/qwen3-coder-abliterated:30b"
MAX_RETRIES = 4
PER_CALL_TIMEOUT = 300  # seconds per LLM call (enough for 30B on GPU)


# ---------------------------------------------------------------------------
# Constrained rule grammar (a deliberately small, expressive vocabulary).
# Each rule is a JSON object. Examples:
#   {"op": "recolor", "map": {"2": "5", "3": "8"}}
#   {"op": "flip", "axis": "h"}
#   {"op": "rotate", "k": 1}
#   {"op": "extend_cells", "direction": "down"}
#   {"op": "tile_2x2_kaleidoscope"}
#   {"op": "extract_subgrid", "corner": "tl", "h": 2, "w": 2}
#   {"op": "fill_between_markers", "marker_color": 1, "fill_color": 2}
#   {"op": "compose", "steps": [<rule1>, <rule2>]}
#   {"op": "keep_color", "color": 5}
#   {"op": "remove_color", "color": 3}
#   {"op": "self_similar_tile", "mask_color": 3}
#   {"op": "split_overlay", "axis": "h", "mode": "or", "fill": null}
#
# The grammar is INTENTIONALLY a superset of our existing primitive library,
# so the LLM can describe what the rule is using categories we know how to
# execute reliably.
# ---------------------------------------------------------------------------

GRAMMAR_DESCRIPTION = """
Output ONLY a JSON object describing the rule. Use these operations:

Simple geometric:
- {"op": "identity"}
- {"op": "flip", "axis": "h"} or "axis": "v"
- {"op": "rotate", "k": N}  (N=1,2,3 for 90,180,270 degrees clockwise)
- {"op": "transpose"}

Recoloring:
- {"op": "recolor", "map": {"OLD": "NEW", ...}}  (e.g., {"2": "5", "3": "8"})
- {"op": "keep_color", "color": N}  (zero out everything except color N)
- {"op": "remove_color", "color": N}  (zero out color N)

Cell-extension (extend each non-zero cell in a direction):
- {"op": "extend_cells", "direction": "down" | "up" | "left" | "right"}

Tiling / kaleidoscope:
- {"op": "tile_2x2_kaleidoscope"}  (output is 2x bigger with mirrored quadrants)
- {"op": "tile_2x2_rotational"}  (same but with rotations)
- {"op": "self_similar_tile", "mask_color": N}  (output is HxH of input-tiles, tile present where input cell == N)

Subgrid extraction:
- {"op": "extract_subgrid", "corner": "tl"|"tr"|"bl"|"br"|"center", "h": H, "w": W}
- {"op": "extract_row", "row": N}
- {"op": "extract_col", "col": N}
- {"op": "extract_half", "which": "top"|"bottom"|"left"|"right"}

Fill / mark:
- {"op": "fill_between_markers", "marker_color": N, "fill_color": M}
- {"op": "draw_frame_inside", "frame_color": N}
- {"op": "draw_frame_outside", "frame_color": N}
- {"op": "fill_enclosed", "color": N}

Object operations:
- {"op": "keep_largest_object"}
- {"op": "keep_smallest_object"}

Composition:
- {"op": "compose", "steps": [<rule1>, <rule2>, ...]}  (apply rules in sequence)
"""


# ---------------------------------------------------------------------------
# Rule compiler — convert structured rule to executable function
# ---------------------------------------------------------------------------

def _g_copy(g):
    return [row[:] for row in g]


def _dims(g):
    return (len(g), len(g[0]) if g else 0)


def compile_rule(rule: dict) -> Callable | None:
    """Compile a structured rule into a Grid -> Grid function. Returns None on
    unrecognized rule (so caller can fall through to LLM retry)."""
    op = rule.get("op")
    if op is None:
        return None

    if op == "identity":
        return lambda g: _g_copy(g)

    if op == "flip":
        axis = rule.get("axis")
        if axis == "h":
            return lambda g: [row[::-1] for row in g]
        if axis == "v":
            return lambda g: g[::-1]
        return None

    if op == "rotate":
        k = int(rule.get("k", 0))
        def rot(g, k=k):
            for _ in range(k % 4):
                g = [list(row) for row in zip(*g[::-1])]
            return [list(row) for row in g]
        return rot

    if op == "transpose":
        return lambda g: [list(row) for row in zip(*g)]

    if op == "recolor":
        m = rule.get("map", {})
        m_int = {int(k): int(v) for k, v in m.items()}
        return lambda g, m=m_int: [[m.get(c, c) for c in row] for row in g]

    if op == "keep_color":
        col = int(rule.get("color", 0))
        return lambda g, col=col: [[c if c == col else 0 for c in row] for row in g]

    if op == "remove_color":
        col = int(rule.get("color", 0))
        return lambda g, col=col: [[0 if c == col else c for c in row] for row in g]

    if op == "extend_cells":
        direction = rule.get("direction")
        from pure_vsa.arc2d_solver import (
            t_extend_each_cell_down, t_extend_each_cell_up,
            t_extend_each_cell_left, t_extend_each_cell_right,
        )
        fns = {"down": t_extend_each_cell_down, "up": t_extend_each_cell_up,
               "left": t_extend_each_cell_left, "right": t_extend_each_cell_right}
        return fns.get(direction)

    if op == "tile_2x2_kaleidoscope":
        from pure_vsa.arc2d_solver import t_kaleidoscope_2x2
        return t_kaleidoscope_2x2

    if op == "tile_2x2_rotational":
        from pure_vsa.arc2d_solver import t_rotational_kaleidoscope_2x2
        return t_rotational_kaleidoscope_2x2

    if op == "self_similar_tile":
        mc = int(rule.get("mask_color", 1))
        from pure_vsa.arc2d_solver import t_self_similar_tile_by_mask
        return lambda g, mc=mc: t_self_similar_tile_by_mask(g, mc)

    if op == "extract_subgrid":
        from pure_vsa.arc2d_solver import (
            t_extract_top_left, t_extract_top_right,
            t_extract_bottom_left, t_extract_bottom_right, t_extract_center,
        )
        corner = rule.get("corner", "tl")
        h = int(rule.get("h", 2)); w = int(rule.get("w", 2))
        fns = {"tl": t_extract_top_left, "tr": t_extract_top_right,
               "bl": t_extract_bottom_left, "br": t_extract_bottom_right,
               "center": t_extract_center}
        fn = fns.get(corner)
        if fn is None:
            return None
        return lambda g, fn=fn, h=h, w=w: fn(g, h, w)

    if op == "extract_row":
        r = int(rule.get("row", 0))
        return lambda g, r=r: [g[r][:]] if 0 <= r < len(g) else None

    if op == "extract_col":
        c = int(rule.get("col", 0))
        def ec(g, c=c):
            h, w = _dims(g)
            if not (0 <= c < w):
                return None
            return [[g[r][c]] for r in range(h)]
        return ec

    if op == "extract_half":
        which = rule.get("which")
        from pure_vsa.arc2d_solver import t_tophalf, t_bottomhalf, t_lefthalf, t_righthalf
        return {"top": t_tophalf, "bottom": t_bottomhalf,
                "left": t_lefthalf, "right": t_righthalf}.get(which)

    if op == "fill_between_markers":
        mc = int(rule.get("marker_color", 1))
        fc = int(rule.get("fill_color", 2))
        from pure_vsa.arc2d_solver import t_fill_between_same_color_markers
        return lambda g, mc=mc, fc=fc: t_fill_between_same_color_markers(g, mc, fc)

    if op == "draw_frame_inside":
        col = int(rule.get("frame_color", 0))
        from pure_vsa.arc2d_solver import t_frame_inside_objects
        return lambda g, col=col: t_frame_inside_objects(g, col)

    if op == "draw_frame_outside":
        col = int(rule.get("frame_color", 0))
        from pure_vsa.arc2d_solver import t_outbox_objects
        return lambda g, col=col: t_outbox_objects(g, col)

    if op == "fill_enclosed":
        col = int(rule.get("color", 0))
        from pure_vsa.arc2d_solver import t_flood_fill_enclosed
        return lambda g, col=col: t_flood_fill_enclosed(g, col)

    if op == "keep_largest_object":
        from pure_vsa.arc2d_solver import t_keep_largest_object
        return lambda g: t_keep_largest_object(g, by_color=True)

    if op == "keep_smallest_object":
        from pure_vsa.arc2d_solver import t_keep_smallest_object
        return lambda g: t_keep_smallest_object(g, by_color=True)

    if op == "compose":
        steps = rule.get("steps", [])
        compiled = [compile_rule(s) for s in steps]
        if any(c is None for c in compiled):
            return None
        def chain(g, compiled=compiled):
            cur = _g_copy(g)
            for fn in compiled:
                cur = fn(cur)
                if cur is None:
                    return None
            return cur
        return chain

    return None


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

PROMPT_PERCEPTION = """You are solving an abstract visual reasoning puzzle.

Given these training (input, output) pairs, describe the transformation rule as a JSON object using ONLY the operations listed below.

{grammar}

Training examples:
{examples}

{retry_feedback}

Output ONLY the JSON object (no prose, no markdown). The rule must work for ALL training examples."""


RETRY_FEEDBACK_TEMPLATE = """\
PREVIOUS ATTEMPT FAILED. Your rule was:
{prev_rule}

It produced wrong output on example {ex_idx}:
- input:    {inp}
- expected: {expected}
- you got:  {got}

Try a different rule."""


def _format_examples(train_pairs) -> str:
    return "\n".join(
        f"Example {i+1}:\n  Input:  {inp}\n  Output: {out}"
        for i, (inp, out) in enumerate(train_pairs)
    )


def _call_ollama(prompt: str, model: str) -> str:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 800},
            },
            timeout=PER_CALL_TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as e:
        return ""


def _extract_json(text: str) -> dict | None:
    # Try to extract the largest JSON object in the text
    candidates = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    for c in sorted(candidates, key=len, reverse=True):
        try:
            obj = json.loads(c)
            if isinstance(obj, dict) and "op" in obj:
                return obj
        except Exception:
            continue
    # Try whole text
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "op" in obj:
            return obj
    except Exception:
        pass
    return None


def _verify_against_training(fn: Callable, train_pairs) -> tuple[bool, int | None, Any, Any]:
    """Run fn against each training pair. Returns (all_match, failed_idx, expected, got)."""
    for i, (inp, out) in enumerate(train_pairs):
        try:
            got = fn(copy.deepcopy(inp))
        except Exception:
            return (False, i, out, None)
        if got != out:
            return (False, i, out, got)
    return (True, None, None, None)


def solve_task_best_of_n(task_data: dict, n_samples: int = 30,
                          model: str = LARGER_MODEL,
                          verbose: bool = False) -> tuple[str, list[list[int]]] | None:
    """AlphaCode-style: sample N candidate rules from the LLM, test EACH against
    training, return the first that matches all training pairs. The LLM is a
    noisy generator; the training pairs are a perfect filter."""
    train = task_data.get("train", [])
    test = task_data.get("test", [])
    if not train or not test:
        return None
    train_pairs = [(t["input"], t["output"]) for t in train]
    test_inp = test[0]["input"]

    prompt = PROMPT_PERCEPTION.format(
        grammar=GRAMMAR_DESCRIPTION,
        examples=_format_examples(train_pairs),
        retry_feedback="",
    )

    for sample_idx in range(n_samples):
        if verbose:
            print(f"  sample {sample_idx + 1}/{n_samples}...")
        # Higher temperature for diversity across samples
        response = _call_ollama_temp(prompt, model, temperature=0.7 if sample_idx > 0 else 0.2)
        rule = _extract_json(response)
        if rule is None:
            continue
        fn = compile_rule(rule)
        if fn is None:
            continue
        ok, _, _, _ = _verify_against_training(fn, train_pairs)
        if ok:
            try:
                result = fn(copy.deepcopy(test_inp))
            except Exception:
                continue
            if result is not None:
                if verbose:
                    print(f"  HIT on sample {sample_idx + 1}: {rule}")
                return ("llm_best_of_n", result)
    return None


def _call_ollama_temp(prompt: str, model: str, temperature: float = 0.3) -> str:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": 800},
            },
            timeout=PER_CALL_TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception:
        return ""


def solve_task_perception(task_data: dict, model: str = DEFAULT_MODEL,
                           verbose: bool = False) -> tuple[str, list[list[int]]] | None:
    train = task_data.get("train", [])
    test = task_data.get("test", [])
    if not train or not test:
        return None
    train_pairs = [(t["input"], t["output"]) for t in train]
    test_inp = test[0]["input"]

    prev_rule = None
    failed_ex = None
    expected = None
    got = None

    for attempt in range(MAX_RETRIES):
        if prev_rule is None:
            retry_feedback = ""
        else:
            retry_feedback = RETRY_FEEDBACK_TEMPLATE.format(
                prev_rule=json.dumps(prev_rule),
                ex_idx=failed_ex + 1,
                inp=train_pairs[failed_ex][0],
                expected=expected,
                got=got,
            )
        prompt = PROMPT_PERCEPTION.format(
            grammar=GRAMMAR_DESCRIPTION,
            examples=_format_examples(train_pairs),
            retry_feedback=retry_feedback,
        )
        if verbose:
            print(f"  attempt {attempt + 1}/{MAX_RETRIES}...")
        t0 = time.time()
        response = _call_ollama(prompt, model)
        if verbose:
            print(f"  got response in {time.time()-t0:.1f}s")

        rule = _extract_json(response)
        if rule is None:
            if verbose:
                print(f"  no parseable JSON")
            continue
        if verbose:
            print(f"  rule: {rule}")

        fn = compile_rule(rule)
        if fn is None:
            if verbose:
                print(f"  rule doesn't compile (unsupported op)")
            prev_rule = rule
            continue

        ok, failed_idx, exp, gt = _verify_against_training(fn, train_pairs)
        if ok:
            try:
                result = fn(copy.deepcopy(test_inp))
            except Exception:
                continue
            if result is None:
                continue
            if verbose:
                print(f"  ✓ rule matched all training, applying to test")
            return ("llm_perception", result)
        else:
            prev_rule = rule
            failed_ex = failed_idx
            expected = exp
            got = gt
            if verbose:
                print(f"  failed on example {failed_idx + 1}")

    return None
