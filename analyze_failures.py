"""Cluster failing ARC-AGI tasks by feature relationships to find the
biggest categories of misses. Output: a tabular breakdown the user can
use to prioritize where to invest primitive engineering.

Features computed per task:
  - in/out shape relationship (same / scale_up Nx / scale_down Nx / 1x1 / smaller / bigger)
  - color counts (input, output, color overlap)
  - object counts
  - whether there's a "marker" (singleton-color cell)
  - whether there are full-row or full-col dividers
  - whether the change is local (few cells differ) or global
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from pure_vsa.arc2d_solver import (
    solve_task, grid_dims, grid_equal, find_objects, colors_in,
    _detect_grid_dividers,
)


def features(data: dict) -> dict:
    pair = data["train"][0]
    inp, out = pair["input"], pair["output"]
    hi, wi = grid_dims(inp); ho, wo = grid_dims(out)
    ic = colors_in(inp); oc = colors_in(out)
    inp_objs = find_objects(inp, by_color=True) if hi * wi < 900 else []
    dividers = _detect_grid_dividers(inp) is not None

    # Shape relation
    if (hi, wi) == (ho, wo):
        shape_rel = "same"
    elif (ho, wo) == (1, 1):
        shape_rel = "to_1x1"
    elif ho > hi or wo > wi:
        if hi > 0 and wi > 0 and ho % hi == 0 and wo % wi == 0:
            shape_rel = f"scale_up_{ho//hi}x{wo//wi}"
        else:
            shape_rel = "bigger"
    elif ho < hi or wo < wi:
        if ho > 0 and wo > 0 and hi % ho == 0 and wi % wo == 0:
            shape_rel = f"scale_down_{hi//ho}x{wi//wo}"
        else:
            shape_rel = "smaller"
    else:
        shape_rel = "same"

    # Cell-level changes (only meaningful for same-shape)
    diffs = None
    if shape_rel == "same":
        diffs = sum(1 for r in range(hi) for c in range(wi) if inp[r][c] != out[r][c])

    # Marker detection (singleton-color non-zero in input)
    from collections import Counter as C
    cc = C(v for row in inp for v in row if v != 0)
    has_marker = any(n == 1 for n in cc.values())

    return {
        "shape_rel": shape_rel,
        "n_in_colors": len(ic),
        "n_out_colors": len(oc),
        "new_colors": len(oc - ic),
        "n_objects": len(inp_objs),
        "has_marker": has_marker,
        "has_dividers": dividers,
        "diffs": diffs,
    }


def cluster_failures() -> None:
    train_dir = Path("data/arc_agi/training")
    failures_by_shape: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    n_solved = 0
    n_total = 0
    for f in sorted(train_dir.glob("*.json")):
        data = json.loads(f.read_text())
        n_total += 1
        sol = solve_task(data, allow_compose=False)
        if sol is not None and grid_equal(sol[1], data["test"][0]["output"]):
            n_solved += 1
            continue
        feats = features(data)
        failures_by_shape[feats["shape_rel"]].append((f.name, feats))

    print(f"=== Solved: {n_solved}/{n_total} ({n_solved/n_total*100:.1f}%) ===")
    print(f"=== Failed: {n_total - n_solved} ===\n")

    print("=== Failures by shape relation ===")
    for rel, items in sorted(failures_by_shape.items(), key=lambda kv: -len(kv[1])):
        print(f"  {rel:<25} {len(items)}")

    # Drill into same-shape failures
    print("\n=== Same-shape failures by # cell changes (local vs global) ===")
    diff_bins: dict[str, int] = defaultdict(int)
    same_shape = failures_by_shape.get("same", [])
    for _, feats in same_shape:
        d = feats["diffs"]
        if d is None or d == 0:
            continue
        if d <= 3:
            bin_name = "1-3 cells (highly local)"
        elif d <= 10:
            bin_name = "4-10 cells (local)"
        elif d <= 30:
            bin_name = "11-30 cells (medium)"
        else:
            bin_name = "31+ cells (global)"
        diff_bins[bin_name] += 1
    for b, n in sorted(diff_bins.items(), key=lambda kv: -kv[1]):
        print(f"  {b:<30} {n}")

    # Same-shape with new colors introduced
    print("\n=== Same-shape failures with NEW colors in output ===")
    n_with_new = sum(1 for _, f in same_shape if f["new_colors"] > 0)
    n_without_new = sum(1 for _, f in same_shape if f["new_colors"] == 0)
    print(f"  with new colors:    {n_with_new}")
    print(f"  without new colors: {n_without_new}")

    # Same-shape failures with marker
    print("\n=== Same-shape failures with vs without marker ===")
    n_marker = sum(1 for _, f in same_shape if f["has_marker"])
    n_no = sum(1 for _, f in same_shape if not f["has_marker"])
    print(f"  with marker:    {n_marker}")
    print(f"  without marker: {n_no}")

    # Same-shape failures with dividers
    n_div = sum(1 for _, f in same_shape if f["has_dividers"])
    print(f"  with divider lines: {n_div}")


if __name__ == "__main__":
    cluster_failures()
