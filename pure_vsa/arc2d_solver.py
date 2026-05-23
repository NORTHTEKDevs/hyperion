"""2D ARC-AGI solver — extends the 1D enumerative program-synthesis approach.

Strategy is the same as `arc1d_solver`: build a library of generic grid
transformations, for each task enumerate programs and select the first that
matches all training examples exactly, then apply to the test input.

This is a baseline. The primitive library here is small — only simple
geometric and color transformations. Real ARC-AGI tasks need object-level
reasoning, enclosed-region detection, scaling, gravity, and more. Those
extensions live in the next iteration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


Grid = list[list[int]]


# ---------------------------------------------------------------------------
# Grid utilities
# ---------------------------------------------------------------------------

def grid_dims(g: Grid) -> tuple[int, int]:
    return len(g), (len(g[0]) if g else 0)


def grid_equal(a: Grid, b: Grid) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if ra != rb:
            return False
    return True


def grid_copy(g: Grid) -> Grid:
    return [row[:] for row in g]


def colors_in(g: Grid) -> set[int]:
    return {c for row in g for c in row}


# ---------------------------------------------------------------------------
# Primitive transformations (input -> output)
# Each returns a new grid or None if not applicable.
# ---------------------------------------------------------------------------

def t_identity(g: Grid) -> Grid:
    return grid_copy(g)


def t_flip_h(g: Grid) -> Grid:
    return [row[::-1] for row in g]


def t_flip_v(g: Grid) -> Grid:
    return g[::-1]


def t_transpose(g: Grid) -> Grid:
    h, w = grid_dims(g)
    return [[g[r][c] for r in range(h)] for c in range(w)]


def t_rotate90(g: Grid) -> Grid:
    # 90 clockwise = transpose then flip horizontal
    return [list(row) for row in zip(*g[::-1])]


def t_rotate180(g: Grid) -> Grid:
    return [row[::-1] for row in g[::-1]]


def t_rotate270(g: Grid) -> Grid:
    # 270 clockwise = 90 counter-clockwise
    return [list(row) for row in zip(*g)][::-1]


def t_recolor(g: Grid, a: int, b: int) -> Grid:
    return [[b if c == a else c for c in row] for row in g]


def t_recolor_all_nonzero(g: Grid, target: int) -> Grid:
    return [[target if c != 0 else 0 for c in row] for row in g]


def t_shift(g: Grid, dr: int, dc: int) -> Grid | None:
    h, w = grid_dims(g)
    out: Grid = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr][nc] = g[r][c]
    return out


def t_tile(g: Grid, kr: int, kc: int) -> Grid:
    """Repeat g kr times vertically and kc times horizontally."""
    out: Grid = []
    for _ in range(kr):
        for row in g:
            out.append(row * kc)
    return out


def t_scale_up(g: Grid, kr: int, kc: int) -> Grid:
    """Scale up: each cell becomes a kr x kc block of itself."""
    out: Grid = []
    for row in g:
        block_rows = [[] for _ in range(kr)]
        for c in row:
            for br in block_rows:
                br.extend([c] * kc)
        out.extend(block_rows)
    return out


def t_scale_down(g: Grid, kr: int, kc: int) -> Grid | None:
    """Inverse of scale_up: take one cell per kr x kc block."""
    h, w = grid_dims(g)
    if h % kr or w % kc:
        return None
    out: Grid = []
    for r in range(0, h, kr):
        out.append([g[r][c] for c in range(0, w, kc)])
    return out


def t_crop_to_bbox(g: Grid) -> Grid | None:
    """Crop to the bounding box of non-zero cells."""
    h, w = grid_dims(g)
    min_r, max_r = h, -1
    min_c, max_c = w, -1
    for r in range(h):
        for c in range(w):
            if g[r][c] != 0:
                if r < min_r: min_r = r
                if r > max_r: max_r = r
                if c < min_c: min_c = c
                if c > max_c: max_c = c
    if max_r < 0:
        return None
    return [row[min_c:max_c + 1] for row in g[min_r:max_r + 1]]


def t_flip_h_concat_right(g: Grid) -> Grid:
    """Concatenate g with its horizontal mirror to the right."""
    mirror = t_flip_h(g)
    return [row + mr for row, mr in zip(g, mirror)]


def t_flip_v_concat_below(g: Grid) -> Grid:
    """Concatenate g with its vertical mirror below."""
    return g + t_flip_v(g)


def t_color_majority(g: Grid) -> int | None:
    """Most common non-zero color (None if no non-zero)."""
    from collections import Counter
    c = Counter(v for row in g for v in row if v != 0)
    if not c:
        return None
    return c.most_common(1)[0][0]


def t_color_minority(g: Grid) -> int | None:
    """Least common non-zero color (None if no non-zero)."""
    from collections import Counter
    c = Counter(v for row in g for v in row if v != 0)
    if not c:
        return None
    return c.most_common()[-1][0]


def t_keep_only_color(g: Grid, color: int) -> Grid:
    """Zero out everything that isn't `color`."""
    return [[c if c == color else 0 for c in row] for row in g]


def t_keep_only_majority(g: Grid) -> Grid:
    col = t_color_majority(g)
    if col is None:
        return grid_copy(g)
    return t_keep_only_color(g, col)


def t_keep_only_minority(g: Grid) -> Grid:
    col = t_color_minority(g)
    if col is None:
        return grid_copy(g)
    return t_keep_only_color(g, col)


def t_gravity(g: Grid, direction: str) -> Grid:
    """Make all non-zero cells fall in the given direction. Cells stack."""
    h, w = grid_dims(g)
    out = [[0] * w for _ in range(h)]
    if direction == "down":
        for c in range(w):
            col = [g[r][c] for r in range(h) if g[r][c] != 0]
            for i, v in enumerate(col):
                out[h - len(col) + i][c] = v
    elif direction == "up":
        for c in range(w):
            col = [g[r][c] for r in range(h) if g[r][c] != 0]
            for i, v in enumerate(col):
                out[i][c] = v
    elif direction == "right":
        for r in range(h):
            row = [g[r][c] for c in range(w) if g[r][c] != 0]
            for i, v in enumerate(row):
                out[r][w - len(row) + i] = v
    elif direction == "left":
        for r in range(h):
            row = [g[r][c] for c in range(w) if g[r][c] != 0]
            for i, v in enumerate(row):
                out[r][i] = v
    return out


def t_complete_symmetry(g: Grid, axis: str) -> Grid:
    """Where g is partially symmetric, complete it by mirroring filled cells
    across the axis (only fills zeros, never overwrites)."""
    h, w = grid_dims(g)
    out = grid_copy(g)
    if axis == "h":  # left-right symmetry
        for r in range(h):
            for c in range(w):
                mirror_c = w - 1 - c
                if out[r][c] == 0 and g[r][mirror_c] != 0:
                    out[r][c] = g[r][mirror_c]
    elif axis == "v":  # top-bottom symmetry
        for r in range(h):
            for c in range(w):
                mirror_r = h - 1 - r
                if out[r][c] == 0 and g[mirror_r][c] != 0:
                    out[r][c] = g[mirror_r][c]
    elif axis == "both":
        out = t_complete_symmetry(out, "h")
        out = t_complete_symmetry(out, "v")
    return out


def t_flood_fill_enclosed(g: Grid, fill_color: int) -> Grid:
    """Find all interior cells (zeros not connected to the border via 4-neighborhood
    through other zeros) and fill them with `fill_color`."""
    h, w = grid_dims(g)
    if h == 0 or w == 0:
        return grid_copy(g)
    # Mark all border-connected zeros
    visited = [[False] * w for _ in range(h)]
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if g[r][c] == 0:
                stack.append((r, c))
        if r in (0, h - 1):
            for c in range(w):
                if g[r][c] == 0:
                    stack.append((r, c))
    while stack:
        r, c = stack.pop()
        if r < 0 or r >= h or c < 0 or c >= w:
            continue
        if visited[r][c] or g[r][c] != 0:
            continue
        visited[r][c] = True
        stack.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])
    out = grid_copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c] == 0 and not visited[r][c]:
                out[r][c] = fill_color
    return out


# ---------------------------------------------------------------------------
# Program-synthesis solver
# ---------------------------------------------------------------------------

@dataclass
class Program:
    name: str
    apply: Any  # function(grid) -> Grid | None


def _shape_preserving_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    """Programs that don't change grid dimensions."""
    progs: list[Program] = [
        Program("identity", t_identity),
        Program("flip_h", t_flip_h),
        Program("flip_v", t_flip_v),
        Program("rotate180", t_rotate180),
    ]
    # Only include square-grid rotations if all training pairs are square AND in/out same shape.
    all_square = all(
        len(i) == len(i[0]) and len(o) == len(o[0]) and len(i) == len(o)
        for i, o in train_pairs
    )
    if all_square:
        progs.append(Program("rotate90", t_rotate90))
        progs.append(Program("rotate270", t_rotate270))
        progs.append(Program("transpose", t_transpose))

    # Shifts — only if all training pairs preserve shape
    same_shape = all(
        len(i) == len(o) and len(i[0]) == len(o[0])
        for i, o in train_pairs
    )
    if same_shape:
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                if dr == 0 and dc == 0:
                    continue
                _dr, _dc = dr, dc
                progs.append(Program(
                    f"shift_{dr}_{dc}",
                    lambda g, dr=_dr, dc=_dc: t_shift(g, dr, dc),
                ))

    # Recolor: any color swap inferred from training
    swaps: set[tuple[int, int]] = set()
    for inp, out in train_pairs:
        h, w = grid_dims(inp)
        if (h, w) != grid_dims(out):
            continue
        for r in range(h):
            for c in range(w):
                a, b = inp[r][c], out[r][c]
                if a != b and a != 0 and b != 0:
                    swaps.add((a, b))
    for a, b in swaps:
        progs.append(Program(f"recolor_{a}_to_{b}", lambda g, a=a, b=b: t_recolor(g, a, b)))

    # Recolor-all-nonzero: detect from training
    for inp, out in train_pairs:
        if grid_dims(inp) != grid_dims(out):
            continue
        # Check if every nonzero in inp became the same color in out
        target_colors = set()
        ok = True
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                a, b = inp[r][c], out[r][c]
                if a != 0:
                    if b == 0:
                        ok = False; break
                    target_colors.add(b)
                else:
                    if b != 0:
                        ok = False; break
            if not ok:
                break
        if ok and len(target_colors) == 1:
            t = next(iter(target_colors))
            progs.append(Program(
                f"recolor_all_nonzero_to_{t}",
                lambda g, t=t: t_recolor_all_nonzero(g, t),
            ))
            break  # only need one such program

    return progs


def _scale_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    """Programs that change dimensions deterministically (tile, scale, crop)."""
    progs: list[Program] = []
    # Detect uniform scaling factor across all training pairs
    factors: set[tuple[int, int]] = set()
    for inp, out in train_pairs:
        hi, wi = grid_dims(inp)
        ho, wo = grid_dims(out)
        if hi == 0 or wi == 0:
            continue
        if ho % hi == 0 and wo % wi == 0:
            factors.add((ho // hi, wo // wi))
    if len(factors) == 1 and (1, 1) not in factors:
        kr, kc = next(iter(factors))
        progs.append(Program(f"tile_{kr}x{kc}", lambda g, kr=kr, kc=kc: t_tile(g, kr, kc)))
        progs.append(Program(f"scale_up_{kr}x{kc}", lambda g, kr=kr, kc=kc: t_scale_up(g, kr, kc)))

    # Detect uniform scale-down factor
    down_factors: set[tuple[int, int]] = set()
    for inp, out in train_pairs:
        hi, wi = grid_dims(inp)
        ho, wo = grid_dims(out)
        if ho == 0 or wo == 0:
            continue
        if hi % ho == 0 and wi % wo == 0:
            down_factors.add((hi // ho, wi // wo))
    if len(down_factors) == 1 and (1, 1) not in down_factors:
        kr, kc = next(iter(down_factors))
        progs.append(Program(f"scale_down_{kr}x{kc}", lambda g, kr=kr, kc=kc: t_scale_down(g, kr, kc)))

    # Crop to bbox (if output dimensions vary but always match the input's bbox)
    all_bbox = True
    for inp, out in train_pairs:
        cropped = t_crop_to_bbox(inp)
        if cropped is None or not grid_equal(cropped, out):
            all_bbox = False; break
    if all_bbox:
        progs.append(Program("crop_to_bbox", t_crop_to_bbox))

    # Mirror concat (output is 2x wider/taller)
    for inp, out in train_pairs:
        hi, wi = grid_dims(inp)
        ho, wo = grid_dims(out)
        if ho == hi and wo == 2 * wi:
            progs.append(Program("flip_h_concat_right", t_flip_h_concat_right))
            break
    for inp, out in train_pairs:
        hi, wi = grid_dims(inp)
        ho, wo = grid_dims(out)
        if wo == wi and ho == 2 * hi:
            progs.append(Program("flip_v_concat_below", t_flip_v_concat_below))
            break

    return progs


def _selection_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    """Programs that select / filter cells based on color frequency."""
    progs: list[Program] = []
    # Only meaningful when output shape == input shape
    if not all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        return progs
    progs.append(Program("keep_only_majority_color", t_keep_only_majority))
    progs.append(Program("keep_only_minority_color", t_keep_only_minority))
    return progs


def _fill_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    """Flood-fill enclosed regions with a learned color."""
    progs: list[Program] = []
    if not all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        return progs
    # Detect the fill color: any color that appears in output but not in input.
    fill_colors: set[int] = set()
    for inp, out in train_pairs:
        new = colors_in(out) - colors_in(inp)
        fill_colors.update(new)
    for fc in fill_colors:
        progs.append(Program(
            f"flood_fill_enclosed_{fc}",
            lambda g, fc=fc: t_flood_fill_enclosed(g, fc),
        ))
    return progs


def _gravity_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    if not all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        return []
    progs: list[Program] = []
    for d in ("down", "up", "left", "right"):
        progs.append(Program(f"gravity_{d}", lambda g, d=d: t_gravity(g, d)))
    return progs


def _symmetry_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    if not all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        return []
    progs: list[Program] = []
    for axis in ("h", "v", "both"):
        progs.append(Program(f"complete_symmetry_{axis}", lambda g, axis=axis: t_complete_symmetry(g, axis)))
    return progs


def candidate_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    return (
        _shape_preserving_programs(train_pairs)
        + _scale_programs(train_pairs)
        + _selection_programs(train_pairs)
        + _fill_programs(train_pairs)
        + _gravity_programs(train_pairs)
        + _symmetry_programs(train_pairs)
    )


def solve_task(task_data: dict) -> tuple[str, Grid] | None:
    train = task_data.get("train", [])
    test = task_data.get("test", [])
    if not train or not test:
        return None
    train_pairs = [(t["input"], t["output"]) for t in train]

    progs = candidate_programs(train_pairs)
    for prog in progs:
        try:
            ok = all(
                (result := prog.apply(inp)) is not None and grid_equal(result, out)
                for inp, out in train_pairs
            )
        except Exception:
            ok = False
        if ok:
            test_inp = test[0]["input"]
            try:
                result = prog.apply(test_inp)
            except Exception:
                continue
            if result is None:
                continue
            return prog.name, result
    return None


def evaluate_directory(root: Path) -> dict[str, list[bool]]:
    """Return {task_name: [correct_bool]} for each task in `root`."""
    results: dict[str, list[bool]] = {}
    for f in sorted(root.glob("*.json")):
        data = json.loads(f.read_text())
        sol = solve_task(data)
        if sol is None:
            results[f.name] = [False]
            continue
        _, pred = sol
        try:
            expected = data["test"][0]["output"]
            results[f.name] = [grid_equal(pred, expected)]
        except Exception:
            results[f.name] = [False]
    return results
