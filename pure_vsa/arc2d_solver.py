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


# ---------------------------------------------------------------------------
# Object detection (connected components)
# ---------------------------------------------------------------------------

def find_objects(g: Grid, by_color: bool = True, diag: bool = False) -> list[dict]:
    """Return list of objects. Each object is a dict:
      { 'color': int, 'cells': set[(r,c)], 'bbox': (r0,c0,r1,c1) }.
    by_color=True: connect cells of same color via 4-neighbor (or 8 if diag).
    by_color=False: connect any non-zero cells regardless of color.
    """
    h, w = grid_dims(g)
    visited = [[False] * w for _ in range(h)]
    objects: list[dict] = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diag:
        neighbors += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for r0 in range(h):
        for c0 in range(w):
            if visited[r0][c0] or g[r0][c0] == 0:
                continue
            color = g[r0][c0]
            cells = set()
            stack = [(r0, c0)]
            while stack:
                r, c = stack.pop()
                if r < 0 or r >= h or c < 0 or c >= w or visited[r][c]:
                    continue
                if g[r][c] == 0:
                    continue
                if by_color and g[r][c] != color:
                    continue
                visited[r][c] = True
                cells.add((r, c))
                for dr, dc in neighbors:
                    stack.append((r + dr, c + dc))
            if cells:
                rs = [r for r, _ in cells]; cs = [c for _, c in cells]
                bbox = (min(rs), min(cs), max(rs), max(cs))
                objects.append({"color": color, "cells": cells, "bbox": bbox})
    return objects


def t_keep_largest_object(g: Grid, by_color: bool = True) -> Grid:
    objs = find_objects(g, by_color=by_color)
    if not objs:
        return grid_copy(g)
    largest = max(objs, key=lambda o: len(o["cells"]))
    h, w = grid_dims(g)
    out = [[0] * w for _ in range(h)]
    for r, c in largest["cells"]:
        out[r][c] = largest["color"]
    return out


def t_keep_smallest_object(g: Grid, by_color: bool = True) -> Grid:
    objs = find_objects(g, by_color=by_color)
    if not objs:
        return grid_copy(g)
    smallest = min(objs, key=lambda o: len(o["cells"]))
    h, w = grid_dims(g)
    out = [[0] * w for _ in range(h)]
    for r, c in smallest["cells"]:
        out[r][c] = smallest["color"]
    return out


def t_crop_to_largest_object(g: Grid, by_color: bool = True) -> Grid | None:
    objs = find_objects(g, by_color=by_color)
    if not objs:
        return None
    largest = max(objs, key=lambda o: len(o["cells"]))
    r0, c0, r1, c1 = largest["bbox"]
    return [row[c0:c1 + 1] for row in g[r0:r1 + 1]]


def t_recolor_by_size(g: Grid, size_to_color: dict[int, int]) -> Grid | None:
    """Recolor each object to a color determined by its size."""
    objs = find_objects(g, by_color=True)
    h, w = grid_dims(g)
    out = grid_copy(g)
    for o in objs:
        sz = len(o["cells"])
        if sz not in size_to_color:
            return None
        nc = size_to_color[sz]
        for r, c in o["cells"]:
            out[r][c] = nc
    return out


def _object_bbox_shape(obj: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = obj["bbox"]
    return (r1 - r0 + 1, c1 - c0 + 1)


def _object_cells_normalized(obj: dict) -> frozenset[tuple[int, int]]:
    r0, c0, _, _ = obj["bbox"]
    return frozenset((r - r0, c - c0) for r, c in obj["cells"])


def t_translate_object_to_marker(g: Grid, obj_color: int, marker_color: int) -> Grid | None:
    """Move the (single) object of `obj_color` so its top-left aligns with the
    single cell of `marker_color`. Removes the marker."""
    objs = find_objects(g, by_color=True)
    obj_list = [o for o in objs if o["color"] == obj_color]
    marker_list = [o for o in objs if o["color"] == marker_color and len(o["cells"]) == 1]
    if len(obj_list) != 1 or len(marker_list) != 1:
        return None
    obj = obj_list[0]
    mr, mc = next(iter(marker_list[0]["cells"]))
    or0, oc0, _, _ = obj["bbox"]
    dr, dc = mr - or0, mc - oc0
    h, w = grid_dims(g)
    out = [[0] * w for _ in range(h)]
    # everything except the moved obj and the marker
    for r in range(h):
        for c in range(w):
            if g[r][c] == obj_color or g[r][c] == marker_color:
                continue
            out[r][c] = g[r][c]
    for r, c in obj["cells"]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out[nr][nc] = obj_color
    return out


def t_copy_object_to_each_marker(g: Grid, obj_color: int, marker_color: int) -> Grid | None:
    """Find the single object of `obj_color` (the 'template') and the singleton
    cells of `marker_color`. Stamp the template at each marker, centered on the
    marker. Remove markers."""
    objs = find_objects(g, by_color=True)
    obj_list = [o for o in objs if o["color"] == obj_color]
    markers = [o for o in objs if o["color"] == marker_color and len(o["cells"]) == 1]
    if len(obj_list) != 1 or not markers:
        return None
    template = obj_list[0]
    th, tw = _object_bbox_shape(template)
    cells_norm = _object_cells_normalized(template)
    h, w = grid_dims(g)
    out = [row[:] for row in g]
    # clear markers
    for m in markers:
        mr, mc = next(iter(m["cells"]))
        out[mr][mc] = 0
    # stamp template centered on each marker
    half_r, half_c = th // 2, tw // 2
    for m in markers:
        mr, mc = next(iter(m["cells"]))
        for dr, dc in cells_norm:
            nr, nc = mr - half_r + dr, mc - half_c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr][nc] = obj_color
    return out


def t_gravity_toward_color(g: Grid, mover_color: int, anchor_color: int, direction: str) -> Grid | None:
    """Move all cells of `mover_color` toward the nearest cell of `anchor_color`
    in the given direction (each column/row independently)."""
    h, w = grid_dims(g)
    out = [[0] * w for _ in range(h)]
    # copy everything that isn't the mover
    for r in range(h):
        for c in range(w):
            if g[r][c] != mover_color:
                out[r][c] = g[r][c]
    if direction in ("up", "down"):
        for c in range(w):
            anchor_rows = [r for r in range(h) if g[r][c] == anchor_color]
            mover_rows = [r for r in range(h) if g[r][c] == mover_color]
            if not anchor_rows or not mover_rows:
                # leave movers where they are
                for r in mover_rows:
                    out[r][c] = mover_color
                continue
            anchor_row = anchor_rows[0] if direction == "up" else anchor_rows[-1]
            # stack movers adjacent to anchor
            for i, _ in enumerate(mover_rows):
                if direction == "down":
                    nr = anchor_row - 1 - i
                else:
                    nr = anchor_row + 1 + i
                if 0 <= nr < h:
                    out[nr][c] = mover_color
    elif direction in ("left", "right"):
        for r in range(h):
            anchor_cols = [c for c in range(w) if g[r][c] == anchor_color]
            mover_cols = [c for c in range(w) if g[r][c] == mover_color]
            if not anchor_cols or not mover_cols:
                for c in mover_cols:
                    out[r][c] = mover_color
                continue
            anchor_col = anchor_cols[0] if direction == "left" else anchor_cols[-1]
            for i, _ in enumerate(mover_cols):
                if direction == "right":
                    nc = anchor_col - 1 - i
                else:
                    nc = anchor_col + 1 + i
                if 0 <= nc < w:
                    out[r][nc] = mover_color
    return out


def t_count_objects_to_color(g: Grid, count_to_color: dict[int, int]) -> Grid | None:
    """Output a 1x1 grid whose color encodes the object count."""
    n = len(find_objects(g, by_color=True))
    if n not in count_to_color:
        return None
    return [[count_to_color[n]]]


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
    progs.append(Program("fill_with_majority_color", t_fill_with_majority_color))
    progs.append(Program("fill_with_majority_nonzero", t_fill_with_majority_nonzero))
    progs.append(Program("fill_with_minority_color", t_fill_with_minority_color))
    progs.append(Program("replace_unique_with_majority", t_replace_unique_color_with_majority))
    progs.append(Program("replace_unique_with_minority", t_replace_unique_color_with_minority))
    progs.append(Program("keep_only_unique_color", t_only_keep_unique_color))

    # Per-row uniform-color detector: try all (uniform_color, nonuniform_color) pairs
    # induced from training output colors
    out_colors: set[int] = set()
    for _, out in train_pairs:
        out_colors.update(colors_in(out))
    for uc in out_colors:
        for nc in out_colors:
            if uc == nc:
                continue
            progs.append(Program(
                f"per_row_uniform_{uc}_other_{nc}",
                lambda g, uc=uc, nc=nc: t_per_row_uniform_to_color(g, uc, nc),
            ))
            progs.append(Program(
                f"per_col_uniform_{uc}_other_{nc}",
                lambda g, uc=uc, nc=nc: t_per_col_uniform_to_color(g, uc, nc),
            ))
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


def t_extract_object_bbox_grid(g: Grid, color: int) -> Grid | None:
    """Crop to the bbox of cells of `color`, returning the bbox area (with all cells, not just color)."""
    h, w = grid_dims(g)
    min_r, max_r = h, -1; min_c, max_c = w, -1
    for r in range(h):
        for c in range(w):
            if g[r][c] == color:
                if r < min_r: min_r = r
                if r > max_r: max_r = r
                if c < min_c: min_c = c
                if c > max_c: max_c = c
    if max_r < 0:
        return None
    return [row[min_c:max_c + 1] for row in g[min_r:max_r + 1]]


def t_extract_largest_object_bbox_grid(g: Grid) -> Grid | None:
    """Crop to the bbox of the largest connected object (returning the bbox area including
    any non-object cells inside the bbox)."""
    objs = find_objects(g, by_color=True)
    if not objs:
        return None
    largest = max(objs, key=lambda o: len(o["cells"]))
    r0, c0, r1, c1 = largest["bbox"]
    return [row[c0:c1 + 1] for row in g[r0:r1 + 1]]


def t_extract_non_majority_subgrid(g: Grid) -> Grid | None:
    """If g is divided into subgrids and one differs from the majority pattern,
    return that 'odd' one. Same as extract_unique_subgrid but tolerates more
    structural differences."""
    subs = _split_into_subgrids(g)
    if subs is None:
        return None
    flat = [s for row in subs for s in row]
    if len(flat) < 3:
        return None
    from collections import Counter
    keys = [tuple(tuple(r) for r in s["grid"]) for s in flat]
    counts = Counter(keys)
    if len(counts) != 2:
        return None
    minority_key, _ = counts.most_common()[-1]
    for s, k in zip(flat, keys):
        if k == minority_key:
            return s["grid"]
    return None


def t_crop_to_color_bbox(g: Grid, color: int) -> Grid | None:
    """Crop to bbox of cells with the given color."""
    h, w = grid_dims(g)
    min_r, max_r = h, -1; min_c, max_c = w, -1
    for r in range(h):
        for c in range(w):
            if g[r][c] == color:
                if r < min_r: min_r = r
                if r > max_r: max_r = r
                if c < min_c: min_c = c
                if c > max_c: max_c = c
    if max_r < 0:
        return None
    return [row[min_c:max_c + 1] for row in g[min_r:max_r + 1]]


def t_keep_only_color_mask(g: Grid, color: int, replace_with: int) -> Grid:
    """Output a mask: cells with `color` -> `replace_with`, others -> 0."""
    return [[replace_with if c == color else 0 for c in row] for row in g]


def _count_objects(g: Grid) -> int:
    return len(find_objects(g, by_color=True))


def _count_color(g: Grid, c: int) -> int:
    return sum(row.count(c) for row in g)


def _count_distinct_colors(g: Grid) -> int:
    return len(colors_in(g) - {0})


def t_property_to_constant_grid(g: Grid, prop_fn, value_to_grid: dict) -> Grid | None:
    """Compute prop_fn(g) and look up the corresponding output grid."""
    v = prop_fn(g)
    if v not in value_to_grid:
        return None
    return [row[:] for row in value_to_grid[v]]


def t_count_color_to_grid(g: Grid, color: int, output_color: int) -> Grid:
    """Output is a 1xN grid filled with `output_color` where N is the count of `color` in input."""
    n = sum(row.count(color) for row in g)
    if n <= 0:
        return [[0]]
    return [[output_color] * n]


def t_dominant_color_grid(g: Grid, h: int, w: int) -> Grid | None:
    """Output an HxW grid filled with the most common non-zero color in input."""
    from collections import Counter
    c = Counter(v for row in g for v in row if v != 0)
    if not c:
        return None
    color = c.most_common(1)[0][0]
    return [[color] * w for _ in range(h)]


def _detect_grid_dividers(g: Grid) -> tuple[list[int], list[int], int] | None:
    """Detect rows/cols that are uniformly filled with a single color (acting
    as grid dividers). Returns (divider_rows, divider_cols, color) or None.
    Requires at least one divider row OR column AND the divider must not be
    the entire grid (otherwise everything is trivially a divider)."""
    h, w = grid_dims(g)
    if h < 3 or w < 3:
        return None
    candidates: dict[int, tuple[list[int], list[int]]] = {}
    for color in range(0, 10):
        rows = [r for r in range(h) if all(g[r][c] == color for c in range(w))]
        cols = [c for c in range(w) if all(g[r][c] == color for r in range(h))]
        if rows or cols:
            candidates[color] = (rows, cols)
    if not candidates:
        return None
    # Pick the color whose divider count is most distinctive (excluding cases
    # where the divider fills MOST of the grid — then it's just a background).
    valid = {col: (rs, cs) for col, (rs, cs) in candidates.items()
             if (len(rs) + len(cs)) < (h + w) - 1 and (rs or cs)}
    if not valid:
        return None
    color = max(valid, key=lambda k: len(valid[k][0]) + len(valid[k][1]))
    rows, cols = valid[color]
    return rows, cols, color


def _split_into_subgrids(g: Grid) -> list[list[dict]] | None:
    """Split g into a 2D list of subgrids based on detected divider lines.
    Each subgrid dict: {'grid': Grid, 'r0': int, 'c0': int}."""
    info = _detect_grid_dividers(g)
    if info is None:
        return None
    rows, cols, _ = info
    h, w = grid_dims(g)
    row_bounds = [-1] + rows + [h]
    col_bounds = [-1] + cols + [w]
    sub_rows: list[list[dict]] = []
    for i in range(len(row_bounds) - 1):
        r0, r1 = row_bounds[i] + 1, row_bounds[i + 1]
        if r1 <= r0:
            continue
        row: list[dict] = []
        for j in range(len(col_bounds) - 1):
            c0, c1 = col_bounds[j] + 1, col_bounds[j + 1]
            if c1 <= c0:
                continue
            sub = [grow[c0:c1] for grow in g[r0:r1]]
            row.append({"grid": sub, "r0": r0, "c0": c0})
        if row:
            sub_rows.append(row)
    if not sub_rows:
        return None
    return sub_rows


def t_extract_unique_subgrid(g: Grid) -> Grid | None:
    """If g is divided into NxM subgrids and exactly ONE subgrid differs from
    the others (by content), return that unique one."""
    subs = _split_into_subgrids(g)
    if subs is None:
        return None
    flat: list[dict] = [s for row in subs for s in row]
    if len(flat) < 2:
        return None
    # Group by content
    from collections import Counter
    keys = [tuple(tuple(r) for r in s["grid"]) for s in flat]
    counts = Counter(keys)
    if len(counts) != 2:
        return None
    # The unique one appears exactly once
    for k, n in counts.items():
        if n == 1:
            for s, kk in zip(flat, keys):
                if kk == k:
                    return s["grid"]
    return None


def t_extract_majority_subgrid(g: Grid) -> Grid | None:
    """Return the subgrid pattern that appears most often (the 'common' one)."""
    subs = _split_into_subgrids(g)
    if subs is None:
        return None
    flat = [s for row in subs for s in row]
    if len(flat) < 2:
        return None
    from collections import Counter
    keys = [tuple(tuple(r) for r in s["grid"]) for s in flat]
    counts = Counter(keys)
    most_common, _ = counts.most_common(1)[0]
    return [list(r) for r in most_common]


def t_extract_subgrid_with_max_color_count(g: Grid, target_color: int) -> Grid | None:
    """Return the subgrid containing the most cells of target_color."""
    subs = _split_into_subgrids(g)
    if subs is None:
        return None
    flat = [s for row in subs for s in row]
    if not flat:
        return None
    best = max(flat, key=lambda s: sum(row.count(target_color) for row in s["grid"]))
    return best["grid"]


def _detect_symmetry_axis(g: Grid) -> str | None:
    """Detect if g has horizontal/vertical/both symmetry (ignoring zeros, i.e.,
    treating any cell as 'compatible' if the mirror is zero)."""
    h, w = grid_dims(g)
    h_sym = all(
        g[r][c] == 0 or g[r][w - 1 - c] == 0 or g[r][c] == g[r][w - 1 - c]
        for r in range(h) for c in range(w // 2)
    )
    v_sym = all(
        g[r][c] == 0 or g[h - 1 - r][c] == 0 or g[r][c] == g[h - 1 - r][c]
        for c in range(w) for r in range(h // 2)
    )
    if h_sym and v_sym:
        return "both"
    if h_sym:
        return "h"
    if v_sym:
        return "v"
    return None


# ---------------------------------------------------------------------------
# Cellular-automaton-style local rule induction
# ---------------------------------------------------------------------------

def _neighbor_signature(g: Grid, r: int, c: int, k: int = 1) -> tuple:
    """Frozen signature of (k×2+1)×(k×2+1) neighborhood, edges padded with -1."""
    h, w = grid_dims(g)
    cells = []
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                cells.append(g[nr][nc])
            else:
                cells.append(-1)
    return tuple(cells)


def _learn_ca_rule(train_pairs: list[tuple[Grid, Grid]]):
    """Induce a per-cell rule mapping (input_color, neighbor_signature) -> output_color.
    Returns the rule dict or None if not consistent across training pairs.
    """
    rule: dict[tuple, int] = {}
    for inp, out in train_pairs:
        if grid_dims(inp) != grid_dims(out):
            return None
        h, w = grid_dims(inp)
        for r in range(h):
            for c in range(w):
                key = (inp[r][c], _neighbor_signature(inp, r, c, k=1))
                val = out[r][c]
                if key in rule:
                    if rule[key] != val:
                        return None
                else:
                    rule[key] = val
    return rule


def _learn_ca_rule_by_color_count(train_pairs: list[tuple[Grid, Grid]]):
    """Simpler CA rule: output cell depends only on (input_color, Counter of neighbor colors).
    Less specific than full neighbor signature, so generalizes better."""
    from collections import Counter
    rule: dict[tuple, int] = {}
    for inp, out in train_pairs:
        if grid_dims(inp) != grid_dims(out):
            return None
        h, w = grid_dims(inp)
        for r in range(h):
            for c in range(w):
                neighbors = []
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            neighbors.append(inp[nr][nc])
                key = (inp[r][c], tuple(sorted(Counter(neighbors).items())))
                val = out[r][c]
                if key in rule and rule[key] != val:
                    return None
                rule[key] = val
    return rule


def t_apply_ca_rule(g: Grid, rule: dict[tuple, int]) -> Grid | None:
    h, w = grid_dims(g)
    out: Grid = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            key = (g[r][c], _neighbor_signature(g, r, c, k=1))
            if key not in rule:
                return None
            out[r][c] = rule[key]
    return out


def t_apply_ca_rule_count(g: Grid, rule: dict[tuple, int]) -> Grid | None:
    from collections import Counter
    h, w = grid_dims(g)
    out: Grid = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            neighbors = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        neighbors.append(g[nr][nc])
            key = (g[r][c], tuple(sorted(Counter(neighbors).items())))
            if key not in rule:
                return None
            out[r][c] = rule[key]
    return out


def t_replace_unique_color_with_majority(g: Grid) -> Grid | None:
    """If the grid has exactly one cell of a color that appears nowhere else,
    replace it with the majority color. Common 'fix the broken cell' pattern."""
    from collections import Counter
    h, w = grid_dims(g)
    c = Counter(v for row in g for v in row)
    # Find a color that appears exactly once
    singletons = [k for k, v in c.items() if v == 1]
    if len(singletons) != 1:
        return None
    target = singletons[0]
    # Majority color (most common, excluding target)
    others = [(k, v) for k, v in c.items() if k != target]
    if not others:
        return None
    majority = max(others, key=lambda x: x[1])[0]
    return [[majority if v == target else v for v in row] for row in g]


def t_replace_unique_color_with_minority(g: Grid) -> Grid | None:
    """Same but replace with the least-common (non-target) color."""
    from collections import Counter
    h, w = grid_dims(g)
    c = Counter(v for row in g for v in row)
    singletons = [k for k, v in c.items() if v == 1]
    if len(singletons) != 1:
        return None
    target = singletons[0]
    others = [(k, v) for k, v in c.items() if k != target]
    if not others:
        return None
    minority = min(others, key=lambda x: x[1])[0]
    return [[minority if v == target else v for v in row] for row in g]


def t_only_keep_unique_color(g: Grid) -> Grid | None:
    """Inverse: keep only the cells that have the unique (singleton) color."""
    from collections import Counter
    c = Counter(v for row in g for v in row if v != 0)
    singletons = [k for k, v in c.items() if v == 1]
    if len(singletons) != 1:
        return None
    target = singletons[0]
    return [[v if v == target else 0 for v in row] for row in g]


def t_complete_detected_symmetry(g: Grid) -> Grid | None:
    """If g has detected partial symmetry, complete it by mirroring filled
    cells across the detected axis (only fills zeros)."""
    axis = _detect_symmetry_axis(g)
    if axis is None:
        return None
    return t_complete_symmetry(g, axis)


def t_fill_with_majority_color(g: Grid) -> Grid | None:
    """Output grid is same shape, filled entirely with the most common color in input."""
    from collections import Counter
    h, w = grid_dims(g)
    c = Counter(v for row in g for v in row)
    if not c:
        return None
    color, _ = c.most_common(1)[0]
    return [[color] * w for _ in range(h)]


def t_fill_with_minority_color(g: Grid) -> Grid | None:
    from collections import Counter
    h, w = grid_dims(g)
    c = Counter(v for row in g for v in row if v != 0)
    if not c:
        return None
    color, _ = c.most_common()[-1]
    return [[color] * w for _ in range(h)]


def t_fill_with_majority_nonzero(g: Grid) -> Grid | None:
    from collections import Counter
    h, w = grid_dims(g)
    c = Counter(v for row in g for v in row if v != 0)
    if not c:
        return None
    color, _ = c.most_common(1)[0]
    return [[color] * w for _ in range(h)]


def _row_uniformity_check(row: list[int]) -> bool:
    """All non-zero cells in row have same color (zeros allowed)."""
    nonzero = [c for c in row if c != 0]
    return len(set(nonzero)) <= 1 and len(nonzero) > 0


def t_per_row_uniform_to_color(g: Grid, uniform_color: int, nonuniform_color: int) -> Grid:
    """For each row: if all non-zero cells have the same color, output a row of
    `uniform_color`; else output a row of `nonuniform_color`."""
    h, w = grid_dims(g)
    out: Grid = []
    for r in range(h):
        nonzero = [c for c in g[r] if c != 0]
        all_same = len(set(nonzero)) <= 1 and len(nonzero) > 0
        v = uniform_color if all_same else nonuniform_color
        out.append([v] * w)
    return out


def t_per_col_uniform_to_color(g: Grid, uniform_color: int, nonuniform_color: int) -> Grid:
    h, w = grid_dims(g)
    cols_uniform = []
    for c in range(w):
        nonzero = [g[r][c] for r in range(h) if g[r][c] != 0]
        all_same = len(set(nonzero)) <= 1 and len(nonzero) > 0
        cols_uniform.append(all_same)
    return [[(uniform_color if cols_uniform[c] else nonuniform_color) for c in range(w)] for _ in range(h)]


def _constant_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    """If all training outputs are identical, output that constant grid."""
    progs: list[Program] = []
    if len(train_pairs) >= 2:
        first_out = train_pairs[0][1]
        if all(grid_equal(o, first_out) for _, o in train_pairs[1:]):
            const = [row[:] for row in first_out]
            progs.append(Program(
                f"constant_output_{len(const)}x{len(const[0]) if const else 0}",
                lambda g, c=const: [row[:] for row in c],
            ))
    return progs


def _property_to_output_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    """If outputs vary across training pairs but each output corresponds to a
    distinct value of some scalar property of the input (count of objects,
    count of a color, count of distinct colors), induce that property -> output
    mapping. Useful for 'output a fixed grid encoding a count' tasks."""
    progs: list[Program] = []
    # Skip if all outputs are the same (handled by _constant_programs)
    outputs = [tuple(tuple(r) for r in o) for _, o in train_pairs]
    if len(set(outputs)) <= 1:
        return progs

    def try_property(name: str, fn):
        nonlocal progs
        value_to_grid: dict = {}
        ok = True
        for inp, out in train_pairs:
            v = fn(inp)
            if v in value_to_grid and value_to_grid[v] != out:
                ok = False; break
            value_to_grid[v] = [row[:] for row in out]
        if ok and value_to_grid:
            vg = {k: [row[:] for row in v] for k, v in value_to_grid.items()}
            progs.append(Program(
                f"prop_{name}_to_grid",
                lambda g, vg=vg, fn=fn: t_property_to_constant_grid(g, fn, vg),
            ))

    try_property("object_count", _count_objects)
    try_property("distinct_colors", _count_distinct_colors)
    for col in range(0, 10):
        try_property(f"count_color_{col}", lambda g, c=col: _count_color(g, c))
    return progs


def _color_permutation_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    """Detect a color permutation (swap, cycle) that maps inputs to outputs."""
    progs: list[Program] = []
    if not all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        return progs
    # Try to fit a permutation by examining all (input_color, output_color) pairs
    pairs_set: set[tuple[int, int]] = set()
    for inp, out in train_pairs:
        h, w = grid_dims(inp)
        for r in range(h):
            for c in range(w):
                pairs_set.add((inp[r][c], out[r][c]))
    # Build a function: each input color maps to ONE output color
    forward: dict[int, int] = {}
    for a, b in pairs_set:
        if a in forward and forward[a] != b:
            return progs  # not a function
        forward[a] = b
    # Check if it's a permutation (each output mapped to by exactly one input)
    reverse: dict[int, int] = {}
    for a, b in forward.items():
        if b in reverse:
            return progs
        reverse[b] = a
    # Skip identity
    if all(k == v for k, v in forward.items()):
        return progs
    m = dict(forward)
    progs.append(Program(
        f"color_perm_{sorted(m.items())}",
        lambda g, m=m: [[m.get(c, c) for c in row] for row in g],
    ))
    return progs


def _input_color_replace_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    """Recolor by mapping: input-color C -> output-color C', uniformly across grid."""
    progs: list[Program] = []
    if not all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        return progs
    # Build per-color mapping by examining all train pairs
    mapping: dict[int, int] = {}
    consistent = True
    for inp, out in train_pairs:
        h, w = grid_dims(inp)
        for r in range(h):
            for c in range(w):
                a = inp[r][c]; b = out[r][c]
                if a in mapping and mapping[a] != b:
                    consistent = False; break
                mapping[a] = b
            if not consistent:
                break
        if not consistent:
            break
    if consistent and mapping and any(k != v for k, v in mapping.items()):
        m = dict(mapping)
        def apply_map(g, m=m):
            return [[m.get(c, c) for c in row] for row in g]
        progs.append(Program(
            f"recolor_map_{sorted(m.items())}",
            apply_map,
        ))
    return progs


def _color_specific_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    progs: list[Program] = []
    # crop to bbox of each color we've seen in training inputs (0..9)
    colors_seen: set[int] = set()
    for inp, _ in train_pairs:
        colors_seen.update(colors_in(inp))
    colors_seen.discard(0)
    for col in colors_seen:
        progs.append(Program(
            f"crop_to_color_{col}_bbox",
            lambda g, col=col: t_crop_to_color_bbox(g, col),
        ))
        progs.append(Program(
            f"extract_color_{col}_bbox_grid",
            lambda g, col=col: t_extract_object_bbox_grid(g, col),
        ))
    progs.append(Program("extract_largest_object_bbox_grid", t_extract_largest_object_bbox_grid))
    progs.append(Program("extract_non_majority_subgrid", t_extract_non_majority_subgrid))
    return progs


def _object_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    progs: list[Program] = []
    # Keep-largest / keep-smallest only make sense when output dimensions match input
    if all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        progs.append(Program("keep_largest_object_bycolor", lambda g: t_keep_largest_object(g, by_color=True)))
        progs.append(Program("keep_largest_object_any",     lambda g: t_keep_largest_object(g, by_color=False)))
        progs.append(Program("keep_smallest_object_bycolor", lambda g: t_keep_smallest_object(g, by_color=True)))
        progs.append(Program("keep_smallest_object_any",     lambda g: t_keep_smallest_object(g, by_color=False)))

    # Crop-to-largest-object: applicable when output is smaller and matches a bbox crop
    progs.append(Program("crop_to_largest_object_bycolor", lambda g: t_crop_to_largest_object(g, by_color=True)))
    progs.append(Program("crop_to_largest_object_any",     lambda g: t_crop_to_largest_object(g, by_color=False)))

    # Induced: size -> color mapping (recolor each object by its area)
    if all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        # Build mapping by examining train pairs: each object in input has a size
        # and (hopefully) a consistent recolor in output.
        size_to_color: dict[int, int] = {}
        consistent = True
        for inp, out in train_pairs:
            objs = find_objects(inp, by_color=True)
            for o in objs:
                sz = len(o["cells"])
                colors_in_out = {out[r][c] for r, c in o["cells"]}
                if len(colors_in_out) != 1:
                    consistent = False; break
                new_color = next(iter(colors_in_out))
                if sz in size_to_color and size_to_color[sz] != new_color:
                    consistent = False; break
                size_to_color[sz] = new_color
            if not consistent:
                break
        if consistent and size_to_color:
            m = dict(size_to_color)
            progs.append(Program(
                f"recolor_by_size_{sorted(m.items())}",
                lambda g, m=m: t_recolor_by_size(g, m),
            ))

    # Induced: object-count -> 1x1 color (e.g., "output a single cell whose color is # of objects+1")
    all_1x1_outputs = all(grid_dims(o) == (1, 1) for _, o in train_pairs)
    if all_1x1_outputs:
        count_to_color: dict[int, int] = {}
        consistent = True
        for inp, out in train_pairs:
            n = len(find_objects(inp, by_color=True))
            c = out[0][0]
            if n in count_to_color and count_to_color[n] != c:
                consistent = False; break
            count_to_color[n] = c
        if consistent and count_to_color:
            m = dict(count_to_color)
            progs.append(Program(
                f"count_objects_to_color_{sorted(m.items())}",
                lambda g, m=m: t_count_objects_to_color(g, m),
            ))

    # Translate-object-to-marker and stamp-template-at-markers: try all
    # (object-color, marker-color) pairs from training inputs.
    if all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        seen_colors: set[int] = set()
        for inp, _ in train_pairs:
            seen_colors.update(colors_in(inp))
        seen_colors.discard(0)
        for oc in seen_colors:
            for mc in seen_colors:
                if oc == mc:
                    continue
                progs.append(Program(
                    f"translate_obj_{oc}_to_marker_{mc}",
                    lambda g, oc=oc, mc=mc: t_translate_object_to_marker(g, oc, mc),
                ))
                progs.append(Program(
                    f"stamp_obj_{oc}_at_markers_{mc}",
                    lambda g, oc=oc, mc=mc: t_copy_object_to_each_marker(g, oc, mc),
                ))
                for d in ("up", "down", "left", "right"):
                    progs.append(Program(
                        f"gravity_{mc}_toward_{oc}_{d}",
                        lambda g, mc=mc, oc=oc, d=d: t_gravity_toward_color(g, mc, oc, d),
                    ))

    return progs


def _pattern_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    progs: list[Program] = []
    if all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        progs.append(Program("complete_tiled_pattern", t_complete_tiled_pattern))
    # extract_tile only valid when output dimensions can be a divisor of input
    progs.append(Program("extract_tile", t_extract_tile))
    return progs


def _drawing_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    """Programs that draw frames/outlines using a learned color."""
    progs: list[Program] = []
    if not all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        return progs
    # Find colors that appear in outputs but not inputs (candidate frame colors)
    new_colors: set[int] = set()
    for inp, out in train_pairs:
        new_colors.update(colors_in(out) - colors_in(inp))
    new_colors.discard(0)
    for col in new_colors:
        progs.append(Program(f"draw_bbox_frame_{col}", lambda g, col=col: t_draw_bbox_frame(g, col)))
        progs.append(Program(f"outline_objects_{col}", lambda g, col=col: t_outline_objects(g, col)))
        progs.append(Program(f"rectangulate_objects_{col}", lambda g, col=col: t_rectangulate_each_object(g, col)))
    progs.append(Program("fill_each_object_bbox", t_fill_each_object_bbox))
    progs.append(Program("keep_only_rectangular_objects", t_keep_only_rectangular_objects))
    return progs


def _diagonal_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    progs: list[Program] = []
    if all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        progs.append(Program("flip_diag_nw_se", t_flip_diag_nw_se))
        progs.append(Program("flip_diag_ne_sw", t_flip_diag_ne_sw))
    return progs


def _line_drawing_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    progs: list[Program] = []
    if not all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        return progs
    # Detect "trigger color" (appears in input but the line might be a new color in output)
    in_colors: set[int] = set()
    new_colors: set[int] = set()
    for inp, out in train_pairs:
        in_colors.update(colors_in(inp))
        new_colors.update(colors_in(out) - colors_in(inp))
    in_colors.discard(0)
    new_colors.discard(0)
    # Also try using the same color as trigger
    line_color_set = new_colors if new_colors else in_colors
    for trig in in_colors:
        for lc in line_color_set:
            if trig == lc and lc not in new_colors:
                continue
            progs.append(Program(
                f"draw_h_line_through_{trig}_color_{lc}",
                lambda g, trig=trig, lc=lc: t_draw_horizontal_line_through_each_color(g, trig, lc),
            ))
            progs.append(Program(
                f"draw_v_line_through_{trig}_color_{lc}",
                lambda g, trig=trig, lc=lc: t_draw_vertical_line_through_each_color(g, trig, lc),
            ))
            progs.append(Program(
                f"draw_cross_through_{trig}_color_{lc}",
                lambda g, trig=trig, lc=lc: t_draw_cross_through_each_color(g, trig, lc),
            ))
            progs.append(Program(
                f"connect_two_{trig}_color_{lc}",
                lambda g, trig=trig, lc=lc: t_connect_two_points_of_color(g, trig, lc),
            ))
    return progs


def _object_filter_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    progs: list[Program] = []
    if not all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        return progs
    # Keep-only-color-X: try each color that appears in any input
    seen: set[int] = set()
    for inp, _ in train_pairs:
        seen.update(colors_in(inp))
    seen.discard(0)
    for col in seen:
        progs.append(Program(f"keep_only_color_{col}", lambda g, col=col: t_keep_only_object_of_color(g, col)))
    progs.append(Program("keep_only_isolated_cells", t_keep_only_isolated_cells))
    progs.append(Program("remove_isolated_cells", t_remove_isolated_cells))
    progs.append(Program("keep_non_rectangular_objects", t_keep_non_rectangular_objects))
    # Top-3 rank picks
    for rank in range(3):
        progs.append(Program(f"keep_object_at_rank_{rank}_size", lambda g, r=rank: t_keep_object_at_rank(g, r, by="size")))
        progs.append(Program(f"keep_object_at_rank_{rank}_area", lambda g, r=rank: t_keep_object_at_rank(g, r, by="area")))
    return progs


def _per_object_transform_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    progs: list[Program] = []
    if not all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        return progs
    for name, fn in [
        ("per_obj_flip_h", t_flip_h),
        ("per_obj_flip_v", t_flip_v),
        ("per_obj_rotate90", t_rotate90),
        ("per_obj_rotate180", t_rotate180),
        ("per_obj_rotate270", t_rotate270),
        ("per_obj_transpose", t_transpose),
    ]:
        progs.append(Program(name, lambda g, fn=fn: t_per_object_transform(g, fn)))
    return progs


def _recolor_size_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    progs: list[Program] = []
    if not all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        return progs
    # Find colors that appear in outputs but not inputs (candidate small/large colors)
    seen: set[int] = set()
    for inp, out in train_pairs:
        seen.update(colors_in(out) - colors_in(inp))
    seen.discard(0)
    for sc in seen:
        for lc in seen:
            if sc == lc:
                continue
            progs.append(Program(
                f"recolor_objs_by_size_small_{sc}_large_{lc}",
                lambda g, sc=sc, lc=lc: t_recolor_objects_by_size(g, sc, lc),
            ))
    return progs


def _subgrid_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    """Programs that decompose input into subgrids (divider-line detected) and
    return one of them."""
    progs: list[Program] = [
        Program("extract_unique_subgrid", t_extract_unique_subgrid),
        Program("extract_majority_subgrid", t_extract_majority_subgrid),
    ]
    # Try each color as a "target" for the subgrid-with-max-color-X selector
    colors_seen: set[int] = set()
    for inp, _ in train_pairs:
        colors_seen.update(colors_in(inp))
    colors_seen.discard(0)
    for col in colors_seen:
        progs.append(Program(
            f"extract_subgrid_with_max_{col}",
            lambda g, col=col: t_extract_subgrid_with_max_color_count(g, col),
        ))
    return progs


def _detected_symmetry_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    if not all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        return []
    return [Program("complete_detected_symmetry", t_complete_detected_symmetry)]


def t_flip_diag_nw_se(g: Grid) -> Grid:
    """Flip across NW-SE diagonal (transpose)."""
    return t_transpose(g)


def t_flip_diag_ne_sw(g: Grid) -> Grid:
    """Flip across NE-SW diagonal (anti-transpose)."""
    h, w = grid_dims(g)
    return [[g[h - 1 - c][w - 1 - r] for c in range(w)] for r in range(h)]


def t_draw_horizontal_line_through_each_color(g: Grid, color: int, line_color: int) -> Grid:
    """For each row that contains `color`, fill the entire row with `line_color`
    (preserving the original color cells)."""
    h, w = grid_dims(g)
    out = grid_copy(g)
    for r in range(h):
        if any(c == color for c in g[r]):
            for c in range(w):
                if out[r][c] == 0:
                    out[r][c] = line_color
    return out


def t_draw_vertical_line_through_each_color(g: Grid, color: int, line_color: int) -> Grid:
    h, w = grid_dims(g)
    out = grid_copy(g)
    for c in range(w):
        if any(g[r][c] == color for r in range(h)):
            for r in range(h):
                if out[r][c] == 0:
                    out[r][c] = line_color
    return out


def t_draw_cross_through_each_color(g: Grid, color: int, line_color: int) -> Grid:
    """Draw both row and column lines through every cell of `color`."""
    out = t_draw_horizontal_line_through_each_color(g, color, line_color)
    return t_draw_vertical_line_through_each_color(out, color, line_color)


def t_connect_two_points_of_color(g: Grid, color: int, line_color: int) -> Grid | None:
    """If exactly two cells of `color` exist, draw a straight line (horizontal,
    vertical, or 45-degree diagonal) connecting them."""
    h, w = grid_dims(g)
    pts = [(r, c) for r in range(h) for c in range(w) if g[r][c] == color]
    if len(pts) != 2:
        return None
    (r1, c1), (r2, c2) = pts
    dr = r2 - r1; dc = c2 - c1
    if dr == 0 or dc == 0 or abs(dr) == abs(dc):
        # straight line
        out = grid_copy(g)
        steps = max(abs(dr), abs(dc))
        sr = (1 if dr > 0 else -1 if dr < 0 else 0)
        sc = (1 if dc > 0 else -1 if dc < 0 else 0)
        for i in range(1, steps):
            r = r1 + i * sr; c = c1 + i * sc
            if 0 <= r < h and 0 <= c < w and out[r][c] == 0:
                out[r][c] = line_color
        return out
    return None


def t_keep_only_object_of_color(g: Grid, color: int) -> Grid:
    """Keep only the cells of `color`; clear everything else."""
    return [[c if c == color else 0 for c in row] for row in g]


def t_keep_only_isolated_cells(g: Grid) -> Grid | None:
    """Keep cells whose 4-neighbors are all 0 (singletons by 4-connectivity)."""
    h, w = grid_dims(g)
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if g[r][c] == 0:
                continue
            isolated = True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and g[nr][nc] != 0:
                    isolated = False; break
            if isolated:
                out[r][c] = g[r][c]
    return out


def t_remove_isolated_cells(g: Grid) -> Grid | None:
    """Inverse: keep cells that have at least one non-zero 4-neighbor."""
    h, w = grid_dims(g)
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if g[r][c] == 0:
                continue
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and g[nr][nc] != 0:
                    out[r][c] = g[r][c]; break
    return out


def t_keep_object_at_rank(g: Grid, rank: int, by: str = "size") -> Grid | None:
    """Keep only the Nth-ranked object. rank=0 = largest, rank=1 = 2nd largest, etc.
    by='size': by cell count. by='area': by bbox area."""
    objs = find_objects(g, by_color=True)
    if rank >= len(objs):
        return None
    if by == "size":
        objs_sorted = sorted(objs, key=lambda o: -len(o["cells"]))
    else:
        objs_sorted = sorted(objs, key=lambda o: -((o["bbox"][2]-o["bbox"][0]+1)*(o["bbox"][3]-o["bbox"][1]+1)))
    target = objs_sorted[rank]
    h, w = grid_dims(g)
    out = [[0] * w for _ in range(h)]
    for r, c in target["cells"]:
        out[r][c] = target["color"]
    return out


def t_keep_non_rectangular_objects(g: Grid) -> Grid | None:
    """Keep only objects whose cells DON'T exactly fill their bbox (non-rectangles)."""
    h, w = grid_dims(g)
    objs = find_objects(g, by_color=True)
    out = [[0] * w for _ in range(h)]
    for o in objs:
        r0, c0, r1, c1 = o["bbox"]
        bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if len(o["cells"]) < bbox_area:
            for r, c in o["cells"]:
                out[r][c] = o["color"]
    return out


def _object_to_grid(o: dict) -> Grid:
    """Extract just the object's bbox as a small grid (color cell vs 0)."""
    r0, c0, r1, c1 = o["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    out: Grid = [[0] * w for _ in range(h)]
    for r, c in o["cells"]:
        out[r - r0][c - c0] = o["color"]
    return out


def _place_grid_at(canvas: Grid, sub: Grid, r0: int, c0: int) -> None:
    """In-place: place `sub` onto `canvas` starting at (r0, c0), overwriting only non-zero."""
    H, W = grid_dims(canvas)
    h, w = grid_dims(sub)
    for r in range(h):
        for c in range(w):
            if sub[r][c] != 0 and 0 <= r0 + r < H and 0 <= c0 + c < W:
                canvas[r0 + r][c0 + c] = sub[r][c]


def t_per_object_transform(g: Grid, fn) -> Grid | None:
    """Apply a small-grid transform to each object's bbox-extract independently,
    then place it back at the same top-left position."""
    h, w = grid_dims(g)
    objs = find_objects(g, by_color=True)
    if not objs:
        return None
    # Clear cells that were part of objects
    out: Grid = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            out[r][c] = g[r][c]
    for o in objs:
        sub = _object_to_grid(o)
        try:
            transformed = fn(sub)
        except Exception:
            return None
        if transformed is None:
            return None
        # clear original cells
        for r, c in o["cells"]:
            out[r][c] = 0
        r0, c0, _, _ = o["bbox"]
        _place_grid_at(out, transformed, r0, c0)
    return out


def t_fill_each_object_bbox(g: Grid) -> Grid | None:
    """For each connected object, fill its bbox with the object's color."""
    h, w = grid_dims(g)
    objs = find_objects(g, by_color=True)
    if not objs:
        return None
    out = grid_copy(g)
    for o in objs:
        r0, c0, r1, c1 = o["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                out[r][c] = o["color"]
    return out


def t_rectangulate_each_object(g: Grid, fill_color: int) -> Grid | None:
    """Replace each object with its bbox filled with fill_color."""
    h, w = grid_dims(g)
    objs = find_objects(g, by_color=True)
    if not objs:
        return None
    out = [[0] * w for _ in range(h)]
    for o in objs:
        r0, c0, r1, c1 = o["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                out[r][c] = fill_color
    return out


def t_keep_only_rectangular_objects(g: Grid) -> Grid | None:
    """Keep only objects whose cells exactly fill their bbox (perfect rectangles)."""
    h, w = grid_dims(g)
    objs = find_objects(g, by_color=True)
    if not objs:
        return None
    out = [[0] * w for _ in range(h)]
    for o in objs:
        r0, c0, r1, c1 = o["bbox"]
        bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if len(o["cells"]) == bbox_area:
            for r, c in o["cells"]:
                out[r][c] = o["color"]
    return out


def t_draw_bbox_frame(g: Grid, frame_color: int) -> Grid:
    """For the bbox of all non-zero cells, draw a frame (border only) in frame_color."""
    h, w = grid_dims(g)
    out = grid_copy(g)
    min_r, max_r = h, -1; min_c, max_c = w, -1
    for r in range(h):
        for c in range(w):
            if g[r][c] != 0:
                if r < min_r: min_r = r
                if r > max_r: max_r = r
                if c < min_c: min_c = c
                if c > max_c: max_c = c
    if max_r < 0:
        return out
    for c in range(min_c, max_c + 1):
        if out[min_r][c] == 0: out[min_r][c] = frame_color
        if out[max_r][c] == 0: out[max_r][c] = frame_color
    for r in range(min_r, max_r + 1):
        if out[r][min_c] == 0: out[r][min_c] = frame_color
        if out[r][max_c] == 0: out[r][max_c] = frame_color
    return out


def t_outline_objects(g: Grid, outline_color: int) -> Grid:
    """For each connected object, draw an outline of cells just outside it."""
    h, w = grid_dims(g)
    out = grid_copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c] != 0:
                continue
            # If any 4-neighbor is non-zero, this cell is an outline cell
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and g[nr][nc] != 0:
                    out[r][c] = outline_color
                    break
    return out


def t_recolor_objects_by_size(g: Grid, small_color: int, large_color: int) -> Grid | None:
    """Recolor the smallest object to small_color and the largest to large_color."""
    objs = find_objects(g, by_color=True)
    if len(objs) < 2:
        return None
    sizes = sorted(set(len(o["cells"]) for o in objs))
    if len(sizes) < 2:
        return None
    smallest_sz, largest_sz = sizes[0], sizes[-1]
    h, w = grid_dims(g)
    out = grid_copy(g)
    for o in objs:
        sz = len(o["cells"])
        if sz == smallest_sz:
            for r, c in o["cells"]:
                out[r][c] = small_color
        elif sz == largest_sz:
            for r, c in o["cells"]:
                out[r][c] = large_color
    return out


def _detect_tile_period(g: Grid) -> tuple[int, int] | None:
    """Detect the period (ph, pw) of a tiled pattern. Returns smallest (ph, pw)
    such that g[r][c] == g[(r%ph)][(c%pw)] for all cells (allowing zeros as 'unknown')."""
    h, w = grid_dims(g)
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            if h % ph != 0 or w % pw != 0:
                continue
            if ph == h and pw == w:
                continue
            # Build the tile from non-zero cells across all instances
            tile: dict[tuple[int, int], int] = {}
            consistent = True
            for r in range(h):
                for c in range(w):
                    v = g[r][c]
                    if v == 0:
                        continue
                    key = (r % ph, c % pw)
                    if key in tile and tile[key] != v:
                        consistent = False; break
                    tile[key] = v
                if not consistent:
                    break
            if consistent and tile:
                return (ph, pw)
    return None


def t_complete_tiled_pattern(g: Grid) -> Grid | None:
    """Detect the period, build the tile from all instances, fill in zeros
    everywhere that should be filled per the period."""
    period = _detect_tile_period(g)
    if period is None:
        return None
    ph, pw = period
    h, w = grid_dims(g)
    tile: dict[tuple[int, int], int] = {}
    for r in range(h):
        for c in range(w):
            if g[r][c] != 0:
                tile[(r % ph, c % pw)] = g[r][c]
    out: Grid = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            key = (r % ph, c % pw)
            if key in tile:
                out[r][c] = tile[key]
    return out


def t_extract_tile(g: Grid) -> Grid | None:
    """If g is tiled, return one tile."""
    period = _detect_tile_period(g)
    if period is None:
        return None
    ph, pw = period
    h, w = grid_dims(g)
    tile: dict[tuple[int, int], int] = {}
    for r in range(h):
        for c in range(w):
            if g[r][c] != 0:
                tile[(r % ph, c % pw)] = g[r][c]
    return [[tile.get((r, c), 0) for c in range(pw)] for r in range(ph)]


def _learn_ca_rule_k(train_pairs: list[tuple[Grid, Grid]], k: int):
    """CA rule with neighborhood radius k."""
    rule: dict[tuple, int] = {}
    for inp, out in train_pairs:
        if grid_dims(inp) != grid_dims(out):
            return None
        h, w = grid_dims(inp)
        for r in range(h):
            for c in range(w):
                key = (inp[r][c], _neighbor_signature(inp, r, c, k=k))
                val = out[r][c]
                if key in rule and rule[key] != val:
                    return None
                rule[key] = val
    return rule


def t_apply_ca_rule_k(g: Grid, rule: dict[tuple, int], k: int) -> Grid | None:
    h, w = grid_dims(g)
    out: Grid = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            key = (g[r][c], _neighbor_signature(g, r, c, k=k))
            if key not in rule:
                return None
            out[r][c] = rule[key]
    return out


def _learn_ca_rule_iterated(train_pairs: list[tuple[Grid, Grid]], n_iter: int):
    """Apply the CA rule n_iter times. Useful when the rule needs propagation."""
    rule = _learn_ca_rule(train_pairs)
    if rule is None:
        return None
    return rule, n_iter


def _ca_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    progs: list[Program] = []
    rule = _learn_ca_rule(train_pairs)
    if rule is not None:
        progs.append(Program("ca_rule_neighbor_sig", lambda g, r=rule: t_apply_ca_rule(g, r)))
    rule_c = _learn_ca_rule_by_color_count(train_pairs)
    if rule_c is not None:
        progs.append(Program("ca_rule_neighbor_count", lambda g, r=rule_c: t_apply_ca_rule_count(g, r)))
    # k=2 (5x5 window) — much more specific, less likely to overfit on training
    # but catches finer-grained local patterns.
    rule_k2 = _learn_ca_rule_k(train_pairs, k=2)
    if rule_k2 is not None:
        progs.append(Program("ca_rule_k2", lambda g, r=rule_k2: t_apply_ca_rule_k(g, r, k=2)))

    # Iterated CA: apply the same simple rule N times. Useful for propagation
    # patterns where one application moves a "boundary" by one cell.
    rule_iter = _learn_ca_rule_by_color_count(train_pairs)
    if rule_iter is not None:
        for n in (2, 3, 5):
            def apply_n(g, r=rule_iter, n=n):
                cur = g
                for _ in range(n):
                    nxt = t_apply_ca_rule_count(cur, r)
                    if nxt is None:
                        return None
                    cur = nxt
                return cur
            progs.append(Program(f"ca_rule_count_iter_{n}", apply_n))
    # Also learn a rule keyed only on the input color (no neighbors) — degenerate
    # to a pure color remap, but useful when the CA rules above fail because of
    # neighbor noise.
    if all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        simple: dict[int, int] = {}
        consistent = True
        for inp, out in train_pairs:
            h, w = grid_dims(inp)
            for r in range(h):
                for c in range(w):
                    a, b = inp[r][c], out[r][c]
                    if a in simple and simple[a] != b:
                        consistent = False; break
                    simple[a] = b
                if not consistent: break
            if not consistent: break
        if consistent and any(k != v for k, v in simple.items()):
            m = dict(simple)
            progs.append(Program(
                f"ca_color_only_{sorted(m.items())}",
                lambda g, m=m: [[m.get(c, c) for c in row] for row in g],
            ))
    return progs


def _property_to_output_programs_constrained(train_pairs):
    """Same as property-to-output but only fires when outputs are all small AND
    the property values are all distinct (so the mapping is non-degenerate)."""
    # Require all outputs to be at most 3x3
    if not all(grid_dims(o)[0] <= 3 and grid_dims(o)[1] <= 3 for _, o in train_pairs):
        return []
    progs = _property_to_output_programs(train_pairs)
    # Wrap each program: if test input's property value is NOT in the learned
    # lookup, return None (so solver continues). This is the default behavior
    # of t_property_to_constant_grid, so we're fine — but only fire on small
    # outputs to limit false-positive risk.
    return progs


def candidate_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    return (
        _constant_programs(train_pairs)
        + _color_permutation_programs(train_pairs)
        + _input_color_replace_programs(train_pairs)
        + _ca_programs(train_pairs)
        + _shape_preserving_programs(train_pairs)
        + _scale_programs(train_pairs)
        + _selection_programs(train_pairs)
        + _fill_programs(train_pairs)
        + _gravity_programs(train_pairs)
        + _symmetry_programs(train_pairs)
        + _detected_symmetry_programs(train_pairs)
        + _color_specific_programs(train_pairs)
        + _object_programs(train_pairs)
        + _subgrid_programs(train_pairs)
        + _pattern_programs(train_pairs)
        + _drawing_programs(train_pairs)
        + _recolor_size_programs(train_pairs)
        + _per_object_transform_programs(train_pairs)
        + _object_filter_programs(train_pairs)
        + _diagonal_programs(train_pairs)
        + _line_drawing_programs(train_pairs)
        + _property_to_output_programs_constrained(train_pairs)  # LAST: high false-positive risk
    )


def _try_program(prog: Program, train_pairs, test_inp):
    """Return (matches_all_training, test_output) — both None on exception."""
    try:
        outputs = []
        for inp, out in train_pairs:
            r = prog.apply(inp)
            if r is None or not grid_equal(r, out):
                return False, None
            outputs.append(r)
        test_r = prog.apply(test_inp)
        if test_r is None:
            return True, None
        return True, test_r
    except Exception:
        return False, None


def solve_task(task_data: dict, allow_compose: bool = True) -> tuple[str, Grid] | None:
    train = task_data.get("train", [])
    test = task_data.get("test", [])
    if not train or not test:
        return None
    train_pairs = [(t["input"], t["output"]) for t in train]
    test_inp = test[0]["input"]

    progs = candidate_programs(train_pairs)
    # First pass: single-primitive
    for prog in progs:
        ok, result = _try_program(prog, train_pairs, test_inp)
        if ok and result is not None:
            return prog.name, result

    # Second pass: composition of two primitives prog_b(prog_a(g))
    if allow_compose:
        # Only compose shape-preserving programs to keep the search tractable
        # and to ensure intermediate grids stay valid.
        composable = [p for p in progs if p.name in {
            "flip_h", "flip_v", "rotate90", "rotate180", "rotate270", "transpose",
            "keep_largest_object_bycolor", "keep_largest_object_any",
            "keep_smallest_object_bycolor", "keep_smallest_object_any",
            "keep_only_majority_color", "keep_only_minority_color",
            "complete_symmetry_h", "complete_symmetry_v", "complete_symmetry_both",
            "complete_detected_symmetry",
            "gravity_up", "gravity_down", "gravity_left", "gravity_right",
        } or p.name.startswith("recolor_") or p.name.startswith("shift_")
              or p.name.startswith("flood_fill_enclosed_")]
        for a in composable:
            for b in composable:
                if a is b:
                    continue
                name = f"compose:{a.name}>>{b.name}"
                def composed(g, a=a, b=b):
                    r = a.apply(g)
                    if r is None:
                        return None
                    return b.apply(r)
                p = Program(name, composed)
                ok, result = _try_program(p, train_pairs, test_inp)
                if ok and result is not None:
                    return name, result

        # Third pass: 3-step composition from a tight CORE library only.
        # Pattern: geometric -> object-select -> recolor (or permutations).
        core_geometric = [p for p in progs if p.name in {
            "flip_h", "flip_v", "rotate90", "rotate180", "rotate270", "transpose",
        }]
        core_select = [p for p in progs if p.name in {
            "keep_largest_object_bycolor", "keep_smallest_object_bycolor",
            "keep_only_majority_color", "keep_only_minority_color",
        }]
        core_recolor = [p for p in progs if p.name.startswith("recolor_")
                        or p.name.startswith("flood_fill_enclosed_")]
        # Try geometric -> select -> recolor and select -> geometric -> recolor
        for a in core_geometric + core_select:
            for b in core_select + core_geometric:
                if a is b:
                    continue
                for c in core_recolor:
                    name = f"compose3:{a.name}>>{b.name}>>{c.name}"
                    def composed3(g, a=a, b=b, c=c):
                        r = a.apply(g)
                        if r is None: return None
                        r = b.apply(r)
                        if r is None: return None
                        return c.apply(r)
                    p = Program(name, composed3)
                    ok, result = _try_program(p, train_pairs, test_inp)
                    if ok and result is not None:
                        return name, result

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
