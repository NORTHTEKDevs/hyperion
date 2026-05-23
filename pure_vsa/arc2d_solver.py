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


def candidate_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    return (
        _constant_programs(train_pairs)
        + _input_color_replace_programs(train_pairs)
        + _shape_preserving_programs(train_pairs)
        + _scale_programs(train_pairs)
        + _selection_programs(train_pairs)
        + _fill_programs(train_pairs)
        + _gravity_programs(train_pairs)
        + _symmetry_programs(train_pairs)
        + _color_specific_programs(train_pairs)
        + _object_programs(train_pairs)
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
