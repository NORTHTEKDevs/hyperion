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


def _per_cell_substitute_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    info = _learn_per_cell_substitute(train_pairs)
    if info is None:
        return []
    kr, kc, table = info
    return [Program(
        f"per_cell_substitute_{kr}x{kc}",
        lambda g, kr=kr, kc=kc, t=table: t_per_cell_substitute(g, kr, kc, t),
    )]


def _kaleidoscope_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    """Kaleidoscope: detect when output is 2x in each dim and try the mirrored tile."""
    progs: list[Program] = []
    for inp, out in train_pairs:
        hi, wi = grid_dims(inp); ho, wo = grid_dims(out)
        if ho == 2 * hi and wo == 2 * wi:
            progs.append(Program("kaleidoscope_2x2", t_kaleidoscope_2x2))
            progs.append(Program("rotational_kaleidoscope_2x2", t_rotational_kaleidoscope_2x2))
            break
    # Self-similar tiling: output is H*H by W*W
    for inp, out in train_pairs:
        hi, wi = grid_dims(inp); ho, wo = grid_dims(out)
        if hi > 0 and wi > 0 and ho == hi * hi and wo == wi * wi:
            # Try each input color as mask
            in_colors: set[int] = set()
            for inp2, _ in train_pairs:
                in_colors.update(colors_in(inp2))
            in_colors.discard(0)
            for col in in_colors:
                progs.append(Program(
                    f"self_similar_tile_by_{col}",
                    lambda g, col=col: t_self_similar_tile_by_mask(g, col),
                ))
            progs.append(Program("self_similar_tile_by_nonzero", t_self_similar_tile_by_nonzero))
            break
    return progs


def _corner_subgrid_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    """Extract a fixed-size subgrid from a corner or center of the input."""
    progs: list[Program] = []
    # Detect output dimensions across training pairs
    out_dims: set[tuple[int, int]] = set()
    for _, out in train_pairs:
        out_dims.add(grid_dims(out))
    if len(out_dims) != 1:
        return progs
    h, w = next(iter(out_dims))
    for name, fn in [
        ("top_left", t_extract_top_left),
        ("top_right", t_extract_top_right),
        ("bottom_left", t_extract_bottom_left),
        ("bottom_right", t_extract_bottom_right),
        ("center", t_extract_center),
    ]:
        progs.append(Program(
            f"extract_{name}_{h}x{w}",
            lambda g, h=h, w=w, fn=fn: fn(g, h, w),
        ))
    return progs


def _keep_only_row_col_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    progs: list[Program] = []
    if not all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        return progs
    progs.append(Program("keep_only_middle_column", t_keep_only_middle_column))
    progs.append(Program("keep_only_middle_row", t_keep_only_middle_row))
    return progs


def _extend_cell_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    progs: list[Program] = []
    if not all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        return progs
    for d, fn in [("down", t_extend_each_cell_down),
                   ("up", t_extend_each_cell_up),
                   ("right", t_extend_each_cell_right),
                   ("left", t_extend_each_cell_left)]:
        progs.append(Program(f"extend_each_cell_{d}", fn))
    return progs


def _pair_rectangle_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    progs: list[Program] = []
    if not all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        return progs
    new_colors: set[int] = set()
    for inp, out in train_pairs:
        new_colors.update(colors_in(out) - colors_in(inp))
    for nc in new_colors:
        progs.append(Program(
            f"fill_pair_bbox_rect_{nc}",
            lambda g, nc=nc: t_fill_pair_bbox_rectangles(g, nc),
        ))
        progs.append(Program(
            f"fill_between_any_color_with_{nc}",
            lambda g, nc=nc: t_fill_between_any_color_markers(g, nc),
        ))
    return progs


def _draw_x_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    progs: list[Program] = []
    if not all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        return progs
    in_colors: set[int] = set()
    new_colors: set[int] = set()
    for inp, out in train_pairs:
        in_colors.update(colors_in(inp))
        new_colors.update(colors_in(out) - colors_in(inp))
    in_colors.discard(0)
    new_colors.discard(0)
    # Try each input color as marker, each color as line
    for mc in in_colors:
        for lc in in_colors | new_colors:
            if lc == 0:
                continue
            progs.append(Program(
                f"draw_x_from_{mc}_color_{lc}",
                lambda g, mc=mc, lc=lc: t_draw_x_from_marker(g, mc, lc),
            ))
    return progs


def _progressive_shift_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    """When output has more rows than input, possibly progressive shift."""
    progs: list[Program] = []
    for inp, out in train_pairs:
        hi, wi = grid_dims(inp); ho, wo = grid_dims(out)
        if hi == 1 and ho > 1 and wo == wi:
            for d in ("right", "left"):
                n = ho
                progs.append(Program(
                    f"progressive_shift_{n}_{d}",
                    lambda g, n=n, d=d: t_progressive_shift(g, n, d),
                ))
            break
    return progs


def _row_col_extract_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    """Extract a specific row or column when output is 1xN or Nx1."""
    progs: list[Program] = []
    for inp, out in train_pairs:
        hi, wi = grid_dims(inp); ho, wo = grid_dims(out)
        if wo == 1 and ho == hi:
            progs.append(Program("extract_first_column", t_extract_first_column))
            progs.append(Program("extract_last_column", t_extract_last_column))
            progs.append(Program("extract_middle_column", t_extract_middle_column))
            break
        if ho == 1 and wo == wi:
            progs.append(Program("extract_first_row", t_extract_first_row))
            progs.append(Program("extract_last_row", t_extract_last_row))
            progs.append(Program("extract_middle_row", t_extract_middle_row))
            break
    return progs


def _split_both_zero_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    """Output is half of input (axis-split), with fill where BOTH halves were 0."""
    progs: list[Program] = []
    halving_axes: set[str] = set()
    for inp, out in train_pairs:
        hi, wi = grid_dims(inp); ho, wo = grid_dims(out)
        if wo * 2 == wi and ho == hi:
            halving_axes.add("h")
        elif ho * 2 == hi and wo == wi:
            halving_axes.add("v")
    new_colors: set[int] = set()
    for _, out in train_pairs:
        new_colors.update(colors_in(out))
    new_colors.discard(0)
    for axis in halving_axes:
        for nc in new_colors:
            progs.append(Program(
                f"split_both_zero_{axis}_fill_{nc}",
                lambda g, axis=axis, nc=nc: t_split_both_zero(g, axis, nc),
            ))
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


def t_extract_unique_color_object_bbox(g: Grid) -> Grid | None:
    """Find the object whose color is unique (appears only once across objects),
    crop to its bbox content."""
    objs = find_objects(g, by_color=True)
    if not objs:
        return None
    from collections import Counter
    color_counts = Counter(o["color"] for o in objs)
    uniques = [o for o in objs if color_counts[o["color"]] == 1]
    if len(uniques) != 1:
        return None
    r0, c0, r1, c1 = uniques[0]["bbox"]
    return [row[c0:c1 + 1] for row in g[r0:r1 + 1]]


def t_extract_smallest_object_bbox_grid(g: Grid) -> Grid | None:
    objs = find_objects(g, by_color=True)
    if not objs:
        return None
    smallest = min(objs, key=lambda o: len(o["cells"]))
    r0, c0, r1, c1 = smallest["bbox"]
    return [row[c0:c1 + 1] for row in g[r0:r1 + 1]]


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


def t_per_cell_substitute(g: Grid, kr: int, kc: int, table: dict[int, Grid]) -> Grid | None:
    """Replace each input cell with the corresponding KxK block from the lookup table."""
    h, w = grid_dims(g)
    out: Grid = [[0] * (w * kc) for _ in range(h * kr)]
    for r in range(h):
        for c in range(w):
            v = g[r][c]
            if v not in table:
                return None
            block = table[v]
            for br in range(kr):
                for bc in range(kc):
                    out[r * kr + br][c * kc + bc] = block[br][bc]
    return out


def _learn_per_cell_substitute(train_pairs: list[tuple[Grid, Grid]]):
    """Detect if output is k× input AND each input value corresponds to a fixed
    output block. Returns (kr, kc, table) or None."""
    # Determine k from the first pair
    if not train_pairs:
        return None
    inp0, out0 = train_pairs[0]
    h0, w0 = grid_dims(inp0)
    H0, W0 = grid_dims(out0)
    if h0 == 0 or w0 == 0 or H0 % h0 != 0 or W0 % w0 != 0:
        return None
    kr = H0 // h0; kc = W0 // w0
    if kr <= 1 and kc <= 1:
        return None
    # Verify all pairs have the same kr, kc
    for inp, out in train_pairs:
        h, w = grid_dims(inp); H, W = grid_dims(out)
        if h * kr != H or w * kc != W:
            return None
    # Build the lookup
    table: dict[int, Grid] = {}
    for inp, out in train_pairs:
        h, w = grid_dims(inp)
        for r in range(h):
            for c in range(w):
                v = inp[r][c]
                block = [out[r * kr + br][c * kc: c * kc + kc] for br in range(kr)]
                if v in table:
                    # All blocks for the same v must be identical
                    if table[v] != block:
                        return None
                else:
                    table[v] = [row[:] for row in block]
    return (kr, kc, table)


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
    progs.append(Program("extract_smallest_object_bbox_grid", t_extract_smallest_object_bbox_grid))
    progs.append(Program("extract_unique_color_object_bbox", t_extract_unique_color_object_bbox))
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


def _fill_between_markers_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    progs: list[Program] = []
    if not all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        return progs
    seen: set[int] = set()
    new_colors: set[int] = set()
    for inp, out in train_pairs:
        seen.update(colors_in(inp))
        new_colors.update(colors_in(out) - colors_in(inp))
    seen.discard(0)
    new_colors.discard(0)
    fill_set = new_colors if new_colors else seen
    for marker in seen:
        for fc in fill_set:
            if marker == fc and fc not in new_colors:
                continue
            progs.append(Program(
                f"fill_between_{marker}_with_{fc}",
                lambda g, m=marker, f=fc: t_fill_between_same_color_markers(g, m, f),
            ))
    for nc in new_colors:
        progs.append(Program(
            f"recolor_non_majority_nonzero_to_{nc}",
            lambda g, nc=nc: t_recolor_non_majority_nonzero(g, nc),
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


def _object_overlay_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    """Overlay all same-shape objects in input into a single output."""
    progs: list[Program] = []
    for mode in ("or", "and", "xor"):
        progs.append(Program(
            f"overlay_objects_{mode}",
            lambda g, mode=mode: t_overlay_objects(g, mode),
        ))
    return progs


def _object_arrange_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    """Programs that rearrange objects within the same-shape canvas."""
    progs: list[Program] = []
    if not all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        return progs
    for axis in ("h", "v"):
        for asc in (True, False):
            progs.append(Program(
                f"sort_objects_by_size_{axis}_{'asc' if asc else 'desc'}",
                lambda g, axis=axis, asc=asc: t_sort_objects_by_size_along_axis(g, axis, asc),
            ))
    for corner in ("tl", "tr", "bl", "br"):
        progs.append(Program(
            f"compact_objects_to_{corner}",
            lambda g, corner=corner: t_compact_objects_to_corner(g, corner),
        ))
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


def _rank_recolor_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    progs: list[Program] = []
    if not all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        return progs
    mapping = _learn_rank_to_color(train_pairs)
    if mapping is not None:
        m = dict(mapping)
        progs.append(Program(
            f"recolor_each_object_by_rank_{sorted(m.items())}",
            lambda g, m=m: t_recolor_each_object_by_rank(g, m),
        ))
    return progs


def _marker_pattern_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    progs: list[Program] = []
    info = _learn_marker_pattern(train_pairs)
    if info is not None:
        mc, pattern = info
        progs.append(Program(
            f"stamp_pattern_at_marker_{mc}",
            lambda g, mc=mc, p=pattern: t_stamp_pattern_at_marker(g, mc, p),
        ))
    return progs


def _noise_removal_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    progs: list[Program] = []
    if not all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        return progs
    # For each color, try removing it (some tasks have a "remove this color" rule)
    seen: set[int] = set()
    for inp, _ in train_pairs:
        seen.update(colors_in(inp))
    seen.discard(0)
    for col in seen:
        progs.append(Program(f"remove_color_{col}", lambda g, col=col: t_remove_objects_by_color(g, col)))
    progs.append(Program("keep_largest_of_each_color", t_keep_only_largest_object_of_each_color))
    progs.append(Program("remove_noise_singletons", t_remove_noise_singletons))

    # Recolor isolated cells / grouped cells to learned color
    new_colors: set[int] = set()
    for inp, out in train_pairs:
        new_colors.update(colors_in(out) - colors_in(inp))
    for nc in new_colors:
        progs.append(Program(f"recolor_isolated_to_{nc}", lambda g, nc=nc: t_recolor_isolated_cells(g, nc)))
        progs.append(Program(f"recolor_grouped_to_{nc}", lambda g, nc=nc: t_recolor_groups_keep_isolated(g, nc)))
        progs.append(Program(f"recolor_smallest_to_{nc}", lambda g, nc=nc: t_recolor_smallest_object(g, nc)))
        progs.append(Program(f"recolor_largest_to_{nc}", lambda g, nc=nc: t_recolor_largest_object(g, nc)))
        progs.append(Program(f"recolor_unique_size_to_{nc}", lambda g, nc=nc: t_recolor_unique_size_object(g, nc)))
    return progs


def _alignment_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    progs: list[Program] = []
    if not all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        return progs
    for edge in ("top", "bottom", "left", "right"):
        progs.append(Program(
            f"align_objects_to_{edge}",
            lambda g, edge=edge: t_align_objects_to_edge(g, edge),
        ))
    return progs


def _mask_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    progs: list[Program] = []
    # Determine the halving axis from training pairs
    halving_axes: set[str] = set()
    for inp, out in train_pairs:
        hi, wi = grid_dims(inp); ho, wo = grid_dims(out)
        if wo * 2 == wi and ho == hi:
            halving_axes.add("h")
        elif ho * 2 == hi and wo == wi:
            halving_axes.add("v")
    # Detect a "recolor target" — color that appears in output but not as a
    # plain overlay of input halves
    new_colors: set[int] = {0}
    for _, out in train_pairs:
        new_colors.update(colors_in(out))
    for axis in halving_axes:
        for mode in ("or", "and", "xor"):
            progs.append(Program(
                f"split_overlay_{axis}_{mode}",
                lambda g, axis=axis, mode=mode: t_split_grid_overlay(g, axis, mode),
            ))
            # Also try with each candidate recolor
            for nc in new_colors:
                if nc == 0:
                    continue
                progs.append(Program(
                    f"split_overlay_{axis}_{mode}_recolor_{nc}",
                    lambda g, axis=axis, mode=mode, nc=nc: t_split_grid_overlay(g, axis, mode, recolor=nc),
                ))
    # Keep the old explicit overlays too
    if "h" in halving_axes:
        progs.append(Program("mask_and_horizontal", t_mask_and))
        progs.append(Program("mask_or_horizontal", t_mask_or))
        progs.append(Program("mask_xor_horizontal", t_mask_xor))
    if "v" in halving_axes:
        progs.append(Program("mask_and_vertical", t_mask_and_vertical))
    # 4-quadrant overlay: output is 1/4 size in both dimensions
    for inp, out in train_pairs:
        hi, wi = grid_dims(inp); ho, wo = grid_dims(out)
        if ho * 2 == hi and wo * 2 == wi:
            for mode in ("or", "and", "xor"):
                progs.append(Program(
                    f"four_quad_overlay_{mode}",
                    lambda g, mode=mode: t_four_quadrant_overlay(g, mode),
                ))
                for nc in new_colors:
                    if nc == 0:
                        continue
                    progs.append(Program(
                        f"four_quad_overlay_{mode}_recolor_{nc}",
                        lambda g, mode=mode, nc=nc: t_four_quadrant_overlay(g, mode, recolor=nc),
                    ))
            break
    return progs


def _radial_symmetry_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    progs: list[Program] = []
    if not all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        return progs
    progs.append(Program("complete_rotational_symmetry", t_complete_rotational_symmetry))
    progs.append(Program("complete_4fold_symmetry", t_complete_4fold_symmetry))
    return progs


def _background_programs(train_pairs: list[tuple[Grid, Grid]]) -> list[Program]:
    progs: list[Program] = []
    if not all(grid_dims(i) == grid_dims(o) for i, o in train_pairs):
        return progs
    new_colors: set[int] = set()
    for inp, out in train_pairs:
        new_colors.update(colors_in(out) - colors_in(inp))
    for nc in new_colors:
        progs.append(Program(f"recolor_background_to_{nc}", lambda g, nc=nc: t_recolor_background(g, nc)))
    progs.append(Program("invert_background", t_invert_background))
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


def t_fill_between_same_color_markers(g: Grid, marker_color: int, fill_color: int) -> Grid:
    """For each row and column, find runs of cells with marker_color separated
    by zeros, fill the zeros BETWEEN consecutive markers with fill_color.
    Handles arbitrary numbers of markers per row/column."""
    h, w = grid_dims(g)
    out = grid_copy(g)
    # Per row
    for r in range(h):
        positions = [c for c in range(w) if g[r][c] == marker_color]
        for i in range(len(positions) - 1):
            c1, c2 = positions[i], positions[i + 1]
            # only fill if intermediate cells were all zero in INPUT
            if all(g[r][c] == 0 for c in range(c1 + 1, c2)):
                for c in range(c1 + 1, c2):
                    if out[r][c] == 0:
                        out[r][c] = fill_color
    # Per column
    for c in range(w):
        positions = [r for r in range(h) if g[r][c] == marker_color]
        for i in range(len(positions) - 1):
            r1, r2 = positions[i], positions[i + 1]
            if all(g[r][c] == 0 for r in range(r1 + 1, r2)):
                for r in range(r1 + 1, r2):
                    if out[r][c] == 0:
                        out[r][c] = fill_color
    return out


def t_recolor_non_majority_nonzero(g: Grid, new_color: int) -> Grid | None:
    """Keep the majority color, recolor all other non-zero cells to new_color."""
    from collections import Counter
    c = Counter(v for row in g for v in row if v != 0)
    if not c:
        return None
    maj, _ = c.most_common(1)[0]
    return [[new_color if v != 0 and v != maj else v for v in row] for row in g]


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


def t_remove_objects_by_color(g: Grid, color: int) -> Grid:
    """Set all cells of `color` to 0 (remove that color entirely)."""
    return [[0 if c == color else c for c in row] for row in g]


def t_keep_only_largest_object_of_each_color(g: Grid) -> Grid | None:
    """For each distinct non-bg color, keep only the largest connected object of that color."""
    h, w = grid_dims(g)
    objs = find_objects(g, by_color=True)
    if not objs:
        return None
    from collections import defaultdict
    by_color: dict[int, list[dict]] = defaultdict(list)
    for o in objs:
        by_color[o["color"]].append(o)
    out: Grid = [[0] * w for _ in range(h)]
    for color, group in by_color.items():
        largest = max(group, key=lambda o: len(o["cells"]))
        for r, c in largest["cells"]:
            out[r][c] = color
    return out


def _learn_marker_pattern(train_pairs: list[tuple[Grid, Grid]]):
    """If each training input has exactly one cell of some 'marker' color, and
    the output has a fixed offset pattern of cells around it, learn that pattern.
    Returns (marker_color, pattern_offsets) or None.
    pattern_offsets is a dict {(dr, dc): color}."""
    marker_color = None
    pattern: dict[tuple[int, int], int] | None = None

    for inp, out in train_pairs:
        if grid_dims(inp) != grid_dims(out):
            return None
        # Find the cell that's in input AND in output and is the marker
        # First, find what color appears EXACTLY ONCE in input
        from collections import Counter
        ic = Counter(v for row in inp for v in row if v != 0)
        singletons = [c for c, n in ic.items() if n == 1]
        if len(singletons) != 1:
            return None
        m_color = singletons[0]
        if marker_color is None:
            marker_color = m_color
        elif marker_color != m_color:
            return None
        # Find marker position
        h, w = grid_dims(inp)
        mr, mc = next((r, c) for r in range(h) for c in range(w) if inp[r][c] == m_color)
        # Compute offsets: cells in output that DIFFER from input
        this_pattern: dict[tuple[int, int], int] = {}
        for r in range(h):
            for c in range(w):
                if out[r][c] != inp[r][c]:
                    this_pattern[(r - mr, c - mc)] = out[r][c]
        if pattern is None:
            pattern = this_pattern
        elif pattern != this_pattern:
            return None
    if pattern is None or marker_color is None:
        return None
    return (marker_color, pattern)


def t_stamp_pattern_at_marker(g: Grid, marker_color: int, pattern: dict[tuple[int, int], int]) -> Grid | None:
    """Find the unique marker cell, stamp the learned pattern at offsets from it."""
    h, w = grid_dims(g)
    from collections import Counter
    c = Counter(v for row in g for v in row if v != 0)
    singletons = [k for k, n in c.items() if n == 1]
    if marker_color not in singletons:
        return None
    mr, mc = next((r, c) for r in range(h) for c in range(w) if g[r][c] == marker_color)
    out = grid_copy(g)
    for (dr, dc), color in pattern.items():
        nr, nc = mr + dr, mc + dc
        if 0 <= nr < h and 0 <= nc < w:
            out[nr][nc] = color
    return out


def t_sort_objects_by_size_along_axis(g: Grid, axis: str, ascending: bool) -> Grid | None:
    """Stack objects along an axis (top→bottom or left→right) sorted by size."""
    h, w = grid_dims(g)
    objs = find_objects(g, by_color=True)
    if len(objs) < 2:
        return None
    # Sort by size
    objs_sorted = sorted(objs, key=lambda o: len(o["cells"]), reverse=not ascending)
    # Place each object's bbox along the axis, starting at top-left
    out: Grid = [[0] * w for _ in range(h)]
    if axis == "v":
        cur_r = 0
        for o in objs_sorted:
            r0, c0, r1, c1 = o["bbox"]
            oh = r1 - r0 + 1
            for r, c in o["cells"]:
                nr = cur_r + (r - r0)
                if 0 <= nr < h:
                    out[nr][c] = o["color"]
            cur_r += oh
    elif axis == "h":
        cur_c = 0
        for o in objs_sorted:
            r0, c0, r1, c1 = o["bbox"]
            ow = c1 - c0 + 1
            for r, c in o["cells"]:
                nc = cur_c + (c - c0)
                if 0 <= nc < w:
                    out[r][nc] = o["color"]
            cur_c += ow
    else:
        return None
    return out


def t_compact_objects_to_corner(g: Grid, corner: str) -> Grid | None:
    """Move all objects to the specified corner, stacking them."""
    h, w = grid_dims(g)
    objs = find_objects(g, by_color=True)
    if not objs:
        return None
    out: Grid = [[0] * w for _ in range(h)]
    if corner == "tl":
        cur_r, cur_c = 0, 0
        for o in sorted(objs, key=lambda o: -len(o["cells"])):
            r0, c0, r1, c1 = o["bbox"]
            for r, c in o["cells"]:
                nr = cur_r + (r - r0); nc = cur_c + (c - c0)
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr][nc] = o["color"]
            cur_r += (r1 - r0 + 1)
    elif corner == "tr":
        cur_c = w
        for o in sorted(objs, key=lambda o: -len(o["cells"])):
            r0, c0, r1, c1 = o["bbox"]
            ow = c1 - c0 + 1
            cur_c -= ow
            for r, c in o["cells"]:
                nr = (r - r0); nc = cur_c + (c - c0)
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr][nc] = o["color"]
    elif corner == "bl":
        cur_r = h
        for o in sorted(objs, key=lambda o: -len(o["cells"])):
            r0, c0, r1, c1 = o["bbox"]
            oh = r1 - r0 + 1
            cur_r -= oh
            for r, c in o["cells"]:
                nr = cur_r + (r - r0); nc = (c - c0)
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr][nc] = o["color"]
    elif corner == "br":
        cur_r, cur_c = h, w
        for o in sorted(objs, key=lambda o: -len(o["cells"])):
            r0, c0, r1, c1 = o["bbox"]
            oh = r1 - r0 + 1; ow = c1 - c0 + 1
            cur_r -= oh; cur_c -= ow
            for r, c in o["cells"]:
                nr = cur_r + (r - r0); nc = cur_c + (c - c0)
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr][nc] = o["color"]
    return out


def t_overlay_objects(g: Grid, mode: str = "or") -> Grid | None:
    """Find all objects with the same bbox shape; overlay them onto a canvas
    of that shape using the given mode."""
    objs = find_objects(g, by_color=True)
    if len(objs) < 2:
        return None
    # Group by bbox shape
    shapes: dict[tuple[int, int], list[dict]] = {}
    for o in objs:
        shapes.setdefault(_object_bbox_shape(o), []).append(o)
    # Find the shape with the most instances
    best = max(shapes.items(), key=lambda kv: len(kv[1]))
    shape, group = best
    if len(group) < 2:
        return None
    sh, sw = shape
    canvas: Grid = [[0] * sw for _ in range(sh)]
    for o in group:
        r0, c0, _, _ = o["bbox"]
        for r, c in o["cells"]:
            dr = r - r0; dc = c - c0
            cur = canvas[dr][dc]
            if mode == "or":
                if cur == 0:
                    canvas[dr][dc] = o["color"]
            elif mode == "and":
                # Mark cells present in ALL members
                pass
            elif mode == "xor":
                canvas[dr][dc] = 0 if cur != 0 else o["color"]
    if mode == "and":
        # Recompute: a cell is set only if every group object has a cell there
        for r in range(sh):
            for c in range(sw):
                if all((r + o["bbox"][0], c + o["bbox"][1]) in o["cells"] for o in group):
                    canvas[r][c] = group[0]["color"]
                else:
                    canvas[r][c] = 0
    return canvas


def t_kaleidoscope_2x2(g: Grid) -> Grid:
    """Output 2x bigger: tl=g, tr=flip_h(g), bl=flip_v(g), br=rotate180(g)."""
    h, w = grid_dims(g)
    out: Grid = [[0] * (w * 2) for _ in range(h * 2)]
    tr = t_flip_h(g)
    bl = t_flip_v(g)
    br = t_rotate180(g)
    for r in range(h):
        for c in range(w):
            out[r][c] = g[r][c]
            out[r][c + w] = tr[r][c]
            out[r + h][c] = bl[r][c]
            out[r + h][c + w] = br[r][c]
    return out


def t_self_similar_tile_by_mask(g: Grid, mask_color: int) -> Grid:
    """Output is HxH copies of input where each tile is input if g[r][c] == mask_color else zeros.
    Only valid for square input; output dims are (H*H, W*W)."""
    h, w = grid_dims(g)
    out: Grid = [[0] * (w * w) for _ in range(h * h)]
    for tr in range(h):
        for tc in range(w):
            if g[tr][tc] == mask_color:
                for r in range(h):
                    for c in range(w):
                        out[tr * h + r][tc * w + c] = g[r][c]
    return out


def t_self_similar_tile_by_nonzero(g: Grid) -> Grid:
    """Each output tile is input where g[r][c] != 0."""
    h, w = grid_dims(g)
    out: Grid = [[0] * (w * w) for _ in range(h * h)]
    for tr in range(h):
        for tc in range(w):
            if g[tr][tc] != 0:
                for r in range(h):
                    for c in range(w):
                        out[tr * h + r][tc * w + c] = g[r][c]
    return out


def t_rotational_kaleidoscope_2x2(g: Grid) -> Grid | None:
    """Output 2x bigger: tl=g, tr=rotate90(g), br=rotate180(g), bl=rotate270(g).
    Only valid for square inputs."""
    h, w = grid_dims(g)
    if h != w:
        return None
    tr = t_rotate90(g)
    br = t_rotate180(g)
    bl = t_rotate270(g)
    out: Grid = [[0] * (w * 2) for _ in range(h * 2)]
    for r in range(h):
        for c in range(w):
            out[r][c] = g[r][c]
            out[r][c + w] = tr[r][c]
            out[r + h][c] = bl[r][c]
            out[r + h][c + w] = br[r][c]
    return out


def t_extract_subgrid_at(g: Grid, r0: int, c0: int, h: int, w: int) -> Grid | None:
    """Extract the h×w subgrid starting at (r0, c0). None if out of bounds."""
    H, W = grid_dims(g)
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return None
    return [row[c0:c0 + w] for row in g[r0:r0 + h]]


def t_extract_top_left(g: Grid, h: int, w: int) -> Grid | None:
    return t_extract_subgrid_at(g, 0, 0, h, w)


def t_extract_top_right(g: Grid, h: int, w: int) -> Grid | None:
    H, W = grid_dims(g)
    return t_extract_subgrid_at(g, 0, W - w, h, w)


def t_extract_bottom_left(g: Grid, h: int, w: int) -> Grid | None:
    H, W = grid_dims(g)
    return t_extract_subgrid_at(g, H - h, 0, h, w)


def t_extract_bottom_right(g: Grid, h: int, w: int) -> Grid | None:
    H, W = grid_dims(g)
    return t_extract_subgrid_at(g, H - h, W - w, h, w)


def t_extract_center(g: Grid, h: int, w: int) -> Grid | None:
    H, W = grid_dims(g)
    return t_extract_subgrid_at(g, (H - h) // 2, (W - w) // 2, h, w)


def t_keep_only_column(g: Grid, c_idx: int) -> Grid:
    """Zero out everything except column c_idx."""
    h, w = grid_dims(g)
    return [[g[r][c] if c == c_idx else 0 for c in range(w)] for r in range(h)]


def t_keep_only_row(g: Grid, r_idx: int) -> Grid:
    h, w = grid_dims(g)
    return [(g[r][:] if r == r_idx else [0] * w) for r in range(h)]


def t_keep_only_middle_column(g: Grid) -> Grid:
    h, w = grid_dims(g)
    return t_keep_only_column(g, w // 2)


def t_keep_only_middle_row(g: Grid) -> Grid:
    h, _ = grid_dims(g)
    return t_keep_only_row(g, h // 2)


def t_extract_first_column(g: Grid) -> Grid:
    """Output is the first column of g as a Hx1 grid."""
    h, _ = grid_dims(g)
    return [[g[r][0]] for r in range(h)]


def t_extract_first_row(g: Grid) -> Grid:
    """Output is the first row of g as a 1xW grid."""
    return [g[0][:]] if g else []


def t_extract_last_column(g: Grid) -> Grid:
    h, w = grid_dims(g)
    return [[g[r][w - 1]] for r in range(h)]


def t_extract_last_row(g: Grid) -> Grid:
    return [g[-1][:]] if g else []


def t_extract_middle_column(g: Grid) -> Grid | None:
    h, w = grid_dims(g)
    if w == 0:
        return None
    return [[g[r][w // 2]] for r in range(h)]


def t_extract_middle_row(g: Grid) -> Grid | None:
    h, _ = grid_dims(g)
    if h == 0:
        return None
    return [g[h // 2][:]]


def t_draw_x_from_marker(g: Grid, marker_color: int, line_color: int) -> Grid | None:
    """Find unique cell of marker_color, draw both diagonals through it in line_color."""
    h, w = grid_dims(g)
    from collections import Counter
    cc = Counter(v for row in g for v in row)
    if cc.get(marker_color, 0) != 1:
        return None
    mr, mc = next((r, c) for r in range(h) for c in range(w) if g[r][c] == marker_color)
    out = grid_copy(g)
    # diagonal /\
    for d in range(-max(h, w), max(h, w) + 1):
        nr, nc = mr + d, mc + d
        if 0 <= nr < h and 0 <= nc < w and (nr, nc) != (mr, mc) and out[nr][nc] == g[mr][mc]:
            # don't overwrite the marker itself — only fill background
            pass
        if 0 <= nr < h and 0 <= nc < w and out[nr][nc] != line_color and out[nr][nc] != marker_color:
            # only overwrite the background (not other colored cells)
            pass
    # Simpler version: replace background cells on the two diagonals with line_color
    bg = _detect_background(g)
    for d in range(-max(h, w), max(h, w) + 1):
        # main diagonal
        nr, nc = mr + d, mc + d
        if 0 <= nr < h and 0 <= nc < w and g[nr][nc] == bg:
            out[nr][nc] = line_color
        # anti diagonal
        nr, nc = mr + d, mc - d
        if 0 <= nr < h and 0 <= nc < w and g[nr][nc] == bg:
            out[nr][nc] = line_color
    return out


def t_fill_between_any_color_markers(g: Grid, fill_color: int) -> Grid:
    """For ANY non-zero color, find pairs of consecutive same-color cells in
    rows and columns, fill the zero cells between them with fill_color."""
    h, w = grid_dims(g)
    out = grid_copy(g)
    colors_present = colors_in(g) - {0}
    for marker_color in colors_present:
        # Per row
        for r in range(h):
            positions = [c for c in range(w) if g[r][c] == marker_color]
            for i in range(len(positions) - 1):
                c1, c2 = positions[i], positions[i + 1]
                if all(g[r][c] == 0 for c in range(c1 + 1, c2)):
                    for c in range(c1 + 1, c2):
                        if out[r][c] == 0:
                            out[r][c] = fill_color
        # Per column
        for c in range(w):
            positions = [r for r in range(h) if g[r][c] == marker_color]
            for i in range(len(positions) - 1):
                r1, r2 = positions[i], positions[i + 1]
                if all(g[r][c] == 0 for r in range(r1 + 1, r2)):
                    for r in range(r1 + 1, r2):
                        if out[r][c] == 0:
                            out[r][c] = fill_color
    return out


def t_fill_pair_bbox_rectangles(g: Grid, fill_color: int) -> Grid:
    """For each pair of same-color cells, fill the rectangle between them with fill_color.
    Cells of different colors do not pair. Single-cell colors are skipped."""
    h, w = grid_dims(g)
    out = grid_copy(g)
    from collections import defaultdict
    by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in range(h):
        for c in range(w):
            v = g[r][c]
            if v != 0:
                by_color[v].append((r, c))
    for color, positions in by_color.items():
        if len(positions) != 2:
            continue
        (r1, c1), (r2, c2) = positions
        r_lo, r_hi = min(r1, r2), max(r1, r2)
        c_lo, c_hi = min(c1, c2), max(c1, c2)
        for r in range(r_lo, r_hi + 1):
            for c in range(c_lo, c_hi + 1):
                if out[r][c] == 0:
                    out[r][c] = fill_color
    return out


def t_extend_each_cell_down(g: Grid) -> Grid:
    """For each non-zero cell, fill the column below it (down to bottom) with the same color."""
    h, w = grid_dims(g)
    out = grid_copy(g)
    for c in range(w):
        # Find topmost non-zero in this column
        for r in range(h):
            if g[r][c] != 0:
                # Fill down from r to bottom with g[r][c] (but don't overwrite different colors)
                color = g[r][c]
                for rr in range(r + 1, h):
                    if out[rr][c] == 0:
                        out[rr][c] = color
                break  # only the topmost cell propagates
    return out


def t_extend_each_cell_up(g: Grid) -> Grid:
    h, w = grid_dims(g)
    out = grid_copy(g)
    for c in range(w):
        for r in range(h - 1, -1, -1):
            if g[r][c] != 0:
                color = g[r][c]
                for rr in range(r - 1, -1, -1):
                    if out[rr][c] == 0:
                        out[rr][c] = color
                break
    return out


def t_extend_each_cell_right(g: Grid) -> Grid:
    h, w = grid_dims(g)
    out = grid_copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c] != 0:
                color = g[r][c]
                for cc in range(c + 1, w):
                    if out[r][cc] == 0:
                        out[r][cc] = color
                break
    return out


def t_extend_each_cell_left(g: Grid) -> Grid:
    h, w = grid_dims(g)
    out = grid_copy(g)
    for r in range(h):
        for c in range(w - 1, -1, -1):
            if g[r][c] != 0:
                color = g[r][c]
                for cc in range(c - 1, -1, -1):
                    if out[r][cc] == 0:
                        out[r][cc] = color
                break
    return out


def t_progressive_shift(g: Grid, n_rows: int, direction: str) -> Grid | None:
    """Output has n_rows rows; row r is input shifted by r positions in direction.
    Assumes input is 1xW (single row)."""
    h, w = grid_dims(g)
    if h != 1:
        return None
    src = g[0]
    out: Grid = []
    for r in range(n_rows):
        if direction == "right":
            shifted = [0] * r + src[:w - r] if r <= w else [0] * w
        elif direction == "left":
            shifted = src[r:] + [0] * r if r <= w else [0] * w
        else:
            return None
        # truncate/pad to w
        shifted = shifted[:w] + [0] * max(0, w - len(shifted))
        out.append(shifted)
    return out


def t_split_both_zero(g: Grid, axis: str, fill_color: int) -> Grid | None:
    """Split g into halves along axis; output is where BOTH halves are 0, filled with fill_color."""
    h, w = grid_dims(g)
    if axis == "v":
        if h % 2 != 0:
            return None
        half = h // 2
        a = g[:half]; b = g[half:]
        out: Grid = [[0] * w for _ in range(half)]
        for r in range(half):
            for c in range(w):
                if a[r][c] == 0 and b[r][c] == 0:
                    out[r][c] = fill_color
        return out
    elif axis == "h":
        if w % 2 != 0:
            return None
        half = w // 2
        out: Grid = [[0] * half for _ in range(h)]
        for r in range(h):
            for c in range(half):
                if g[r][c] == 0 and g[r][c + half] == 0:
                    out[r][c] = fill_color
        return out
    return None


def t_overlay_grids(a: Grid, b: Grid, mode: str = "or") -> Grid | None:
    """Stack two equally-shaped grids, with one of (or, and, xor) semantics."""
    if grid_dims(a) != grid_dims(b):
        return None
    h, w = grid_dims(a)
    out: Grid = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            av, bv = a[r][c], b[r][c]
            if mode == "or":
                out[r][c] = av if av != 0 else bv
            elif mode == "and":
                if av != 0 and bv != 0:
                    out[r][c] = av
            elif mode == "xor":
                if (av != 0) != (bv != 0):
                    out[r][c] = av if av != 0 else bv
    return out


def t_four_quadrant_overlay(g: Grid, mode: str, recolor: int | None = None) -> Grid | None:
    """Split grid into 4 quadrants (split both H and V), overlay all four."""
    h, w = grid_dims(g)
    if h % 2 != 0 or w % 2 != 0:
        return None
    hh, hw = h // 2, w // 2
    quads = [
        [row[:hw] for row in g[:hh]],
        [row[hw:] for row in g[:hh]],
        [row[:hw] for row in g[hh:]],
        [row[hw:] for row in g[hh:]],
    ]
    result = quads[0]
    for q in quads[1:]:
        result = t_overlay_grids(result, q, mode)
        if result is None:
            return None
    if recolor is None:
        return result
    return [[recolor if c != 0 else 0 for c in row] for row in result]


def t_split_grid_overlay(g: Grid, axis: str, mode: str, recolor: int | None = None) -> Grid | None:
    """Split grid into two halves along axis, overlay them with given mode.
    Optionally recolor any non-zero in result to `recolor`."""
    h, w = grid_dims(g)
    if axis == "h":
        if w % 2 != 0:
            return None
        half = w // 2
        a = [row[:half] for row in g]
        b = [row[half:] for row in g]
    elif axis == "v":
        if h % 2 != 0:
            return None
        half = h // 2
        a = g[:half]
        b = g[half:]
    else:
        return None
    result = t_overlay_grids(a, b, mode)
    if result is None or recolor is None:
        return result
    return [[recolor if c != 0 else 0 for c in row] for row in result]


def t_recolor_smallest_object(g: Grid, new_color: int) -> Grid | None:
    """Recolor the smallest connected object to new_color, leave others."""
    objs = find_objects(g, by_color=True)
    if not objs:
        return None
    smallest = min(objs, key=lambda o: len(o["cells"]))
    out = grid_copy(g)
    for r, c in smallest["cells"]:
        out[r][c] = new_color
    return out


def t_recolor_largest_object(g: Grid, new_color: int) -> Grid | None:
    objs = find_objects(g, by_color=True)
    if not objs:
        return None
    largest = max(objs, key=lambda o: len(o["cells"]))
    out = grid_copy(g)
    for r, c in largest["cells"]:
        out[r][c] = new_color
    return out


def t_recolor_unique_size_object(g: Grid, new_color: int) -> Grid | None:
    """Recolor the object whose size is unique (only one object of that size)."""
    objs = find_objects(g, by_color=True)
    if not objs:
        return None
    from collections import Counter
    size_counts = Counter(len(o["cells"]) for o in objs)
    uniques = [o for o in objs if size_counts[len(o["cells"])] == 1]
    if len(uniques) != 1:
        return None
    out = grid_copy(g)
    for r, c in uniques[0]["cells"]:
        out[r][c] = new_color
    return out


def t_recolor_isolated_cells(g: Grid, new_color: int) -> Grid:
    """Recolor cells that have no non-zero 4-neighbor (singletons) to new_color."""
    h, w = grid_dims(g)
    out = grid_copy(g)
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
                out[r][c] = new_color
    return out


def t_recolor_groups_keep_isolated(g: Grid, new_color: int) -> Grid:
    """Inverse: recolor cells that have at least one non-zero 4-neighbor to new_color."""
    h, w = grid_dims(g)
    out = grid_copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c] == 0:
                continue
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and g[nr][nc] != 0:
                    out[r][c] = new_color
                    break
    return out


def t_remove_noise_singletons(g: Grid) -> Grid:
    """Remove isolated cells (no 4-neighbor of any color), keep larger objects."""
    h, w = grid_dims(g)
    out = grid_copy(g)
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
                out[r][c] = 0
    return out


def t_align_objects_to_edge(g: Grid, edge: str) -> Grid | None:
    """Move each object to the specified edge (top, bottom, left, right) of the
    grid, preserving its shape and relative position along the perpendicular axis."""
    h, w = grid_dims(g)
    objs = find_objects(g, by_color=True)
    if not objs:
        return None
    out: Grid = [[0] * w for _ in range(h)]
    for o in objs:
        r0, c0, r1, c1 = o["bbox"]
        oh = r1 - r0 + 1; ow = c1 - c0 + 1
        if edge == "top":
            new_r0 = 0; new_c0 = c0
        elif edge == "bottom":
            new_r0 = h - oh; new_c0 = c0
        elif edge == "left":
            new_r0 = r0; new_c0 = 0
        elif edge == "right":
            new_r0 = r0; new_c0 = w - ow
        else:
            return None
        for r, c in o["cells"]:
            nr = new_r0 + (r - r0); nc = new_c0 + (c - c0)
            if 0 <= nr < h and 0 <= nc < w:
                out[nr][nc] = o["color"]
    return out


def t_complete_rotational_symmetry(g: Grid) -> Grid | None:
    """If g has approximate 180-degree rotational symmetry (about center), complete it."""
    h, w = grid_dims(g)
    out = grid_copy(g)
    for r in range(h):
        for c in range(w):
            mr, mc = h - 1 - r, w - 1 - c
            if out[r][c] == 0 and g[mr][mc] != 0:
                out[r][c] = g[mr][mc]
    return out


def t_complete_4fold_symmetry(g: Grid) -> Grid | None:
    """Complete H + V + rotational symmetries together (4-fold radial)."""
    h, w = grid_dims(g)
    if h != w:
        return None
    out = grid_copy(g)
    for r in range(h):
        for c in range(w):
            mirrors = [(r, c), (r, w - 1 - c), (h - 1 - r, c), (h - 1 - r, w - 1 - c)]
            colors = [g[mr][mc] for mr, mc in mirrors if g[mr][mc] != 0]
            if colors and len(set(colors)) == 1:
                out[r][c] = colors[0]
    return out


def t_mask_and(g: Grid) -> Grid | None:
    """Split grid horizontally into two halves; output is AND-mask (cells that are
    non-zero in both halves get color from left half, others 0)."""
    h, w = grid_dims(g)
    if w % 2 != 0:
        return None
    half = w // 2
    out: Grid = [[0] * half for _ in range(h)]
    for r in range(h):
        for c in range(half):
            if g[r][c] != 0 and g[r][c + half] != 0:
                out[r][c] = g[r][c]
    return out


def t_mask_or(g: Grid) -> Grid | None:
    h, w = grid_dims(g)
    if w % 2 != 0:
        return None
    half = w // 2
    out: Grid = [[0] * half for _ in range(h)]
    for r in range(h):
        for c in range(half):
            left = g[r][c]; right = g[r][c + half]
            if left != 0:
                out[r][c] = left
            elif right != 0:
                out[r][c] = right
    return out


def t_mask_xor(g: Grid) -> Grid | None:
    h, w = grid_dims(g)
    if w % 2 != 0:
        return None
    half = w // 2
    out: Grid = [[0] * half for _ in range(h)]
    for r in range(h):
        for c in range(half):
            left = g[r][c]; right = g[r][c + half]
            if (left != 0) != (right != 0):
                out[r][c] = left if left != 0 else right
    return out


def t_mask_and_vertical(g: Grid) -> Grid | None:
    """Same as t_mask_and but split vertically."""
    h, w = grid_dims(g)
    if h % 2 != 0:
        return None
    half = h // 2
    out: Grid = [[0] * w for _ in range(half)]
    for r in range(half):
        for c in range(w):
            if g[r][c] != 0 and g[r + half][c] != 0:
                out[r][c] = g[r][c]
    return out


def _detect_background(g: Grid) -> int:
    """Most common color, treated as background."""
    from collections import Counter
    c = Counter(v for row in g for v in row)
    if not c:
        return 0
    return c.most_common(1)[0][0]


def t_recolor_background(g: Grid, new_bg: int) -> Grid:
    """Recolor the (auto-detected) background to new_bg, keep others."""
    bg = _detect_background(g)
    if bg == new_bg:
        return grid_copy(g)
    return [[new_bg if c == bg else c for c in row] for row in g]


def t_invert_background(g: Grid) -> Grid:
    """Make all background cells the most common non-background color, and vice versa."""
    bg = _detect_background(g)
    from collections import Counter
    other = Counter(v for row in g for v in row if v != bg)
    if not other:
        return grid_copy(g)
    fg = other.most_common(1)[0][0]
    return [[fg if c == bg else (bg if c == fg else c) for c in row] for row in g]


def t_recolor_each_object_by_rank(g: Grid, rank_to_color: dict[int, int]) -> Grid | None:
    """Sort objects by size descending. Object at rank R gets recolored to rank_to_color[R]."""
    objs = find_objects(g, by_color=True)
    if not objs:
        return None
    objs_sorted = sorted(objs, key=lambda o: -len(o["cells"]))
    h, w = grid_dims(g)
    out = grid_copy(g)
    for rank, o in enumerate(objs_sorted):
        if rank not in rank_to_color:
            return None
        nc = rank_to_color[rank]
        for r, c in o["cells"]:
            out[r][c] = nc
    return out


def _learn_rank_to_color(train_pairs: list[tuple[Grid, Grid]]):
    """Learn rank -> color mapping from training pairs."""
    mapping: dict[int, int] = {}
    for inp, out in train_pairs:
        if grid_dims(inp) != grid_dims(out):
            return None
        objs = find_objects(inp, by_color=True)
        if not objs:
            return None
        objs_sorted = sorted(objs, key=lambda o: -len(o["cells"]))
        for rank, o in enumerate(objs_sorted):
            colors_at_obj = {out[r][c] for r, c in o["cells"]}
            if len(colors_at_obj) != 1:
                return None
            color = next(iter(colors_at_obj))
            if rank in mapping and mapping[rank] != color:
                return None
            mapping[rank] = color
    return mapping if mapping else None


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
        + _kaleidoscope_programs(train_pairs)
        + _row_col_extract_programs(train_pairs)
        + _extend_cell_programs(train_pairs)
        + _draw_x_programs(train_pairs)
        + _pair_rectangle_programs(train_pairs)
        + _keep_only_row_col_programs(train_pairs)
        + _corner_subgrid_programs(train_pairs)
        + _progressive_shift_programs(train_pairs)
        + _split_both_zero_programs(train_pairs)
        + _per_cell_substitute_programs(train_pairs)
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
        + _rank_recolor_programs(train_pairs)
        + _background_programs(train_pairs)
        + _radial_symmetry_programs(train_pairs)
        + _mask_programs(train_pairs)
        + _object_overlay_programs(train_pairs)
        + _object_arrange_programs(train_pairs)
        + _alignment_programs(train_pairs)
        + _noise_removal_programs(train_pairs)
        + _marker_pattern_programs(train_pairs)
        + _per_object_transform_programs(train_pairs)
        + _object_filter_programs(train_pairs)
        + _diagonal_programs(train_pairs)
        + _line_drawing_programs(train_pairs)
        + _fill_between_markers_programs(train_pairs)
        + _property_to_output_programs_constrained(train_pairs)  # LAST: high false-positive risk
    )


def _try_program(prog: Program, train_pairs, test_inp):
    """Return (matches_all_training, test_output) — both None on exception."""
    try:
        # Early termination: check the FIRST training pair first; if it fails,
        # skip the rest. This is the hot path — most programs fail on the first
        # check.
        first_inp, first_out = train_pairs[0]
        r0 = prog.apply(first_inp)
        if r0 is None or not grid_equal(r0, first_out):
            return False, None
        # Check the rest
        for inp, out in train_pairs[1:]:
            r = prog.apply(inp)
            if r is None or not grid_equal(r, out):
                return False, None
        test_r = prog.apply(test_inp)
        if test_r is None:
            return True, None
        return True, test_r
    except Exception:
        return False, None


def _input_features(g: Grid) -> dict:
    """Compute features used to prioritize which program families to try."""
    h, w = grid_dims(g)
    colors = colors_in(g)
    objs = find_objects(g, by_color=True) if h * w < 900 else []  # skip for huge grids
    return {
        "h": h, "w": w, "n_cells": h * w,
        "n_colors": len(colors), "n_nonzero_colors": len(colors - {0}),
        "n_objects": len(objs),
        "is_square": h == w,
        "has_zero": 0 in colors,
    }


def _shape_relation(train_pairs) -> str:
    """Classify train pairs' input vs output shape relationship.
    Returns one of: 'same', 'scale_up', 'scale_down', 'to_1x1', 'bigger', 'smaller', 'mixed'."""
    if not train_pairs:
        return "unknown"
    relations = set()
    for inp, out in train_pairs:
        hi, wi = grid_dims(inp); ho, wo = grid_dims(out)
        if (hi, wi) == (ho, wo):
            relations.add("same")
        elif (ho, wo) == (1, 1):
            relations.add("to_1x1")
        elif ho > hi or wo > wi:
            if hi > 0 and wi > 0 and ho % hi == 0 and wo % wi == 0:
                relations.add("scale_up")
            else:
                relations.add("bigger")
        elif ho < hi or wo < wi:
            if ho > 0 and wo > 0 and hi % ho == 0 and wi % wo == 0:
                relations.add("scale_down")
            else:
                relations.add("smaller")
    if len(relations) == 1:
        return next(iter(relations))
    return "mixed"


# ---------------------------------------------------------------------------
# Approach 3 (sketch): scene-graph extraction + object-diff
# ---------------------------------------------------------------------------

def _scene(g: Grid) -> dict:
    """Compute a scene description: list of objects with normalized shape sigs.

    Returns: {
      'dims': (h, w),
      'objects': [
         {'color': c, 'cells': frozenset((r,c)),
          'bbox': (r0,c0,r1,c1),
          'shape_sig': frozenset((dr,dc) normalized to bbox top-left),
          'size': int}
      ],
      'bg_color': int (most common color),
    }
    """
    h, w = grid_dims(g)
    bg = _detect_background(g)
    objs = find_objects(g, by_color=True)
    enriched = []
    for o in objs:
        r0, c0, _, _ = o["bbox"]
        sig = frozenset((r - r0, c - c0) for r, c in o["cells"])
        enriched.append({
            "color": o["color"],
            "cells": frozenset(o["cells"]),
            "bbox": o["bbox"],
            "shape_sig": sig,
            "size": len(o["cells"]),
        })
    return {"dims": (h, w), "objects": enriched, "bg_color": bg}


def _diff_scenes(scene_in: dict, scene_out: dict) -> dict:
    """Compare input scene to output scene; classify the transformation.

    Returns: { 'transformation_type': str, 'details': dict }
    Possible types:
      - 'identity': scenes match exactly
      - 'recolor_only': same shapes, recolored
      - 'shape_unchanged_moved': same shapes, different positions
      - 'one_to_one': bijection of input objects to output objects
      - 'subset_kept': output objects are a subset of input objects
      - 'objects_added': output has more objects than input
      - 'objects_removed': fewer
      - 'unclassified': none of the above
    """
    in_objs = scene_in["objects"]
    out_objs = scene_out["objects"]
    if len(in_objs) == len(out_objs):
        # Check if shape_sigs match in some bijection
        in_shapes = sorted(o["shape_sig"] for o in in_objs)
        out_shapes = sorted(o["shape_sig"] for o in out_objs)
        if in_shapes == out_shapes:
            in_colors = sorted(o["color"] for o in in_objs)
            out_colors = sorted(o["color"] for o in out_objs)
            in_bboxes = sorted(o["bbox"] for o in in_objs)
            out_bboxes = sorted(o["bbox"] for o in out_objs)
            if in_colors == out_colors and in_bboxes == out_bboxes:
                return {"transformation_type": "identity"}
            if in_bboxes == out_bboxes:
                return {"transformation_type": "recolor_only", "details": {"color_map": "TODO"}}
            if in_colors == out_colors:
                return {"transformation_type": "shape_unchanged_moved"}
            return {"transformation_type": "one_to_one"}
    if len(out_objs) < len(in_objs):
        in_sigs = [o["shape_sig"] for o in in_objs]
        out_sigs_set = set(o["shape_sig"] for o in out_objs)
        if all(s in in_sigs for s in [o["shape_sig"] for o in out_objs]):
            return {"transformation_type": "subset_kept"}
        return {"transformation_type": "objects_removed"}
    if len(out_objs) > len(in_objs):
        return {"transformation_type": "objects_added"}
    return {"transformation_type": "unclassified"}


def _classify_task(train_pairs) -> dict:
    """Compute the dominant transformation type across training pairs."""
    from collections import Counter
    types: list[str] = []
    for inp, out in train_pairs:
        si = _scene(inp); so = _scene(out)
        if grid_dims(inp) != grid_dims(out):
            types.append("shape_change")
            continue
        d = _diff_scenes(si, so)
        types.append(d["transformation_type"])
    c = Counter(types)
    return {"types": types, "dominant": c.most_common(1)[0][0]}


def _learn_output_constraints(train_pairs, test_inp):
    """From training pairs, predict CONSTRAINTS the test output should satisfy.

    Returns dict with possible fields:
      - 'expected_dims': (h, w) if predictable from test_inp
      - 'expected_colors': frozenset of colors that should be in output (or None)
      - 'output_subset_of_input_colors': bool (output never adds colors)
    """
    constraints: dict = {}
    test_h, test_w = grid_dims(test_inp)

    # Detect dim relation: same / scale / fixed / function-of-input-dim
    dim_relations: set[str] = set()
    for inp, out in train_pairs:
        hi, wi = grid_dims(inp); ho, wo = grid_dims(out)
        if (hi, wi) == (ho, wo):
            dim_relations.add("same")
        elif hi > 0 and wi > 0 and ho % hi == 0 and wo % wi == 0:
            dim_relations.add(f"scale_up_{ho//hi}_{wo//wi}")
        elif ho > 0 and wo > 0 and hi % ho == 0 and wi % wo == 0:
            dim_relations.add(f"scale_down_{hi//ho}_{wi//wo}")
        elif (ho, wo) == (1, 1):
            dim_relations.add("to_1x1")
        else:
            dim_relations.add(f"fixed_{ho}_{wo}")
    if len(dim_relations) == 1:
        rel = next(iter(dim_relations))
        if rel == "same":
            constraints["expected_dims"] = (test_h, test_w)
        elif rel.startswith("scale_up_"):
            kr, kc = map(int, rel.split("_")[2:])
            constraints["expected_dims"] = (test_h * kr, test_w * kc)
        elif rel.startswith("scale_down_"):
            kr, kc = map(int, rel.split("_")[2:])
            if test_h % kr == 0 and test_w % kc == 0:
                constraints["expected_dims"] = (test_h // kr, test_w // kc)
        elif rel == "to_1x1":
            constraints["expected_dims"] = (1, 1)
        elif rel.startswith("fixed_"):
            h, w = map(int, rel.split("_")[1:])
            constraints["expected_dims"] = (h, w)

    # Detect color subset: output colors ⊆ input colors in EVERY training pair?
    subset_holds = all(
        colors_in(out) <= colors_in(inp) for inp, out in train_pairs
    )
    constraints["output_subset_of_input_colors"] = subset_holds

    # Detect when output colors are CONSTANT across training pairs
    out_color_sets = [colors_in(out) for _, out in train_pairs]
    if len(set(frozenset(s) for s in out_color_sets)) == 1:
        constraints["expected_colors"] = frozenset(out_color_sets[0])

    return constraints


def _expected_output_signature(train_pairs):
    """Legacy: compute features of training outputs (used by old smart_rank)."""
    sigs = []
    for inp, out in train_pairs:
        sigs.append({
            "colors": frozenset(colors_in(out)),
            "dims": grid_dims(out),
            "n_nonzero": sum(1 for row in out for c in row if c != 0),
        })
    return sigs


def _candidate_passes_constraints(candidate_out: Grid, constraints: dict, test_inp: Grid) -> bool:
    """Hard filter: candidate must satisfy learned constraints."""
    if "expected_dims" in constraints:
        if grid_dims(candidate_out) != constraints["expected_dims"]:
            return False
    if constraints.get("output_subset_of_input_colors"):
        in_colors = colors_in(test_inp)
        out_colors = colors_in(candidate_out)
        # Allow 0 (background) always
        if not (out_colors - {0}) <= in_colors:
            return False
    if "expected_colors" in constraints:
        cand_colors = colors_in(candidate_out)
        if cand_colors != constraints["expected_colors"]:
            return False
    return True


def _program_priority(prog_name: str) -> int:
    """Higher = preferred when multiple programs match training.

    Heuristic: prefer LEARNED primitives (parameters induced from training
    data) over hand-coded geometric/fixed ops. Within learned, prefer more
    specific over more general.
    """
    # Strong: programs that learn parameters from training
    if prog_name.startswith("ca_rule_neighbor_sig"): return 100
    if prog_name.startswith("ca_rule_neighbor_count"): return 95
    if prog_name.startswith("ca_rule_k2"): return 90
    if prog_name.startswith("per_cell_substitute"): return 88
    if prog_name.startswith("recolor_map_"): return 85
    if prog_name.startswith("recolor_by_length"): return 84
    if prog_name.startswith("color_perm_"): return 83
    if prog_name.startswith("recolor_each_object_by_rank_"): return 82
    if prog_name.startswith("stamp_pattern_at_marker"): return 80

    # Strong: pattern + structural
    if prog_name.startswith("complete_tiled_pattern"): return 75
    if prog_name.startswith("recolor_oe_"): return 72
    if prog_name.startswith("recolor_longest_to_"): return 70
    if prog_name.startswith("fill_between_"): return 68
    if prog_name.startswith("recolor_non_majority_nonzero_to_"): return 66
    if prog_name.startswith("split_overlay_"): return 65
    if prog_name.startswith("four_quad_overlay_"): return 63

    # Medium: object-level
    if prog_name.startswith("keep_largest_object"): return 60
    if prog_name.startswith("keep_smallest_object"): return 59
    if prog_name.startswith("extract_unique_color_object"): return 58
    if prog_name.startswith("extract_majority_subgrid"): return 57
    if prog_name.startswith("extract_unique_subgrid"): return 56
    if prog_name.startswith("crop_to_color_"): return 55
    if prog_name.startswith("crop_to_largest_object"): return 54
    if prog_name.startswith("crop_to_bbox"): return 53

    # Medium: simple but data-derived
    if prog_name.startswith("flood_fill_enclosed_"): return 50
    if prog_name.startswith("recolor_objs_by_size_"): return 48
    if prog_name.startswith("draw_bbox_frame_"): return 46
    if prog_name.startswith("outline_objects_"): return 44

    # Geometric (no parameter learning)
    if prog_name == "complete_symmetry_h": return 35
    if prog_name == "complete_symmetry_v": return 34
    if prog_name == "complete_symmetry_both": return 33
    if prog_name == "complete_detected_symmetry": return 32
    if prog_name in ("flip_h", "flip_v", "rotate90", "rotate180", "rotate270", "transpose"): return 30
    if prog_name.startswith("shift_"): return 28
    if prog_name.startswith("gravity_"): return 25

    # Recolor variants
    if prog_name.startswith("recolor_"): return 22

    # Generic / fallback
    if prog_name == "identity": return 1
    if prog_name.startswith("constant_output_"): return 5
    if prog_name.startswith("prop_"): return 10  # property-to-output runs LAST anyway
    if prog_name.startswith("compose:"): return 15
    if prog_name.startswith("compose3:"): return 12

    # Default for everything else
    return 20


def _score_candidate(candidate_out: Grid, training_sigs, prog_name: str = "") -> float:
    """Score a candidate. Higher = better.

    Combines: (a) program-family priority (learned > geometric),
              (b) color set similarity to training outputs,
              (c) non-zero density similarity.
    """
    score = 0.0

    # (a) Program priority (dominant factor)
    score += _program_priority(prog_name) * 10.0  # weight = 10 so priorities dominate

    # (b) + (c) Output similarity to training outputs
    cand_colors = frozenset(colors_in(candidate_out))
    cand_nz = sum(1 for row in candidate_out for c in row if c != 0)
    overlap_scores = []
    for sig in training_sigs:
        if sig["colors"] or cand_colors:
            jac = len(cand_colors & sig["colors"]) / max(1, len(cand_colors | sig["colors"]))
            overlap_scores.append(jac)
        denom = max(1, max(cand_nz, sig["n_nonzero"]))
        diff = abs(cand_nz - sig["n_nonzero"]) / denom
        overlap_scores.append(1.0 - diff)
    if overlap_scores:
        score += sum(overlap_scores) / len(overlap_scores)
    return score


def solve_task(task_data: dict, allow_compose: bool = True,
               smart_rank: bool = False, use_constraints: bool = True) -> tuple[str, Grid] | None:
    train = task_data.get("train", [])
    test = task_data.get("test", [])
    if not train or not test:
        return None
    train_pairs = [(t["input"], t["output"]) for t in train]
    test_inp = test[0]["input"]

    progs = candidate_programs(train_pairs)

    # Learn output constraints from training pairs (Approach 2: backward filtering)
    constraints = _learn_output_constraints(train_pairs, test_inp) if use_constraints else {}

    if smart_rank:
        training_sigs = _expected_output_signature(train_pairs)
        candidates = []
        for prog in progs:
            ok, result = _try_program(prog, train_pairs, test_inp)
            if ok and result is not None:
                if constraints and not _candidate_passes_constraints(result, constraints, test_inp):
                    continue
                candidates.append((prog, result))
                if len(candidates) >= 20:
                    break
        if candidates:
            best = max(candidates, key=lambda pr: _score_candidate(pr[1], training_sigs, pr[0].name))
            return best[0].name, best[1]
    else:
        # First-match-that-passes-constraints wins
        for prog in progs:
            ok, result = _try_program(prog, train_pairs, test_inp)
            if ok and result is not None:
                if constraints and not _candidate_passes_constraints(result, constraints, test_inp):
                    continue
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
