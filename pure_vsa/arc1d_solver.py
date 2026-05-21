"""1D-ARC solver: program-synthesis approach over a small library of grid
transformations, induced from the training input-output examples per task.

For each task, we observe ~3-5 (input, output) example pairs and must
predict the output for a held-out test input. The transformations are
deterministic per task type.

Strategy: search a small library of parameterized programs. A program
is one of:
  - identity:        output = input
  - shift(k):        cyclically shift cells by k positions
  - flip_block:      reverse the colored block, swap marker to opposite end
  - reverse:         reverse the entire colored region
  - recolor(a, b):   replace color a with color b
  - mirror:          mirror about center
  - fill_between:    fill cells between two marker cells with a color
  - copy_n(n):       repeat the colored block n times

For each test task, we enumerate programs and select the one that matches
ALL training examples exactly. If found, apply to the test input.

This is NOT pure VSA — it's classical program-induction. The unifying
mechanism with Hyperion is: extract a SHARED rule from training examples,
then apply it to held-out test input. VSA-flavored generalization, but
the algorithm here is direct enumeration + match-check, not algebra.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


Grid = list[list[int]]  # 1xN grid; outer list always length 1 for 1D-ARC


def grid_row(g: Grid) -> list[int]:
    return g[0]


def to_grid(row: list[int]) -> Grid:
    return [row]


# ----------------------------------------------------------------------
# Transformation primitives
# ----------------------------------------------------------------------

def t_identity(row: list[int]) -> list[int]:
    return list(row)


def t_reverse(row: list[int]) -> list[int]:
    return list(reversed(row))


def t_shift(row: list[int], k: int) -> list[int]:
    n = len(row)
    if n == 0:
        return row
    k = k % n
    return row[-k:] + row[:-k] if k else list(row)


def t_recolor(row: list[int], a: int, b: int) -> list[int]:
    return [b if c == a else c for c in row]


def t_mirror(row: list[int]) -> list[int]:
    return list(reversed(row))


def find_block(row: list[int]) -> tuple[int, int, int] | None:
    """Find the largest connected run of a single non-zero color. Returns
    (start, end, color) or None if no block."""
    best: tuple[int, int, int] | None = None
    i = 0
    n = len(row)
    while i < n:
        if row[i] != 0:
            j = i
            while j < n and row[j] == row[i]:
                j += 1
            run_len = j - i
            if best is None or run_len > best[1] - best[0]:
                best = (i, j, row[i])
            i = j
        else:
            i += 1
    return best


def find_marker_and_block(row: list[int]) -> tuple[int, int, int, int, int] | None:
    """Find a single marker cell adjacent to a colored block.
    Returns (marker_pos, marker_color, block_start, block_end, block_color).
    """
    n = len(row)
    for i, c in enumerate(row):
        if c == 0:
            continue
        # Is i a marker (single cell) adjacent to a different-colored block?
        is_singleton = (i == 0 or row[i - 1] != c) and (i == n - 1 or row[i + 1] != c)
        if not is_singleton:
            continue
        # Check left neighbor
        if i + 1 < n and row[i + 1] != 0 and row[i + 1] != c:
            color = row[i + 1]
            j = i + 1
            while j < n and row[j] == color:
                j += 1
            return (i, c, i + 1, j, color)
        if i - 1 >= 0 and row[i - 1] != 0 and row[i - 1] != c:
            color = row[i - 1]
            j = i - 1
            while j >= 0 and row[j] == color:
                j -= 1
            return (i, c, j + 1, i, color)
    return None


def t_flip_marker(row: list[int]) -> list[int] | None:
    """Swap marker to opposite end of its adjacent block.

    Input pattern: ..0 M B B B 0..  or  ..0 B B B M 0..
    Output pattern: same length, but marker and end-of-block swap.
    """
    info = find_marker_and_block(row)
    if info is None:
        return None
    m_pos, m_color, b_start, b_end, b_color = info
    # b_end is exclusive
    n = len(row)
    out = [0] * n
    # Preserve original non-marker non-block cells (should be zeros)
    if m_pos < b_start:
        # marker before block, new: block at [m_pos .. b_end-1], marker at b_end-1
        for k in range(m_pos, b_end - 1):
            out[k] = b_color
        out[b_end - 1] = m_color
    elif m_pos >= b_end:
        # marker after block, new: marker at b_start, block at [b_start+1 .. m_pos]
        out[b_start] = m_color
        for k in range(b_start + 1, m_pos + 1):
            if k < n:
                out[k] = b_color
    else:
        return None
    return out


def t_recolor_all_nonzero(row: list[int], target: int) -> list[int]:
    return [target if c != 0 else 0 for c in row]


# ----------------------------------------------------------------------
# Program-synthesis solver
# ----------------------------------------------------------------------

@dataclass
class Program:
    name: str
    apply: Any  # function(row) -> list[int]


def t_denoise_singletons(row: list[int]) -> list[int]:
    """Remove singleton colored cells; keep only runs of length >= 2."""
    n = len(row)
    out = list(row)
    i = 0
    while i < n:
        if row[i] == 0:
            i += 1
            continue
        j = i
        while j < n and row[j] == row[i]:
            j += 1
        if j - i == 1:
            out[i] = 0
        i = j
    return out


def t_keep_only_singletons(row: list[int]) -> list[int]:
    """Inverse: keep singletons, zero out long runs."""
    n = len(row)
    out = list(row)
    i = 0
    while i < n:
        if row[i] == 0:
            i += 1
            continue
        j = i
        while j < n and row[j] == row[i]:
            j += 1
        if j - i > 1:
            for k in range(i, j):
                out[k] = 0
        i = j
    return out


def t_fill_between_markers(row: list[int]) -> list[int] | None:
    """If there are exactly two non-zero cells of the same color separated
    by zeros, fill the zeros between them with that color."""
    nonzeros = [(i, c) for i, c in enumerate(row) if c != 0]
    if len(nonzeros) != 2:
        return None
    (i1, c1), (i2, c2) = nonzeros
    if c1 != c2:
        return None
    out = list(row)
    for k in range(i1, i2 + 1):
        out[k] = c1
    return out


def t_hollow(row: list[int]) -> list[int]:
    """Replace the interior of each connected run with 0; keep only the
    two endpoints of each run."""
    n = len(row)
    out = list(row)
    i = 0
    while i < n:
        if row[i] == 0:
            i += 1
            continue
        j = i
        while j < n and row[j] == row[i]:
            j += 1
        # i..j is a run; zero out i+1 .. j-2 (keep endpoints)
        for k in range(i + 1, j - 1):
            out[k] = 0
        i = j
    return out


def t_mirror_about_marker(row: list[int]) -> list[int] | None:
    """If there's a singleton marker cell of color M (M unique in row),
    reflect the colored block on one side of M to the OTHER side, preserving
    the same distance from M."""
    from collections import Counter
    nonzero = [(i, c) for i, c in enumerate(row) if c != 0]
    if not nonzero:
        return None
    counts = Counter(c for _, c in nonzero)
    # Find a singleton color
    markers = [c for c, n in counts.items() if n == 1]
    if len(markers) != 1:
        return None
    marker_color = markers[0]
    marker_pos = next(i for i, c in nonzero if c == marker_color)
    block_cells = [(i, c) for i, c in nonzero if c != marker_color]
    if not block_cells:
        return None
    block_color = block_cells[0][1]
    if any(c != block_color for _, c in block_cells):
        return None
    out = [0] * len(row)
    out[marker_pos] = marker_color
    for i, _ in block_cells:
        # reflect across marker_pos: new_pos = 2*marker_pos - i
        new_pos = 2 * marker_pos - i
        if 0 <= new_pos < len(row):
            out[new_pos] = block_color
        else:
            return None  # block doesn't fit
    return out


def t_fill_run(row: list[int]) -> list[int]:
    """Find the colored run and FILL any 0s inside it (between its first
    and last non-zero cell of that color)."""
    nonzero_positions = [i for i, c in enumerate(row) if c != 0]
    if not nonzero_positions:
        return list(row)
    # Determine the dominant color among non-zeros
    from collections import Counter
    colors = Counter(row[i] for i in nonzero_positions)
    color, _ = colors.most_common(1)[0]
    # Find first and last position of that color
    color_pos = [i for i in nonzero_positions if row[i] == color]
    first, last = color_pos[0], color_pos[-1]
    out = list(row)
    for k in range(first, last + 1):
        out[k] = color
    return out


def _runs(row: list[int]) -> list[tuple[int, int, int]]:
    """Return list of (start, end_exclusive, color) for each maximal non-zero run."""
    out = []
    n = len(row)
    i = 0
    while i < n:
        if row[i] == 0:
            i += 1
            continue
        j = i
        while j < n and row[j] == row[i]:
            j += 1
        out.append((i, j, row[i]))
        i = j
    return out


def _find_block_and_marker(row: list[int]):
    """Find ONE multi-cell colored block and ONE singleton-marker of different color.
    Returns (b_start, b_end, b_color, m_pos, m_color) or None."""
    runs = _runs(row)
    if len(runs) != 2:
        return None
    (s1, e1, c1), (s2, e2, c2) = runs
    if c1 == c2:
        return None
    len1, len2 = e1 - s1, e2 - s2
    if len1 >= 2 and len2 == 1:
        return (s1, e1, c1, s2, c2)
    if len2 >= 2 and len1 == 1:
        return (s2, e2, c2, s1, c1)
    return None


def t_move_adjacent_to_marker(row: list[int]) -> list[int] | None:
    """Move colored block to be directly adjacent to marker (block stays adjacent to it)."""
    info = _find_block_and_marker(row)
    if info is None:
        return None
    b_start, b_end, b_color, m_pos, m_color = info
    n = len(row)
    block_len = b_end - b_start
    out = [0] * n
    out[m_pos] = m_color
    if b_end - 1 < m_pos:
        # block left of marker, move right so block ends at m_pos - 1
        new_start = m_pos - block_len
        if new_start < 0:
            return None
        for k in range(new_start, m_pos):
            out[k] = b_color
    else:
        # block right of marker, move left so block starts at m_pos + 1
        new_start = m_pos + 1
        if new_start + block_len > n:
            return None
        for k in range(new_start, new_start + block_len):
            out[k] = b_color
    return out


def t_move_k_toward_marker(row: list[int], k: int) -> list[int] | None:
    """Move colored block by k cells toward the marker (block keeps its size)."""
    info = _find_block_and_marker(row)
    if info is None:
        return None
    b_start, b_end, b_color, m_pos, m_color = info
    n = len(row)
    out = [0] * n
    out[m_pos] = m_color
    if b_end - 1 < m_pos:
        new_start = b_start + k
        new_end = b_end + k
        if new_end - 1 >= m_pos or new_start < 0:
            return None
        for j in range(new_start, new_end):
            out[j] = b_color
    else:
        new_start = b_start - k
        new_end = b_end - k
        if new_start <= m_pos or new_end > n:
            return None
        for j in range(new_start, new_end):
            out[j] = b_color
    return out


def t_scale_to_marker(row: list[int]) -> list[int] | None:
    """Extend colored block to reach (but not overlap) the marker. Block stays anchored at its far end."""
    info = _find_block_and_marker(row)
    if info is None:
        return None
    b_start, b_end, b_color, m_pos, m_color = info
    n = len(row)
    out = [0] * n
    out[m_pos] = m_color
    if b_end - 1 < m_pos:
        # block left of marker, extend right to m_pos - 1
        for j in range(b_start, m_pos):
            out[j] = b_color
    else:
        # block right of marker, extend left to m_pos + 1
        for j in range(m_pos + 1, b_end):
            out[j] = b_color
    return out


def t_pcopy_same_color(row: list[int]) -> list[int] | None:
    """Source block (longest run) + singletons of SAME color. Replace each singleton
    with a copy of the source block, centered on the singleton's position."""
    runs = _runs(row)
    if len(runs) < 2:
        return None
    colors = set(c for _, _, c in runs)
    if len(colors) != 1:
        return None
    longest = max(runs, key=lambda r: r[1] - r[0])
    src_len = longest[1] - longest[0]
    if src_len < 2:
        return None
    color = longest[2]
    singletons = [r for r in runs if r[1] - r[0] == 1]
    if not singletons:
        return None
    n = len(row)
    out = [0] * n
    # Keep source
    for k in range(longest[0], longest[1]):
        out[k] = color
    half = src_len // 2
    for s_start, _, _ in singletons:
        new_start = s_start - half
        new_end = new_start + src_len
        if new_start < 0 or new_end > n:
            return None
        for k in range(new_start, new_end):
            out[k] = color
    return out


def t_pcopy_multi_color(row: list[int]) -> list[int] | None:
    """Source block (longest run) of color C0 + singletons of various colors.
    Replace each singleton with a copy of source-block-shape, in the singleton's color."""
    runs = _runs(row)
    if len(runs) < 2:
        return None
    longest = max(runs, key=lambda r: r[1] - r[0])
    src_len = longest[1] - longest[0]
    if src_len < 2:
        return None
    src_color = longest[2]
    singletons = [r for r in runs if r[1] - r[0] == 1]
    if not singletons:
        return None
    n = len(row)
    out = [0] * n
    for k in range(longest[0], longest[1]):
        out[k] = src_color
    half = src_len // 2
    for s_start, _, s_color in singletons:
        new_start = s_start - half
        new_end = new_start + src_len
        if new_start < 0 or new_end > n:
            return None
        for k in range(new_start, new_end):
            out[k] = s_color
    return out


def t_padded_fill_pairs(row: list[int]) -> list[int] | None:
    """Markers (singletons same color) come in pairs (m_0, m_1), (m_2, m_3), ...
    Fill from each even-indexed marker to the next odd-indexed marker (inclusive)."""
    nonzero = [(i, c) for i, c in enumerate(row) if c != 0]
    if len(nonzero) < 2 or len(nonzero) % 2 != 0:
        return None
    colors = set(c for _, c in nonzero)
    if len(colors) != 1:
        return None
    color = next(iter(colors))
    # Singletons only
    runs = _runs(row)
    if any(e - s != 1 for s, e, _ in runs):
        return None
    positions = [i for i, _ in nonzero]
    n = len(row)
    out = [0] * n
    for k in range(0, len(positions), 2):
        if k + 1 >= len(positions):
            return None
        a, b = positions[k], positions[k + 1]
        for j in range(a, b + 1):
            out[j] = color
    return out


def make_recolor_by_length(length_to_color: dict[int, int]):
    """Recolor each run based on its length using the given mapping."""
    def fn(row: list[int]) -> list[int] | None:
        runs = _runs(row)
        out = list(row)
        for s, e, _c in runs:
            run_len = e - s
            if run_len not in length_to_color:
                return None
            new_color = length_to_color[run_len]
            for k in range(s, e):
                out[k] = new_color
        return out
    return fn


def make_recolor_longest(new_color: int):
    """Recolor the longest run(s) to new_color, leave others alone."""
    def fn(row: list[int]) -> list[int] | None:
        runs = _runs(row)
        if not runs:
            return None
        max_len = max(e - s for s, e, _ in runs)
        out = list(row)
        for s, e, _c in runs:
            if e - s == max_len:
                for k in range(s, e):
                    out[k] = new_color
        return out
    return fn


def make_recolor_oe(odd_color: int, even_color: int):
    """Recolor each run: odd-length → odd_color, even-length → even_color."""
    def fn(row: list[int]) -> list[int] | None:
        runs = _runs(row)
        out = list(row)
        for s, e, _c in runs:
            run_len = e - s
            new_color = odd_color if run_len % 2 == 1 else even_color
            for k in range(s, e):
                out[k] = new_color
        return out
    return fn


def _learn_oe_colors(train_examples):
    """Induce (odd_color, even_color) if all odd-length runs map to one color
    and all even-length runs map to another."""
    odd_color = None
    even_color = None
    for inp, out in train_examples:
        in_runs = _runs(inp)
        out_runs = _runs(out)
        if len(in_runs) != len(out_runs):
            return None
        for (s_i, e_i, _), (s_o, e_o, c_o) in zip(in_runs, out_runs):
            if (s_i, e_i) != (s_o, e_o):
                return None
            run_len = e_i - s_i
            if run_len % 2 == 1:
                if odd_color is None:
                    odd_color = c_o
                elif odd_color != c_o:
                    return None
            else:
                if even_color is None:
                    even_color = c_o
                elif even_color != c_o:
                    return None
    if odd_color is None or even_color is None or odd_color == even_color:
        return None
    return (odd_color, even_color)


def _learn_length_to_color(train_examples):
    """From training pairs, induce {run_length -> output_color} mapping.
    Each run in input must have a consistent output color."""
    mapping: dict[int, int] = {}
    for inp, out in train_examples:
        in_runs = _runs(inp)
        out_runs = _runs(out)
        if len(in_runs) != len(out_runs):
            return None
        for (s_i, e_i, _), (s_o, e_o, c_o) in zip(in_runs, out_runs):
            if (s_i, e_i) != (s_o, e_o):
                return None
            run_len = e_i - s_i
            if run_len in mapping and mapping[run_len] != c_o:
                return None
            mapping[run_len] = c_o
    return mapping


def _learn_longest_color(train_examples):
    """From training pairs, induce the color used to recolor longest run(s).
    Other runs must remain unchanged."""
    target_colors = set()
    for inp, out in train_examples:
        in_runs = _runs(inp)
        out_runs = _runs(out)
        if len(in_runs) != len(out_runs):
            return None
        max_len = max(e - s for s, e, _ in in_runs)
        for (s_i, e_i, c_i), (s_o, e_o, c_o) in zip(in_runs, out_runs):
            if (s_i, e_i) != (s_o, e_o):
                return None
            if e_i - s_i == max_len:
                target_colors.add(c_o)
            else:
                if c_o != c_i:
                    return None
    if len(target_colors) != 1:
        return None
    return next(iter(target_colors))


def candidate_programs(train_examples: list[tuple[list[int], list[int]]]) -> list[Program]:
    """Return a list of candidate programs to try."""
    progs: list[Program] = [
        Program("identity", t_identity),
        Program("reverse", t_reverse),
        Program("denoise_singletons", t_denoise_singletons),
        Program("keep_only_singletons", t_keep_only_singletons),
        Program("hollow", t_hollow),
        Program("fill_run", t_fill_run),
        Program("fill_between_markers",
                lambda r: t_fill_between_markers(r) if t_fill_between_markers(r) else r),
        Program("mirror_about_marker",
                lambda r: t_mirror_about_marker(r) if t_mirror_about_marker(r) else r),
    ]
    # Try shifts of various sizes
    for k in range(1, 8):
        kk = k
        progs.append(Program(f"shift_{k}", lambda r, kk=kk: t_shift(r, kk)))
        progs.append(Program(f"shift_-{k}", lambda r, kk=kk: t_shift(r, -kk)))

    # Recolor pairs we see in training: any color->color swap
    color_swaps: set[tuple[int, int]] = set()
    for inp, out in train_examples:
        for a, b in zip(inp, out):
            if a != b and a != 0 and b != 0:
                color_swaps.add((a, b))
    for a, b in color_swaps:
        progs.append(Program(f"recolor_{a}_to_{b}", lambda r, a=a, b=b: t_recolor(r, a, b)))

    # Marker-flip
    def flip_wrap(r):
        out = t_flip_marker(r)
        return out if out is not None else r
    progs.append(Program("flip_marker", flip_wrap))

    # Block-with-marker movement programs
    def wrap(fn):
        def inner(r):
            out = fn(r)
            return out if out is not None else r
        return inner

    progs.append(Program("move_adjacent_to_marker", wrap(t_move_adjacent_to_marker)))
    progs.append(Program("scale_to_marker", wrap(t_scale_to_marker)))
    for k in range(1, 5):
        kk = k
        progs.append(Program(f"move_{kk}_toward_marker",
                             wrap(lambda r, kk=kk: t_move_k_toward_marker(r, kk))))

    # Pattern-copy (singletons -> source-block-shape)
    progs.append(Program("pcopy_same_color", wrap(t_pcopy_same_color)))
    progs.append(Program("pcopy_multi_color", wrap(t_pcopy_multi_color)))

    # Padded fill (paired markers)
    progs.append(Program("padded_fill_pairs", wrap(t_padded_fill_pairs)))

    # Induced recolor programs — order matters; more general first
    longest_color = _learn_longest_color(train_examples)
    if longest_color is not None:
        progs.append(Program(
            f"recolor_longest_to_{longest_color}",
            wrap(make_recolor_longest(longest_color)),
        ))
    oe = _learn_oe_colors(train_examples)
    if oe is not None:
        odd_c, even_c = oe
        progs.append(Program(
            f"recolor_oe_odd{odd_c}_even{even_c}",
            wrap(make_recolor_oe(odd_c, even_c)),
        ))
    length_map = _learn_length_to_color(train_examples)
    if length_map is not None:
        progs.append(Program(
            f"recolor_by_length_{sorted(length_map.items())}",
            wrap(make_recolor_by_length(length_map)),
        ))

    return progs


def solve_task(task_data: dict) -> tuple[str, list[int]] | None:
    """Given a parsed 1D-ARC task JSON, return (program_name, predicted_test_output_row)
    or None if no program matches all training examples.
    """
    train = task_data.get("train", [])
    test = task_data.get("test", [])
    if not train or not test:
        return None
    train_pairs = [(grid_row(t["input"]), grid_row(t["output"])) for t in train]

    progs = candidate_programs(train_pairs)
    for prog in progs:
        try:
            ok = all(prog.apply(inp) == out for inp, out in train_pairs)
        except Exception:
            ok = False
        if ok:
            test_inp = grid_row(test[0]["input"])
            try:
                result = prog.apply(test_inp)
            except Exception:
                continue
            if result is None:
                continue
            return prog.name, result
    return None


def evaluate_directory(arc1d_root: Path) -> dict:
    """Run solver on every 1D-ARC task under root, return summary stats."""
    task_dirs = [d for d in arc1d_root.iterdir() if d.is_dir()]
    per_type: dict[str, list[bool]] = {}
    for task_dir in task_dirs:
        results = []
        for f in task_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            sol = solve_task(data)
            if sol is None:
                results.append(False)
                continue
            _, pred = sol
            # Compare to expected test output
            try:
                expected = grid_row(data["test"][0]["output"])
                results.append(pred == expected)
            except Exception:
                results.append(False)
        per_type[task_dir.name] = results
    return per_type
