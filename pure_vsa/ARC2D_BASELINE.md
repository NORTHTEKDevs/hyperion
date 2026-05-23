# 2D ARC-AGI baseline — initial port

**Result:** the same enumerative program-synthesis approach that hits 100% on 1D-ARC extends to 2D, but with a small primitive library only reaches **6.0% on the public training set (24 / 400)** and **0.5% on the held-out evaluation set (2 / 400)**.

This is a baseline, not a flagship result. The point is to establish that the *mechanism* works on 2D and to measure how far a simple primitive library gets you. Real ARC-AGI competitive systems use much larger DSLs with object-level reasoning, composition operators, and search heuristics.

## What was done

- Downloaded the official ARC-AGI training + evaluation sets (400 + 400 tasks) from `fchollet/ARC-AGI`.
- Ported the 1D solver architecture to 2D: same `solve_task(task) → (program_name, grid)` shape, same "enumerate programs, pick first that matches all training pairs" loop.
- Built a small primitive library covering geometric transforms, recolors, scaling, gravity, symmetry completion, and flood-fill of enclosed regions.

## The library (~25 primitives)

**Geometric:** identity, flip_h, flip_v, rotate90/180/270, transpose, shift_*.

**Recolor:** recolor_a_to_b (learned from training color swaps), recolor_all_nonzero_to_X.

**Scale:** tile_KxL, scale_up_KxL, scale_down_KxL (factor learned from train), crop_to_bbox, flip_h_concat_right, flip_v_concat_below.

**Selection:** keep_only_majority_color, keep_only_minority_color.

**Fill:** flood_fill_enclosed_X (fill color inferred as colors in output but not input).

**Gravity:** gravity_{up,down,left,right}.

**Symmetry:** complete_symmetry_{h,v,both}.

All parameter values (color swaps, scale factors, fill colors) are induced from training pairs only. The test output is used solely for the final `pred == expected` comparison.

## Honest numbers

| Split | Solved | Total | % |
|---|---|---|---|
| training | 24 | 400 | **6.00%** |
| evaluation (held out) | 2 | 400 | **0.50%** |

For reference, the **public state of the art** on ARC-AGI is around 55-60% (best public submissions to the ARC Prize, mostly LLM-based with extensive search). The Hodel-style hand-coded DSL approaches with ~150 primitives reach 30-40% on training. This 6% baseline with ~25 primitives is a starting point, not a destination.

## What would push it higher (in rough order of expected impact)

1. **Object-level reasoning** — connected-components extraction by color. Per-object operations: largest/smallest by area, move to center, move to alignment with marker, recolor by size. This alone probably gets 10-15% absolute on training.
2. **Composition** — chain primitives (e.g., `crop_to_bbox ∘ flip_h`). Currently the solver picks one primitive. A 2-step search over the library would multiply coverage.
3. **Pattern continuation** — many tasks involve extending a partial pattern (e.g., extending a line, completing a shape).
4. **Color-by-property** — recolor an object based on its size, position, or shape category.
5. **Mask operations** — overlay, AND, OR, XOR of binary masks.
6. **Larger primitive library** — Hodel's ARC-DSL has ~150 primitives. We have 25. Most of that gap is reachable mechanically (rote translation of his Python primitives into our library).

## Comparison to 1D-ARC

| Benchmark | Primitives | Result |
|---|---|---|
| 1D-ARC | 25 | 100% (901/901) |
| 2D ARC-AGI (training) | 25 | 6% (24/400) |
| 2D ARC-AGI (evaluation) | 25 | 0.5% (2/400) |

The same code architecture; the 2D version just doesn't have enough primitives yet. 1D-ARC was tractable with simple primitives because the search space is dramatically smaller (rows vs grids) and the patterns simpler. The gap is the primitive library, not the algorithm.

## Verdict

The architecture transfers cleanly. The path to a competitive 2D ARC-AGI result is a richer primitive library + composition operators + search budget — all of which is incremental engineering, not a fundamental rethink. Whether to invest that engineering effort is the open question.

## Reproducing

```bash
python data/arc_agi/download_and_prep.py   # ~5 minutes, downloads 800 tasks
python -c "
from pathlib import Path
from pure_vsa.arc2d_solver import evaluate_directory
for split in ('training', 'evaluation'):
    r = evaluate_directory(Path(f'data/arc_agi/{split}'))
    c = sum(sum(rs) for rs in r.values()); t = len(r)
    print(f'{split}: {c}/{t} = {c/t*100:.2f}%')
"
```
