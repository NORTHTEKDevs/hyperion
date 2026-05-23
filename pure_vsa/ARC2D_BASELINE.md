# 2D ARC-AGI baseline

**Current result:** the same enumerative program-synthesis approach that hits 100% on 1D-ARC, extended to 2D grids with a ~60-primitive library + composition, reaches **8.75% on the public training set (35 / 400)** and **0.75% on the held-out evaluation set (3 / 400)**.

This is a baseline, not a flagship. The point is to establish that the *mechanism* works on 2D and to measure how far an honest primitive library + composition gets you without specialized search heuristics.

## Iteration log

| Iteration | Library size | Training | Evaluation |
|---|---|---|---|
| 1. Simple geometric + recolor + shift | ~15 | 2.50% (10/400) | 0.00% (0/400) |
| 2. + tile, scale, crop-to-bbox, mirror-concat, fill-enclosed | ~25 | 5.00% (20/400) | 0.50% (2/400) |
| 3. + gravity, symmetry completion | ~30 | 6.00% (24/400) | 0.50% (2/400) |
| 4. + connected-components object reasoning | ~38 | 7.25% (29/400) | 0.75% (3/400) |
| 5. + composition (chain 2 primitives) | ~38 + composed | 7.75% (31/400) | 0.75% (3/400) |
| 6. + constant-output + per-color recolor map | ~40 | 8.25% (33/400) | 0.75% (3/400) |
| 7. + object-level coordinate transforms (translate to marker, stamp at markers, gravity-toward-color) | ~55 | 8.50% (34/400) | 0.75% (3/400) |
| 8. + fill-with-majority/minority-color (whole grid) | ~60 | **8.75% (35/400)** | **0.75% (3/400)** |

Per-program-family wins after the final iteration:

```
flip                 6    rotate180            2
recolor_map          4    rotate270            1
crop                 3    transpose            2
recolor              3    compose              2
scale_up             3    shift                1
flood_fill           2    count                1
gravity              2    tile                 1
symmetry             2    scale_down           1
```

## The library (~40 primitives)

**Geometric:** identity, flip_h, flip_v, rotate90/180/270, transpose, shift_{-3..3}.

**Recolor:** recolor_a_to_b (learned from training color swaps), recolor_all_nonzero_to_X, recolor_map (full per-color mapping induced from train).

**Scale / shape change:** tile_KxL, scale_up_KxL, scale_down_KxL, crop_to_bbox, crop_to_color_X_bbox, flip_h_concat_right, flip_v_concat_below.

**Selection:** keep_only_majority_color, keep_only_minority_color.

**Fill:** flood_fill_enclosed_X (color induced from output \ input).

**Gravity:** gravity_{up,down,left,right}.

**Symmetry:** complete_symmetry_{h,v,both}.

**Object-level (connected components, 4-neighbor):** keep_largest_object_{bycolor,any}, keep_smallest_object_{bycolor,any}, crop_to_largest_object_{bycolor,any}, recolor_by_size (size→color induced), count_objects_to_color (for 1x1 outputs).

**Constant:** constant_output_HxW (output is the same fixed grid as training).

**Composition:** chain 2 shape-preserving primitives; tried only when no single primitive matches.

All parameter values (color swaps, scale factors, fill colors, size→color maps) are induced from training pairs only. Audit-grade discipline: test outputs are used solely for the final `pred == expected` comparison, never during solving.

## What we are NOT doing (and what it would buy)

For context, here are the major categories of ARC primitives this library does not yet implement, and roughly what they'd add:

1. **Object-level coordinate transformations** (translate object to align with another, move object until touching a wall, gravity-toward-marker). Probably adds 4-8%.
2. **Per-object property-based selection** (find object with N cells, find rectangular object, find object touching the border). Probably adds 3-6%.
3. **Pattern matching** (find sub-pattern in input, copy it elsewhere). Adds 2-4%.
4. **Per-row/per-col operations** (sort rows by something, repeat top row, transpose specific rows). Adds 2-4%.
5. **Multi-step composition** (3+ primitives in sequence). Adds 1-3% but search blows up; needs pruning.
6. **Conditional rules** (if-then on a per-cell or per-object property). Adds 3-5%.

Realistic ceiling for an enumerative-search approach with ~150 primitives + 2-step composition: ~25-30% on training, much lower on evaluation. To exceed that requires either learned heuristics (which violates the "no neural training" principle) or a substantially smarter search procedure (Bayesian inference over programs, MCTS, etc.).

## How this compares to the field

- Hyperion baseline (this work): 8.25% / 0.75%
- Hodel ARC-DSL (~150 hand-coded primitives, enumerative search): ~30-40% training
- Best public ARC Prize submissions (LLM-based with extensive search): ~55-60%
- Human-level performance (the bar Chollet defined): ~85%

The 8.25% is honest. It's a real baseline, reproducible by running `evaluate_directory(Path('data/arc_agi/training'))`. It demonstrates the architecture transfers from 1D to 2D, and quantifies the gap that more primitive engineering would close.

## Honest scope

1. The library is hand-coded primitives. The system *learns* parameter values (which color maps to which, which size maps to which color) but not which primitives to include.
2. Composition is 2-deep only. Real ARC programs often need 3-5 primitives chained.
3. No search heuristics — pure enumeration in a fixed order. A smarter search (informed by training-pair shape, color, object count) would do better.
4. Object detection uses 4-neighbor only. Some tasks need 8-neighbor diagonal connectivity.

## Reproducing

```bash
python data/arc_agi/download_and_prep.py   # downloads 800 tasks (~5 min)
python -c "
from pathlib import Path
from pure_vsa.arc2d_solver import evaluate_directory
for split in ('training', 'evaluation'):
    r = evaluate_directory(Path(f'data/arc_agi/{split}'))
    c = sum(sum(rs) for rs in r.values()); t = len(r)
    print(f'{split}: {c}/{t} = {c/t*100:.2f}%')
"
```
