# 2D ARC-AGI baseline

**Current result:** the same enumerative program-synthesis approach that hits 100% on 1D-ARC, extended to 2D grids with a ~155-primitive library + composition + cellular-automaton rule induction + input-property-to-output induction + per-cell-substitute + fill-between-markers, reaches **12.50% on the public training set (50 / 400)** and **1.75% on the held-out evaluation set (7 / 400)**.

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
| 9. + subgrid decomposition, detected-symmetry, 3-step composition, hole-fill | ~75 | 8.75% (35/400) | 0.75% (3/400) |
| 10. + cellular-automaton rule induction (neighbor signature, neighbor count) | ~78 | 9.75% (39/400) | 0.75% (3/400) |
| 11. + tiled-pattern completion, drawing/outline primitives, size-recolor, color permutation | ~90 | 10.25% (41/400) | 1.25% (5/400) |
| 12. + color-0-divider support, per-row/col uniformity, rectangle/bbox-fill, larger CA window | ~100 | 11.00% (44/400) | 1.25% (5/400) |
| 13. + per-object transforms, object filters, diagonal flips, line/cross drawing | ~125 | 11.00% (44/400) | 1.25% (5/400) |
| 14. + input-property-to-output induction (constrained: small outputs only, runs last) | ~135 | **11.75% (47/400)** | **1.50% (6/400)** |
| 15. + bbox-grid extraction, non-majority subgrid, iterated CA | ~140 | 11.75% (47/400) | 1.50% (6/400) |
| 16. + per-cell-substitute (each input cell -> learned KxL output block) | ~145 | 11.75% (47/400) | **1.75% (7/400)** |
| 17. + fill-between-same-color-markers, recolor-non-majority-nonzero | ~155 | **12.50% (50/400)** | 1.75% (7/400) |

**Plateau broken at iteration 10** by the CA-rule induction primitive (the first one that actually learns a per-cell transformation from training data instead of being a hand-coded geometric op). It's the most general primitive in the library and unlocked patterns the geometric primitives couldn't reach.

**Plateau reached at iteration 9 (now superseded).** Each of these additions is real scaffolding (subgrid extraction primitives, symmetry-axis detection, 3-step composition with a tight core library, "fix the broken cell" primitives), but none caught new tasks. The honest reading: the cheap-primitive approach has reached its ceiling. Further progress requires:

- A **substantially larger primitive library** (Hodel's ARC-DSL has ~150; we have ~75).
- **Pattern matching** as a first-class operation (detect a motif in input, match against a template).
- **Search heuristics** that guide enumeration by input properties (size, color count, object structure) rather than fixed order.
- **Per-cell neighbor rules** (cellular-automaton style transformations).

Each of these is multi-day engineering work, not a single-session add.

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
