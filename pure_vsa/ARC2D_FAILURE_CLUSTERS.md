# ARC-AGI 2D failure cluster analysis (2026-05-24)

Run `python analyze_failures.py` to regenerate.

## Headline

- Solved: **53 / 400 = 13.2%** of training tasks (at time of analysis; current state may differ slightly)
- Failed: **347**

## Failures by input→output shape relation

| Shape relation | Failures | Notes |
|---|---|---|
| **same** | **227** | Biggest cluster. Output dimensions match input. |
| smaller | 67 | Output is smaller than input but not a clean divisor. Extraction tasks. |
| scale_up_2x2 | 14 | Each cell becomes a 2x2 block. Partially covered by `per_cell_substitute`. |
| to_1x1 | 5 | Output is a single cell. Counting/property tasks. |
| scale_up_3x3 | 4 | |
| scale_down_3x3 | 4 | |
| bigger / smaller / various_scale | 17 | Long tail of dimension-changing tasks. |

## Same-shape failures sub-clustered

By number of cells that change between input and output:

| Magnitude | Failures | Hypothesis |
|---|---|---|
| 1-3 cells (highly local) | 16 | Single-cell transformations. Needs `find-the-special-cell` primitives. |
| 4-10 cells (local) | 61 | Object-level changes. Needs per-object property reasoning. |
| **11-30 cells (medium)** | **98** | **Biggest sub-cluster.** Multi-object or region transformations. |
| 31+ cells (global) | 52 | Whole-grid transformations. Needs CA-style or whole-grid operations. |

By whether output introduces new colors:

| | Failures |
|---|---|
| Output introduces NEW colors | 91 |
| Output uses ONLY input colors | 136 |

The 136 "no new colors" tasks are pure rearrangements/recolorings within the input color palette. Common patterns expected: move objects, sort by property, reflect each object internally.

By presence of marker:

| | Failures |
|---|---|
| Input has marker (singleton-color cell) | 58 |
| No marker | 169 |

## Implications for primitive engineering

The current ~200-primitive library catches the "easy" patterns but misses:

1. **Medium-magnitude same-shape transformations (98 tasks).** These need multi-object reasoning that our object primitives don't currently capture (e.g., "merge adjacent objects of same color", "objects in same row become a single rectangle").
2. **Pure rearrangements (136 same-shape no-new-colors).** Object compaction, sorting, alignment — we have some primitives but they don't fire because the exact pattern differs from our hand-coded versions.
3. **Smaller-other extraction (67 tasks).** Output dimensions don't cleanly divide input. These need pattern matching: "find the sub-region of input that LOOKS like the expected output shape" — currently impossible without smarter search.

## What this argues for

The next 2-3 percent on ARC-AGI training requires **either**:

(a) **A much richer object-level DSL** — operations over scene-graphs (objects with properties, relationships, transformations). Hodel-style. Multi-day effort.

(b) **Smarter search** — instead of "first match wins", rank candidate programs by training-test feature alignment, then beam-search top-K. Could surface compositions and primitives that get blocked by earlier false-positives.

(c) **A categorically different mechanism** — like CA-rule induction was. The most promising candidates:
  - Object-pair relationship induction (`A is to B as C is to ?`)
  - Top-down hypothesis generation (given output shape, what input transformations could produce it?)
  - Constraint satisfaction (declare desired output properties, search backward)

Pure primitive grinding past 14-15% is unlikely to be efficient.
