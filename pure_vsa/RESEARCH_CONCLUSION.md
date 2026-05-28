# Hyperion ARC-AGI Research — Conclusion

**Period:** 2026-04 through 2026-05-25
**Hardware:** AMD Strix Halo (Radeon 8060S iGPU, no CUDA)
**Repo:** `~/projects/active/hyperion-release`

## Headline numbers (verified, reproducible)

| Benchmark | Method | Result | Status |
|---|---|---|---|
| SCAN (all 7 splits) | pure_vsa enumerative + grammar | **27,141 / 27,141 = 100%** | Paper-ready (`pure_vsa/PAPER.md`) |
| ARC-AGI 2D training (400) | pure_vsa enumerative solver | **18.75% (75/400)** | Best on-machine result |
| ARC-AGI 2D evaluation (400) | pure_vsa enumerative solver | **4.50% (18/400)** | Best on-machine result |

Repro:
```
pytest tests/test_scan_hyperion.py   # 100% SCAN
pytest tests/test_arc2d.py            # 18.5% train bar, 4.5% eval bar
```

## What worked

1. **VSA + enumerative DSL on bounded grammars.** SCAN is fully solved with 0
   trained parameters. This is a genuine, reportable result.
2. **Hand-grown ARC primitive library (~250 ops).** 22 iterations of failure
   clustering + targeted primitive additions got us from 0% to 18.75% on ARC
   training.
3. **Output-constraint backward filtering.** Learning legal output signatures
   from training pairs and pruning candidates pre-test pushed up the constraint
   layer's hit rate without raising compute.

## What didn't work (null results, all documented)

| Approach | Result | Why it failed | File |
|---|---|---|---|
| VSA-based analogical reasoning on ARC | 0 / 400 | Continuous hypervectors don't capture discrete grid semantics | `pure_vsa/arc2d_vsa.py` |
| Single-shot LLM proposer (qwen 7B, temp 0.2) | 0 / 18 sampled | No diversity, no DSL context | `pure_vsa/arc2d_llm_proposer.py` |
| AlphaCode high-N free-Python proposer, 7B | 0 / 5 in 80 samples | **Diagnostic showed 6/6 samples produced runnable code that got 0/3 training pairs right** — pure capability ceiling, not a sampling problem | `pure_vsa/arc2d_alphacode.py`, `pure_vsa/ALPHACODE_NULL_RESULT.md` |
| AlphaCode high-N free-Python, 30B | Infeasible | >10 min per sample on Vulkan; >36 days for one eval pass | same |
| Akyürek-style Test-Time Training (LoRA) | Not runnable | Requires CUDA; AMD iGPU + Windows has no path | `pure_vsa/arc2d_ttt.py` (code shipped, gated on hardware) |
| Smart-rank reranker on enum candidates | No movement past 18.75% | Bottleneck is "no matcher exists," not "wrong matcher chosen" | `analyze_failures.py` |

## The hardware reality

The Corsair AI Workstation in this configuration is AMD-based (Strix Halo APU,
Radeon 8060S iGPU, 4 GB VRAM, no NVIDIA card). All published ARC-AGI results
in the 50%+ range use one of:
- GPT-4o-tier or stronger external API
- Multi-A100/H100 CUDA training for TTT
- o3-tier reasoning at very high sample counts

None of those run on this box. That's not a limitation of the approach — it's
a hardware constraint, and any honest path past ~20% has to acknowledge it.

## What 18.75% / 4.50% actually means

- It is a fully-deterministic, 0-parameter, fully-explainable system. Every
  solved task ships with the exact program that solved it. No black box.
- It beats the *first generation* of published LLM-only ARC baselines (which
  hovered ~0-5% on the eval set without test-time training).
- It is **not** competitive with the current ARC Prize leaderboard, which is
  in the 50-87% regime via GPT-4o-class compute + TTT + ensembling.
- The compositional reasoning + SCAN result is the genuinely publishable
  contribution. The ARC numbers are an honest secondary result that
  demonstrates the approach scales beyond toy grammars but hits a capability
  wall without external models.

## Lanes left open (none pursued, all viable next steps)

1. **Restricted-choice DSL via local LLM.** Expose all 250 primitives as a
   typed menu; LLM picks compositions rather than writing code. Plausibly
   reaches 25-35% on this hardware. ~1 day build.
2. **Neural-guided enumerative search.** Tiny CPU scorer trained on solved
   tasks, prunes the program-synthesis search. DreamCoder pattern. ~3-5 days.
3. **Cloud GPU rental, $10 RunPod A100 for one TTT pass.** Published ceiling
   47-62%. The cheapest path to a real number.
4. **External API for AlphaCode (Claude/GPT).** ~$5-20 for one 400-task pass.
   Greenblatt recipe verbatim. Published ceiling ~50%.

## Recommended next move (if research is resumed)

Pursue Lane 3 ($10, 6 hours, $1M-prize-relevant numbers) before further
on-machine engineering. The on-machine work has demonstrated the
capability gap is real — the next dollar of investment should be on the
constraint that's actually binding.

## Files of record

- `pure_vsa/PAPER.md` — SCAN paper draft (100% result)
- `pure_vsa/ARC2D_BASELINE.md` — 22-iteration ARC improvement log
- `pure_vsa/ARC2D_FAILURE_CLUSTERS.md` — failure analysis (227 same-shape,
  67 smaller-output, etc.)
- `pure_vsa/ALPHACODE_NULL_RESULT.md` — local LLM AlphaCode null
- `pure_vsa/arc2d_solver.py` — the actual 18.75% solver (4,427 lines, 250
  primitives)
- `WHAT_WE_BUILT.md` — plain-English summary
- `AUDIT_REPORT.md` — five-phase adversarial audit (numbers verified)
