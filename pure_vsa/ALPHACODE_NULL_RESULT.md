# AlphaCode-style Free-Python Proposer — Null Result on Local Hardware

**Date:** 2026-05-25
**Hardware:** AMD Strix Halo (Radeon 8060S iGPU, 4 GB VRAM, no CUDA)
**Module:** `pure_vsa/arc2d_alphacode.py`

## What was tried

Greenblatt-style high-N free-Python proposer via Ollama:
- 3 rotating prompt templates (raw, DSL-hint, few-shot exemplars)
- Temperature schedule 0.2 → 1.0
- Subprocess sandbox per candidate (2s timeout, kills runaways)
- Strict verification: program must match ALL training pairs exactly
- Hybrid wrapper: enumerative-first, AlphaCode-fallback

## Results

### 7B (`huihui_ai/qwen2.5-coder-abliterate:7b`) on Vulkan
- Smoke: 5 tasks × 16 samples = 80 samples, 0 hits in ~13 min
- Diagnostic (1 task × 6 samples): **6/6 produced runnable code that got 0/3 training pairs right**
- Per-sample latency: 4-20 s

### 30B (`huihui_ai/qwen3-coder-abliterated:30b`) on Vulkan
- First sample timed out at 10 min wall-clock — infeasible per-task budget
- At even 10 min/sample × 16 samples × 325 enum-misses = 36+ days for ONE eval pass

## Conclusion

Local LLM free-Python sampling cannot reach the Greenblatt 50% regime on Strix
Halo. The 7B has insufficient capability (generates plausible-but-wrong code on
the first training pair), and the 30B has insufficient throughput (>10 min per
sample) regardless of capability.

The published AlphaCode-style results on ARC (Greenblatt 50%, MindsAI 55%,
o3 75-87%) used GPT-4o-tier or stronger models at high sample counts. Local
models in the 7-30B range, on integrated AMD GPUs without ROCm, do not
substitute.

## What this rules in / out

**Ruled out:**
- Free-Python proposer on local 7B (any N)
- Free-Python proposer on local 30B (any N) — throughput
- Akyürek TTT as written (needs CUDA)

**Still viable on this hardware:**
1. **Restricted-choice DSL composition** — expand `arc2d_perception.py`'s tiny
   JSON grammar to expose all ~250 primitives from `arc2d_solver.py` as named
   ops. LLM picks compositions instead of inventing code. Massively lower
   capability bar.
2. **Neural-guided enumerative search** — train a small CPU-feasible scorer to
   prune the program-synthesis search in `arc2d_solver.py`. DreamCoder pattern.
3. **Cloud GPU for one TTT pass** — ~$10 on RunPod A100 for ~6 hours.
4. **External API for AlphaCode** — Claude/GPT API at high N for ~$5-20 on a
   ~400-task eval. This is the Greenblatt recipe verbatim.

## Recommendation

Lane 1 (restricted-choice DSL) is the next move that runs entirely on this
hardware and has a credible capability story. The 7B can pick from a menu of
named ops even when it can't write code from scratch.
