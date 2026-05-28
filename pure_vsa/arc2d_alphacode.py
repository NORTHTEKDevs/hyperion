"""AlphaCode-style high-N free-Python proposer for ARC-AGI 2D tasks.

Why this exists
---------------
The single-shot LLM proposer (`arc2d_llm_proposer.py`) got 0/18 because temp=0.2
+ 1 sample = nearly deterministic with no exploration. The `solve_task_best_of_n`
in `arc2d_perception.py` uses a constrained JSON grammar that limits what the
model can express. This module is the third arm: free-Python, high-N, diverse
temperatures, diverse prompts, strict verifier — the published Greenblatt
recipe that hit 50% on ARC-Pub with GPT-4 at high N.

Hardware target: AMD Strix Halo + Ollama (Vulkan). No CUDA required.

Pipeline per task
-----------------
1. Skip if the enumerative solver already solved it (caller responsibility).
2. Sample N candidate `def transform(grid):` programs from Ollama, rotating
   prompt template and temperature for diversity.
3. For each candidate: exec in an isolated subprocess (timeout=2s) and verify
   it matches ALL training pairs exactly.
4. First verified candidate → apply to test input → return result.

The training pairs are a perfect filter; the LLM is a noisy generator.

Compute budget
--------------
- 7B model on Vulkan: ~5-10s per sample × 32 samples × ~325 misses ≈ 15-30h overnight.
- 30B model: ~30-60s per sample × 32 × ~325 ≈ multiple days.
Default: 7B with N=32 for tractable overnight runs.

CLI
---
  python -m pure_vsa.arc2d_alphacode --smoke           # 5 known failures
  python -m pure_vsa.arc2d_alphacode --eval --max-tasks 20
  python -m pure_vsa.arc2d_alphacode --eval --n-samples 64 --model 30b
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODELS = {
    "7b": "huihui_ai/qwen2.5-coder-abliterate:7b",
    "30b": "huihui_ai/qwen3-coder-abliterated:30b",
}
DEFAULT_MODEL = MODELS["7b"]
DEFAULT_N_SAMPLES = 32
SAMPLE_TIMEOUT_SEC = 90       # max wall-clock per LLM call (covers 7B)
EXEC_TIMEOUT_SEC = 2          # max wall-clock per candidate transform()
ARC_TRAIN_DIR = Path("data/arc_agi/training")
ARC_EVAL_DIR = Path("data/arc_agi/evaluation")


# ---------------------------------------------------------------------------
# Prompt templates — three flavors rotated across samples for diversity.
# ---------------------------------------------------------------------------

_PROMPT_RAW = """You are solving an ARC-AGI visual puzzle. Given training pairs, write a Python function `def transform(grid):` that maps each input to its expected output. The grid is a list of lists of ints 0-9 (0 is background).

Training pairs:
{examples}

Rules:
- Return a 2D list of ints.
- Must work for ALL training examples shown.
- Use only Python standard library. No imports of numpy/scipy/etc.
- Output ONLY the code inside a ```python block.
"""

_PROMPT_DSL_HINT = """You are solving an ARC-AGI puzzle. Write `def transform(grid):` that maps input to output.

Common ARC transformations to consider:
- Geometric: rotate 90/180/270, flip horizontal/vertical, transpose
- Color: recolor map, keep one color, zero out one color, swap two colors
- Object detection: find connected components of nonzero cells, count them, sort by size
- Tiling: tile output as HxW grid of input copies, mask by input cells
- Crop: extract bounding box of non-background pixels
- Symmetry: complete a symmetric pattern, mirror around an axis
- Cellular: each cell becomes a function of its neighbors (Conway-like rules)
- Marker fill: find marker color, fill area between or around it

Training pairs:
{examples}

Output ONLY ```python code that defines `transform(grid)`. No prose. Use standard library only.
"""

_PROMPT_FEWSHOT = """You are solving an ARC-AGI puzzle. Examples of correct transform() functions from other tasks:

{exemplars}

Now your task. Write `def transform(grid):` for these training pairs:
{examples}

Output ONLY ```python code. No imports beyond standard library.
"""

PROMPT_TEMPLATES = [_PROMPT_RAW, _PROMPT_DSL_HINT, _PROMPT_FEWSHOT]

# Temperature schedule across N samples. Cycles diversity 0.2 → 1.0.
def _temperature_for(idx: int) -> float:
    schedule = [0.2, 0.4, 0.6, 0.8, 1.0]
    return schedule[idx % len(schedule)]


# ---------------------------------------------------------------------------
# Few-shot exemplar cache. Hand-curated for now; covers common patterns the
# enumerative solver handles. Used only by the FEWSHOT prompt template.
# ---------------------------------------------------------------------------

_FEWSHOT_EXEMPLARS = [
    {
        "examples": "Input:  [[1,0],[0,1]]\n  Output: [[0,1],[1,0]]",
        "code": "def transform(grid):\n    return [row[::-1] for row in grid]",
    },
    {
        "examples": "Input:  [[2,2,0],[0,2,0],[0,0,0]]\n  Output: [[0,0,0],[0,2,0],[2,2,0]]",
        "code": "def transform(grid):\n    return grid[::-1]",
    },
    {
        "examples": "Input:  [[3,0,0],[0,3,0],[0,0,3]]\n  Output: [[5,0,0],[0,5,0],[0,0,5]]",
        "code": ("def transform(grid):\n"
                 "    return [[5 if c == 3 else c for c in row] for row in grid]"),
    },
    {
        "examples": "Input:  [[1,2],[3,4]]\n  Output: [[1,2,2,1],[3,4,4,3],[3,4,4,3],[1,2,2,1]]",
        "code": ("def transform(grid):\n"
                 "    h, w = len(grid), len(grid[0])\n"
                 "    out = [[0]*(2*w) for _ in range(2*h)]\n"
                 "    for r in range(h):\n"
                 "        for c in range(w):\n"
                 "            v = grid[r][c]\n"
                 "            out[r][c] = v\n"
                 "            out[r][2*w-1-c] = v\n"
                 "            out[2*h-1-r][c] = v\n"
                 "            out[2*h-1-r][2*w-1-c] = v\n"
                 "    return out"),
    },
]


def _format_exemplars() -> str:
    parts = []
    for i, ex in enumerate(_FEWSHOT_EXEMPLARS):
        parts.append(f"Task {i+1}:\n  {ex['examples']}\n  Solution:\n```python\n{ex['code']}\n```")
    return "\n\n".join(parts)


def _format_examples(train_pairs: list[tuple[list[list[int]], list[list[int]]]]) -> str:
    lines = []
    for i, (inp, out) in enumerate(train_pairs):
        lines.append(f"Example {i + 1}:")
        lines.append(f"  Input:  {inp}")
        lines.append(f"  Output: {out}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, model: str, temperature: float,
                 num_predict: int = 1024) -> str:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                    "top_p": 0.95,
                },
            },
            timeout=SAMPLE_TIMEOUT_SEC,
        )
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL)


def _extract_code(text: str) -> str | None:
    m = _CODE_RE.search(text)
    if m:
        code = m.group(1).strip()
        if "def transform" in code:
            return code
    # Fallback: raw response starts with def transform
    stripped = text.strip()
    if stripped.startswith("def transform"):
        return stripped
    return None


# ---------------------------------------------------------------------------
# Sandboxed execution via subprocess. Kills runaways on Windows reliably.
# ---------------------------------------------------------------------------

_RUNNER_TEMPLATE = """\
import json, sys, signal
USER_CODE = {code!r}
INPUTS = {inputs!r}
try:
    ns = {{}}
    exec(USER_CODE, ns)
    fn = ns.get('transform')
    if fn is None:
        print(json.dumps({{'error': 'no transform'}}))
        sys.exit(0)
    outs = []
    for inp in INPUTS:
        outs.append(fn(inp))
    print(json.dumps({{'outs': outs}}))
except Exception as e:
    print(json.dumps({{'error': str(e)[:200]}}))
"""


def _run_transform(code: str, inputs: list[list[list[int]]]) -> list[list[list[int]]] | None:
    """Execute the candidate code in a fresh subprocess. Returns list of outputs or None."""
    script = _RUNNER_TEMPLATE.format(code=code, inputs=inputs)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=EXEC_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return None
    if "error" in result:
        return None
    return result.get("outs")


def _verify_and_apply(code: str, train_pairs: list[tuple[list[list[int]], list[list[int]]]],
                      test_inp: list[list[int]]) -> list[list[int]] | None:
    """Run code on (all_train_inputs + test_inp) in one subprocess. If outputs
    match training, return the test output. Else None."""
    all_inputs = [tp[0] for tp in train_pairs] + [test_inp]
    outs = _run_transform(code, all_inputs)
    if outs is None or len(outs) != len(all_inputs):
        return None
    for i, (_, expected) in enumerate(train_pairs):
        if outs[i] != expected:
            return None
    test_out = outs[-1]
    if not isinstance(test_out, list) or not test_out or not isinstance(test_out[0], list):
        return None
    return test_out


# ---------------------------------------------------------------------------
# Per-task AlphaCode loop
# ---------------------------------------------------------------------------

def solve_task_alphacode(task_data: dict, n_samples: int = DEFAULT_N_SAMPLES,
                         model: str = DEFAULT_MODEL,
                         verbose: bool = False) -> tuple[str, list[list[int]]] | None:
    train = task_data.get("train", [])
    test = task_data.get("test", [])
    if not train or not test:
        return None
    train_pairs = [(t["input"], t["output"]) for t in train]
    test_inp = test[0]["input"]
    examples = _format_examples(train_pairs)
    exemplars = _format_exemplars()

    seen_codes: set[str] = set()
    for i in range(n_samples):
        template = PROMPT_TEMPLATES[i % len(PROMPT_TEMPLATES)]
        prompt = template.format(examples=examples, exemplars=exemplars)
        temp = _temperature_for(i)
        if verbose:
            print(f"  sample {i+1}/{n_samples} (template={i%3}, temp={temp:.1f})...", flush=True)
        t0 = time.time()
        resp = _call_ollama(prompt, model, temp)
        if verbose:
            print(f"    LLM {time.time()-t0:.1f}s, {len(resp)} chars", flush=True)
        code = _extract_code(resp)
        if code is None:
            continue
        if code in seen_codes:
            continue
        seen_codes.add(code)
        test_out = _verify_and_apply(code, train_pairs, test_inp)
        if test_out is not None:
            if verbose:
                print(f"  HIT on sample {i+1} (template={i%3}, temp={temp:.1f})", flush=True)
            return ("alphacode", test_out)
    return None


# ---------------------------------------------------------------------------
# Hybrid: enumerative → AlphaCode fallback
# ---------------------------------------------------------------------------

def solve_hybrid(task_data: dict, n_samples: int = DEFAULT_N_SAMPLES,
                 model: str = DEFAULT_MODEL, verbose: bool = False
                 ) -> tuple[str, list[list[int]]] | None:
    from pure_vsa.arc2d_solver import solve_task
    sol = solve_task(task_data)
    if sol is not None:
        return sol
    return solve_task_alphacode(task_data, n_samples=n_samples, model=model, verbose=verbose)


# ---------------------------------------------------------------------------
# Eval / smoke
# ---------------------------------------------------------------------------

def evaluate_directory(arc_root: Path, n_samples: int = DEFAULT_N_SAMPLES,
                       model: str = DEFAULT_MODEL, max_tasks: int | None = None,
                       skip_solved: bool = True, verbose: bool = True) -> dict:
    from pure_vsa.arc2d_solver import solve_task
    results = {"enum": 0, "alphacode": 0, "total": 0, "attempted_llm": 0, "failures": []}
    files = sorted(arc_root.glob("*.json"))
    if max_tasks:
        files = files[:max_tasks]
    for i, f in enumerate(files):
        data = json.loads(f.read_text())
        expected = data["test"][0]["output"]
        sol = solve_task(data)
        results["total"] += 1
        if sol is not None and sol[1] == expected:
            results["enum"] += 1
            if verbose:
                print(f"[{i+1}/{len(files)}] {f.name} ENUM ({sol[0]})", flush=True)
            continue
        if not skip_solved:
            pass
        results["attempted_llm"] += 1
        if verbose:
            print(f"[{i+1}/{len(files)}] {f.name} -> AlphaCode (N={n_samples})", flush=True)
        t0 = time.time()
        sol2 = solve_task_alphacode(data, n_samples=n_samples, model=model, verbose=verbose)
        elapsed = time.time() - t0
        if sol2 is not None and sol2[1] == expected:
            results["alphacode"] += 1
            if verbose:
                print(f"    -> ALPHACODE HIT ({elapsed:.1f}s)", flush=True)
        else:
            results["failures"].append(f.name)
            if verbose:
                print(f"    -> miss ({elapsed:.1f}s)", flush=True)
    return results


# Curated smoke set: 5 enum-misses from the training split.
_SMOKE_TASKS = [
    "007bbfb7.json",   # tile-by-input pattern
    "017c7c7b.json",   # tile vertically + recolor
    "025d127b.json",   # diagonal stripe extension
    "045e512c.json",   # multi-arm extension
    "0520fde7.json",   # split + overlay
]


def smoke(model: str = DEFAULT_MODEL, n_samples: int = 16) -> dict:
    """Run the AlphaCode proposer on 5 known enumerative misses. ~5-15 min on 7B."""
    if not ARC_TRAIN_DIR.exists():
        print(f"ARC training dir not found at {ARC_TRAIN_DIR}", flush=True)
        return {"error": "no data"}
    results = {"hits": 0, "tried": 0, "details": []}
    for name in _SMOKE_TASKS:
        path = ARC_TRAIN_DIR / name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        expected = data["test"][0]["output"]
        results["tried"] += 1
        print(f"\n=== {name} (N={n_samples}) ===", flush=True)
        t0 = time.time()
        sol = solve_task_alphacode(data, n_samples=n_samples, model=model, verbose=True)
        elapsed = time.time() - t0
        hit = sol is not None and sol[1] == expected
        if hit:
            results["hits"] += 1
        results["details"].append({"task": name, "hit": hit, "elapsed_sec": round(elapsed, 1)})
        print(f"=== {name}: {'HIT' if hit else 'miss'} ({elapsed:.1f}s) ===", flush=True)
    print(f"\nSmoke: {results['hits']}/{results['tried']} on N={n_samples}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    ap.add_argument("--model", default="7b", choices=list(MODELS.keys()))
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--split", default="training", choices=["training", "evaluation"])
    args = ap.parse_args()
    model = MODELS[args.model]
    if args.smoke:
        smoke(model=model, n_samples=args.n_samples)
        return
    if args.eval:
        root = ARC_TRAIN_DIR if args.split == "training" else ARC_EVAL_DIR
        out = evaluate_directory(root, n_samples=args.n_samples, model=model,
                                 max_tasks=args.max_tasks, verbose=True)
        print(json.dumps({k: v for k, v in out.items() if k != "failures"}, indent=2))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
