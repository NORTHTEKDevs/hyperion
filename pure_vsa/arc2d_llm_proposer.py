"""Small-LLM-as-primitive-proposer for ARC tasks the enumerative solver missed.

Pipeline:
  1. For each unsolved task, show the training (input, output) pairs to a
     small local LLM via Ollama.
  2. Ask it to write a Python function `def transform(grid):` that maps input
     to output.
  3. Compile the function in a sandboxed namespace.
  4. Run it on every training input; reject if any output doesn't match.
  5. If all training matches, run on test input and return the answer.

The LLM never sees the test answer — only training pairs. This is the same
contract as the rest of the solver: extract a rule from training, apply to test.

Compute cost: tens of seconds per call on CPU, free. Local LLM (Ollama
serving qwen2.5-coder:7b or similar).
"""

from __future__ import annotations

import json
import re
import signal
import time
from pathlib import Path
from typing import Any

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "huihui_ai/qwen2.5-coder-abliterate:7b"
TIMEOUT_PER_TASK_SEC = 60  # max wall-clock per LLM call
TIMEOUT_PER_FN_SEC = 5     # max wall-clock for a single transform() call


PROMPT_TEMPLATE = """You are solving a visual abstract reasoning puzzle.

Given these training examples, write a Python function `def transform(grid):` that converts each input to its expected output. The grid is a list of lists of ints 0-9. The function must work for ALL training examples.

Training examples:
{examples}

Output ONLY the Python code for `transform`. No prose. The function must:
- Take a 2D list of ints as input
- Return a 2D list of ints
- Work for all training cases shown
- Use only Python standard library (no numpy, no imports)

Return your code in a ```python code block."""


def _format_examples(train_pairs: list[tuple[list[list[int]], list[list[int]]]]) -> str:
    lines = []
    for i, (inp, out) in enumerate(train_pairs):
        lines.append(f"Example {i + 1}:")
        lines.append(f"  Input:  {inp}")
        lines.append(f"  Output: {out}")
    return "\n".join(lines)


def _extract_python_code(text: str) -> str | None:
    """Extract the first python code block from the LLM response."""
    m = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: assume entire response is code
    m = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # If it just starts with `def transform`
    if text.strip().startswith("def transform"):
        return text.strip()
    return None


def _call_ollama(prompt: str, model: str) -> str:
    """Synchronous call to local Ollama. Returns generated text or empty on error."""
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 1024},
            },
            timeout=TIMEOUT_PER_TASK_SEC,
        )
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as e:
        print(f"  ollama error: {e}")
        return ""


def _compile_and_test(code: str, train_pairs: list[tuple[list[list[int]], list[list[int]]]]) -> Any | None:
    """Compile the code and verify it matches all training pairs.
    Returns the transform callable if it matches all training, else None."""
    namespace: dict[str, Any] = {}
    try:
        exec(code, namespace)
    except Exception:
        return None
    fn = namespace.get("transform")
    if fn is None or not callable(fn):
        return None
    # Verify on all training pairs
    for inp, out in train_pairs:
        try:
            # deepcopy input so the fn can't mutate ours
            import copy
            in_copy = copy.deepcopy(inp)
            got = fn(in_copy)
        except Exception:
            return None
        if got != out:
            return None
    return fn


def solve_task_with_llm(task_data: dict, model: str = DEFAULT_MODEL,
                        verbose: bool = False) -> tuple[str, list[list[int]]] | None:
    """Attempt to solve via LLM-generated transform function."""
    train = task_data.get("train", [])
    test = task_data.get("test", [])
    if not train or not test:
        return None
    train_pairs = [(t["input"], t["output"]) for t in train]
    test_inp = test[0]["input"]

    prompt = PROMPT_TEMPLATE.format(examples=_format_examples(train_pairs))
    if verbose:
        print(f"  asking {model}...")
    t0 = time.time()
    response = _call_ollama(prompt, model)
    if verbose:
        print(f"  got response in {time.time()-t0:.1f}s, {len(response)} chars")

    code = _extract_python_code(response)
    if code is None:
        return None

    fn = _compile_and_test(code, train_pairs)
    if fn is None:
        return None

    # Apply to test
    try:
        import copy
        result = fn(copy.deepcopy(test_inp))
    except Exception:
        return None

    return ("llm_proposer", result)


def evaluate_directory_llm(arc_root: Path, fallback_solver=None, model: str = DEFAULT_MODEL,
                            max_tasks: int | None = None, verbose: bool = True) -> dict:
    """For each task: first try fallback_solver (if provided), else LLM-only."""
    from pure_vsa.arc2d_solver import solve_task, grid_equal

    results: dict[str, list[bool]] = {}
    n = 0
    llm_attempted = 0
    llm_solved = 0
    for f in sorted(arc_root.glob("*.json")):
        if max_tasks and n >= max_tasks:
            break
        data = json.loads(f.read_text())

        # First try the enumerative solver
        sol = None
        if fallback_solver is not None:
            sol = fallback_solver(data)
        if sol is None or sol[1] != data["test"][0]["output"]:
            # Fallback failed; try LLM
            llm_attempted += 1
            if verbose:
                print(f"[{n+1}] {f.name}: trying LLM...")
            sol = solve_task_with_llm(data, model=model, verbose=verbose)
            if sol is not None and sol[1] == data["test"][0]["output"]:
                llm_solved += 1
                if verbose:
                    print(f"  → LLM solved!")

        if sol is None:
            results[f.name] = [False]
            continue
        results[f.name] = [sol[1] == data["test"][0]["output"]]
        n += 1
    if verbose:
        print(f"\nLLM: attempted {llm_attempted}, solved {llm_solved}")
    return results
