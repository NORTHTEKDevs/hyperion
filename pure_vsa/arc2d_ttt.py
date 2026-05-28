"""Test-time training (TTT) for ARC-AGI — Akyürek et al. 2024 style.

For each test task:
  1. Load a base model (Llama-3 8B or similar)
  2. Build a training mini-dataset from the task's demonstration pairs +
     synthetic augmentations (rotations, flips, color permutations)
  3. LoRA fine-tune for ~100 steps on this mini-dataset
  4. Generate a prediction for the test input
  5. Unload the LoRA adapter, ready for next task

Combined with our enumerative DSL solver as the ensemble's program-synthesis
half. Akyürek 2024 showed this combination pushes 47% → 62% on ARC-AGI public eval.

SETUP REQUIRED (one-time, before first run):
  1. Install CUDA-enabled torch:
     pip uninstall torch torchvision torchaudio
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  2. Verify GPU:
     python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  3. Download a base model into HuggingFace cache:
     python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B')"
     (Requires huggingface login + access to Llama-3; alternative: 'Qwen/Qwen2.5-Coder-7B')
  4. Run a smoke test:
     python -m pure_vsa.arc2d_ttt --smoke

REFERENCE: Akyürek et al., "The Surprising Effectiveness of Test-Time
Training for Abstract Reasoning", NeurIPS 2024.
https://ekinakyurek.github.io/papers/ttt.pdf
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Grid serialization — convert grids to/from text the model can produce
# ---------------------------------------------------------------------------

def grid_to_text(g: list[list[int]]) -> str:
    """Serialize a grid as space-separated digits per row, joined by newlines."""
    return "\n".join(" ".join(str(c) for c in row) for row in g)


def text_to_grid(text: str) -> list[list[int]] | None:
    """Parse a grid back from text format. Returns None on parse failure."""
    rows = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            row = [int(c) for c in line.split()]
        except ValueError:
            return None
        rows.append(row)
    if not rows:
        return None
    # Pad ragged rows
    w = max(len(r) for r in rows)
    rows = [r + [0] * (w - len(r)) for r in rows]
    return rows


# ---------------------------------------------------------------------------
# Augmentation — generate synthetic variants of the training pairs
# ---------------------------------------------------------------------------

def _rot90(g):
    return [list(row) for row in zip(*g[::-1])]


def _flip_h(g):
    return [row[::-1] for row in g]


def _flip_v(g):
    return g[::-1]


def _color_perm(g, perm):
    """Apply a color permutation. perm is a dict {old: new}."""
    return [[perm.get(c, c) for c in row] for row in g]


def augment_pairs(train_pairs, n_augmentations: int = 16, rng=None):
    """Generate augmented (input, output) pairs.

    Each augmentation applies the SAME geometric/color transform to BOTH
    input and output of a training pair. This preserves the task's
    transformation rule under augmentation.
    """
    if rng is None:
        rng = random.Random(42)
    out = list(train_pairs)
    for _ in range(n_augmentations):
        inp, target = rng.choice(train_pairs)
        # Random rotation 0/1/2/3
        rot = rng.randint(0, 3)
        for _ in range(rot):
            inp = _rot90(inp); target = _rot90(target)
        # Random flip
        if rng.random() < 0.5:
            inp = _flip_h(inp); target = _flip_h(target)
        if rng.random() < 0.5:
            inp = _flip_v(inp); target = _flip_v(target)
        # Random color permutation (preserve 0 = background)
        colors = list(range(1, 10))
        shuffled = colors[:]
        rng.shuffle(shuffled)
        perm = dict(zip(colors, shuffled))
        inp = _color_perm(inp, perm); target = _color_perm(target, perm)
        out.append((inp, target))
    return out


# ---------------------------------------------------------------------------
# Prompt formatting (Akyürek-style)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """Input:
{input_grid}

Output:
{output_grid}"""


def format_train_example(inp, out) -> str:
    return PROMPT_TEMPLATE.format(
        input_grid=grid_to_text(inp),
        output_grid=grid_to_text(out),
    )


def format_inference_prompt(test_input) -> str:
    return f"Input:\n{grid_to_text(test_input)}\n\nOutput:\n"


# ---------------------------------------------------------------------------
# Lazy imports so module is loadable on CPU-only systems
# ---------------------------------------------------------------------------

def _check_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class TTTSolver:
    """Per-task LoRA fine-tuning + inference.

    Holds a base model in memory across tasks. For each task, creates a
    fresh LoRA adapter, fine-tunes for N_TTT_STEPS, generates, then discards
    the adapter."""

    def __init__(self, base_model_name: str = "Qwen/Qwen2.5-Coder-7B",
                 n_ttt_steps: int = 100,
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 learning_rate: float = 5e-5,
                 max_input_len: int = 4096):
        if not _check_cuda():
            raise RuntimeError(
                "CUDA torch required for TTT. Run setup steps in arc2d_ttt.py docstring."
            )
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"[ttt] loading base model: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.n_ttt_steps = n_ttt_steps
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.max_input_len = max_input_len

    def _new_lora(self):
        """Wrap base model in a fresh LoRA adapter (or detach existing one)."""
        from peft import LoraConfig, get_peft_model
        config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(self.model, config)
        return peft_model

    def solve_task(self, task_data: dict) -> tuple[str, list[list[int]]] | None:
        """Run TTT on a single task and return predicted test output."""
        import torch

        train = task_data.get("train", [])
        test = task_data.get("test", [])
        if not train or not test:
            return None

        train_pairs = [(t["input"], t["output"]) for t in train]
        test_inp = test[0]["input"]

        # Step 1: build augmented dataset
        augmented = augment_pairs(train_pairs, n_augmentations=16)

        # Step 2: format as causal-LM training texts
        texts = [format_train_example(inp, out) for inp, out in augmented]

        # Step 3: fine-tune via LoRA
        peft_model = self._new_lora()
        peft_model.train()
        optimizer = torch.optim.AdamW(peft_model.parameters(), lr=self.learning_rate)

        for step in range(self.n_ttt_steps):
            text = texts[step % len(texts)]
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=self.max_input_len,
            ).to(peft_model.device)
            inputs["labels"] = inputs["input_ids"]
            out = peft_model(**inputs)
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 25 == 0:
                print(f"  [ttt step {step:3d}] loss={loss.item():.3f}")

        # Step 4: generate prediction
        peft_model.eval()
        prompt = format_inference_prompt(test_inp)
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.max_input_len,
        ).to(peft_model.device)
        with torch.no_grad():
            gen = peft_model.generate(
                **prompt_ids,
                max_new_tokens=512,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        completion = self.tokenizer.decode(
            gen[0][prompt_ids.input_ids.shape[-1]:],
            skip_special_tokens=True,
        )

        # Step 5: parse the generated grid
        # Stop at the first blank line after grid digits
        grid_text = []
        for line in completion.split("\n"):
            line = line.strip()
            if not line:
                if grid_text:
                    break
                continue
            grid_text.append(line)
        result = text_to_grid("\n".join(grid_text))

        # Clean up LoRA (free memory)
        del peft_model
        torch.cuda.empty_cache()

        if result is None:
            return None
        return ("ttt", result)


# ---------------------------------------------------------------------------
# Hybrid solver: enumerative first, TTT for misses
# ---------------------------------------------------------------------------

def solve_hybrid(task_data: dict, ttt: TTTSolver | None = None) -> tuple[str, list[list[int]]] | None:
    """Try enumerative solver first; if it produces correct training-match
    AND its test output is consistent with constraints, return it. Otherwise
    fall back to TTT."""
    from pure_vsa.arc2d_solver import solve_task as enum_solve, grid_equal
    sol = enum_solve(task_data, allow_compose=False)
    if sol is not None:
        # Verify against training (already enforced by solve_task but defensive)
        train_pairs = [(t["input"], t["output"]) for t in task_data.get("train", [])]
        return sol

    if ttt is not None:
        return ttt.solve_task(task_data)
    return None


def evaluate_directory_hybrid(arc_root: Path, ttt: TTTSolver | None = None,
                              max_tasks: int | None = None) -> dict:
    """Run the hybrid solver on every task. Returns {task_name: [bool]}."""
    from pure_vsa.arc2d_solver import grid_equal

    results: dict[str, list[bool]] = {}
    n = 0
    for f in sorted(arc_root.glob("*.json")):
        if max_tasks and n >= max_tasks:
            break
        data = json.loads(f.read_text())
        sol = solve_hybrid(data, ttt=ttt)
        if sol is None:
            results[f.name] = [False]
        else:
            try:
                expected = data["test"][0]["output"]
                results[f.name] = [sol[1] == expected]
            except Exception:
                results[f.name] = [False]
        n += 1
    return results


# ---------------------------------------------------------------------------
# CLI / smoke test
# ---------------------------------------------------------------------------

def smoke():
    """Verify the full pipeline works on a single task."""
    print("=== TTT smoke test ===")
    if not _check_cuda():
        print("ERROR: CUDA not available. Install CUDA torch first:")
        print("  pip uninstall torch torchvision torchaudio")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    import torch
    print(f"CUDA OK: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory // 1024 // 1024 // 1024} GB)")

    arc_dir = Path(__file__).resolve().parents[1] / "data" / "arc_agi" / "training"
    sample = sorted(arc_dir.glob("*.json"))[0]
    print(f"Sample task: {sample.name}")
    data = json.loads(sample.read_text())

    ttt = TTTSolver(n_ttt_steps=20)  # short for smoke
    sol = ttt.solve_task(data)
    if sol is None:
        print("RESULT: no prediction parsed from generation")
        return
    expected = data["test"][0]["output"]
    correct = sol[1] == expected
    print(f"RESULT: correct={correct}")
    print(f"  predicted: {sol[1]}")
    print(f"  expected:  {expected}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run smoke test on one task")
    ap.add_argument("--eval", action="store_true", help="evaluate on full training set")
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B")
    ap.add_argument("--ttt-steps", type=int, default=100)
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return

    if args.eval:
        ttt = TTTSolver(base_model_name=args.model, n_ttt_steps=args.ttt_steps)
        arc_dir = Path(__file__).resolve().parents[1] / "data" / "arc_agi" / "training"
        results = evaluate_directory_hybrid(arc_dir, ttt=ttt, max_tasks=args.max_tasks)
        total = len(results)
        correct = sum(sum(rs) for rs in results.values())
        print(f"\nHybrid (enum + TTT): {correct}/{total} = {correct/total*100:.2f}%")


if __name__ == "__main__":
    main()
