"""Perturbation tests. If the system is honest, deliberately wrong
ground-truth labels should drop accuracy to ~0.

Tests:
  T1. Shuffle the COGS gen ground-truth outputs randomly across rows.
      Honest system: accuracy ~= 1/N (random chance, basically 0).
      Cheating system: stays high.

  T2. Substitute random unseen proper nouns in COGS gen inputs.
      Honest system: still produces correct output (just with the new name).
      Cheating system: degrades (because it memorized specific names).

  T3. Recolor 1D-ARC test inputs with a never-seen color.
      Honest system: should still solve (color is a parameter).
      Cheating system: might fail if it memorized specific color sequences.

  T4. Random sample of 1D-ARC tasks where we swap test input <-> a random
      OTHER task's test input. Honest system: should fail (since the wrong
      input goes against the wrong rule).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from pure_vsa.cogs_template_learner import COGSTemplateLearner
from pure_vsa.cogs_hyperion import load_cogs_tsv
from pure_vsa.arc1d_solver import solve_task


HERE = Path(__file__).parent


def my_normalize(s: str) -> list[str]:
    return (
        s.replace("(", " ( ")
         .replace(")", " ) ")
         .replace(",", " , ")
         .replace(";", " ; ")
         .split()
    )


def t1_shuffle_cogs_outputs() -> None:
    train = load_cogs_tsv(HERE / "data/cogs/raw/train.tsv")
    gen = load_cogs_tsv(HERE / "data/cogs/raw/gen.tsv")
    learner = COGSTemplateLearner()
    learner.fit([(t[0], t[1]) for t in train])

    rng = random.Random(42)
    outputs = [g[1] for g in gen]
    shuffled_outputs = outputs.copy()
    rng.shuffle(shuffled_outputs)

    correct = 0
    for (inp, _real_out, _cat), shuffled_out in zip(gen, shuffled_outputs):
        pred = learner.predict(inp)
        if pred == my_normalize(shuffled_out):
            correct += 1
    print(f"  T1 COGS shuffled ground truth: {correct}/{len(gen)} = {correct/len(gen)*100:.4f}%")
    print(f"     Expected: << 1%. If it's high, something is wrong.")


def t2_substitute_proper_nouns_cogs() -> None:
    import re
    PROP = re.compile(r"^[A-Z][a-z]+$")
    train = load_cogs_tsv(HERE / "data/cogs/raw/train.tsv")
    gen = load_cogs_tsv(HERE / "data/cogs/raw/gen.tsv")
    learner = COGSTemplateLearner()
    learner.fit([(t[0], t[1]) for t in train])

    # Pick a never-seen proper noun
    seen_props = set()
    for t in train:
        for tok in t[0].split():
            if PROP.match(tok) and tok not in {"A", "The"}:
                seen_props.add(tok)

    novel = "Zelldakor"  # made up
    assert novel not in seen_props, "novel name accidentally in train"

    correct = 0
    swapped = 0
    for inp, gt, _cat in gen:
        toks = inp.split()
        # pick the first proper noun in the input
        replaced = False
        for i, t in enumerate(toks):
            if PROP.match(t) and t not in {"A", "The"} and not replaced:
                old = t
                toks[i] = novel
                # also replace in expected output
                replaced = True
                old_name = old
                break
        if not replaced:
            continue
        swapped += 1
        new_inp = " ".join(toks)
        new_gt = gt.replace(old_name, novel)
        pred = learner.predict(new_inp)
        if pred == my_normalize(new_gt):
            correct += 1
    print(f"  T2 COGS with novel proper noun '{novel}': {correct}/{swapped} = {correct/max(1,swapped)*100:.4f}%")
    print(f"     Honest system should be ~= original accuracy (system handles names by COPY slot, not memorization).")


def t3_recolor_arc1d() -> None:
    arc_dir = HERE / "data/arc1d"
    rng = random.Random(7)

    novel_color = 99  # 1D-ARC uses 0-9. 99 is novel.

    # Sample 50 tasks, recolor all non-zero cells to novel_color in BOTH train and test
    # The rule should still work (it's color-agnostic for most types) or fail cleanly.
    sampled = 0
    correct = 0
    for task_dir in sorted(arc_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        tasks = list(task_dir.glob("*.json"))
        rng.shuffle(tasks)
        for f in tasks[:3]:
            data = json.loads(f.read_text())

            def recolor(grid):
                return [[novel_color if c != 0 else 0 for c in row] for row in grid]

            # Need to recolor the train pairs too so the rule is still detectable
            for pair in data["train"]:
                pair["input"] = recolor(pair["input"])
                pair["output"] = recolor(pair["output"])
            data["test"][0]["input"] = recolor(data["test"][0]["input"])
            data["test"][0]["output"] = recolor(data["test"][0]["output"])

            sampled += 1
            sol = solve_task(data)
            if sol is not None:
                _, pred = sol
                if pred == data["test"][0]["output"][0]:
                    correct += 1
    print(f"  T3 1D-ARC with novel color 99: {correct}/{sampled} = {correct/max(1,sampled)*100:.4f}%")
    print(f"     Honest system should be close to original (color is a parameter, not memorized).")


def t4_swap_arc1d_test_inputs() -> None:
    """Swap test inputs across tasks: take task A's training data + task B's
    test input. Honest system: should fail (A's rule applied to B's input,
    different ground truth). If accuracy stays near 100%, system is cheating
    (e.g., somehow looking up by file name)."""
    arc_dir = HERE / "data/arc1d"
    all_tasks = []
    for task_dir in sorted(arc_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        for f in sorted(task_dir.glob("*.json")):
            all_tasks.append(json.loads(f.read_text()))

    rng = random.Random(2026)
    rng.shuffle(all_tasks)
    n = len(all_tasks)

    # Pair adjacent: task_i gets task_{(i+1)%n}'s test
    correct = 0
    for i in range(n):
        a = all_tasks[i]
        b = all_tasks[(i + 1) % n]
        mutated = {
            "train": a["train"],
            "test": [{"input": b["test"][0]["input"]}],
        }
        sol = solve_task(mutated)
        if sol is not None:
            _, pred = sol
            # compare to A's expected output (the rule from A applied to B's input)
            # — but we don't know what A's rule applied to B's input should produce.
            # Instead check: does it match A's original test output? (No, that was for A's input.)
            # The honest expectation: pred is DIFFERENT from B's expected output (different rule)
            # AND different from A's original test output (different input).
            # We measure: how often does pred coincidentally match B's expected? Should be near 0.
            b_expected = b["test"][0]["output"][0]
            if pred == b_expected:
                correct += 1
    print(f"  T4 1D-ARC with swapped test inputs (task A's rule on task B's input): {correct}/{n} matched task B's expected output")
    print(f"     Expected: near 0. The rule is A's, so applying it to B's input should rarely produce B's expected output.")


if __name__ == "__main__":
    print("=== Perturbation tests (adversarial) ===\n")
    print("T1 — shuffled COGS ground-truth labels:")
    t1_shuffle_cogs_outputs()
    print("\nT2 — novel proper noun substitution:")
    t2_substitute_proper_nouns_cogs()
    print("\nT3 — 1D-ARC with novel color:")
    t3_recolor_arc1d()
    print("\nT4 — 1D-ARC with swapped test inputs:")
    t4_swap_arc1d_test_inputs()
