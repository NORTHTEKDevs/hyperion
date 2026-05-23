"""Independent evaluator — does NOT use the repo's coverage_stats or
evaluate_directory functions. Rebuilds accuracy from scratch.

If the repo's official evaluator is doing anything funny, this catches it."""

from __future__ import annotations

import json
import re
from pathlib import Path


# --- Independent reimplementation of normalize_output ---
def my_normalize(s: str) -> list[str]:
    return (
        s.replace("(", " ( ")
         .replace(")", " ) ")
         .replace(",", " , ")
         .replace(";", " ; ")
         .split()
    )


def my_load_cogs_tsv(path: Path) -> list[tuple[str, str, str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            out.append((parts[0], parts[1], parts[2]))
        elif len(parts) == 2:
            out.append((parts[0], parts[1], "?"))
    return out


def audit_cogs() -> None:
    from pure_vsa.cogs_template_learner import COGSTemplateLearner

    here = Path(__file__).parent
    train = my_load_cogs_tsv(here / "data/cogs/raw/train.tsv")
    gen = my_load_cogs_tsv(here / "data/cogs/raw/gen.tsv")

    print(f"  Loaded {len(train)} train / {len(gen)} gen rows (independent loader)")

    learner = COGSTemplateLearner()
    # Fit ONLY on train (no gen, no test)
    learner.fit([(t[0], t[1]) for t in train])

    # Independent eval — call predict per row, manually compare
    correct = 0
    total = len(gen)
    for inp, ground_truth, _cat in gen:
        pred = learner.predict(inp)  # returns list[str] or []
        expected = my_normalize(ground_truth)
        if pred == expected:
            correct += 1
    print(f"  COGS gen: {correct}/{total} = {correct/total*100:.4f}% (independent)")


def audit_arc1d() -> None:
    from pure_vsa.arc1d_solver import solve_task

    here = Path(__file__).parent
    arc_dir = here / "data/arc1d"
    correct = 0
    total = 0
    for task_dir in sorted(arc_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        for f in sorted(task_dir.glob("*.json")):
            data = json.loads(f.read_text())
            total += 1
            sol = solve_task(data)
            if sol is None:
                continue
            _, pred = sol
            try:
                expected = data["test"][0]["output"][0]
                if pred == expected:
                    correct += 1
            except Exception:
                pass
    print(f"  1D-ARC: {correct}/{total} = {correct/total*100:.4f}% (independent)")


if __name__ == "__main__":
    print("=== Independent audit ===\n")
    print("COGS:")
    audit_cogs()
    print("\n1D-ARC:")
    audit_arc1d()
