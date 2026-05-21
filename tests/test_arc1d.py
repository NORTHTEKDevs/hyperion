"""Tests for the minimal 1D-ARC solver (program-induction baseline)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pure_vsa.arc1d_solver import evaluate_directory, solve_task


REPO_ROOT = Path(__file__).resolve().parents[1]
ARC1D_DIR = REPO_ROOT / "data" / "arc1d"


def _data_available() -> bool:
    return ARC1D_DIR.exists() and any(ARC1D_DIR.iterdir())


def test_solve_task_shift_pattern():
    """Solver should handle a simple shift task."""
    task = {
        "train": [
            {"input": [[1, 2, 3, 0, 0]], "output": [[0, 1, 2, 3, 0]]},
            {"input": [[5, 6, 7, 0, 0]], "output": [[0, 5, 6, 7, 0]]},
        ],
        "test": [{"input": [[8, 9, 0, 0, 0]], "output": [[0, 8, 9, 0, 0]]}],
    }
    result = solve_task(task)
    assert result is not None
    name, pred = result
    assert pred == [0, 8, 9, 0, 0]


@pytest.mark.skipif(not _data_available(), reason="1D-ARC data not downloaded")
def test_arc1d_move_tasks_solved_perfectly():
    """The 3 simple-move task types should be solved at 100% by the shift program."""
    results = evaluate_directory(ARC1D_DIR)
    for task_type in ["1d_move_1p", "1d_move_2p", "1d_move_3p"]:
        if task_type not in results:
            continue
        rs = results[task_type]
        if not rs:
            continue
        acc = sum(rs) / len(rs)
        assert acc == 1.0, f"{task_type}: {sum(rs)}/{len(rs)} = {acc:.4f}"


@pytest.mark.skipif(not _data_available(), reason="1D-ARC data not downloaded")
def test_arc1d_overall_perfect():
    """All 18 task types should be solved by the program-synthesis library."""
    results = evaluate_directory(ARC1D_DIR)
    total_correct = sum(sum(rs) for rs in results.values())
    total = sum(len(rs) for rs in results.values())
    if total == 0:
        pytest.skip("no tasks found")
    acc = total_correct / total
    assert acc >= 0.99, f"overall {total_correct}/{total} = {acc:.4f}, expected >=99%"


@pytest.mark.skipif(not _data_available(), reason="1D-ARC data not downloaded")
def test_arc1d_each_task_type_perfect():
    """Each task type should hit 100% — fail loud on any regression."""
    results = evaluate_directory(ARC1D_DIR)
    for task_type, rs in results.items():
        if not rs:
            continue
        acc = sum(rs) / len(rs)
        assert acc == 1.0, f"{task_type}: {sum(rs)}/{len(rs)} = {acc:.4f}"
