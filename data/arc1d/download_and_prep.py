"""Download 1D-ARC (Khalil et al. 2024) task JSONs.

1D-ARC is a one-dimensional simplification of the ARC-AGI benchmark, designed
to make compositional-reasoning challenges tractable while preserving the
core difficulty. 18 task types, 50 tasks each (~900 total).

Source: https://github.com/khalil-research/1D-ARC

Each task is a JSON file with `train` and `test` lists; each list contains
{`input`, `output`} pairs where input/output are 1xN colored grids.

Usage:
  python data/arc1d/download_and_prep.py

Files end up at data/arc1d/<task_type>/<task_type>_<idx>.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests


REPO = "khalil-research/1D-ARC"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/main/dataset"
API_BASE = f"https://api.github.com/repos/{REPO}/contents/dataset"

HERE = Path(__file__).parent.resolve()


def _gh_list(rel_path: str) -> list[dict]:
    url = f"{API_BASE}/{rel_path}" if rel_path else API_BASE
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def _download_file(rel_path: str, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    url = f"{RAW_BASE}/{rel_path}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(r.content)


def main() -> None:
    print(f"Fetching task-type list from {REPO} ...")
    top_entries = _gh_list("")
    task_dirs = [e for e in top_entries if e.get("type") == "dir" and e["name"].startswith("1d_")]
    print(f"Found {len(task_dirs)} task types.")

    total = 0
    for entry in task_dirs:
        tt = entry["name"]
        print(f"  - {tt} ...", end="", flush=True)
        files = _gh_list(tt)
        for f in files:
            if f["name"].endswith(".json"):
                out = HERE / tt / f["name"]
                _download_file(f"{tt}/{f['name']}", out)
                total += 1
        print(f" {len([f for f in files if f['name'].endswith('.json')])} tasks")

    print(f"\nDone. Downloaded {total} task files to {HERE}.")


if __name__ == "__main__":
    main()
