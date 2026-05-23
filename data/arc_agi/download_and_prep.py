"""Download ARC-AGI training tasks (Chollet 2019).

The full 2D ARC-AGI benchmark — 400 training tasks, 400 evaluation tasks.
Each task is a JSON file with `train` and `test` lists; each list contains
{`input`, `output`} pairs where input/output are 2D colored grids (HxW, 0-9).

Source: https://github.com/fchollet/ARC-AGI

Usage:
  python data/arc_agi/download_and_prep.py
"""

from __future__ import annotations

from pathlib import Path
import requests


REPO = "fchollet/ARC-AGI"
RAW = f"https://raw.githubusercontent.com/{REPO}/master/data"
API = f"https://api.github.com/repos/{REPO}/contents/data"
HERE = Path(__file__).parent.resolve()


def _gh_list(rel: str):
    r = requests.get(f"{API}/{rel}", timeout=30)
    r.raise_for_status()
    return r.json()


def _fetch(rel: str, out: Path):
    if out.exists() and out.stat().st_size > 0:
        return
    r = requests.get(f"{RAW}/{rel}", timeout=30)
    r.raise_for_status()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)


def main() -> None:
    for split in ("training", "evaluation"):
        print(f"Fetching {split} split...")
        files = _gh_list(split)
        for entry in files:
            if entry["name"].endswith(".json"):
                out = HERE / split / entry["name"]
                _fetch(f"{split}/{entry['name']}", out)
        print(f"  {len(files)} tasks downloaded")
    print(f"Done. Stored under {HERE}.")


if __name__ == "__main__":
    main()
