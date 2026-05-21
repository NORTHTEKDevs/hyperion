"""Download PCFG SET (Hupkes et al. 2020).

PCFG SET tests compositional generalization on nested string-edit operations
like `reverse(copy(append(...)))`. 10 primitive operations, nested up to depth 8.

Source: https://github.com/i-machine-think/am-i-compositional

Format: parallel src / tgt files. We mirror the upstream layout under
data/pcfg/.

Usage:
  python data/pcfg/download_and_prep.py
"""

from __future__ import annotations

from pathlib import Path

import requests


BASE_URL = "https://raw.githubusercontent.com/i-machine-think/am-i-compositional/master/data/pcfgset/pcfgset"
HERE = Path(__file__).parent.resolve()

FILES = [
    ("train.src", "train.src"),
    ("train.tgt", "train.tgt"),
    ("dev.src",   "dev.src"),
    ("dev.tgt",   "dev.tgt"),
    ("test.src",  "test.src"),
    ("test.tgt",  "test.tgt"),
]


def _download(rel_url: str, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    url = f"{BASE_URL}/{rel_url}"
    print(f"Fetching {url} -> {out_path.name}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    out_path.write_text(r.text, encoding="utf-8")


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    for rel, out in FILES:
        _download(rel, HERE / out)
    print("Done.")


if __name__ == "__main__":
    main()
