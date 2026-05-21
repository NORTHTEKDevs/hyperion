"""Download COGS (Kim & Linzen 2020) and tokenize.

COGS tests compositional generalization on natural-language-to-logical-form
parsing. Train (~24K examples) covers a fixed set of constructions; the
generalization split (~21K examples) contains novel compositions of the same
primitives. Transformers reach ~30-50% on the generalization split.

Source: https://github.com/najoungkim/COGS

Format: TSV, one example per line: <input>\\t<logical_form>\\t<category>

We reformulate as a decoder-only LM sequence:
  <BOS> <input tokens> <SEP> <logical-form tokens> <EOS>
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import requests


BASE_URL = "https://raw.githubusercontent.com/najoungkim/COGS/main/data"
HERE = Path(__file__).parent.resolve()

FILES = {
    "train": "train.tsv",
    "dev":   "dev.tsv",
    "test":  "test.tsv",
    "gen":   "gen.tsv",
}

SPECIALS = ["<PAD>", "<BOS>", "<SEP>", "<EOS>", "<UNK>"]


def _download(rel_url: str, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    url = f"{BASE_URL}/{rel_url}"
    print(f"Fetching {url} -> {out_path}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(r.text, encoding="utf-8")


def _parse(raw: str) -> list[tuple[list[str], list[str]]]:
    """COGS lines: input\\tlogical_form\\tcategory. We ignore category."""
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        in_str, out_str = parts[0], parts[1]
        # COGS logical forms use a custom syntax; split whitespace-then-punct.
        in_tokens = in_str.split()
        # tokenize the logical form by inserting spaces around brackets/commas/dots.
        lf = out_str
        for sym in [",", "(", ")", ".", ";"]:
            lf = lf.replace(sym, f" {sym} ")
        out_tokens = [t for t in lf.split() if t]
        out.append((in_tokens, out_tokens))
    return out


def _build_vocab(examples: list[tuple[list[str], list[str]]]) -> dict[str, int]:
    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    next_id = len(vocab)
    for in_tok, out_tok in examples:
        for t in in_tok + out_tok:
            if t not in vocab:
                vocab[t] = next_id
                next_id += 1
    return vocab


def _encode(examples: list[tuple[list[str], list[str]]], vocab: dict[str, int]) -> list[list[int]]:
    bos, sep, eos, unk = vocab["<BOS>"], vocab["<SEP>"], vocab["<EOS>"], vocab["<UNK>"]
    seqs = []
    for in_tok, out_tok in examples:
        ids = [bos]
        ids += [vocab.get(t, unk) for t in in_tok]
        ids.append(sep)
        ids += [vocab.get(t, unk) for t in out_tok]
        ids.append(eos)
        seqs.append(ids)
    return seqs


def _pad_and_pack(seqs: list[list[int]], pad_id: int, max_len: int) -> np.ndarray:
    arr = np.full((len(seqs), max_len), pad_id, dtype=np.uint16)
    for i, s in enumerate(seqs):
        L = min(len(s), max_len)
        arr[i, :L] = s[:L]
    return arr


def main() -> None:
    raw_dir = HERE / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, rel in FILES.items():
        _download(rel, raw_dir / rel)

    parsed = {
        name: _parse((raw_dir / rel).read_text(encoding="utf-8"))
        for name, rel in FILES.items()
    }
    all_examples = []
    for ex_list in parsed.values():
        all_examples.extend(ex_list)
    vocab = _build_vocab(all_examples)
    print(f"COGS vocab size: {len(vocab)}")

    max_len = max(
        max(len(s) for s in _encode(parsed[name], vocab))
        for name in FILES
    )
    print(f"COGS max sequence length: {max_len}")
    pad_id = vocab["<PAD>"]

    for name in FILES:
        seqs = _encode(parsed[name], vocab)
        arr = _pad_and_pack(seqs, pad_id=pad_id, max_len=max_len)
        arr.tofile(HERE / f"{name}.bin")
        print(f"  {name}: {arr.shape[0]:,} examples")

    meta = {
        "vocab": vocab,
        "vocab_size": len(vocab),
        "max_len": int(max_len),
        "pad_id": int(pad_id),
        "bos_id": int(vocab["<BOS>"]),
        "sep_id": int(vocab["<SEP>"]),
        "eos_id": int(vocab["<EOS>"]),
        "splits": {name: int(len(parsed[name])) for name in FILES},
    }
    (HERE / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
