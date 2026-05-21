"""Download SCAN compositional generalization splits and tokenize them.

SCAN (Lake & Baroni 2018, https://github.com/brendenlake/SCAN) tests
systematic compositionality. The "add-primitive jump" split is THE test:
trains on all examples WITHOUT 'jump', evaluates on all examples WITH 'jump'.
Transformers get ~2% accuracy. HYMN's target is >60% (per v1.1 audit).

We download:
  - simple_split:        random 80/20 (sanity check)
  - add_prim_split/jump: systematic compositional split (the real test)
  - length_split:        longer test sequences than train (the other real test)

Each example: "IN: <command> OUT: <action sequence>" on one line.
We reformat to a single decoder-only sequence:
  <BOS> <command tokens> <SEP> <action tokens> <EOS>
so an autoregressive LM can be trained on it directly.

Writes:
  data/scan/<split_name>/train.bin, test.bin (uint16 token ids)
  data/scan/<split_name>/meta.json (vocab, max lengths)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import requests


BASE_URL = "https://raw.githubusercontent.com/brendenlake/SCAN/master"
HERE = Path(__file__).parent.resolve()

SPLITS = {
    "simple": {
        "train": "simple_split/tasks_train_simple.txt",
        "test":  "simple_split/tasks_test_simple.txt",
    },
    "addprim_jump": {
        "train": "add_prim_split/tasks_train_addprim_jump.txt",
        "test":  "add_prim_split/tasks_test_addprim_jump.txt",
    },
    "length": {
        "train": "length_split/tasks_train_length.txt",
        "test":  "length_split/tasks_test_length.txt",
    },
}

# Special tokens (low IDs so they don't collide with the small natural vocab).
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


def _parse_lines(raw: str) -> list[tuple[list[str], list[str]]]:
    """Parse 'IN: a b c OUT: X Y Z' into (input_tokens, output_tokens) pairs."""
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if "IN:" not in line or "OUT:" not in line:
            continue
        # split on OUT: first to avoid collisions with arbitrary tokens
        in_part, out_part = line.split("OUT:", 1)
        in_tokens = in_part.replace("IN:", "").strip().split()
        out_tokens = out_part.strip().split()
        out.append((in_tokens, out_tokens))
    return out


def _build_vocab(examples: list[tuple[list[str], list[str]]]) -> dict[str, int]:
    """Build a unified vocab. Input and output tokens share a namespace; specials first."""
    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    next_id = len(vocab)
    for in_tok, out_tok in examples:
        for t in in_tok + out_tok:
            if t not in vocab:
                vocab[t] = next_id
                next_id += 1
    return vocab


def _encode(examples: list[tuple[list[str], list[str]]], vocab: dict[str, int]) -> list[list[int]]:
    """Encode each example as <BOS> in_tokens <SEP> out_tokens <EOS>."""
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


def _pad_and_pack(seqs: list[list[int]], pad_id: int, max_len: int | None = None) -> np.ndarray:
    if max_len is None:
        max_len = max(len(s) for s in seqs)
    arr = np.full((len(seqs), max_len), pad_id, dtype=np.uint16)
    for i, s in enumerate(seqs):
        L = min(len(s), max_len)
        arr[i, :L] = s[:L]
    return arr


def prepare_split(split_name: str) -> None:
    cfg = SPLITS[split_name]
    out_dir = HERE / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_raw = out_dir / "train.txt"
    test_raw = out_dir / "test.txt"
    _download(cfg["train"], train_raw)
    _download(cfg["test"], test_raw)

    train_ex = _parse_lines(train_raw.read_text(encoding="utf-8"))
    test_ex = _parse_lines(test_raw.read_text(encoding="utf-8"))
    print(f"[{split_name}] train: {len(train_ex)}, test: {len(test_ex)}")

    # Build vocab from train + test to avoid <UNK>s at eval (SCAN tokens overlap).
    vocab = _build_vocab(train_ex + test_ex)
    print(f"[{split_name}] vocab size: {len(vocab)}")

    train_enc = _encode(train_ex, vocab)
    test_enc = _encode(test_ex, vocab)
    max_len = max(max(len(s) for s in train_enc), max(len(s) for s in test_enc))
    print(f"[{split_name}] max sequence length: {max_len}")

    pad_id = vocab["<PAD>"]
    train_arr = _pad_and_pack(train_enc, pad_id=pad_id, max_len=max_len)
    test_arr = _pad_and_pack(test_enc, pad_id=pad_id, max_len=max_len)

    train_arr.tofile(out_dir / "train.bin")
    test_arr.tofile(out_dir / "test.bin")

    meta = {
        "vocab": vocab,
        "vocab_size": len(vocab),
        "max_len": int(max_len),
        "n_train": int(train_arr.shape[0]),
        "n_test": int(test_arr.shape[0]),
        "pad_id": int(pad_id),
        "bos_id": int(vocab["<BOS>"]),
        "sep_id": int(vocab["<SEP>"]),
        "eos_id": int(vocab["<EOS>"]),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[{split_name}] wrote {out_dir}/{{train,test}}.bin + meta.json")


def main() -> None:
    for split in SPLITS:
        prepare_split(split)


if __name__ == "__main__":
    main()
