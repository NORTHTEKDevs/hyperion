"""PCFG SET solver: extends the pure-VSA approach to sequence transformations.

PCFG SET (Hupkes et al. 2020) is structurally different from SCAN:
  - SCAN: small grammar, fixed-template outputs
  - PCFG: 10 string-edit operations on variable-length argument lists, with
    nesting up to 8 levels deep, outputs up to 736 tokens

This module handles SINGLE-operation examples (the simplest 1411 of 81010
training examples). Multi-operation nesting requires procedural composition
of operations (compose like SCAN's `and`/`after`). The 10 operations:

    copy <args>             -> <args>                          identity
    reverse <args>          -> <args reversed>                 position-permute
    echo <args>             -> <args> <last_arg>               extend
    swap_first_last <args>  -> <last> <middle> <first>         swap two positions
    repeat <args>           -> <args> <args>                   self-concat
    shift <args>            -> <args[1:]> <args[0]>            cyclic shift
    remove_first <a> , <b>  -> <b>                             keep second
    remove_second <a> , <b> -> <a>                             keep first
    append <a> , <b>        -> <a> <b>                         concat
    prepend <a> , <b>       -> <b> <a>                         concat reversed

Each operation has a known length-dependent output mapping. For pure VSA,
we encode inputs as bundle of position bindings, apply the operation via
position-permutation rules, and decode.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from vsa_core import bind, permute as vsa_permute, unbind
from vsa_core.cleanup import similarity
from vsa_core.codebook import Codebook


@dataclass
class PCFGConfig:
    d: int = 1024            # PCFG arg-token vocab is 520; 1024 is comfortable
    max_input_len: int = 80  # PCFG max input length is 53; pad up
    max_output_len: int = 800  # PCFG max output length is 736; pad up
    seed: int = 0


# the 10 PCFG string-edit operations
OPERATIONS = {
    "copy", "reverse", "echo", "swap_first_last", "repeat",
    "shift", "remove_first", "remove_second", "append", "prepend",
}


def tokenize_pcfg(line: str) -> tuple[str, list[list[str]]]:
    """Tokenize a single PCFG src line into (operation, list_of_arg_groups).

    Most operations take a single arg group. `append`/`prepend`/`remove_*`
    take two arg groups separated by ','.

    Returns:
        ("reverse", [["Q2", "P20", "X14"]])
        ("append",  [["T10", "E6", "W17"], ["J13", "L14", "S13"]])
    """
    toks = line.strip().split()
    if not toks:
        raise ValueError("empty line")
    op = toks[0]
    if op not in OPERATIONS:
        raise ValueError(f"unknown operation: {op}")
    rest = toks[1:]
    # split on ','
    groups: list[list[str]] = [[]]
    for t in rest:
        if t == ",":
            groups.append([])
        else:
            groups[-1].append(t)
    return op, groups


def apply_pcfg_operation(op: str, groups: list[list[str]]) -> list[str]:
    """Reference interpreter for one PCFG operation. Used for oracle testing
    and as ground truth during eval.
    """
    if op == "copy":
        return list(groups[0])
    if op == "reverse":
        return list(reversed(groups[0]))
    if op == "echo":
        return list(groups[0]) + [groups[0][-1]]
    if op == "swap_first_last":
        if len(groups[0]) < 2:
            return list(groups[0])
        return [groups[0][-1]] + groups[0][1:-1] + [groups[0][0]]
    if op == "repeat":
        return list(groups[0]) * 2
    if op == "shift":
        if not groups[0]:
            return []
        return groups[0][1:] + [groups[0][0]]
    if op == "remove_first":
        return list(groups[1]) if len(groups) >= 2 else []
    if op == "remove_second":
        return list(groups[0])
    if op == "append":
        return list(groups[0]) + (list(groups[1]) if len(groups) >= 2 else [])
    if op == "prepend":
        return (list(groups[1]) if len(groups) >= 2 else []) + list(groups[0])
    raise ValueError(f"unknown op: {op}")


class PCFGHyperion:
    """Pure-VSA solver for single-operation PCFG examples.

    Built on the same primitives as SCANHyperion. Each operation maps to a
    procedural sequence transform implemented in VSA via:
      - encode input as bundle of `bind(role_i, token_i)` terms
      - apply per-operation position-permutation rule
      - decode output positions via unbind + cleanup against token codebook
    """

    def __init__(self, cfg: PCFGConfig) -> None:
        self.cfg = cfg
        # Token codebook: built lazily as we see tokens in training/inference.
        self._token_to_idx: dict[str, int] = {}
        self._idx_to_token: list[str] = []
        # Set during predict_nested to restrict cleanup search to the current
        # example's input tokens (essential for long-output precision).
        self._current_input_indices: list[int] | None = None
        self.symbols: Codebook = Codebook(2048, cfg.d, seed=cfg.seed)  # 520 args + buffer
        # Output role HVs: permute-derived from a single base
        base = Codebook(1, cfg.d, seed=cfg.seed + 1)[0]
        self.output_roles: Tensor = torch.stack(
            [vsa_permute(base, shift=i) for i in range(cfg.max_output_len)]
        )

    def _restrict_indices(self) -> list[int] | None:
        """Return the cleanup-restriction indices for the current prediction."""
        return self._current_input_indices

    def _token_idx(self, tok: str) -> int:
        if tok not in self._token_to_idx:
            idx = len(self._idx_to_token)
            if idx >= self.symbols.n:
                raise RuntimeError(f"codebook size {self.symbols.n} exceeded; bump cfg.d / codebook size")
            self._token_to_idx[tok] = idx
            self._idx_to_token.append(tok)
        return self._token_to_idx[tok]

    def _encode_sequence(self, tokens: list[str]) -> Tensor:
        terms = [
            bind(self.output_roles[i], self.symbols[self._token_idx(t)])
            for i, t in enumerate(tokens)
        ]
        return torch.stack(terms).sum(dim=0) if terms else torch.zeros(self.cfg.d)

    def _decode_sequence(
        self, hv: Tensor, length: int, restrict_to_indices: list[int] | None = None
    ) -> list[str]:
        """Decode `length` output positions from `hv`.

        If `restrict_to_indices` is given, cleanup is restricted to those
        codebook entries. This is critical for PCFG: every output token in
        a PCFG example comes from the INPUT tokens of that example. Limiting
        cleanup to the ~5-10 input-token entries per example (vs the full
        520-entry codebook) dramatically suppresses cleanup error at the
        precision floor for long outputs.
        """
        out: list[str] = []
        n_known = len(self._idx_to_token)
        if n_known == 0:
            return []
        if restrict_to_indices is not None and restrict_to_indices:
            cb_idx = torch.tensor(restrict_to_indices, dtype=torch.long)
            codebook_subset = self.symbols.all()[cb_idx]
            local_to_global = restrict_to_indices
        else:
            codebook_subset = self.symbols.all()[:n_known]
            local_to_global = list(range(n_known))
        for i in range(length):
            slot_hv = unbind(hv, self.output_roles[i])
            sims = similarity(slot_hv, codebook_subset)
            best_local = int(sims.argmax(dim=-1).item())
            out.append(self._idx_to_token[local_to_global[best_local]])
        return out

    def fit(self, examples: list[tuple[str, list[str]]]) -> None:
        """Register all tokens seen in (src, tgt) pairs. No rules to learn --
        the operations are procedural.
        """
        for src, tgt in examples:
            for t in src.split():
                if t in OPERATIONS or t == ",":
                    continue
                self._token_idx(t)
            for t in tgt:
                self._token_idx(t)

    def predict(self, src: str) -> list[str]:
        """Predict output tokens for a single PCFG src line."""
        op, groups = tokenize_pcfg(src)
        # Get the result via the per-operation procedural transform on encoded inputs.
        if op == "copy":
            return self._predict_copy(groups[0])
        if op == "reverse":
            return self._predict_reverse(groups[0])
        if op == "echo":
            return self._predict_echo(groups[0])
        if op == "swap_first_last":
            return self._predict_swap_first_last(groups[0])
        if op == "repeat":
            return self._predict_repeat(groups[0])
        if op == "shift":
            return self._predict_shift(groups[0])
        if op == "remove_first":
            return self._predict_remove(groups, keep_index=1)
        if op == "remove_second":
            return self._predict_remove(groups, keep_index=0)
        if op == "append":
            return self._predict_append(groups, reverse=False)
        if op == "prepend":
            return self._predict_append(groups, reverse=True)
        raise ValueError(f"unknown op: {op}")

    # --- per-operation procedural VSA transforms ---

    def _predict_copy(self, args: list[str]) -> list[str]:
        hv = self._encode_sequence(args)
        return self._decode_sequence(hv, len(args), self._restrict_indices())

    def _predict_reverse(self, args: list[str]) -> list[str]:
        # Build reversed encoding: at position i in output, place args[L-1-i].
        L = len(args)
        terms = [
            bind(self.output_roles[i], self.symbols[self._token_idx(args[L - 1 - i])])
            for i in range(L)
        ]
        hv = torch.stack(terms).sum(dim=0) if terms else torch.zeros(self.cfg.d)
        return self._decode_sequence(hv, L, self._restrict_indices())

    def _predict_echo(self, args: list[str]) -> list[str]:
        L = len(args)
        # echo: args + [args[-1]]; equivalently encode args and add bind(role_L, args[-1])
        hv = self._encode_sequence(args)
        if args:
            hv = hv + bind(self.output_roles[L], self.symbols[self._token_idx(args[-1])])
        return self._decode_sequence(hv, L + 1 if args else 0, self._restrict_indices())

    def _predict_swap_first_last(self, args: list[str]) -> list[str]:
        L = len(args)
        if L < 2:
            return self._predict_copy(args)
        # Place args[0] at end, args[-1] at start, middle unchanged.
        terms = []
        terms.append(bind(self.output_roles[0], self.symbols[self._token_idx(args[-1])]))
        for i in range(1, L - 1):
            terms.append(bind(self.output_roles[i], self.symbols[self._token_idx(args[i])]))
        terms.append(bind(self.output_roles[L - 1], self.symbols[self._token_idx(args[0])]))
        hv = torch.stack(terms).sum(dim=0)
        return self._decode_sequence(hv, L, self._restrict_indices())

    def _predict_repeat(self, args: list[str]) -> list[str]:
        # repeat: concat args+args = encode args then add permute(encoded, L)
        L = len(args)
        hv = self._encode_sequence(args)
        hv = hv + vsa_permute(hv, shift=L)
        return self._decode_sequence(hv, L * 2, self._restrict_indices())

    def _predict_shift(self, args: list[str]) -> list[str]:
        # shift: args[1:] + args[0]; cyclic left shift by 1
        L = len(args)
        if L == 0:
            return []
        rotated = args[1:] + [args[0]]
        return self._predict_copy(rotated)

    def _predict_remove(self, groups: list[list[str]], keep_index: int) -> list[str]:
        if keep_index >= len(groups):
            return []
        return self._predict_copy(groups[keep_index])

    def predict_nested(self, src: str) -> list[str]:
        """Parse + recursively evaluate, using single-op VSA transforms at each node.

        The cleanup at decode time is restricted to the input tokens of this
        example (typically 5-20 tokens vs the full 520-entry codebook). This
        is essential for accuracy at the longest output sequences.
        """
        tree = parse_pcfg(src)
        # collect input tokens for cleanup restriction
        toks = src.strip().split()
        input_token_idxs = [
            self._token_idx(t) for t in toks
            if t not in OPERATIONS and t != ","
        ]
        # dedup while preserving order
        seen = set()
        self._current_input_indices = []
        for idx in input_token_idxs:
            if idx not in seen:
                seen.add(idx)
                self._current_input_indices.append(idx)
        try:
            return self._eval_tree(tree)
        finally:
            self._current_input_indices = None

    def _eval_tree(self, tree) -> list[str]:
        kind = tree[0]
        if kind == "PRIM":
            return list(tree[1])
        _, op, arg_trees = tree
        groups = [self._eval_tree(g) for g in arg_trees]
        # Reuse the single-op procedural transforms.
        if op == "copy":
            return self._predict_copy(groups[0])
        if op == "reverse":
            return self._predict_reverse(groups[0])
        if op == "echo":
            return self._predict_echo(groups[0])
        if op == "swap_first_last":
            return self._predict_swap_first_last(groups[0])
        if op == "repeat":
            return self._predict_repeat(groups[0])
        if op == "shift":
            return self._predict_shift(groups[0])
        if op == "remove_first":
            return self._predict_remove(groups, keep_index=1)
        if op == "remove_second":
            return self._predict_remove(groups, keep_index=0)
        if op == "append":
            return self._predict_append(groups, reverse=False)
        if op == "prepend":
            return self._predict_append(groups, reverse=True)
        raise ValueError(f"unknown op: {op}")

    def _predict_append(self, groups: list[list[str]], reverse: bool) -> list[str]:
        if len(groups) < 2:
            return list(groups[0]) if groups else []
        a, b = groups[0], groups[1]
        first, second = (b, a) if reverse else (a, b)
        # build encoded output: first then second
        L1 = len(first)
        hv = self._encode_sequence(first)
        second_hv = self._encode_sequence(second)
        hv = hv + vsa_permute(second_hv, shift=L1)
        return self._decode_sequence(hv, L1 + len(second), self._restrict_indices())


def accuracy(reasoner: PCFGHyperion, examples: list[tuple[str, list[str]]]) -> dict:
    correct = 0
    failures = []
    for src, tgt in examples:
        try:
            pred = reasoner.predict(src)
        except Exception:
            pred = []
        if pred == tgt:
            correct += 1
        elif len(failures) < 5:
            failures.append((src, pred, tgt))
    return {"correct": correct, "total": len(examples), "acc": correct / len(examples), "failures": failures}


def load_pcfg_split(src_path, tgt_path) -> list[tuple[str, list[str]]]:
    with open(src_path) as f:
        src_lines = [line.strip() for line in f if line.strip()]
    with open(tgt_path) as f:
        tgt_lines = [line.strip() for line in f if line.strip()]
    return list(zip(src_lines, [line.split() for line in tgt_lines]))


def is_single_op(src: str) -> bool:
    """Return True iff the src line consists of exactly one PCFG operation
    (no nested operations)."""
    toks = src.strip().split()
    op_count = sum(1 for t in toks if t in OPERATIONS)
    return op_count == 1


# ----------------------------------------------------------------------
# Nested (multi-operation) parsing + recursive evaluation
# ----------------------------------------------------------------------

def _parse_arg_group(tokens: list[str], i: int) -> tuple[list, int]:
    """Parse a single argument group starting at tokens[i].

    An arg group is either:
      - a primitive sequence of non-op, non-',' tokens (the base case)
      - an op followed by 1 or 2 arg groups

    Returns: (parse_tree, index_after_group).

    Parse tree forms:
      ('PRIM', [tokens...])
      ('OP', op_name, [arg_group_1, ...])
    """
    if i >= len(tokens):
        return ("PRIM", []), i
    if tokens[i] in OPERATIONS:
        op = tokens[i]
        i += 1
        # 1-arg ops (most) vs 2-arg ops (append/prepend/remove_*)
        if op in {"append", "prepend", "remove_first", "remove_second"}:
            g1, i = _parse_arg_group(tokens, i)
            if i < len(tokens) and tokens[i] == ",":
                i += 1
                g2, i = _parse_arg_group(tokens, i)
                return ("OP", op, [g1, g2]), i
            return ("OP", op, [g1]), i
        # 1-arg op
        g1, i = _parse_arg_group(tokens, i)
        return ("OP", op, [g1]), i
    # primitive sequence: consume tokens until next ',' or end
    start = i
    while i < len(tokens) and tokens[i] != "," and tokens[i] not in OPERATIONS:
        i += 1
    return ("PRIM", tokens[start:i]), i


def parse_pcfg(src: str):
    """Parse a PCFG src into a recursive parse tree.

    Examples:
      'copy X Y'        -> ('OP', 'copy', [('PRIM', ['X','Y'])])
      'reverse copy X Y' -> ('OP', 'reverse', [('OP', 'copy', [('PRIM', ['X','Y'])])])
      'append copy X , reverse Y Z' -> ('OP', 'append', [
            ('OP', 'copy', [('PRIM', ['X'])]),
            ('OP', 'reverse', [('PRIM', ['Y', 'Z'])])
        ])
    """
    toks = src.strip().split()
    tree, _ = _parse_arg_group(toks, 0)
    return tree


def evaluate_pcfg(tree, reasoner: "PCFGHyperion | None" = None) -> list[str]:
    """Reference interpreter (oracle) for a parsed PCFG tree."""
    kind = tree[0]
    if kind == "PRIM":
        return list(tree[1])
    # OP
    _, op, arg_trees = tree
    groups = [evaluate_pcfg(g, reasoner) for g in arg_trees]
    return apply_pcfg_operation(op, groups)

