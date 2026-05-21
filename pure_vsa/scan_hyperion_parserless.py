"""Parser-free SCAN: SCANHyperion that uses a *discovered* grammar.

This replaces `pure_vsa.scan_runner.parse_scan` (which is hand-written for
SCAN's exact grammar) with a discovered-grammar parser built from training
data alone via `pure_vsa.scan_grammar_discovery.discover_grammar`.

The compositional generalization mechanism is unchanged; only the way the
grammar structure is obtained changes from "hardcoded in Python" to
"discovered from data".
"""

from __future__ import annotations

from pathlib import Path

from pure_vsa.scan_grammar_discovery import DiscoveredGrammar, discover_grammar
from pure_vsa.scan_hyperion import SCANConfig, SCANHyperion
from pure_vsa.scan_runner import Atom, Clause, ParsedSCAN


class DiscoveredParser:
    """Parser that uses a DiscoveredGrammar instead of hand-written rules."""

    def __init__(self, grammar: DiscoveredGrammar) -> None:
        self.g = grammar

    def parse(self, input_str: str) -> ParsedSCAN:
        tokens = input_str.strip().split()
        # split on conjunction (whichever comes first)
        for conn in self.g.conjunctions:
            if conn in tokens:
                i = tokens.index(conn)
                return ParsedSCAN(
                    clause1=self._parse_clause(tokens[:i]),
                    clause2=self._parse_clause(tokens[i + 1:]),
                    connective=conn,
                )
        return ParsedSCAN(clause1=self._parse_clause(tokens))

    def _parse_clause(self, tokens: list[str]) -> Clause:
        modifier = None
        if tokens and tokens[-1] in self.g.modifiers:
            modifier = tokens[-1]
            tokens = tokens[:-1]
        return Clause(atom=self._parse_atom(tokens), modifier=modifier)

    def _parse_atom(self, tokens: list[str]) -> Atom:
        if not tokens:
            raise ValueError("empty atom")
        verb = tokens[0]
        if verb not in self.g.content_tokens:
            raise ValueError(f"first token {verb!r} not a discovered verb")
        if len(tokens) == 1:
            return Atom(verb=verb)
        if len(tokens) == 2:
            d = tokens[1]
            if d not in self.g.directions:
                raise ValueError(f"second token {d!r} not a discovered direction")
            return Atom(verb=verb, direction=d)
        if len(tokens) == 3:
            sp = tokens[1]
            d = tokens[2]
            if sp not in self.g.spatial:
                raise ValueError(f"second token {sp!r} not a discovered spatial modifier")
            if d not in self.g.directions:
                raise ValueError(f"third token {d!r} not a discovered direction")
            return Atom(verb=verb, spatial=sp, direction=d)
        raise ValueError(f"unexpected atom tokens: {tokens}")


class ParserlessSCANHyperion(SCANHyperion):
    """SCANHyperion that uses a discovered grammar parser instead of parse_scan.

    Architecturally identical to SCANHyperion -- only the parser changes from
    a hand-written one (parse_scan in scan_runner.py) to a discovered one
    built from training data (DiscoveredParser above).
    """

    def __init__(self, cfg: SCANConfig) -> None:
        super().__init__(cfg)
        self._discovered_parser: DiscoveredParser | None = None

    def fit(self, examples: list[tuple[str, list[str]]]) -> None:
        # Step 1: discover the grammar from training data alone.
        grammar = discover_grammar(examples)
        self._discovered_parser = DiscoveredParser(grammar)
        self._discovered_grammar = grammar
        # Step 2: temporarily monkey-patch parse_scan to use our discovered parser.
        # We do this via a module-level shim that the parent fit() uses.
        import pure_vsa.scan_runner as scan_runner_mod
        original_parse_scan = scan_runner_mod.parse_scan
        try:
            scan_runner_mod.parse_scan = self._discovered_parser.parse
            super().fit(examples)
        finally:
            scan_runner_mod.parse_scan = original_parse_scan

    def accuracy(self, examples):
        # Use the discovered parser for evaluation too.
        import pure_vsa.scan_runner as scan_runner_mod
        original_parse_scan = scan_runner_mod.parse_scan
        try:
            scan_runner_mod.parse_scan = self._discovered_parser.parse
            return super().accuracy(examples)
        finally:
            scan_runner_mod.parse_scan = original_parse_scan

    def predict(self, parsed):
        return super().predict(parsed)


def fit_and_eval_parserless(train_path: Path, test_path: Path, *, d: int = 8192, seed: int = 0) -> dict:
    """Convenience: load a split, fit ParserlessSCANHyperion, eval on test."""
    from pure_vsa.scan_runner import load_scan_split
    train = load_scan_split(train_path)
    test = load_scan_split(test_path)
    r = ParserlessSCANHyperion(SCANConfig(d=d, seed=seed, max_output_len=80))
    r.fit(train)
    result = r.accuracy(test)
    return {
        "split": train_path.parent.name,
        "n_train": len(train),
        "n_test": len(test),
        "correct": result["correct"],
        "total": result["total"],
        "acc": result["acc"],
        "discovered_grammar": r._discovered_grammar.summary(),
    }
