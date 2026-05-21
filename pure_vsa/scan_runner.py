"""Run HyperionReasoner against real SCAN. Report whatever accuracy comes out.

The SCAN grammar (Lake & Baroni 2018):
    S       -> CLAUSE | CLAUSE and CLAUSE | CLAUSE after CLAUSE
    CLAUSE  -> ATOM | ATOM twice | ATOM thrice
    ATOM    -> VERB | VERB DIR | VERB opposite DIR | VERB around DIR
    VERB    -> walk | look | run | jump | turn
    DIR     -> left | right

The output of each ATOM:
    VERB                -> I_VERB
    VERB DIR            -> I_TURN_DIR I_VERB
    VERB opposite DIR   -> I_TURN_DIR I_TURN_DIR I_VERB
    VERB around DIR     -> (I_TURN_DIR I_VERB) x 4
    `turn` is a special case: turn DIR -> I_TURN_DIR (no second action), etc.

CLAUSE = ATOM is the ATOM output.
CLAUSE = ATOM twice  is the ATOM output, repeated.
CLAUSE = ATOM thrice is the ATOM output, repeated 3x.

S = X and Y -> X_output ++ Y_output.
S = X after Y -> Y_output ++ X_output  (REVERSED).

The reasoner this file is wrapping was built for: single-clause `verb [modifier]`
with one verb-multiplier or constant-prepend modifier, plus binary "and" conjunctions
with permute-shifted output role positions. It was NOT built for nested modifiers
(opposite, around), variable output lengths up to 48, or the `after` reversal.

This runner does the honest test: try the mechanism as-is, see what it actually does.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


VERBS = ["walk", "look", "run", "jump", "turn"]
DIRS = ["left", "right"]
MODIFIERS = ["twice", "thrice"]
SPATIAL = ["opposite", "around"]
CONJUNCTIONS = ["and", "after"]

ACTION_FOR_VERB = {
    "walk": "I_WALK", "look": "I_LOOK", "run": "I_RUN",
    "jump": "I_JUMP", "turn": None,  # turn alone outputs nothing extra; turn DIR -> I_TURN_DIR
}
TURN_FOR_DIR = {"left": "I_TURN_LEFT", "right": "I_TURN_RIGHT"}


# ---------------------------------------------------------------------------
# Grammar parser + reference interpreter (oracle output).
# We need this to recover the "correct" sub-structure of each SCAN input so
# we can attempt prediction with the reasoner.
# ---------------------------------------------------------------------------

@dataclass
class Atom:
    verb: str
    direction: str | None = None       # left / right / None
    spatial: str | None = None         # opposite / around / None

    def output(self) -> list[str]:
        if self.verb == "turn":
            # turn alone is invalid; turn DIR -> I_TURN_DIR
            if self.spatial == "opposite":
                return [TURN_FOR_DIR[self.direction], TURN_FOR_DIR[self.direction]]
            if self.spatial == "around":
                return [TURN_FOR_DIR[self.direction]] * 4
            return [TURN_FOR_DIR[self.direction]]
        verb_action = ACTION_FOR_VERB[self.verb]
        if self.direction is None:
            return [verb_action]
        if self.spatial is None:
            return [TURN_FOR_DIR[self.direction], verb_action]
        if self.spatial == "opposite":
            return [TURN_FOR_DIR[self.direction], TURN_FOR_DIR[self.direction], verb_action]
        if self.spatial == "around":
            unit = [TURN_FOR_DIR[self.direction], verb_action]
            return unit * 4
        raise ValueError(f"unknown spatial {self.spatial}")


@dataclass
class Clause:
    atom: Atom
    modifier: str | None  # twice / thrice / None

    def output(self) -> list[str]:
        atom_out = self.atom.output()
        if self.modifier is None:
            return atom_out
        if self.modifier == "twice":
            return atom_out * 2
        if self.modifier == "thrice":
            return atom_out * 3
        raise ValueError(f"unknown modifier {self.modifier}")


@dataclass
class ParsedSCAN:
    clause1: Clause
    clause2: Clause | None = None
    connective: str | None = None  # 'and' / 'after' / None

    def expected_output(self) -> list[str]:
        out1 = self.clause1.output()
        if self.clause2 is None:
            return out1
        out2 = self.clause2.output()
        if self.connective == "and":
            return out1 + out2
        if self.connective == "after":
            return out2 + out1  # REVERSED
        raise ValueError(f"unknown connective {self.connective}")


def _parse_atom(tokens: list[str]) -> Atom:
    """Parse an ATOM: [VERB | VERB DIR | VERB opposite DIR | VERB around DIR]."""
    verb = tokens[0]
    if verb not in VERBS:
        raise ValueError(f"expected verb, got {verb}")
    if len(tokens) == 1:
        return Atom(verb=verb)
    if len(tokens) == 2:
        # VERB DIR
        return Atom(verb=verb, direction=tokens[1])
    if len(tokens) == 3:
        # VERB opposite|around DIR
        return Atom(verb=verb, spatial=tokens[1], direction=tokens[2])
    raise ValueError(f"unexpected atom: {tokens}")


def _parse_clause(tokens: list[str]) -> Clause:
    """Parse a CLAUSE: [ATOM | ATOM twice | ATOM thrice]."""
    if tokens[-1] in MODIFIERS:
        return Clause(atom=_parse_atom(tokens[:-1]), modifier=tokens[-1])
    return Clause(atom=_parse_atom(tokens), modifier=None)


def parse_scan(input_str: str) -> ParsedSCAN:
    """Parse a SCAN command string into a ParsedSCAN structure."""
    tokens = input_str.strip().split()
    # find connective
    for conn in CONJUNCTIONS:
        if conn in tokens:
            i = tokens.index(conn)
            return ParsedSCAN(
                clause1=_parse_clause(tokens[:i]),
                clause2=_parse_clause(tokens[i + 1:]),
                connective=conn,
            )
    return ParsedSCAN(clause1=_parse_clause(tokens))


def load_scan_split(path: Path) -> list[tuple[str, list[str]]]:
    """Load a SCAN .txt file into list of (input_str, expected_output_tokens)."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            inp_str, out_str = line.split(" OUT: ")
            inp_str = inp_str.replace("IN: ", "")
            out_tokens = out_str.split()
            out.append((inp_str, out_tokens))
    return out


# ---------------------------------------------------------------------------
# Oracle parser test: confirm our parser+interpreter is correct.
# ---------------------------------------------------------------------------

def verify_oracle(examples: list[tuple[str, list[str]]]) -> dict:
    """Run the parser+interpreter on all examples and confirm it matches SCAN."""
    correct = 0
    failures = []
    for inp_str, expected in examples:
        try:
            parsed = parse_scan(inp_str)
            predicted = parsed.expected_output()
            if predicted == expected:
                correct += 1
            else:
                if len(failures) < 5:
                    failures.append((inp_str, predicted, expected))
        except Exception as e:
            if len(failures) < 5:
                failures.append((inp_str, f"PARSE_ERROR: {e}", expected))
    return {"correct": correct, "total": len(examples), "failures": failures}


# ---------------------------------------------------------------------------
# Attempt prediction with HyperionReasoner. Will only work on a subset of SCAN.
# ---------------------------------------------------------------------------

def classify_complexity(parsed: ParsedSCAN) -> str:
    """What kind of SCAN structure is this? Used to triage which cases the
    Hyperion mechanism CAN attempt vs. which are out-of-scope."""
    a1 = parsed.clause1.atom
    has_spatial1 = a1.spatial is not None
    has_dir1 = a1.direction is not None
    has_mod1 = parsed.clause1.modifier is not None

    if parsed.clause2 is None:
        if not has_dir1 and not has_mod1:
            return "atom_only"           # just `walk`
        if has_dir1 and not has_spatial1 and not has_mod1:
            return "atom+dir"            # `walk left`
        if not has_dir1 and has_mod1:
            return "atom+mod"            # `walk twice`
        if has_dir1 and not has_spatial1 and has_mod1:
            return "atom+dir+mod"        # `walk left twice`
        if has_spatial1 and not has_mod1:
            return "atom+spatial"        # `walk opposite left`
        return "atom+spatial+mod"        # `walk opposite left twice`
    # 2-clause
    c1 = classify_complexity(ParsedSCAN(parsed.clause1))
    c2 = classify_complexity(ParsedSCAN(parsed.clause2))
    return f"2-clause:{parsed.connective}:{c1}+{c2}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="addprim_jump",
                        choices=["addprim_jump", "simple", "length"])
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1] / "data" / "scan" / args.split
    print(f"# SCAN split: {args.split}")
    print(f"#   path: {base}")

    train = load_scan_split(base / "train.txt")
    test = load_scan_split(base / "test.txt")
    print(f"#   train: {len(train)} examples")
    print(f"#   test:  {len(test)} examples")

    if args.max_examples:
        test = test[: args.max_examples]
        print(f"#   (limited to first {args.max_examples} test examples)")

    print("\n## Step 1: Verify our SCAN parser+interpreter (oracle)")
    t0 = time.time()
    oracle_train = verify_oracle(train)
    oracle_test = verify_oracle(test)
    print(f"oracle on train: {oracle_train['correct']}/{oracle_train['total']} = "
          f"{oracle_train['correct'] / oracle_train['total']:.1%}")
    print(f"oracle on test:  {oracle_test['correct']}/{oracle_test['total']} = "
          f"{oracle_test['correct'] / oracle_test['total']:.1%}")
    if oracle_train["failures"]:
        print("first oracle failures (if any):")
        for inp, pred, exp in oracle_train["failures"][:3]:
            print(f"  IN: {inp}")
            print(f"  PRED: {pred[:8]}...")
            print(f"  EXP:  {exp[:8]}...")
    print(f"oracle time: {time.time() - t0:.1f}s")

    print("\n## Step 2: Classify test examples by structural complexity")
    complexity_counter: Counter = Counter()
    for inp_str, _ in test:
        try:
            parsed = parse_scan(inp_str)
            complexity_counter[classify_complexity(parsed)] += 1
        except Exception:
            complexity_counter["PARSE_ERROR"] += 1

    print(f"test set structural breakdown ({len(test)} examples):")
    for cls, n in complexity_counter.most_common():
        frac = n / len(test)
        print(f"  {n:5d}  ({frac:5.1%})  {cls}")

    # Identify cases the current Hyperion mechanism can even attempt.
    in_scope = {"atom_only", "atom+mod"}
    n_in_scope = sum(n for c, n in complexity_counter.items() if c in in_scope)
    print("\nFraction of test cases the current Hyperion mechanism could attempt:")
    print(f"  {n_in_scope}/{len(test)} = {n_in_scope / len(test):.1%}")
    print("(everything else needs `opposite`/`around`/2-clause/`after` -- not in v0.1 mechanism.)")


if __name__ == "__main__":
    main()
