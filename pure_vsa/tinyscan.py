"""TinySCAN: a deliberately small compositional benchmark.

Grammar:
    S        -> ACTION | ACTION MODIFIER
    ACTION   -> walk | jump | run | swim | look
    MODIFIER -> twice | thrice | left | right

Outputs (token sequences, encoded as slot bindings out_1, out_2, ...):
    walk          -> [W]
    walk twice    -> [W, W]
    walk thrice   -> [W, W, W]
    walk left     -> [LEFT, W]
    walk right    -> [RIGHT, W]

Held-out split: swim + any modifier. swim alone is in train; swim + modifier
combinations are only in test. A pure lookup memory fails this -- there's no
exact key match for the held-out compositions.
"""

from __future__ import annotations

from dataclasses import dataclass


# token symbols (input vocabulary)
ACTIONS = ["walk", "jump", "run", "swim", "look"]
MODIFIERS = ["twice", "thrice", "left", "right"]

# output symbols
OUTPUT_TOKENS = ["W", "J", "R", "S", "L", "LEFT", "RIGHT"]  # one per action, plus LEFT/RIGHT
ACTION_TO_OUT = {"walk": "W", "jump": "J", "run": "R", "swim": "S", "look": "L"}

# Combined codebook indices: symbols 0..n_actions-1 are actions,
# n_actions..n_actions+n_mods-1 are modifiers,
# remaining are output tokens.
N_ACTIONS = len(ACTIONS)
N_MODIFIERS = len(MODIFIERS)
N_OUTPUT_TOKENS = len(OUTPUT_TOKENS)
TOTAL_SYMBOLS = N_ACTIONS + N_MODIFIERS + N_OUTPUT_TOKENS


def symbol_index(symbol: str) -> int:
    if symbol in ACTIONS:
        return ACTIONS.index(symbol)
    if symbol in MODIFIERS:
        return N_ACTIONS + MODIFIERS.index(symbol)
    if symbol in OUTPUT_TOKENS:
        return N_ACTIONS + N_MODIFIERS + OUTPUT_TOKENS.index(symbol)
    raise ValueError(f"unknown symbol: {symbol}")


# role slots
ROLE_VERB = 0
ROLE_MOD = 1
ROLE_OUT1 = 2
ROLE_OUT2 = 3
ROLE_OUT3 = 4
N_ROLES = 5

OUTPUT_ROLES = [ROLE_OUT1, ROLE_OUT2, ROLE_OUT3]


@dataclass
class TinySCANExample:
    action: str
    modifier: str | None  # None means bare action
    output_tokens: list[str]

    def input_slots(self) -> dict[int, int]:
        slots = {ROLE_VERB: symbol_index(self.action)}
        if self.modifier is not None:
            slots[ROLE_MOD] = symbol_index(self.modifier)
        return slots

    def output_slots(self) -> dict[int, int]:
        return {
            OUTPUT_ROLES[i]: symbol_index(tok)
            for i, tok in enumerate(self.output_tokens)
        }


def _apply_grammar(action: str, modifier: str | None) -> list[str]:
    """The ground-truth grammar."""
    a_out = ACTION_TO_OUT[action]
    if modifier is None:
        return [a_out]
    if modifier == "twice":
        return [a_out, a_out]
    if modifier == "thrice":
        return [a_out, a_out, a_out]
    if modifier == "left":
        return ["LEFT", a_out]
    if modifier == "right":
        return ["RIGHT", a_out]
    raise ValueError(f"unknown modifier: {modifier}")


def make_dataset(held_out_action: str = "swim") -> tuple[
    list[TinySCANExample], list[TinySCANExample]
]:
    """Return (train, test).

    train = all bare actions + all (action, modifier) pairs whose action is not held_out_action.
    test  = (held_out_action, modifier) for every modifier.
    """
    train, test = [], []
    for action in ACTIONS:
        bare = TinySCANExample(action, None, _apply_grammar(action, None))
        # Bare swim IS in the training set; only swim+modifier is held out.
        train.append(bare)
        for modifier in MODIFIERS:
            ex = TinySCANExample(action, modifier, _apply_grammar(action, modifier))
            if action == held_out_action:
                test.append(ex)
            else:
                train.append(ex)
    return train, test


def exact_match(predicted: list[str], expected: list[str]) -> bool:
    return predicted == expected
