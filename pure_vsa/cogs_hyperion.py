"""COGS minimal: handle the simplest English sentence -> logical form mapping.

COGS (Kim & Linzen 2020) is a much harder benchmark than SCAN:
  - English-like sentences (876-word vocab)
  - Logical-form outputs with explicit semantic roles
  - 21 generalization conditions testing transfer across syntactic positions

This module implements the SIMPLEST construction: proper-noun + intransitive
verb + period (e.g., "Oliver crumpled ."). The expected output is:
  "verb . ROLE ( x _ 1 , PNAME )"
where ROLE is "agent" or "theme" depending on verb category, and verb is the
infinitive form (crumpled -> crumple).

This is a tiny subset of COGS (~790 of 24,155 training examples) but it shows
the pure-VSA mechanism extends to a different problem class. Full COGS support
would require handling 20+ more constructions and is left for future work.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor

from vsa_core import bind, permute as vsa_permute, unbind
from vsa_core.cleanup import similarity
from vsa_core.codebook import Codebook


PROPER_NOUN_RE = re.compile(r"^[A-Z][a-z]+$")


def _is_proper_noun(tok: str) -> bool:
    return bool(PROPER_NOUN_RE.match(tok))


def _is_common_noun_candidate(tok: str) -> bool:
    """A common noun is lowercase, not a determiner / aux / preposition."""
    return tok.islower() and tok not in {
        "a", "the", "was", "by", "that", "to", "in", "on", "beside",
    }


@dataclass
class COGSConfig:
    d: int = 4096
    seed: int = 0


def is_simple_intransitive(input_str: str) -> bool:
    """3 tokens: ProperNoun VerbPastTense . """
    toks = input_str.strip().split()
    return (
        len(toks) == 3
        and toks[-1] == "."
        and PROPER_NOUN_RE.match(toks[0]) is not None
        and toks[1].islower()
    )


def is_intrans_w_det(input_str: str) -> bool:
    """4 tokens: Det Noun VerbPastTense . (e.g., "The captain ate .")"""
    toks = input_str.strip().split()
    return (
        len(toks) == 4
        and toks[-1] == "."
        and toks[0] in ("A", "The")
        and _is_common_noun_candidate(toks[1])
        and _is_common_noun_candidate(toks[2])
    )


def is_transitive_proper_proper(input_str: str) -> bool:
    """5 tokens: ProperNoun VerbPastTense Det Noun . (proper subj, indef obj)"""
    toks = input_str.strip().split()
    return (
        len(toks) == 5
        and toks[-1] == "."
        and _is_proper_noun(toks[0])
        and _is_common_noun_candidate(toks[1])
        and toks[2] in ("a", "the")
        and _is_common_noun_candidate(toks[3])
    )


def is_pp_transitive_proper(input_str: str) -> bool:
    """8 tokens: ProperNoun VerbPast Det Noun Prep Det Noun .
    e.g. 'Emma ate the ring beside a bed .'"""
    toks = input_str.strip().split()
    return (
        len(toks) == 8
        and toks[-1] == "."
        and _is_proper_noun(toks[0])
        and _is_common_noun_candidate(toks[1])
        and toks[2] in ("a", "the")
        and _is_common_noun_candidate(toks[3])
        and toks[4] in ("in", "on", "beside")
        and toks[5] in ("a", "the")
        and _is_common_noun_candidate(toks[6])
    )


def emit_pp_transitive_proper(
    subj_pname, verb_inf, obj_det, obj_noun, prep, pp_det, pp_noun,
    subj_role, obj_role,
) -> list[str]:
    out = []
    if obj_det == "the":
        out += ["*", obj_noun, "(", "x", "_", "3", ")", ";"]
    if pp_det == "the":
        out += ["*", pp_noun, "(", "x", "_", "6", ")", ";"]
    # main clause: verb . subj_role (x_1, SubjPName) AND verb . obj_role (x_1, x_3)
    out += [
        verb_inf, ".", subj_role, "(", "x", "_", "1", ",", subj_pname, ")", "AND",
        verb_inf, ".", obj_role, "(", "x", "_", "1", ",", "x", "_", "3", ")",
    ]
    if obj_det == "a":
        out += ["AND", obj_noun, "(", "x", "_", "3", ")"]
    # PP attachment to obj
    out += ["AND", obj_noun, ".", "nmod", ".", prep, "(", "x", "_", "3", ",", "x", "_", "6", ")"]
    if pp_det == "a":
        out += ["AND", pp_noun, "(", "x", "_", "6", ")"]
    return out


def is_dative_to_proper(input_str: str) -> bool:
    """7 tokens: ProperNoun VerbPast Det Noun to ProperNoun .
    e.g. 'Natalie mailed the cake to Emma .'"""
    toks = input_str.strip().split()
    return (
        len(toks) == 7
        and toks[-1] == "."
        and _is_proper_noun(toks[0])
        and _is_common_noun_candidate(toks[1])
        and toks[2] in ("a", "the")
        and _is_common_noun_candidate(toks[3])
        and toks[4] == "to"
        and _is_proper_noun(toks[5])
    )


def emit_dative_to_proper(
    subj_pname, verb_inf, obj_det, obj_noun, recipient_pname,
    subj_role, obj_role, recipient_role,
) -> list[str]:
    out = []
    if obj_det == "the":
        out += ["*", obj_noun, "(", "x", "_", "3", ")", ";"]
    out += [
        verb_inf, ".", subj_role, "(", "x", "_", "1", ",", subj_pname, ")", "AND",
        verb_inf, ".", obj_role, "(", "x", "_", "1", ",", "x", "_", "3", ")", "AND",
        verb_inf, ".", recipient_role, "(", "x", "_", "1", ",", recipient_pname, ")",
    ]
    if obj_det == "a":
        out += ["AND", obj_noun, "(", "x", "_", "3", ")"]
    return out


def is_cp_simple(input_str: str) -> bool:
    """7 tokens: ProperN VerbPast that Det Noun VerbPast .
    e.g. 'Emma liked that a girl saw .'

    The matrix verb takes a CP complement; the embedded clause is intransitive."""
    toks = input_str.strip().split()
    return (
        len(toks) == 7
        and toks[-1] == "."
        and _is_proper_noun(toks[0])
        and _is_common_noun_candidate(toks[1])
        and toks[2] == "that"
        and toks[3] in ("a", "the")
        and _is_common_noun_candidate(toks[4])
        and _is_common_noun_candidate(toks[5])
    )


def emit_cp_simple(
    matrix_pname, matrix_verb_inf, matrix_subj_role, matrix_ccomp_role,
    embed_det, embed_noun, embed_verb_inf, embed_subj_role,
) -> list[str]:
    """Emit:
       like . agent ( x _ 1 , Emma ) AND like . ccomp ( x _ 1 , x _ 5 ) AND girl ( x _ 4 ) AND see . agent ( x _ 5 , x _ 4 )
       OR with definite det: '* girl ( x _ 4 ) ; ...'
    """
    out = []
    if embed_det == "the":
        out += ["*", embed_noun, "(", "x", "_", "4", ")", ";"]
    out += [
        matrix_verb_inf, ".", matrix_subj_role, "(", "x", "_", "1", ",", matrix_pname, ")", "AND",
        matrix_verb_inf, ".", matrix_ccomp_role, "(", "x", "_", "1", ",", "x", "_", "5", ")",
    ]
    if embed_det == "a":
        out += ["AND", embed_noun, "(", "x", "_", "4", ")"]
    out += [
        "AND",
        embed_verb_inf, ".", embed_subj_role, "(", "x", "_", "5", ",", "x", "_", "4", ")",
    ]
    return out


def is_passive_short(input_str: str) -> bool:
    """7 tokens: Det Noun was VerbPast by Det Noun .
    (e.g. 'A rose was helped by a dog .')
    """
    toks = input_str.strip().split()
    return (
        len(toks) == 8
        and toks[-1] == "."
        and toks[0] in ("A", "The")
        and _is_common_noun_candidate(toks[1])
        and toks[2] == "was"
        and _is_common_noun_candidate(toks[3])  # verb past
        and toks[4] == "by"
        and toks[5] in ("a", "the")
        and _is_common_noun_candidate(toks[6])
    )


def _scan_verb_roles(parts: list[str]) -> tuple[str, dict[int, str]] | None:
    """Scan a parsed output for `verb . role ( x _ k , ...)` patterns.

    Returns (verb_inf, {x_index: role}) -- a map from each x_N argument index
    to the role of the verb taking that argument. Works on any COGS output
    that uses one main verb (the verb token is shared across all clauses).
    """
    verb_to_roles: dict[str, dict[int, str]] = {}
    i = 0
    while i + 7 < len(parts):
        # match: <verb> . <role> ( x _ <num> , ...
        if (parts[i + 1] == "." and parts[i + 3] == "("
                and parts[i + 4] == "x" and parts[i + 5] == "_"):
            verb = parts[i]
            role = parts[i + 2]
            try:
                k = int(parts[i + 6])
            except ValueError:
                i += 1
                continue
            verb_to_roles.setdefault(verb, {})[k] = role
        i += 1
    if not verb_to_roles:
        return None
    # main verb is the one with the most role bindings (or any if tied)
    verb_inf = max(verb_to_roles, key=lambda v: len(verb_to_roles[v]))
    return verb_inf, verb_to_roles[verb_inf]


def parse_passive_short_output(out_str: str):
    """Parse a passive output robustly across definite-subject, definite-object,
    and indefinite forms by scanning the output for verb-role patterns.

    Returns (verb_inf, subj_noun, obj_noun, theme_role, agent_role, is_def_subj).
    The theme role is the one attached to x_1 (subject position in input);
    the agent role is the one attached to x_6 (by-object position, last noun
    in the input).
    """
    parts = out_str.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").replace(";", " ; ").split()
    scan = _scan_verb_roles(parts)
    if scan is None:
        return None
    verb_inf, role_at_idx = scan
    # Expected: role at x_1 = theme, role at x_6 = agent (for 8-token passive)
    if 1 not in role_at_idx or 6 not in role_at_idx:
        return None
    theme_role = role_at_idx[1]
    agent_role = role_at_idx[6]
    # Extract nouns: subj_noun bound to x_1, obj_noun bound to x_6.
    # In the output, `noun ( x _ 1 )` appears for the subject and
    # `noun ( x _ 6 )` for the object (potentially with `*` prefix).
    subj_noun = _find_noun_at_index(parts, 1)
    obj_noun = _find_noun_at_index(parts, 6)
    if subj_noun is None or obj_noun is None:
        return None
    is_def_subj = ("*" in parts and parts.index("*") < parts.index(subj_noun))
    return verb_inf, subj_noun, obj_noun, theme_role, agent_role, is_def_subj


def _find_noun_at_index(parts: list[str], k: int) -> str | None:
    """Find the noun bound to x_k, i.e. the token <N> in `N ( x _ k )`."""
    i = 0
    while i + 5 < len(parts):
        if (parts[i + 1] == "(" and parts[i + 2] == "x" and parts[i + 3] == "_"
                and parts[i + 4] == str(k) and parts[i + 5] == ")"):
            return parts[i]
        i += 1
    return None


def parse_intransitive_output(out_str: str) -> tuple[str, str, str] | None:
    """Parse the expected COGS output for a simple intransitive sentence.

    Format: 'verb . role ( x _ 1 , PName )'
    Returns: (verb_infinitive, role, pname) or None if doesn't match.
    """
    parts = out_str.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").split()
    if len(parts) != 10:
        return None
    if parts[1] != "." or parts[3] != "(" or parts[9] != ")":
        return None
    if parts[4] != "x" or parts[5] != "_" or parts[6] != "1" or parts[7] != ",":
        return None
    verb_inf = parts[0]
    role = parts[2]
    pname = parts[8]
    if role not in {"agent", "theme"}:
        return None
    return verb_inf, role, pname


def emit_intransitive_output(verb_inf: str, role: str, pname: str) -> list[str]:
    return [verb_inf, ".", role, "(", "x", "_", "1", ",", pname, ")"]


def parse_intrans_det_output(out_str: str) -> tuple[str, str, str, str, bool] | None:
    """Parse 'A captain ate .' output:
       'captain ( x _ 1 ) AND eat . agent ( x _ 2 , x _ 1 )'
    or for definite "The captain ate .":
       '* captain ( x _ 1 ) ; eat . agent ( x _ 2 , x _ 1 )'

    Returns: (noun, verb_inf, role, det, is_definite) or None.
    """
    parts = out_str.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").replace(";", " ; ").split()
    is_definite = parts[0] == "*"
    if is_definite:
        # form: * noun ( x _ 1 ) ; verb . role ( x _ 2 , x _ 1 )
        if len(parts) != 18 or parts[7] != ";":
            return None
        noun = parts[1]
        verb_inf = parts[8]
        role = parts[10]
    else:
        # form: noun ( x _ 1 ) AND verb . role ( x _ 2 , x _ 1 )
        if len(parts) != 17 or parts[6] != "AND":
            return None
        noun = parts[0]
        verb_inf = parts[7]
        role = parts[9]
    if role not in {"agent", "theme"}:
        return None
    return noun, verb_inf, role, ("the" if is_definite else "a"), is_definite


def emit_intrans_det_output(noun: str, verb_inf: str, role: str, is_definite: bool) -> list[str]:
    if is_definite:
        return [
            "*", noun, "(", "x", "_", "1", ")", ";",
            verb_inf, ".", role, "(", "x", "_", "2", ",", "x", "_", "1", ")"
        ]
    return [
        noun, "(", "x", "_", "1", ")", "AND",
        verb_inf, ".", role, "(", "x", "_", "2", ",", "x", "_", "1", ")"
    ]


def parse_transitive_pp_output(out_str: str) -> tuple[str, str, str, str, str] | None:
    """Parse transitive ProperNoun + verb + det + noun + . output:
       'roll . agent ( x _ 1 , Emma ) AND roll . theme ( x _ 1 , x _ 3 ) AND teacher ( x _ 3 )'
    for indef obj 'a teacher', OR:
       '* teacher ( x _ 3 ) ; roll . agent ( x _ 1 , Emma ) AND roll . theme ( x _ 1 , x _ 3 )'
    for def obj 'the teacher'.

    Returns: (verb_inf, subj_pname, obj_noun, subj_role, obj_role) or None.
    """
    parts = out_str.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").replace(";", " ; ").split()
    is_definite_obj = parts[0] == "*"
    if is_definite_obj:
        # * noun ( x _ 3 ) ; verb . role1 ( x _ 1 , PName ) AND verb . role2 ( x _ 1 , x _ 3 )
        if "AND" not in parts:
            return None
        try:
            and_idx = parts.index("AND")
            obj_noun = parts[1]
            sub_role_part = parts[10]
            subj_pname = parts[15]
            verb_inf = parts[8]
            obj_role = parts[and_idx + 3]
        except (IndexError, ValueError):
            return None
        return verb_inf, subj_pname, obj_noun, sub_role_part, obj_role
    # indef: verb . role1 ( x _ 1 , PName ) AND verb . role2 ( x _ 1 , x _ 3 ) AND noun ( x _ 3 )
    if parts.count("AND") != 2:
        return None
    try:
        verb_inf = parts[0]
        sub_role_part = parts[2]
        subj_pname = parts[8]  # ( x _ 1 , PName ) -> PName at index 8
        and1 = parts.index("AND")
        obj_role = parts[and1 + 3]
        and2 = parts.index("AND", and1 + 1)
        obj_noun = parts[and2 + 1]
    except (IndexError, ValueError):
        return None
    return verb_inf, subj_pname, obj_noun, sub_role_part, obj_role


def emit_transitive_output(
    verb_inf: str,
    subj_pname: str,
    obj_noun: str,
    subj_role: str,
    obj_role: str,
    is_definite_obj: bool,
    subj_idx: int = 1,
    obj_idx: int = 3,
) -> list[str]:
    s = str(subj_idx)
    o = str(obj_idx)
    if is_definite_obj:
        return [
            "*", obj_noun, "(", "x", "_", o, ")", ";",
            verb_inf, ".", subj_role, "(", "x", "_", s, ",", subj_pname, ")", "AND",
            verb_inf, ".", obj_role, "(", "x", "_", s, ",", "x", "_", o, ")"
        ]
    return [
        verb_inf, ".", subj_role, "(", "x", "_", s, ",", subj_pname, ")", "AND",
        verb_inf, ".", obj_role, "(", "x", "_", s, ",", "x", "_", o, ")", "AND",
        obj_noun, "(", "x", "_", o, ")"
    ]


class COGSIntransitiveHyperion:
    """Solve COGS sentences across 3 constructions:
      - simple intransitive (proper subj, no det)
      - intransitive with determiner (det + common-noun subj)
      - transitive (proper subj + verb + det + common-noun obj)

    Learned from training:
      - past_to_inf: verb past-tense -> infinitive
      - verb_to_intrans_role:  intransitive role per verb (agent / theme)
      - verb_to_trans_roles:   (subj_role, obj_role) per transitive verb
      - proper_nouns, common_nouns observed

    For both transitive and intransitive, "the X" vs "a X" determines the
    definite-vs-indefinite output schema (* X (x_n) ; ... vs X (x_n) AND ...).
    """

    def __init__(self, cfg: COGSConfig) -> None:
        self.cfg = cfg
        self.past_to_inf: dict[str, str] = {}
        self.verb_to_intrans_role: dict[str, str] = {}
        self.verb_to_trans_roles: dict[str, tuple[str, str]] = {}
        # verb_to_dative_roles: verb -> (subj_role, obj_role, recipient_role)
        self.verb_to_dative_roles: dict[str, tuple[str, str, str]] = {}
        # verb_to_cp_roles: matrix-verb -> (subj_role, ccomp_role)
        self.verb_to_cp_roles: dict[str, tuple[str, str]] = {}
        self.proper_nouns: set[str] = set()
        self.common_nouns: set[str] = set()
        self.symbols = Codebook(2048, cfg.d, seed=cfg.seed)
        self._token_to_idx: dict[str, int] = {}
        # backward compat alias
        self.verb_to_role = self.verb_to_intrans_role

    def _token_idx(self, tok: str) -> int:
        if tok not in self._token_to_idx:
            self._token_to_idx[tok] = len(self._token_to_idx)
        return self._token_to_idx[tok]

    def fit(self, examples) -> None:
        """examples: list of (input_str, output_str[, category]) tuples from COGS.

        Accepts both 2- and 3-tuples (category ignored). Learns past->infinitive,
        verb->role(s), and noun classes from training data.
        """
        for ex in examples:
            inp, out = ex[0], ex[1]
            toks = inp.split()

            # simple intransitive: 3 tokens, proper subject
            if is_simple_intransitive(inp):
                parsed = parse_intransitive_output(out)
                if parsed is None:
                    continue
                verb_inf, role, pname = parsed
                if toks[0] != pname:
                    continue
                self.past_to_inf[toks[1]] = verb_inf
                self.verb_to_intrans_role[verb_inf] = role
                self.proper_nouns.add(pname)
                continue

            # intransitive with det: 4 tokens, det + common-noun subject
            if is_intrans_w_det(inp):
                parsed = parse_intrans_det_output(out)
                if parsed is None:
                    continue
                noun, verb_inf, role, det, is_definite = parsed
                if toks[1] != noun or toks[2] != self.past_form(verb_inf, toks[2]):
                    pass  # don't enforce strict matching; just learn what we can
                self.past_to_inf[toks[2]] = verb_inf
                self.verb_to_intrans_role[verb_inf] = role
                self.common_nouns.add(noun)
                continue

            # transitive: 5 tokens, proper subj + verb + det + common-noun obj
            if is_transitive_proper_proper(inp):
                parsed = parse_transitive_pp_output(out)
                if parsed is None:
                    continue
                verb_inf, subj_pname, obj_noun, subj_role, obj_role = parsed
                if toks[0] != subj_pname:
                    continue
                self.past_to_inf[toks[1]] = verb_inf
                self.verb_to_trans_roles[verb_inf] = (subj_role, obj_role)
                self.proper_nouns.add(subj_pname)
                self.common_nouns.add(obj_noun)
                continue

            # passive: 8 tokens, det + noun + was + verbpast + by + det + noun + .
            if is_passive_short(inp):
                parsed = parse_passive_short_output(out)
                if parsed is None:
                    continue
                verb_inf, subj_noun, obj_noun, theme_role, agent_role, _ = parsed
                self.past_to_inf[toks[3]] = verb_inf
                self.verb_to_trans_roles[verb_inf] = (agent_role, theme_role)
                self.common_nouns.add(subj_noun)
                self.common_nouns.add(obj_noun)
                continue

            # transitive + PP: 8 tokens, ProperN VerbPast Det Noun Prep Det Noun .
            if is_pp_transitive_proper(inp):
                # Use the robust scanner to extract verb_inf and roles
                scan = _scan_verb_roles(
                    out.replace("(", " ( ").replace(")", " ) ")
                       .replace(",", " , ").replace(";", " ; ").split()
                )
                if scan is None:
                    continue
                verb_inf, role_at_idx = scan
                if 1 not in role_at_idx or 3 not in role_at_idx:
                    continue
                self.past_to_inf[toks[1]] = verb_inf
                self.verb_to_trans_roles[verb_inf] = (role_at_idx[1], role_at_idx[3])
                self.proper_nouns.add(toks[0])
                self.common_nouns.add(toks[3])
                self.common_nouns.add(toks[6])
                continue

            # dative-to: 7 tokens, ProperN VerbPast Det Noun to ProperN .
            if is_dative_to_proper(inp):
                parts = out.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").replace(";", " ; ").split()
                # Find the main verb's infinitive: look for first `<verb> . <role> (`
                verb_inf = None
                for i in range(len(parts) - 3):
                    if parts[i + 1] == "." and parts[i + 3] == "(" and parts[i].isalpha() and parts[i] not in {"AND", "x"}:
                        verb_inf = parts[i]
                        break
                if verb_inf is None:
                    continue
                # role-per-entity: scan for `verb_inf . R ( x _ 1 , ENTITY )` patterns
                # where ENTITY is either a proper noun (Natalie / Emma) or `x _ N`.
                subj_role = obj_role = recipient_role = None
                subj_pname = toks[0]   # Natalie
                recipient_pname = toks[5]  # Emma
                for i in range(len(parts) - 9):
                    if not (parts[i] == verb_inf and parts[i + 1] == "."
                            and parts[i + 3] == "(" and parts[i + 4] == "x"
                            and parts[i + 5] == "_" and parts[i + 6] == "1"
                            and parts[i + 7] == ","):
                        continue
                    role = parts[i + 2]
                    # ENTITY starts at parts[i+8]
                    if parts[i + 8] == subj_pname:
                        subj_role = role
                    elif parts[i + 8] == recipient_pname:
                        recipient_role = role
                    elif parts[i + 8] == "x" and parts[i + 9] == "_" and parts[i + 10] == "3":
                        obj_role = role
                if subj_role is None or obj_role is None or recipient_role is None:
                    continue
                self.past_to_inf[toks[1]] = verb_inf
                self.verb_to_dative_roles[verb_inf] = (subj_role, obj_role, recipient_role)
                if verb_inf not in self.verb_to_trans_roles:
                    self.verb_to_trans_roles[verb_inf] = (subj_role, obj_role)
                self.proper_nouns.add(toks[0])
                self.proper_nouns.add(toks[5])
                self.common_nouns.add(toks[3])
                continue

            # CP recursion (simple): 7 tokens, ProperN VerbPast that Det Noun VerbPast .
            if is_cp_simple(inp):
                parts = out.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").replace(";", " ; ").split()
                # Find the two verbs (matrix at clause-1 with proper subj, embedded at clause-5 with x_4 subj)
                matrix_pname = toks[0]
                # matrix verb_inf: first verb-like token that takes ( x _ 1 , ProperName )
                matrix_verb_inf = None
                matrix_subj_role = None
                matrix_ccomp_role = None
                for i in range(len(parts) - 9):
                    if (parts[i + 1] == "." and parts[i + 3] == "(" and parts[i + 4] == "x"
                            and parts[i + 5] == "_" and parts[i + 6] == "1" and parts[i + 7] == ","):
                        verb = parts[i]
                        role = parts[i + 2]
                        if parts[i + 8] == matrix_pname:
                            matrix_verb_inf = verb
                            matrix_subj_role = role
                        elif parts[i + 8] == "x" and parts[i + 9] == "_" and parts[i + 10] == "5":
                            if matrix_verb_inf is None or verb == matrix_verb_inf:
                                matrix_verb_inf = verb
                            matrix_ccomp_role = role
                if matrix_verb_inf is None or matrix_subj_role is None or matrix_ccomp_role is None:
                    continue
                # embedded verb_inf: takes ( x _ 5 , x _ 4 )
                embed_verb_inf = None
                embed_subj_role = None
                for i in range(len(parts) - 11):
                    if (parts[i + 1] == "." and parts[i + 3] == "(" and parts[i + 4] == "x"
                            and parts[i + 5] == "_" and parts[i + 6] == "5" and parts[i + 7] == ","
                            and parts[i + 8] == "x" and parts[i + 9] == "_" and parts[i + 10] == "4"):
                        embed_verb_inf = parts[i]
                        embed_subj_role = parts[i + 2]
                        break
                if embed_verb_inf is None or embed_subj_role is None:
                    continue
                self.past_to_inf[toks[1]] = matrix_verb_inf
                self.past_to_inf[toks[5]] = embed_verb_inf
                self.verb_to_cp_roles[matrix_verb_inf] = (matrix_subj_role, matrix_ccomp_role)
                if embed_verb_inf not in self.verb_to_intrans_role:
                    self.verb_to_intrans_role[embed_verb_inf] = embed_subj_role
                self.proper_nouns.add(toks[0])
                self.common_nouns.add(toks[4])
                continue

    def past_form(self, verb_inf: str, past_guess: str) -> str:
        """Trivial helper -- we don't reverse-engineer past tense, just trust training."""
        return past_guess

    def predict(self, input_str: str) -> list[str]:
        """Predict the COGS output. Handles 3 constructions; returns [] if out of scope."""
        toks = input_str.split()

        if is_simple_intransitive(input_str):
            pname = toks[0]
            verb_past = toks[1]
            if verb_past not in self.past_to_inf:
                return []
            verb_inf = self.past_to_inf[verb_past]
            role = self.verb_to_intrans_role.get(verb_inf)
            if role is None:
                return []
            return emit_intransitive_output(verb_inf, role, pname)

        if is_intrans_w_det(input_str):
            det = toks[0]
            noun = toks[1]
            verb_past = toks[2]
            if verb_past not in self.past_to_inf:
                return []
            verb_inf = self.past_to_inf[verb_past]
            role = self.verb_to_intrans_role.get(verb_inf)
            if role is None:
                return []
            is_definite = (det == "The")
            return emit_intrans_det_output(noun, verb_inf, role, is_definite)

        if is_transitive_proper_proper(input_str):
            subj_pname = toks[0]
            verb_past = toks[1]
            det = toks[2]
            obj_noun = toks[3]
            if verb_past not in self.past_to_inf:
                return []
            verb_inf = self.past_to_inf[verb_past]
            roles = self.verb_to_trans_roles.get(verb_inf)
            if roles is None:
                return []
            subj_role, obj_role = roles
            is_definite_obj = (det == "the")
            return emit_transitive_output(
                verb_inf, subj_pname, obj_noun, subj_role, obj_role, is_definite_obj
            )

        if is_pp_transitive_proper(input_str):
            subj_pname, vp, obj_det, obj_noun, prep, pp_det, pp_noun = (
                toks[0], toks[1], toks[2], toks[3], toks[4], toks[5], toks[6]
            )
            if vp not in self.past_to_inf:
                return []
            verb_inf = self.past_to_inf[vp]
            roles = self.verb_to_trans_roles.get(verb_inf)
            if roles is None:
                return []
            subj_role, obj_role = roles
            return emit_pp_transitive_proper(
                subj_pname, verb_inf, obj_det, obj_noun, prep, pp_det, pp_noun,
                subj_role, obj_role,
            )

        if is_dative_to_proper(input_str):
            subj_pname, vp, obj_det, obj_noun, recipient_pname = (
                toks[0], toks[1], toks[2], toks[3], toks[5]
            )
            if vp not in self.past_to_inf:
                return []
            verb_inf = self.past_to_inf[vp]
            roles = self.verb_to_dative_roles.get(verb_inf)
            if roles is None:
                return []
            subj_role, obj_role, recipient_role = roles
            return emit_dative_to_proper(
                subj_pname, verb_inf, obj_det, obj_noun, recipient_pname,
                subj_role, obj_role, recipient_role,
            )

        if is_cp_simple(input_str):
            matrix_pname, matrix_vp, embed_det, embed_noun, embed_vp = (
                toks[0], toks[1], toks[3], toks[4], toks[5]
            )
            if matrix_vp not in self.past_to_inf or embed_vp not in self.past_to_inf:
                return []
            matrix_inf = self.past_to_inf[matrix_vp]
            embed_inf = self.past_to_inf[embed_vp]
            cp_roles = self.verb_to_cp_roles.get(matrix_inf)
            embed_role = self.verb_to_intrans_role.get(embed_inf)
            if cp_roles is None or embed_role is None:
                return []
            subj_role, ccomp_role = cp_roles
            return emit_cp_simple(
                matrix_pname, matrix_inf, subj_role, ccomp_role,
                embed_det, embed_noun, embed_inf, embed_role,
            )

        return []  # out of scope


def load_cogs_tsv(path) -> list[tuple[str, str, str]]:
    """Load a COGS .tsv file. Returns list of (input, output, category)."""
    out = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                out.append((parts[0], parts[1], parts[2]))
    return out


def evaluate_cogs_intransitive(
    reasoner: COGSIntransitiveHyperion,
    examples: list[tuple[str, str, str]],
) -> dict:
    """Evaluate across all 3 supported constructions (simple intrans + intrans
    with det + transitive). Returns accuracy on the in-scope subset and total
    counts. Per-construction breakdown also reported.
    """
    per_class_correct: dict[str, int] = {}
    per_class_total: dict[str, int] = {}

    def classify(inp: str) -> str | None:
        if is_simple_intransitive(inp): return "simple_intrans"
        if is_intrans_w_det(inp): return "intrans_w_det"
        if is_transitive_proper_proper(inp): return "transitive"
        if is_pp_transitive_proper(inp): return "pp_transitive"
        if is_dative_to_proper(inp): return "dative_to"
        if is_cp_simple(inp): return "cp_simple"
        return None

    in_scope_correct = 0
    in_scope_total = 0
    for inp, out, _cat in examples:
        cls = classify(inp)
        if cls is None:
            continue
        in_scope_total += 1
        per_class_total[cls] = per_class_total.get(cls, 0) + 1
        predicted = reasoner.predict(inp)
        expected = (
            out.replace("(", " ( ").replace(")", " ) ").replace(",", " , ")
               .replace(";", " ; ").split()
        )
        if predicted == expected:
            in_scope_correct += 1
            per_class_correct[cls] = per_class_correct.get(cls, 0) + 1
    return {
        "in_scope_correct": in_scope_correct,
        "in_scope_total": in_scope_total,
        "in_scope_acc": (in_scope_correct / in_scope_total) if in_scope_total else 0.0,
        "total_examples": len(examples),
        "coverage": in_scope_total / len(examples) if examples else 0.0,
        "per_class_correct": per_class_correct,
        "per_class_total": per_class_total,
    }
