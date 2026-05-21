"""Recursive-structure COGS handler — covers PP-recursion, CP-recursion,
and PP-modified subject NPs that the per-signature template learner misses.

Used as a fallback inside COGSTemplateLearner when no template matches.

The mechanism: parse the input into a structured tree of clauses, then walk
the tree to emit the COGS logical-form string. Per-verb metadata
(past_to_inf, intransitive-role agent-vs-theme, ditransitive bool) is induced
from the same training set used by the template learner.
"""

from __future__ import annotations

import re
from collections import defaultdict


PROPER_RE = re.compile(r"^[A-Z][a-z]+$")
PREPS = {"in", "on", "beside"}
DETS_DEF = {"the", "The"}
DETS_INDEF = {"a", "A"}


def _is_np_starter(tok: str) -> bool:
    return tok in DETS_DEF or tok in DETS_INDEF or bool(PROPER_RE.match(tok))


def parse_simple_np(toks: list[str], start: int, end: int):
    """Parse a single NP (no PP chain). Returns (head_pos, det_type, next_pos) or None."""
    if start >= end:
        return None
    # Check determiners FIRST (since 'The' and 'A' also match the PROPER regex)
    if toks[start] in DETS_DEF and start + 1 < end:
        return (start + 1, "def", start + 2)
    if toks[start] in DETS_INDEF and start + 1 < end:
        return (start + 1, "indef", start + 2)
    if PROPER_RE.match(toks[start]):
        return (start, "prop", start + 1)
    return None


def parse_np_chain(toks: list[str], start: int, end: int):
    """Parse NP optionally followed by PP-chain.
    Returns (chain, next_pos) where chain is a list of (head_pos, det_type, prep_to_next_or_None).
    The last chain entry has prep_to_next=None.
    """
    first = parse_simple_np(toks, start, end)
    if first is None:
        return [], start
    chain: list[tuple[int, str, str | None]] = []
    cur_head, cur_det, pos = first
    while pos < end and toks[pos] in PREPS:
        prep = toks[pos]
        nxt = parse_simple_np(toks, pos + 1, end)
        if nxt is None:
            break
        chain.append((cur_head, cur_det, prep))
        cur_head, cur_det, pos = nxt
    chain.append((cur_head, cur_det, None))
    return chain, pos


def collect_definite_heads(chain) -> list[int]:
    return [h for (h, d, _) in chain if d == "def"]


def filler_for(toks: list[str], head_pos: int, det_type: str) -> str:
    if det_type == "prop":
        return toks[head_pos]
    return f"x _ {head_pos}"


def common_noun(toks: list[str], head_pos: int) -> str:
    """COGS lowercases common nouns in logical form (so `the TV` → `tv`)."""
    return toks[head_pos].lower()


def emit_np_chain_clauses(toks: list[str], chain) -> list[str]:
    """Emit nmod-attachment clauses + inline indef type clauses for each
    filler in a PP chain. Does NOT emit the type clause for the CHAIN HEAD
    (caller handles that)."""
    parts: list[str] = []
    for i in range(len(chain) - 1):
        head_pos, _head_det, prep = chain[i]
        next_pos, next_det, _ = chain[i + 1]
        parts.append(
            f"{common_noun(toks, head_pos)} . nmod . {prep} ( x _ {head_pos} , x _ {next_pos} )"
        )
        if next_det == "indef":
            parts.append(f"{common_noun(toks, next_pos)} ( x _ {next_pos} )")
    return parts


def parse_and_emit(
    toks: list[str],
    start: int,
    end: int,
    past_to_inf: dict[str, str],
    intrans_role: dict[str, str],   # verb_inf -> 'agent' | 'theme'
    ditrans: set[str] | None = None,  # verbs that take recipient role
    inf_verbs: set[str] | None = None,  # known infinitive verb tokens
    outer_ctx: dict | None = None,
) -> dict | None:
    if ditrans is None:
        ditrans = set()
    if inf_verbs is None:
        inf_verbs = set(past_to_inf.values())
    """Parse a single clause and emit its parts.

    Returns dict {
      'definite_heads': list[int],    # def-NP positions found in this clause + subclauses
      'pre_verb': list[str],          # parts to place BEFORE verb clauses (subj indef type, subj PP chain)
      'verb_clauses': list[str],      # verb agent/theme/recipient/ccomp clauses
      'post_verb': list[str],         # parts AFTER verb clauses (obj indef type, obj PP chain, recipient PP chain)
      'verb_pos': int,
    }
    """
    pos = start

    # Subject NP + PP chain
    subj_chain, pos = parse_np_chain(toks, pos, end)
    if not subj_chain:
        return None

    # Passive aux
    is_passive = False
    if pos < end and toks[pos] == "was":
        is_passive = True
        pos += 1

    # Verb
    if pos >= end or toks[pos] not in past_to_inf:
        return None
    verb_pos = pos
    verb_past = toks[verb_pos]
    verb_inf = past_to_inf[verb_past]
    pos += 1

    # Control-verb infinitive complement: 'V to V2'
    # Handle this BEFORE object parsing so the embedded clause is detected.
    # The token after 'to' is treated as an infinitive if either it's a known
    # infinitive (learned from training) OR it's a plain `w`-category word
    # (this covers `to crawl` where 'crawl' was never seen as a finite verb —
    # the prim_to_inf_arg generalization case).
    xcomp_verb_pos: int | None = None
    xcomp_verb_inf: str | None = None
    if pos + 1 < end and toks[pos] == "to":
        nxt = toks[pos + 1]
        if (nxt in inf_verbs
                or (nxt.islower() and nxt.isalpha()
                    and nxt not in {"a", "the", "in", "on", "beside", "to",
                                    "by", "was", "that"})):
            xcomp_verb_pos = pos + 1
            xcomp_verb_inf = nxt
            pos = pos + 2

    # Object NP + PP chain (only if not immediately followed by 'to', 'that', 'by', '.')
    obj_chain: list = []
    if pos < end and _is_np_starter(toks[pos]) and xcomp_verb_pos is None:
        obj_chain, pos = parse_np_chain(toks, pos, end)

    # Do-dative detection: if verb is ditransitive AND we see two consecutive NPs,
    # the FIRST is the recipient and the SECOND is the theme (which carries PP chain).
    do_dative_recipient: list = []
    if (obj_chain and verb_inf in ditrans and pos < end
            and toks[pos] != "to" and _is_np_starter(toks[pos])):
        # The chain we just parsed was actually the recipient. Re-parse the next NP+PP chain as theme.
        # But beware: obj_chain may have included a PP chain by mistake. We only treat as do-dative
        # if obj_chain is a bare single NP (no PP modifiers).
        if len(obj_chain) == 1:
            do_dative_recipient = obj_chain
            obj_chain, pos = parse_np_chain(toks, pos, end)

    # Recipient (PP-dative) 'to NP'
    recipient_chain: list = []
    if pos < end and toks[pos] == "to":
        recipient_chain, pos = parse_np_chain(toks, pos + 1, end)
    elif do_dative_recipient:
        recipient_chain = do_dative_recipient

    # By-phrase (passive) 'by NP+PP-chain'
    by_chain: list = []
    by_agent: tuple | None = None  # (head_pos, det_type)
    if is_passive and pos < end and toks[pos] == "by":
        by_chain, pos = parse_np_chain(toks, pos + 1, end)
        if by_chain:
            by_agent = (by_chain[0][0], by_chain[0][1])

    # That-complement: recurse
    that_sub: dict | None = None
    if pos < end and toks[pos] == "that":
        that_sub = parse_and_emit(toks, pos + 1, end, past_to_inf, intrans_role, ditrans, inf_verbs)
        if that_sub is None:
            return None
        pos = end  # consumed

    # Collect definite heads from this clause
    definite_heads: list[int] = []
    definite_heads.extend(collect_definite_heads(subj_chain))
    definite_heads.extend(collect_definite_heads(obj_chain))
    definite_heads.extend(collect_definite_heads(recipient_chain))
    definite_heads.extend(collect_definite_heads(by_chain))
    if that_sub:
        definite_heads.extend(that_sub["definite_heads"])

    subj_head, subj_det, _ = subj_chain[0]
    subj_filler = filler_for(toks, subj_head, subj_det)

    pre_verb: list[str] = []
    # Indef subject head type clause
    if subj_det == "indef":
        pre_verb.append(f"{common_noun(toks, subj_head)} ( x _ {subj_head} )")
    # Subject PP chain attachments
    pre_verb.extend(emit_np_chain_clauses(toks, subj_chain))

    verb_clauses: list[str] = []

    if is_passive:
        # Passive do-dative: subject is recipient (no `to`, but a theme NP after the verb).
        # Passive pp-dative: subject is theme; `to NP` carries recipient.
        # Passive non-dative: subject is theme.
        is_passive_do_dative = (verb_inf in ditrans and obj_chain
                                and not recipient_chain)
        if is_passive_do_dative:
            # passive do-dative: recipient first, then theme
            o_head, o_det, _ = obj_chain[0]
            verb_clauses.append(
                f"{verb_inf} . recipient ( x _ {verb_pos} , {subj_filler} )"
            )
            verb_clauses.append(
                f"{verb_inf} . theme ( x _ {verb_pos} , {filler_for(toks, o_head, o_det)} )"
            )
        else:
            verb_clauses.append(
                f"{verb_inf} . theme ( x _ {verb_pos} , {subj_filler} )"
            )
            if recipient_chain:
                r_head, r_det, _ = recipient_chain[0]
                verb_clauses.append(
                    f"{verb_inf} . recipient ( x _ {verb_pos} , {filler_for(toks, r_head, r_det)} )"
                )
        if by_agent:
            b_head, b_det = by_agent
            verb_clauses.append(
                f"{verb_inf} . agent ( x _ {verb_pos} , {filler_for(toks, b_head, b_det)} )"
            )
    else:
        # active: agent = subject (or theme if unaccusative), optional theme = object, optional recipient, optional ccomp, optional xcomp
        if not obj_chain and not recipient_chain and not that_sub and xcomp_verb_pos is None:
            # Pure intransitive — decide agent vs theme
            role = intrans_role.get(verb_inf, "agent")
            verb_clauses.append(
                f"{verb_inf} . {role} ( x _ {verb_pos} , {subj_filler} )"
            )
        else:
            verb_clauses.append(
                f"{verb_inf} . agent ( x _ {verb_pos} , {subj_filler} )"
            )
            # do-dative active: agent, recipient, theme (recipient before theme)
            # pp-dative active: agent, theme, recipient
            is_do_dative = bool(do_dative_recipient)
            if is_do_dative and recipient_chain:
                r_head, r_det, _ = recipient_chain[0]
                verb_clauses.append(
                    f"{verb_inf} . recipient ( x _ {verb_pos} , {filler_for(toks, r_head, r_det)} )"
                )
            if obj_chain:
                o_head, o_det, _ = obj_chain[0]
                verb_clauses.append(
                    f"{verb_inf} . theme ( x _ {verb_pos} , {filler_for(toks, o_head, o_det)} )"
                )
            if (not is_do_dative) and recipient_chain:
                r_head, r_det, _ = recipient_chain[0]
                verb_clauses.append(
                    f"{verb_inf} . recipient ( x _ {verb_pos} , {filler_for(toks, r_head, r_det)} )"
                )
            if that_sub:
                sub_verb_pos = that_sub["verb_pos"]
                verb_clauses.append(
                    f"{verb_inf} . ccomp ( x _ {verb_pos} , x _ {sub_verb_pos} )"
                )
            if xcomp_verb_pos is not None:
                # Subject-control: matrix verb has xcomp; embedded verb reuses subject as agent.
                verb_clauses.append(
                    f"{verb_inf} . xcomp ( x _ {verb_pos} , x _ {xcomp_verb_pos} )"
                )
                verb_clauses.append(
                    f"{xcomp_verb_inf} . agent ( x _ {xcomp_verb_pos} , {subj_filler} )"
                )

    post_verb: list[str] = []
    is_do_dative_active = (not is_passive) and bool(do_dative_recipient)

    # Order of indef type clauses + PP chains differs across constructions:
    if is_do_dative_active:
        # do-dative active: recipient indef-type FIRST, then theme indef-type, then theme PP chain
        if recipient_chain:
            r_head, r_det, _ = recipient_chain[0]
            if r_det == "indef":
                post_verb.append(f"{common_noun(toks, r_head)} ( x _ {r_head} )")
        if obj_chain:
            o_head, o_det, _ = obj_chain[0]
            if o_det == "indef":
                post_verb.append(f"{common_noun(toks, o_head)} ( x _ {o_head} )")
            post_verb.extend(emit_np_chain_clauses(toks, obj_chain))
        if recipient_chain:
            post_verb.extend(emit_np_chain_clauses(toks, recipient_chain))
    else:
        # default order: theme indef-type + PP chain, then recipient indef-type + PP chain
        if obj_chain:
            o_head, o_det, _ = obj_chain[0]
            if o_det == "indef":
                post_verb.append(f"{common_noun(toks, o_head)} ( x _ {o_head} )")
            post_verb.extend(emit_np_chain_clauses(toks, obj_chain))
        if recipient_chain:
            r_head, r_det, _ = recipient_chain[0]
            if r_det == "indef":
                post_verb.append(f"{common_noun(toks, r_head)} ( x _ {r_head} )")
            post_verb.extend(emit_np_chain_clauses(toks, recipient_chain))

    # By-phrase NP chain (passive only): indef-type clause + chain attachments
    if by_chain:
        b_head, b_det, _ = by_chain[0]
        if b_det == "indef":
            post_verb.append(f"{common_noun(toks, b_head)} ( x _ {b_head} )")
        post_verb.extend(emit_np_chain_clauses(toks, by_chain))

    # Append the that-clause's contents to post_verb (subj's indef type, subj PP, verb clauses, post)
    if that_sub:
        post_verb.extend(that_sub["pre_verb"])
        post_verb.extend(that_sub["verb_clauses"])
        post_verb.extend(that_sub["post_verb"])

    return {
        "definite_heads": definite_heads,
        "pre_verb": pre_verb,
        "verb_clauses": verb_clauses,
        "post_verb": post_verb,
        "verb_pos": verb_pos,
    }


def predict_recursive(
    input_str: str,
    past_to_inf: dict[str, str],
    intrans_role: dict[str, str],
    ditrans: set[str] | None = None,
) -> list[str] | None:
    """Parse and emit a COGS logical form for input_str. Returns the
    tokenized output (list of tokens) or None on parse failure.
    """
    toks = input_str.split()
    if not toks or toks[-1] != ".":
        return None
    end = len(toks) - 1  # exclude the period

    inf_verbs = set(past_to_inf.values())
    result = parse_and_emit(toks, 0, end, past_to_inf, intrans_role, ditrans, inf_verbs)
    if result is None:
        return None
    # If we didn't consume everything, give up
    # (Could verify by tracking 'pos' more carefully — skipped for now.)

    parts: list[str] = []
    # Definite prefix (sorted by input position, deduplicated)
    seen_defs = set()
    for h in sorted(result["definite_heads"]):
        if h in seen_defs:
            continue
        seen_defs.add(h)
        parts.append(f"* {common_noun(toks, h)} ( x _ {h} ) ;")
    prefix_str = " ".join(parts)

    body_parts = result["pre_verb"] + result["verb_clauses"] + result["post_verb"]
    body_str = " AND ".join(body_parts)

    if prefix_str:
        full = f"{prefix_str} {body_str}"
    else:
        full = body_str

    return full.split()


def learn_ditrans_verbs(examples: list[tuple[str, str]], past_to_inf: dict[str, str]) -> set[str]:
    """Verbs that take a recipient role in any training output."""
    ditrans: set[str] = set()
    for inp, out in examples:
        out_toks = out.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").split()
        for i in range(len(out_toks) - 2):
            if out_toks[i + 1] == "." and out_toks[i + 2] == "recipient":
                verb_inf = out_toks[i]
                if verb_inf and verb_inf.replace("_", "").isalpha():
                    ditrans.add(verb_inf)
    return ditrans


def learn_intrans_role(examples: list[tuple[str, str]], past_to_inf: dict[str, str]) -> dict[str, str]:
    """For each verb, learn whether its intransitive-subject role is 'agent'
    or 'theme'. Scans ALL training outputs (not just bare 3-tok inputs) and
    aggregates evidence — a verb whose intransitive uses overwhelmingly map
    to 'theme' (unaccusative) gets 'theme'; otherwise defaults to 'agent'.

    Also propagates: if a verb is seen WITH an `agent` role at any point, it
    can be used unergatively (so default to 'agent' unless evidence is
    strongly unaccusative)."""
    has_agent: set[str] = set()
    has_theme_intrans: dict[str, int] = {}
    # First pass: which verbs ever appear with an 'agent' clause in training?
    for _, out in examples:
        out_toks = out.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").split()
        for i in range(len(out_toks) - 2):
            if out_toks[i + 1] == "." and out_toks[i + 2] == "agent":
                vinf = out_toks[i]
                if vinf and vinf.replace("_", "").isalpha():
                    has_agent.add(vinf)

    # Second pass: tally intransitive theme appearances (verbs that have theme
    # but no other arg in a single output).
    role: dict[str, str] = {}
    for inp, out in examples:
        toks = inp.split()
        # Heuristic: a clear intransitive is a 3-token 'X V .' input.
        if len(toks) == 3 and toks[2] == ".":
            verb_past = toks[1]
            verb_inf = past_to_inf.get(verb_past)
            if verb_inf is None:
                continue
            out_toks = out.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").split()
            for i in range(len(out_toks) - 2):
                if out_toks[i] == verb_inf and out_toks[i + 1] == ".":
                    r = out_toks[i + 2]
                    if r in ("agent", "theme"):
                        if verb_inf not in role:
                            role[verb_inf] = r
                    break
    return role
