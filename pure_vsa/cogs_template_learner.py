"""COGS template learner: auto-extract (input-signature, output-template) pairs
from training data so we don't need a hand-written handler per construction.

For each training pair (input_str, output_str):
  1. Compute the input signature: replace each token with its CATEGORY
     (PROP / DET / aux-word like 'was', 'by' / VERB_PAST / NOUN / .).
  2. Compute the output template: each output token is either
     - LITERAL: ".", "(", ")", ",", ";", "AND", "x", "_", "1", "2", ...,
       known role labels (agent, theme, recipient, ccomp, xcomp, ...),
       "*"
     - COPY[i]: a verbatim copy of input token at position i
     - INF[i]: the infinitive form of the verb-past token at input position i
  3. Group examples by input signature; within each group, every output
     should follow the same template (the SCAN/COGS grammar is regular).

At test time: classify the input by signature, look up the template,
fill slots from the test input's positions.

This is grammar induction. The verb→infinitive mapping is learned
separately from training (same way the per-construction handlers learned it).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass


PROPER_RE = re.compile(r"^[A-Z][a-z]+$")
FUNCTION_WORDS = {"a", "the", "A", "The", "was", "by", "that", "to",
                  "in", "on", "beside"}


def token_category(tok: str) -> str:
    """Classify an input token into a structural category."""
    if tok == ".":
        return "."
    if tok in FUNCTION_WORDS:
        return tok  # function words keep their identity
    if PROPER_RE.match(tok):
        return "PROP"
    return "w"


def input_signature(input_str: str) -> tuple[str, ...]:
    return tuple(token_category(t) for t in input_str.split())


# Output tokens that are always literal regardless of input
KNOWN_LITERALS = {
    ".", "(", ")", ",", ";", "AND", "x", "_", "*",
    # numerals
    *[str(i) for i in range(20)],
    # role labels — these appear in COGS outputs and are not input tokens
    "agent", "theme", "recipient", "ccomp", "xcomp",
    "nmod", "in", "on", "beside",  # PP role names
}


@dataclass
class TemplateSlot:
    kind: str  # 'LIT' | 'COPY' | 'INF' | 'VERB_COND'
    value: str
    # For VERB_COND: value is the unique slot-key "sig|pos|n" used to look up
    # the per-verb token in self.verb_conditioned_lookup.


@dataclass
class Template:
    signature: tuple[str, ...]
    slots: list[TemplateSlot]
    n_examples: int


def output_to_template_slots(
    input_toks: list[str],
    output_toks: list[str],
    past_to_inf: dict[str, str],
) -> list[TemplateSlot] | None:
    """For each output token, decide if it's LIT, COPY[i], or INF[i].

    Returns None if any output token can't be classified consistently.
    """
    slots: list[TemplateSlot] = []
    for ot in output_toks:
        # 1. Literal token?
        if ot in KNOWN_LITERALS:
            slots.append(TemplateSlot("LIT", ot))
            continue
        # 2. Infinitive of an input verb-past token? (Checked BEFORE COPY so
        # verbs whose past form == infinitive (e.g., "put"/"put") classify as
        # INF consistently. Without this, identical surface tokens at the
        # verb position would split between COPY and INF.)
        inf_idx = None
        for i, it in enumerate(input_toks):
            if past_to_inf.get(it) == ot:
                inf_idx = i
                break
        if inf_idx is not None:
            slots.append(TemplateSlot("INF", str(inf_idx)))
            continue
        # 3. Verbatim copy of an input token?
        copy_idx = None
        for i, it in enumerate(input_toks):
            if it == ot:
                copy_idx = i
                break
        if copy_idx is not None:
            slots.append(TemplateSlot("COPY", str(copy_idx)))
            continue
        # Unrecognizable; can't classify this output token.
        return None
    return slots


def normalize_output(output_str: str) -> list[str]:
    """Tokenize an output string with punctuation expanded."""
    return (
        output_str
        .replace("(", " ( ")
        .replace(")", " ) ")
        .replace(",", " , ")
        .replace(";", " ; ")
        .split()
    )


class COGSTemplateLearner:
    """Learn (input_signature -> output_template) mappings from training data.

    Iteratively bootstraps: starts by extracting past->inf from bare-verb
    examples, then learns templates that depend on past->inf, then learns
    more past->inf from those examples, repeat until stable.
    """

    def __init__(self) -> None:
        self.past_to_inf: dict[str, str] = {}
        self.templates: dict[tuple[str, ...], Template] = {}
        # Track signature conflicts (multiple distinct templates per signature)
        self.signature_conflicts: set[tuple[str, ...]] = set()
        # Verb-conditioned lookup: slot_key -> {verb_inf -> token}
        # slot_key is "sig|pos|verb_inp_pos", value is the actual token
        # to emit when this verb is at this signature position.
        self.verb_conditioned_lookup: dict[str, dict[str, str]] = {}
        # Recursive-structure fallback metadata
        self.intrans_role: dict[str, str] = {}
        self.ditrans_verbs: set[str] = set()

    def _fit_recursive_metadata(self, examples: list[tuple[str, str]]) -> None:
        """Learn the auxiliary maps used by the recursive-structure fallback."""
        # Import lazily to avoid circular imports
        from pure_vsa.cogs_recursive import learn_intrans_role, learn_ditrans_verbs
        self.intrans_role = learn_intrans_role(examples, self.past_to_inf)
        self.ditrans_verbs = learn_ditrans_verbs(examples, self.past_to_inf)

    def fit(self, examples: list[tuple[str, str]]) -> None:
        """examples: list of (input_str, output_str) tuples."""
        # Bootstrap pass 1: past->inf from bare-verb examples (3-tok PROP w .)
        for inp, out in examples:
            toks = inp.split()
            out_toks = normalize_output(out)
            if len(toks) == 3 and PROPER_RE.match(toks[0]) and toks[2] == ".":
                if len(out_toks) >= 1:
                    verb_past = toks[1]
                    verb_inf_candidate = out_toks[0]
                    self.past_to_inf.setdefault(verb_past, verb_inf_candidate)

        # Iterative bootstrap: expand past_to_inf using verb-position discovery.
        # For each example, locate the verb-past token via _find_verb_input_pos,
        # then look up its infinitive in the output (it appears in the patterns
        # `<verb_inf> . <role> (...)`). The verb_inf is always at a fixed
        # output token: the first token after stripping any leading definite
        # marker `* noun ( x _ N ) ;`.
        for _ in range(5):
            new_past_to_inf: dict[str, str] = {}
            for inp, out in examples:
                toks = inp.split()
                sig = input_signature(inp)
                verb_pos = self._find_verb_input_pos(sig)
                if verb_pos is None or verb_pos >= len(toks):
                    continue
                verb_past = toks[verb_pos]
                if verb_past in self.past_to_inf:
                    continue
                # Search the output for the first `<verb> . <role> (` pattern.
                # The verb_inf is the same across all uses of this verb in
                # one example (matrix verb in active/passive, etc.), so any
                # occurrence works.
                out_toks = normalize_output(out)
                candidate = None
                role_labels = {"agent", "theme", "recipient", "ccomp", "xcomp"}
                for j in range(len(out_toks) - 3):
                    if (out_toks[j + 1] == "."
                            and out_toks[j + 2] in role_labels
                            and out_toks[j + 3] == "("):
                        tok = out_toks[j]
                        if tok not in KNOWN_LITERALS:
                            candidate = tok
                            break
                if candidate is not None:
                    new_past_to_inf[verb_past] = candidate
            if not new_past_to_inf:
                break
            self.past_to_inf.update(new_past_to_inf)

        # Now extract templates per signature
        sig_to_slot_lists: dict[tuple[str, ...], list[list[TemplateSlot]]] = defaultdict(list)
        for inp, out in examples:
            toks = inp.split()
            out_toks = normalize_output(out)
            slots = output_to_template_slots(toks, out_toks, self.past_to_inf)
            if slots is None:
                continue
            sig = input_signature(inp)
            sig_to_slot_lists[sig].append(slots)

        # We also need the original input toks per slot_list to learn
        # verb-conditioned values. Re-collect with toks attached.
        sig_to_slot_with_input: dict[tuple[str, ...], list[tuple[list[TemplateSlot], list[str]]]] = defaultdict(list)
        for inp, out in examples:
            toks = inp.split()
            out_toks = normalize_output(out)
            slots = output_to_template_slots(toks, out_toks, self.past_to_inf)
            if slots is None:
                continue
            sig = input_signature(inp)
            sig_to_slot_with_input[sig].append((slots, toks))

        for sig, slot_with_inputs in sig_to_slot_with_input.items():
            slot_lists = [s for s, _ in slot_with_inputs]
            input_lists = [t for _, t in slot_with_inputs]
            # Check all slot_lists have same length
            lengths = {len(sl) for sl in slot_lists}
            if len(lengths) != 1:
                self.signature_conflicts.add(sig)
                continue
            n = lengths.pop()

            # Build the unified template, allowing per-position verb conditioning.
            verb_input_pos = self._find_verb_input_pos(sig)

            template_slots: list[TemplateSlot] = []
            consistent = True
            for pos in range(n):
                values = {(sl[pos].kind, sl[pos].value) for sl in slot_lists}
                if len(values) == 1:
                    # Consistent; use directly
                    template_slots.append(slot_lists[0][pos])
                    continue
                # Conflict at this position. Can verb-conditioning resolve it?
                if verb_input_pos is None:
                    consistent = False
                    break
                # Check if the slot kind is LIT (i.e., a role label or similar literal
                # token that varies with the verb). VERB_COND only makes sense for LIT.
                kinds = {sl[pos].kind for sl in slot_lists}
                if kinds != {"LIT"}:
                    consistent = False
                    break
                # Build verb -> token lookup at this position
                slot_key = f"{'|'.join(sig)}|{pos}|{verb_input_pos}"
                per_verb_token: dict[str, str] = {}
                for sl, toks in slot_with_inputs:
                    verb_past = toks[verb_input_pos]
                    verb_inf = self.past_to_inf.get(verb_past)
                    if verb_inf is None:
                        consistent = False
                        break
                    tok = sl[pos].value
                    if verb_inf in per_verb_token and per_verb_token[verb_inf] != tok:
                        # Same verb produces different tokens at this position — give up
                        consistent = False
                        break
                    per_verb_token[verb_inf] = tok
                if not consistent:
                    break
                self.verb_conditioned_lookup[slot_key] = per_verb_token
                template_slots.append(TemplateSlot("VERB_COND", slot_key))

            if not consistent:
                self.signature_conflicts.add(sig)
                continue
            self.templates[sig] = Template(
                signature=sig,
                slots=template_slots,
                n_examples=len(slot_lists),
            )

        # Fit recursive fallback metadata
        self._fit_recursive_metadata(examples)

    def _find_verb_input_pos(self, sig: tuple[str, ...]) -> int | None:
        """Heuristic: find the main-verb input position.

        Patterns:
          PROP w ...                -> verb at 1
          DET w w ...               -> verb at 2
          DET w was w ...           -> verb at 3 (passive)
          PROP w PROP ...           -> verb at 1
          DET w w to ...            -> verb at 2 (control)
          PROP w to ...             -> verb at 1
        """
        n = len(sig)
        if n == 0:
            return None
        # Skip subject
        if sig[0] == "PROP":
            i = 1
        elif n >= 2 and sig[0] in ("A", "The") and sig[1] == "w":
            i = 2
        else:
            return None
        # Skip auxiliary "was" if present (passive)
        if i < n and sig[i] == "was":
            i += 1
        # Now i should be the verb position
        if i < n and sig[i] == "w":
            return i
        return None

    def predict(self, input_str: str) -> list[str]:
        """Apply the learned template for this input's signature.

        Strategy:
        - Single-example templates (n_examples == 1) are vulnerable to
          spurious-correlation overfit (e.g., a training example where the
          outer and inner CP verbs happen to be the same forces the
          template to emit INF[1] at the inner verb's slot, which fails
          when the test has different verbs). In that case, defer to the
          recursive parser if it succeeds.
        - Otherwise prefer the (multi-example-validated) template.
        - Always fall back to the recursive parser for signatures with no template.
        """
        toks = input_str.split()
        sig = tuple(token_category(t) for t in toks)
        template = self.templates.get(sig)

        def _recursive():
            from pure_vsa.cogs_recursive import predict_recursive
            return predict_recursive(
                input_str, self.past_to_inf, self.intrans_role, self.ditrans_verbs
            )

        if template is None:
            r = _recursive()
            return r if r is not None else []
        # If template was learned from a single example, try the parser first
        if template.n_examples == 1:
            r = _recursive()
            if r is not None:
                return r
        out: list[str] = []
        for slot in template.slots:
            if slot.kind == "LIT":
                out.append(slot.value)
            elif slot.kind == "COPY":
                idx = int(slot.value)
                if idx >= len(toks):
                    return []
                out.append(toks[idx])
            elif slot.kind == "INF":
                idx = int(slot.value)
                if idx >= len(toks):
                    return []
                verb_past = toks[idx]
                inf = self.past_to_inf.get(verb_past)
                if inf is None:
                    return []
                out.append(inf)
            elif slot.kind == "VERB_COND":
                slot_key = slot.value
                # parse verb_input_pos from key
                parts = slot_key.split("|")
                if not parts:
                    return []
                verb_input_pos = int(parts[-1])
                if verb_input_pos >= len(toks):
                    return []
                verb_past = toks[verb_input_pos]
                verb_inf = self.past_to_inf.get(verb_past)
                if verb_inf is None:
                    return []
                lookup = self.verb_conditioned_lookup.get(slot_key)
                if lookup is None or verb_inf not in lookup:
                    return []
                out.append(lookup[verb_inf])
        return out

    def coverage_stats(self, examples) -> dict:
        """Evaluate coverage and accuracy on a list of (input, output[, cat]) tuples.
        An example is considered 'in scope' if either a template matches its
        signature OR the recursive fallback returns a non-empty prediction.
        """
        n = len(examples)
        in_scope = 0
        correct = 0
        per_cat_correct: dict[str, int] = defaultdict(int)
        per_cat_total: dict[str, int] = defaultdict(int)
        for ex in examples:
            inp, out = ex[0], ex[1]
            cat = ex[2] if len(ex) > 2 else "?"
            per_cat_total[cat] += 1
            pred = self.predict(inp)
            if pred:
                in_scope += 1
                expected = normalize_output(out)
                if pred == expected:
                    correct += 1
                    per_cat_correct[cat] += 1
        return {
            "n": n,
            "in_scope": in_scope,
            "coverage": in_scope / n if n else 0.0,
            "correct": correct,
            "in_scope_acc": correct / in_scope if in_scope else 0.0,
            "per_cat_correct": dict(per_cat_correct),
            "per_cat_total": dict(per_cat_total),
        }
