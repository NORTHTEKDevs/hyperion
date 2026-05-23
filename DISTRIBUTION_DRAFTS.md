# Hyperion distribution drafts

Use or ignore. These are starting points, not final copy.

---

## Hacker News submission

**Title (one of):**

- Hyperion: Compositional reasoning at 100% / 99.98% / 99.75% / 100% on four benchmarks, no neural training
- Show HN: Compositional reasoning benchmarks solved by ~1000 lines of Python (no gradient descent)
- Hyperion: 100% on 1D-ARC and SCAN, 99.75% on COGS gen, no neural networks

**URL:** https://github.com/NORTHTEKDevs/hyperion

**Lead comment (post immediately after submission):**

> I built this as a research bet: how much of "compositional generalization" — the kind LLMs famously struggle with — actually needs neural networks?
>
> The answer on these four benchmarks turned out to be "none of it." Vector Symbolic Architecture algebra handles SCAN and PCFG. Template induction + a recursive descent parser handles COGS. Program synthesis with an Occam-prior over a small primitive library handles 1D-ARC.
>
> Numbers:
> - SCAN: 100.00% on all 7 splits (27,141 / 27,141)
> - PCFG: 99.98%
> - COGS gen: 99.75% (Kim & Linzen 2020 transformer baseline: ~35%)
> - 1D-ARC: 100% on all 18 task types (Joffe & Eliasmith 2025 hand-crafted VSA: 83%)
>
> All tests gate the percentages — clone, run pytest, see for yourself.
>
> What it isn't: a ChatGPT replacement. It works in domains with learnable rule structure (formal languages, structured puzzles, simplified grammars). Not general English, not arbitrary text. The COGS parser knows the COGS output schema; what it learns from data is the lexicon and parameter values.
>
> Honest limitation: 2D ARC-AGI (the real one) is the next test. Open question whether the approach extends.
>
> README explains it for non-technical readers, with verification steps. Writeup at https://northtek.io/labs/hyperion

---

## Twitter / X thread

**Tweet 1 (lead):**

> Spent a few sessions building a non-neural compositional reasoning system.
>
> Results on four published academic benchmarks:
> - SCAN: 100%
> - PCFG: 99.98%
> - COGS gen: 99.75%
> - 1D-ARC: 100%
>
> Zero gradient descent. ~1,000 lines of Python.
>
> Open source: github.com/NORTHTEKDevs/hyperion

**Tweet 2:**

> For comparison:
>
> - Vanilla transformers on COGS gen: ~35% (Kim & Linzen 2020)
> - Best published neuro-symbolic on COGS: 60-80%
> - Joffe & Eliasmith 2025 on 1D-ARC (hand-crafted VSA): 83%
>
> The bar isn't 100% on these. The bar is showing it can be done without a billion-param model.

**Tweet 3:**

> How:
>
> SCAN/PCFG → Vector Symbolic Architecture (bipolar 8192-dim, bind/bundle/permute)
>
> COGS → template induction + recursive descent parser
>
> 1D-ARC → program synthesis over a small primitive library, Occam-prior ordering, parameters induced from training pairs

**Tweet 4:**

> What it isn't:
>
> Not a ChatGPT replacement. Won't summarize PDFs. Won't write poetry.
>
> Works in domains with learnable rule structure: formal languages, structured puzzles, simplified grammars.
>
> 2D ARC-AGI is the next test. Open question whether the approach extends.

**Tweet 5 (close):**

> Tests gate the numbers. Clone the repo, run pytest, see for yourself.
>
> Writeup with the full story (in plain English):
> https://northtek.io/labs/hyperion
>
> Code:
> https://github.com/NORTHTEKDevs/hyperion
>
> MIT licensed.

---

## LinkedIn post (longer, professional framing)

> **What I learned this week: a chunk of "AI reasoning" might not need neural networks.**
>
> Spent a few overnight sessions building Hyperion — an open-source compositional reasoning system. On four published academic benchmarks (the standardized tests researchers use to measure whether an AI can apply rules to new situations), it scores at or near 100%:
>
> SCAN: 100% (27,141 of 27,141)
> PCFG: 99.98%
> COGS gen: 99.75% (vs. ~35% for vanilla transformers in the original Kim & Linzen 2020 paper)
> 1D-ARC: 100% on all 18 task types (vs. 83% for the prior best published method)
>
> No GPU. No training. No neural networks. About 1,000 lines of Python that runs in seconds on a laptop.
>
> This is not a ChatGPT replacement. It works on domains with learnable rule structure — formal languages, structured puzzles, simplified grammars. Not general English. The "research finding" is narrower than it sounds: it's evidence that compositional generalization on these specific benchmarks might be a smaller problem than the field has assumed, solvable with structured search over the right hypothesis space.
>
> Open-sourced under MIT. Writeup explains it in plain English with verification steps:
>
> https://northtek.io/labs/hyperion
>
> Code: https://github.com/NORTHTEKDevs/hyperion
>
> Built as part of Northtek Labs.

---

## Reddit (r/MachineLearning, "[R]" tag)

**Title:** [R] Compositional generalization benchmarks (SCAN, PCFG, COGS, 1D-ARC) solved at 99.75-100% by VSA algebra + template induction + program synthesis, no gradient descent

**Body:**

> Open-sourcing a system that hits:
>
> - SCAN: 100% on all 7 splits
> - PCFG: 99.98%
> - COGS gen: 99.75% (all 21 categories ≥99%)
> - 1D-ARC: 100% on all 18 task types
>
> Repo: https://github.com/NORTHTEKDevs/hyperion (MIT)
>
> **Mechanism per benchmark:**
> - SCAN/PCFG: bipolar 8192-dim hypervectors with bind/bundle/permute; grammar hand-encoded, composition algebraic.
> - COGS: template induction (auto-learned input-signature → output-template map with LIT/COPY/INF/VERB_COND slot types) + a recursive descent parser fallback that handles PP/CP recursion, control verbs, ditransitive variants. Parameters (past→inf, intransitive role per verb, ditransitive verb set) learned from training only.
> - 1D-ARC: enumerative program synthesis over ~25 generic grid primitives, parameter induction for color mappings (length→color, longest→color, parity→color), Occam-prior ordering to break ties.
>
> **Honest scope:**
> - The COGS parser hand-codes knowledge of the output schema (role labels, definite-prefix conventions). What's learned from data is the lexicon and the parameters. This is data-driven parameter induction over a structured hypothesis space, not "discover the language from scratch."
> - 1D-ARC is much easier than 2D ARC-AGI. Extending the program-synthesis approach to 2D is the obvious next experiment.
> - "Comparison to transformer baselines at ~35% on COGS gen" is from Kim & Linzen 2020. I have not directly benchmarked GPT-4 or similar.
>
> Tests gate the headline numbers. `pytest tests/` asserts each threshold.
>
> Plain-English writeup with verification instructions: https://northtek.io/labs/hyperion
>
> Curious for criticism, especially on the "what's encoded vs. learned" framing in the README.

---

## When to post what

**Highest signal, fastest:** Hacker News on a weekday morning Pacific time. Single submission. Honest title (no clickbait). Stay in the thread for the first hour to answer questions.

**LinkedIn:** Once. Tag Northtek. Lets the agency-customer audience see you ship real research.

**Twitter/X:** The thread above. Pin to profile.

**Reddit r/MachineLearning:** Use the [R] tag, post the version above. The audience there will appreciate the "honest scope" section.

**Don't:** mass-post the same thing in 10 places in one day. Pick one or two, let it breathe.
