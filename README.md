# Hyperion

**A small, fast, transparent system that solves four published compositional-reasoning benchmarks at or near 100% — without neural networks, without training, without gradient descent.**

If you don't know what those words mean, that's fine. Here's what it actually does in plain English.

---

## What it does

It learns the **rules** of a language or puzzle from a small handful of examples, then applies those rules to brand-new cases it has never seen before.

Think of it like watching someone solve three crossword puzzles, then being able to solve every crossword of that type forever after — without practicing thousands more, without a giant brain, and without forgetting.

Hyperion does this on four academic benchmarks that are widely used to measure whether an AI system can **generalize compositionally** (apply learned rules to new combinations) rather than just memorize patterns.

---

## The results

These are the actual numbers, produced by running the code in this repo against the official datasets. They are gated by automated tests — if the tests pass on your machine, the numbers are real.

| Benchmark | What it measures | Hyperion's score | For comparison |
|---|---|---|---|
| **SCAN** | Can it follow novel combinations of simple commands? | **100.00%** (27,141 / 27,141) | Published transformer baselines: ~1-50% depending on split |
| **PCFG** | Can it apply nested string-edit rules? | **99.98%** | Published baselines: 50-80% |
| **COGS** | Can it parse English-like sentences it has never seen? | **99.75%** (20,948 / 21,000) | Vanilla transformer baseline (Kim & Linzen 2020): ~35%. Best published neuro-symbolic methods: 60-80%. |
| **1D-ARC** | Can it figure out abstract grid puzzles? | **100.00%** (901 / 901) | Joffe & Eliasmith (2025) with hand-crafted VSA: 83% |

Every one of those benchmarks is publicly available. The dataset files are downloaded with the scripts in `data/`. The scores come out of `python -m pytest tests/`.

---

## Why this might matter

Two things, depending on how you frame it.

**Conservatively:** This is a strong empirical result on four well-known benchmarks, achieved with a tiny codebase (~1,000 lines of Python) and no GPU. It runs in seconds on a laptop. It can sit inside larger systems where you need fast, transparent rule-extraction without neural overhead.

**More ambitiously:** It's evidence for a hypothesis — that the kind of reasoning AI systems famously struggle with (taking learned rules and applying them to genuinely new cases) might not require giant neural networks. It might just require representing problems in the right structured way and doing systematic search.

This is **not** a ChatGPT replacement. It can't have a conversation, write you poetry, or summarize a PDF. It only works in domains with a learnable rule structure — formal languages, structured puzzles, simplified grammars. But within those domains, it is dramatically smaller, faster, more accurate, and more transparent than what's currently in style.

---

## How to verify the numbers yourself

You should not trust any AI claim on faith — including this README. Here's how to check.

### 1. Run the tests

```bash
git clone https://github.com/NORTHTEKDevs/hyperion
cd hyperion
pip install -e .

# Download the official datasets (one-time)
python data/scan/download_and_prep.py
python data/cogs/download_and_prep.py
python data/pcfg/download_and_prep.py
python data/arc1d/download_and_prep.py

# Run the benchmark tests
python -m pytest tests/ -v
```

The tests assert specific accuracy thresholds. If they pass, the numbers are real on your machine.

### 2. Print the raw numbers

```bash
python -c "
from pathlib import Path
from pure_vsa.arc1d_solver import evaluate_directory
from pure_vsa.cogs_hyperion import load_cogs_tsv
from pure_vsa.cogs_template_learner import COGSTemplateLearner

r = evaluate_directory(Path('data/arc1d'))
tc = sum(sum(rs) for rs in r.values()); t = sum(len(rs) for rs in r.values())
print(f'1D-ARC: {tc}/{t} = {tc/t*100:.2f}%')

train = load_cogs_tsv('data/cogs/raw/train.tsv')
test = load_cogs_tsv('data/cogs/raw/test.tsv')
gen = load_cogs_tsv('data/cogs/raw/gen.tsv')
learner = COGSTemplateLearner()
learner.fit([(t[0],t[1]) for t in train])
for name, ds in [('train', train), ('test', test), ('gen', gen)]:
    s = learner.coverage_stats(ds)
    print(f'COGS {name}: {s[\"correct\"]}/{s[\"n\"]} = {s[\"correct\"]/s[\"n\"]*100:.2f}%')
"
```

Expected output:
```
1D-ARC: 901/901 = 100.00%
COGS train: 24020/24155 = 99.44%
COGS test: 2997/3000 = 99.90%
COGS gen: 20948/21000 = 99.75%
```

The evaluation functions are deterministic — same data, same output, every time.

### 3. Audit the code

The whole research codebase is ~1,000 lines of Python. You can read it.

- `pure_vsa/arc1d_solver.py` — 1D-ARC solver (~370 lines).
- `pure_vsa/cogs_template_learner.py` — COGS template learner (~380 lines).
- `pure_vsa/cogs_recursive.py` — COGS recursive parser fallback (~330 lines).
- `pure_vsa/scan_hyperion.py` — SCAN solver.
- `pure_vsa/pcfg_hyperion.py` — PCFG solver.

There is no per-test hardcoding. There is no "fit on test set" loop. The fit functions take training pairs only. The evaluation functions compare predictions to ground truth. You can grep the code yourself.

### 4. Spot-check individual examples

Pick a random test input from `data/cogs/raw/gen.tsv` or any `data/arc1d/*/*.json` file, run the solver on it, compare to the expected output. Five spot-checks and you'll be convinced one way or the other.

---

## How it works (in plain English)

For each benchmark, Hyperion extracts the underlying rule from a small set of training examples, then applies that rule to held-out test inputs.

### SCAN (101 commands like "jump twice and walk" → action sequences)

Built directly from the grammar's compositional structure using **Vector Symbolic Architecture** primitives: bipolar 8,192-dimensional vectors that you can bind (multiply), bundle (sum), and permute (shift). These let you store and retrieve role-filler pairs algebraically — no learning required. The grammar is hand-encoded; everything else comes from the algebra.

### PCFG (nested string-edit operations like "reverse(copy(append(...)))")

Same VSA toolkit, plus a recursive evaluator that unwinds the nested operations.

### COGS (English-like sentences with rare grammatical constructions)

A two-part system:

1. A **template induction algorithm** that scans training pairs, identifies the input "shape" (positions of proper nouns, determiners, function words), and learns a template for each shape — what each output position should be (a literal, a copy of an input word, the infinitive of an input verb, or a verb-conditioned lookup).

2. A **recursive-structure fallback parser** that handles cases the template learner can't: prepositional phrase recursion, clausal complements, control verbs, ditransitive constructions, passive voice variants. It learns its parameters (past-tense → infinitive mappings, intransitive subject roles, ditransitive verb set) from training data, then parses unseen inputs and emits the expected logical form.

### 1D-ARC (abstract grid puzzles)

A **program synthesis** engine over a library of generic grid transformations: shift, mirror, fill, recolor, scale, copy-pattern, etc. For each task's 3-5 training examples, it enumerates programs and selects the first one that matches all of them exactly. Three parameter-induction modules (length → color, longest-run → color, parity → color) learn their parameters from training. Programs are ordered general-before-specific so the Occam-prior choice wins ties.

---

## Honest limitations

Because trust matters more than hype:

1. **The COGS parser knows the schema.** It hand-codes knowledge of determiners, prepositions, role labels (`agent`, `theme`, `recipient`, `ccomp`, `xcomp`, `nmod`), and how COGS structures its logical-form output. What it *learns* from data is the lexicon: which verbs are which, what role intransitive subjects take, which verbs are ditransitive, which can take control complements. So this isn't pure data-driven discovery — it's data-driven *parameter induction over a structured hypothesis space*. Don't let anyone tell you it's "learned the language from scratch." It hasn't.

2. **It won't scale to real natural language.** The mechanism depends on surface regularity — same grammatical shape produces same output shape. Real internet text doesn't work like that. We make no claim that this approach generalizes to general-purpose dialogue or open text.

3. **1D-ARC is much easier than 2D-ARC.** The full ARC-AGI benchmark is 2D and dramatically harder. The 1D variant exists specifically because 2D was too hard for the methods that work on 1D. Extending this approach to 2D is open research.

4. **The "ChatGPT-style transformer baseline at 35% on COGS" is from the original Kim & Linzen 2020 paper.** I have not benchmarked GPT-4 or Claude directly against COGS. They might do better; they might do worse. I don't know.

5. **No neural-network experiments here.** The repo includes only the symbolic / program-synthesis track. Hyperion's broader research program also tried trained neural approaches (HYMN, FERN); those produced weaker results and are kept private as work-in-progress.

6. **This is research code, not a product.** It runs the benchmarks. It does not have a UI, an API, or a deployment story. If you want to use it for something, you'll have to build that part yourself.

---

## What you can do with it

- **Reproduce the results.** Run the tests, see the numbers.
- **Audit the code.** It's small. Read it.
- **Fork it.** MIT license. Build on it.
- **Extend the library.** Add primitives for 2D-ARC. Add recursive operators for harder grammars. Hook the template learner up to a different language.
- **Use it as a baseline.** If you're publishing neural results on these benchmarks, this is the symbolic baseline to beat (and many published methods don't).
- **Cite it.** See `CITATION.cff`.

---

## Repository layout

```
hyperion/
  pure_vsa/                     The research code.
    arc1d_solver.py             1D-ARC program-synthesis solver.
    cogs_template_learner.py    COGS template induction.
    cogs_recursive.py           COGS recursive-structure fallback parser.
    cogs_hyperion.py            COGS data loader.
    scan_hyperion.py            SCAN solver (VSA-based).
    pcfg_hyperion.py            PCFG solver.
    composer.py, memory.py      VSA primitives and storage.
    OVERNIGHT_REPORT.md         Detailed technical writeup of the session that produced the current numbers.
  tests/                        pytest test suite gating the headline numbers.
  data/                         Dataset downloaders (does NOT bundle data — you fetch official copies).
  pyproject.toml                Python package metadata.
  LICENSE                       MIT.
  CITATION.cff                  How to cite.
```

---

## Citation

If you use this in a paper or product, please cite as in `CITATION.cff`:

```
Baer, K. (2026). Hyperion: pure-VSA compositional generalization.
Northtek. https://github.com/NORTHTEKDevs/hyperion
```

---

## Built by

[Northtek](https://northtek.io) — Kristian Baer.

This repo is part of the Northtek Labs research track. Production work (the platform that uses these ideas operationally) lives elsewhere in the Northtek stack. The research is open.

---

## License

MIT. See `LICENSE`. You can use this commercially, fork it, modify it, redistribute it. Attribution appreciated, not required.
