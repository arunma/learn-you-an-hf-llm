# CLAUDE.md — context for AI assistants

This file is read automatically by Claude Code when working in this repo. It captures the project's purpose, conventions, and the user's working preferences so an assistant can pick up cold without re-deriving everything.

---

## What this repo is

A **learning project**. Build a tiny GPT-style LLM from scratch using the Hugging Face stack (`transformers`, `tokenizers`, `datasets`, `safetensors`) in five small staged scripts. Companion repo to `learn-you-a-nanochat` which does the same in plain PyTorch.

Goal of the repo: the user understands every line of how a modern LLM is trained. Not production training. Not state-of-the-art. **Pedagogy through copywork.**

---

## Repo layout (memorise this)

```
learn-you-an-hf-llm/
├── README.md              ← public-facing entry doc
├── pyproject.toml         ← deps; pip install -e .
├── CLAUDE.md              ← this file
├── hf_nanochat/model.py   ← THE model: NanoChatConfig + NanoChatModel
├── hf_pipeline/           ← answer files (5 staged scripts + run_all.py)
├── copywork/              ← user types into these next to the answer files
│   ├── README.md          ← copywork method, tier list
│   ├── ROADMAP.md         ← staged journey with done-checklist
│   └── 0X_*/copywork_*.py ← the user's hand-typed copies
├── sections/              ← short conceptual notes
└── journal/               ← consolidated reference book (00..06_*.md)
```

---

## How the user learns (and how you should help)

The user follows a **copywork** workflow:

1. Read `journal/0X_*.md` for the concept
2. Open the answer file (`hf_pipeline/0X_*.py` or `hf_nanochat/model.py`) next to the empty `copywork/0X_*/copywork_*.py`
3. **Type** the answer file line by line
4. Drop `# Q: <question>` comments where unclear
5. Ask Claude (you) about the `# Q:` comments
6. After the section: `diff` against the answer

**Your role when they ask a `# Q:`:**

- Explain the concept first (intuition before math)
- Define every term
- Walk through any math step-by-step with a worked example
- Then **consolidate the explanation into `journal/0X_*.md`** as a topical sub-section under "Topics covered" — *not* as a Q&A log entry. The journal is a reference book, not a transcript.

---

## User profile (saved memory)

The user is a **software engineer making a first detailed pass at LLM internals**. Math comfort: **not advanced**. Assume:

- No measure theory, no advanced linear algebra beyond matmul, no probability theory background
- No RL background (so explain expected value, KL divergence, etc. when they appear)
- Comfortable with Python, PyTorch tensor ops, and reading framework code

**How to communicate:**

- **Lead with concrete intuition / analogy, not formulas.** Math comes after the picture is clear.
- When math is necessary, walk through it step by step with a worked example. Never drop a formula and move on.
- Prefer "what / why / when" framing over "given a probability distribution P over...".
- Use diagrams, tables, and code examples liberally; equations sparingly.
- **Always define technical terms on first use, every time** — even ones that came up earlier.
- Calibrate length to "good details but just enough for good intuition" — neither surface-level nor textbook-deep.

---

## Documentation style (saved feedback)

For the `journal/` reference book: **integrate each new piece of understanding into the relevant `journal/0X_*.md` file as consolidated topical prose. Don't append dated Q&A logs.**

The user wants the journal to read like a reference book they can look back on, not a transcript. When a question comes up:

- Each `journal/0X_*.md` has a `### Topics covered` section that grows with topical sub-sections (`#### topic name`)
- Refine and expand existing sub-sections rather than appending new dated entries
- Cross-link related concepts across files
- Style: prose-first concept explanation, `[PT]` / `[NC]` / `[HF]` annotations on technical names, tables, shape traces, step-by-step worked examples

---

## Copywork tier list (from `copywork/README.md`)

When helping the user copywork, distinguish **architectural code** (worth typing) from **configuration plumbing** (skip):

| Tier | Examples | Action |
|---|---|---|
| **High** — copy verbatim | `CausalSelfAttention`, `MLP`, `Block`, `forward()`, RoPE precomputation, generation loop, custom config class | Type every line. `# Q:` for unfamiliar lines. Diff after. |
| **Medium** — paraphrase | `__init__` of the model class, `_init_weights`, scheduler logic, `TrainingArguments` setup | Read, close, write a summary in your own words. |
| **Skip** — read once for the *why* | Multi-param-group optimizer setups (e.g., nanochat's `setup_optimizer`), distributed init, argparse, `__main__` demos, file-path handling | Hyperparameter values aren't memorable; concepts are. Don't waste time copying numeric literals. |

If the user is bored copying numeric literals (`betas=(0.8, 0.96)`, `weight_decay=0.05`), they're in tier 3 — call this out and suggest reading for the concept instead.

---

## Working principles for this project

- **Working > complete.** A 5M-param model that trains for 200 steps with loss decreasing is more valuable than a 100M-param model with twelve advanced features that don't work yet.
- **YAGNI for round one.** Multiple param groups, Muon, sliding-window attention, value embeddings, smear gates, backout connections, KV cache, FlashAttention, gradient checkpointing — *none* of these are needed for the smoke-test loop. They're upgrades for after the basic loop works.
- **Disk-passing between stages.** The five `hf_pipeline/0X_*.py` scripts read inputs from `~/.cache/hf_pipeline/` and write outputs there. Each runs standalone; you can rerun any stage in isolation.

---

## Cache layout (where artifacts live)

`~/.cache/hf_pipeline/` — outside the repo, persists across runs:

```
~/.cache/hf_pipeline/
├── train.txt              ← from stage 1
├── val.txt                ← from stage 1
├── tokenizer.json         ← from stage 2
├── model_trained/         ← from stage 3 (HF save_pretrained format)
│   ├── config.json
│   └── model.safetensors
└── my-first-llm/          ← from stage 5 (model + tokenizer combined)
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

When debugging "why doesn't stage X work", always check that stage X-1's outputs exist here.

---

## Writing code in this repo

- Python 3.10+. Type annotations on all function signatures.
- PEP 8 via `black` / `ruff` (see `pyproject.toml [project.optional-dependencies] dev`).
- Imports are top-level — `from hf_nanochat.model import NanoChatConfig, NanoChatModel` works because the package is installed editable.
- For copywork files (`copywork/0X_*/copywork_*.py`), the user types their own version. **Don't pre-fill these.** They get a docstring stub explaining the answer file location and the import-rebind trick.
- Keep functions short, files focused. The whole repo is small on purpose.

---

## Don't do

- **Don't auto-fill `copywork/*.py` files.** The whole point is the user types them.
- **Don't add features to `hf_pipeline/`** beyond what the smoke-test loop needs. The simplicity is pedagogical.
- **Don't append dated Q&A entries to journal files.** Always consolidate into topical sub-sections.
- **Don't use emojis** in code or docs unless the user explicitly asks.
- **Don't add `print()` statements** to production-style code; existing scripts use them appropriately for stage progress reporting, but new code in `hf_nanochat/` should use `logging`.
- **Don't break the `journal/` cross-references** without auditing — they link by relative `[label](XX_*.md)` pattern, no `hf_` prefix.

---

## Common tasks and where they fit

| Task | Likely files to touch |
|---|---|
| User asks "why does HF need X?" or "what does Y do?" | Answer + consolidate into `journal/0X_*.md` |
| User shows a `# Q:` comment in their copywork | Same — answer + consolidate |
| User shows a bug in their copywork | Compare line-by-line against the answer file in `hf_pipeline/` or `hf_nanochat/`. Don't write the fix for them; describe the gap. |
| User wants to upgrade (bigger model, real GPU, etc.) | Point at the upgrade table in `copywork/ROADMAP.md` and the relevant `journal/06_deep_dive_internals.md` topic. |
| User wants to add a new stage | Modify `hf_pipeline/`. Update the section map in `copywork/README.md` and the journey in `copywork/ROADMAP.md`. |
| Tokenizer or data corpus change | Touch `hf_pipeline/01_data.py` or `02_tokenizer.py`. Cache layout doesn't change. |

---

## Companion repo

`learn-you-a-nanochat` (sibling at `../learn-you-a-nanochat`) — same architecture in plain PyTorch with no HF abstractions. Useful when the user asks "how does HF do X under the hood?" — show them the corresponding nanochat code if applicable.
