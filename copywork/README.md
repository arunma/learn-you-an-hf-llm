# HF Copywork

Active practice for internalising the Hugging Face wrappers around a transformer. Read the journal → close source → write the canonical code line-by-line in your own file, asking questions as you go. Diff against source. Re-do tomorrow.

This README is the **philosophy + workflow + section map**. Each numbered section directory under here is the workspace for that stage.

---

## Why copywork

Reading is passive. Typing is active. When you type a line you didn't quite understand, your hand pauses — and **that pause is the gap you didn't know you had**. Copywork makes those gaps visible and converts them into questions, which become understanding, which becomes the journal.

Same method as Karpathy's nanochat vault: read → close → write → diff → repeat.

---

## Principle 1 — copy what teaches; skip what configures

Not all code is created equal. Some lines encode **architectural ideas**; others are just **configuration values**. Copywork should target the ideas.

| Tier | What | Action |
|---|---|---|
| **High value** | `CausalSelfAttention`, `MLP`, `Block`, model `forward()`, RoPE precomputation, generation loop, custom config class | **Type every line.** Mark unfamiliar with `# Q: ...`. Diff after. |
| **Medium value** | `__init__` of the GPT model, `_init_weights`, scheduler logic, `TrainingArguments` setup | **Read, close, paraphrase.** Write a summary in your own words: what it does, what each knob controls. |
| **Skip** | `setup_optimizer`-style param-group plumbing, distributed init, argparse, `__main__` demos, file-path handling | **Read the docstring once for the *why*; never copy.** Hyperparameter values aren't memorable; concepts are. |

**Self-check:** if you find yourself bored copying numeric literals (`betas=(0.8, 0.96)`, `weight_decay=0.05`), you're in tier 3. Stop typing and read for the concept instead.

---

## Principle 2 — working before complete

For a first LLM, **step by step beats full swing**:

1. **Working > complete.** A 5M-param model that trains for 200 steps with loss decreasing is more valuable than a 100M-param model with twelve advanced features you never wire up.
2. **YAGNI for round one.** Multiple param groups, Muon, sliding-window attention, value embeddings, smear gates, backout connections — research polish that won't move the needle on a tiny TinyStories run, and *will* burn your weekend on bugs.

Get one round-trip end-to-end first. Upgrade pieces only when you have a measured reason.

---

## The loop (per section)

For each numbered section directory:

1. **Read the journal first.** Open `~/projects/ai/learn-you-an-llm/journal/hf/hf_XX_*.md` and absorb the concepts (~10 min).
2. **Open source + copywork side by side.** Source is whatever the section maps to (see table below). Copywork file is `vault/copywork/hf/XX_*/copywork_*.py`.
3. **Type, don't paste.** Line by line. For every line that's not obvious, drop a `# Q: ...` comment with the question.
4. **Ask the assistant.** Each `# Q` becomes a focused conversation. The understanding gets consolidated into the matching `journal/hf/hf_XX_*.md` topic — as a refined topical sub-section, not a Q&A log.
5. **Diff.** When the section is done:
   ```bash
   diff vault/copywork/hf/02_config_and_model/copywork_model.py hf_nanochat/model.py
   ```
   Note what you missed. Add corrections in the file as comments: `# Missed: <X> — <reason>`.
6. **Re-do tomorrow.** Fresh file, no source open. Diff again — deltas should be smaller. When you can re-write the high-value parts close to verbatim and paraphrase the medium parts cleanly, you've internalised the section.

---

## Section map

The numbered directories mirror the journal sections. Each one points at a specific source file as its "answer."

| # | Section | Journal | Source ("the answer") | Copywork target |
|---|---|---|---|---|
| 01 | Data & tokenizer | `hf_01_data_and_tokenizer.md` | `hf_pipeline/01_data.py` + `hf_pipeline/02_tokenizer.py` | `01_data_and_tokenizer/copywork_data.py`, `copywork_tokenizer.py` |
| 02 | Config & model | `hf_02_model_and_config.md` | `hf_nanochat/model.py` (`NanoChatConfig` + `NanoChatModel`, 306 lines) | `02_config_and_model/copywork_config.py`, `copywork_model.py` |
| 03 | Training loop | `hf_03_training_loop.md` | `hf_pipeline/03_train.py` (`get_batch`, training loop, save) | `03_training_loop/copywork_train.py` |
| 04 | Evaluation & generation | `hf_04_evaluation_and_generation.md` | `hf_pipeline/04_eval_and_generate.py` (`evaluate`, `generate_samples`) | `04_evaluation_and_generation/copywork_eval.py` |
| 05 | Save / load / Hub | `hf_05_save_load_hub.md` | `hf_pipeline/05_save_share.py` (`PreTrainedTokenizerFast` wrap, round-trip, Hub upload) | `05_save_load_hub/copywork_save.py` |

### Two levels of "answer file"

`hf_pipeline/run_all.py` is the **abstracted** end-to-end script — it uses `LlamaConfig` / `LlamaForCausalLM`. Keep it as a comparison reference.

`hf_pipeline/01_data.py` through `05_save_share.py` are the **per-stage scripts** using your custom `NanoChatConfig` / `NanoChatModel` from `hf_nanochat/model.py`. These are the answer files you copywork against. Each one is runnable on its own and reads its inputs from `~/.cache/hf_pipeline/`, so you can rerun any stage without rerunning earlier ones.

The recommended path:

1. **Smoke-test first.** Run `python -m hf_pipeline.run_all` as-is to confirm env works, MPS is detected, loss decreases, save/reload round-trips. ~5–10 min. Do this **before** any copywork.
2. **Then walk through the staged scripts in order:** `01_data.py` → `02_tokenizer.py` → … → `05_save_share.py`. For each, copywork the corresponding `vault/copywork/hf/0X_*/copywork_*.py` line-by-line against the staged script as the answer.
3. **Section 02 is the deepest one** — the model class itself. `hf_nanochat/model.py` is the answer for that section; the other stages just *use* the classes defined there.
4. **End state:** you've written your own `0X_*.py` files in `vault/copywork/hf/` mirroring `hf_pipeline/0X_*.py`. Diff them. The deltas are the gaps to revisit.

`mkdir` each copywork section directory when you reach it; don't pre-create them.

### Section 02 split — what goes in each file

`hf_nanochat/model.py` is one 306-line file. It splits naturally into two copywork files because config and model are independent HF abstractions.

**`copywork_config.py`** — just the config class, ~40 lines.

| Source lines | Content |
|---|---|
| 26–32 | imports needed by config (`PretrainedConfig`) |
| 35–71 | `class NanoChatConfig(PretrainedConfig)` |

What you internalise: `PretrainedConfig` inheritance pattern, `model_type = "nanochat"` requirement, `**kwargs` forward to `super().__init__()`, the `num_hidden_layers` alias (HF naming vs nanochat naming).

**`copywork_model.py`** — imports, helpers, all four model classes, ~200 lines.

| Source lines | Content |
|---|---|
| 26–32 | full imports (`torch`, `nn`, `F`, `PreTrainedModel`, `GenerationMixin`, `CausalLMOutput`) |
| 74–88 | helper functions `norm()`, `apply_rotary_emb()` |
| 91–127 | `class CausalSelfAttention` |
| 130–142 | `class MLP` |
| 145–156 | `class Block` |
| 163–248 | `class NanoChatModel` |

What you internalise: Q/K/V/O projections + RoPE + `scaled_dot_product_attention`; pre-norm residual block pattern; `PreTrainedModel` extras (`post_init`, `_init_weights`, `save_pretrained` for free); how `forward` returns `CausalLMOutput` for HF compatibility.

**Skip** (tier-3 from the principle above — read once for API surface, don't copy line by line):

| Source lines | Content |
|---|---|
| 255–306 | `if __name__ == "__main__":` demo block — shows `model.generate()`, `save_pretrained`, `from_pretrained`, `num_parameters` |

Suggested order: `copywork_config.py` first (~30 min, short and self-contained), then split `copywork_model.py` across two sittings — sitting A = imports + helpers + `CausalSelfAttention`; sitting B = `MLP` + `Block` + `NanoChatModel`.

After both, eyeball-compare your copywork ranges against the source ranges. `diff` won't line up cleanly across the file split — visual comparison is faster.

---

## Marking questions in copywork files

Convention: `# Q: ...` for questions, `# A: ...` once we work them through. Once consolidated into the journal, leaving `# Q/A` in the copywork file as a personal trace is fine — or delete them, your call.

```python
class NanoChatConfig(PretrainedConfig):
    model_type = "nanochat"   # Q: why does HF need a model_type string? what registers with what?
                              # A: required so AutoConfig.from_pretrained() knows which class to instantiate.
                              #    Consolidated → journal/hf/hf_02_model_and_config.md "Auto class registration".

    def __init__(
        self,
        sequence_len=2048,
        ...
        **kwargs,             # Q: what does HF actually pass via **kwargs?
                              # A: things like torch_dtype, transformers_version, _name_or_path.
                              #    Forward via super().__init__(**kwargs); don't try to enumerate.
    ):
```

The point isn't to make the copywork file a textbook — it's to leave a personal trail of "what tripped me up here". Future-you will thank present-you.

---

## When a section is "done"

A section is internalised when:

- [ ] You can rewrite the **high-value** parts from a blank file, close to verbatim. Diffs are minor (variable naming, whitespace).
- [ ] You can paraphrase the **medium-value** parts cleanly: *"this function builds the optimizer; it groups params, gives matrices a different LR, …"*.
- [ ] You've read the **skip** parts once and can name the concept without recalling the values.
- [ ] The matching `journal/hf/hf_XX_*.md` "Topics covered" section feels complete to you.

You don't need all four to start the next section — start when boxes 2 and 4 feel solid. Come back to box 1 over multiple passes.

---

## Diff tooling

Plain diff:
```bash
diff vault/copywork/hf/02_config_and_model/copywork_model.py hf_nanochat/model.py
```

VS Code visual diff:
```bash
code --diff vault/copywork/hf/02_config_and_model/copywork_model.py hf_nanochat/model.py
```

Obsidian: install the **Diff** community plugin to compare two notes side by side.

---

## Where things live

| Repo | Path | Purpose |
|---|---|---|
| `learn-you-a-nanochat` | `vault/copywork/hf/` (this dir) | Your hand-typed code + Q/A traces |
| `learn-you-a-nanochat` | `vault/sections/hf-pipeline/` | Conceptual section notes |
| `learn-you-a-nanochat` | `hf_nanochat/`, `hf_pipeline/` | The "answer" code |
| `learn-you-an-llm` | `journal/hf/hf_*.md` | Reference book — consolidated topical understanding |

Pattern: copywork is the *practice*, journal is the *reference*. Practice generates questions; answers refine the reference.
