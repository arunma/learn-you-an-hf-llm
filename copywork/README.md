# Copywork — the type-it-yourself companion

Active practice for internalising the Hugging Face training pipeline. Read the journal section → close it → write the canonical code line-by-line into your own file → diff against the answer → re-do tomorrow without looking.

This README is the **method, philosophy, and section map**. The numbered subdirectories are the workspaces.

---

## What "copywork" means here

You have two kinds of files for each section:

| | What it is | Where it lives |
|---|---|---|
| **Answer file** | The finished, runnable script | `hf_pipeline/0X_*.py` (or `hf_nanochat/model.py` for section 02) |
| **Copywork file** | A blank or stub file you type into | `copywork/0X_*/copywork_*.py` |

You open both side by side in your editor, **type** the answer-file content into the copywork file, and drop `# Q: <question>` comments on anything unclear. When the section is done you `diff` your file against the answer.

You're not trying to memorise. You're trying to make your hands pause on the lines you don't really understand. The pause is the signal.

---

## Why typing beats reading

Reading is passive. Typing is active. When you read `q = self.c_q(x).view(B, T, self.n_head, self.head_dim)`, your eye glides over it. When you type that same line, your hand pauses over `.view(B, T, ...)` and you suddenly notice you don't quite remember what shape `q` was before the view, or why we reshape into `n_head` instead of one big `(B, T, n_embd)` tensor.

That pause is the gap you didn't know you had. Copywork makes those gaps visible.

The method is borrowed from Karpathy's nanochat learning vault.

---

## Principle 1 — copy what teaches; skip what configures

Not every line is worth typing. Some lines encode **architectural ideas** (worth typing slowly). Others are just **configuration values** (worth reading once, never typing).

| Tier | What | What to do |
|---|---|---|
| **High value** | `CausalSelfAttention`, `MLP`, `Block`, model `forward()`, RoPE precomputation, generation loop, custom config class | **Type every line.** Mark unfamiliar with `# Q: ...`. Diff after. |
| **Medium value** | `__init__` of the model class, `_init_weights`, scheduler logic, `TrainingArguments` setup | **Read, close, paraphrase.** Write a summary in your own words: what it does, what each knob controls. |
| **Skip** | Multi-param-group optimizer setups, distributed init, argparse, `if __name__ == "__main__":` demos, file-path handling | **Read the docstring once for the *why*; never copy.** Hyperparameter values aren't memorable; concepts are. |

**Self-check:** if you find yourself bored copying numeric literals (`betas=(0.8, 0.96)`, `weight_decay=0.05`), you're in tier 3. Stop typing and read for the concept instead.

---

## Principle 2 — working before complete

For a first LLM, **step by step beats full swing**:

1. **Working > complete.** A 5M-param model that trains for 200 steps with loss decreasing is more valuable than a 100M-param model with twelve advanced features you never wire up.
2. **YAGNI for round one.** Multiple param groups, Muon, sliding-window attention, value embeddings, smear gates, backout connections — all research polish that won't move the needle on a tiny WikiText run, and *will* burn your weekend on bugs.

Get one round-trip end-to-end first. Upgrade pieces only when you have a measured reason.

---

## The loop (per section)

For each numbered section directory:

1. **Read the journal first.** Open the matching `journal/0X_*.md` and absorb the concepts (~10 min).
2. **Open source + copywork side by side.** Source = the answer file (see the section map below). Copywork file = `copywork/0X_*/copywork_*.py`.
3. **Type, don't paste.** Line by line. For every line that's not obvious, drop a `# Q: ...` comment with the question.
4. **Ask the assistant** (or work it out yourself). Each `# Q` becomes a focused conversation. The understanding gets consolidated into the matching `journal/0X_*.md` topic.
5. **Diff.** When the section is done:
   ```bash
   diff copywork/02_config_and_model/copywork_model.py hf_nanochat/model.py
   ```
   Note what you missed. Add corrections in the file as comments: `# Missed: <X> — <reason>`.
6. **Re-do tomorrow.** Fresh file, no source open. Diff again — the deltas should be smaller. When you can re-write the high-value parts close to verbatim and paraphrase the medium parts cleanly, you've internalised the section.

---

## Section map

Each numbered directory mirrors a journal section. Each one points at a specific source file as its "answer."

| # | Section | Journal | Answer file | Copywork target |
|---|---|---|---|---|
| 01 | Data & tokenizer | `journal/01_data_and_tokenizer.md` | `hf_pipeline/01_data.py` + `hf_pipeline/02_tokenizer.py` | `01_data_and_tokenizer/copywork_data.py`, `copywork_tokenizer.py` |
| 02 | Config & model | `journal/02_model_and_config.md` | `hf_nanochat/model.py` (`NanoChatConfig` + `NanoChatModel`, ~300 lines) | `02_config_and_model/copywork_config.py`, `copywork_model.py` |
| 03 | Training loop | `journal/03_training_loop.md` | `hf_pipeline/03_train.py` (`get_batch`, training loop, save) | `03_training_loop/copywork_train.py` |
| 04 | Evaluation & generation | `journal/04_evaluation_and_generation.md` | `hf_pipeline/04_eval_and_generate.py` (`evaluate`, `generate_samples`) | `04_evaluation_and_generation/copywork_eval.py` |
| 05 | Save / load / Hub | `journal/05_save_load_hub.md` | `hf_pipeline/05_save_share.py` (`PreTrainedTokenizerFast` wrap, round-trip, Hub upload) | `05_save_load_hub/copywork_save.py` |

### Two levels of "answer file"

`hf_pipeline/run_all.py` is the **abstracted** end-to-end script — it uses `LlamaConfig` and `LlamaForCausalLM` (HF's prebuilt classes). Keep it as a comparison reference.

`hf_pipeline/01_data.py` through `05_save_share.py` are the **per-stage scripts** that use your custom `NanoChatConfig` / `NanoChatModel` from `hf_nanochat/model.py`. These are the answer files you copywork against. Each one is runnable on its own and reads its inputs from `~/.cache/hf_pipeline/`, so you can rerun any stage without rerunning earlier ones.

The recommended path:

1. **Smoke-test first.** Run `python -m hf_pipeline.run_all` to confirm your env works, MPS is detected, loss decreases, save/reload round-trips. ~5 min. Do this **before** any copywork.
2. **Then walk the staged scripts in order:** `01_data.py` → `02_tokenizer.py` → … → `05_save_share.py`. For each one, copywork the corresponding `copywork/0X_*/copywork_*.py` line-by-line.
3. **Section 02 is the deepest.** The model class itself. `hf_nanochat/model.py` is its answer; the other stages just *use* the classes defined there.
4. **End state:** you've written your own `0X_*.py` files in `copywork/` mirroring `hf_pipeline/0X_*.py`. Diff them. The deltas are the gaps to revisit.

### Section 02 split — what goes in each file

`hf_nanochat/model.py` is one ~300-line file. It splits naturally into two copywork files because config and model are independent HF abstractions.

**`copywork_config.py`** — just the config class, ~40 lines.

| Source lines | Content |
|---|---|
| ~26–32 | imports needed by config (`PretrainedConfig`) |
| ~35–71 | `class NanoChatConfig(PretrainedConfig)` |

What you internalise: `PretrainedConfig` inheritance pattern, `model_type = "nanochat"` requirement, `**kwargs` forward to `super().__init__()`, the `num_hidden_layers` alias (HF naming vs nanochat naming).

**`copywork_model.py`** — imports, helpers, all four model classes, ~200 lines.

| Source lines | Content |
|---|---|
| ~26–32 | full imports (`torch`, `nn`, `F`, `PreTrainedModel`, `GenerationMixin`, `CausalLMOutput`) |
| ~74–88 | helper functions `norm()`, `apply_rotary_emb()` |
| ~91–127 | `class CausalSelfAttention` |
| ~130–142 | `class MLP` |
| ~145–156 | `class Block` |
| ~163–248 | `class NanoChatModel` |

What you internalise: Q/K/V/O projections + RoPE + `scaled_dot_product_attention`; pre-norm residual block pattern; `PreTrainedModel` extras (`post_init`, `_init_weights`, `save_pretrained` for free); how `forward` returns `CausalLMOutput` for HF compatibility.

**Skip** (tier-3 from the principle above — read once for API surface, don't copy line by line):

| Source lines | Content |
|---|---|
| ~255–306 | `if __name__ == "__main__":` demo block — shows `model.generate()`, `save_pretrained`, `from_pretrained`, `num_parameters` |

Suggested order: `copywork_config.py` first (~30 min, short and self-contained), then split `copywork_model.py` across two sittings — sitting A = imports + helpers + `CausalSelfAttention`; sitting B = `MLP` + `Block` + `NanoChatModel`.

After both, eyeball-compare your copywork ranges against the source ranges. `diff` won't line up cleanly across the file split — visual comparison is faster.

---

## Marking questions in copywork files

Convention: `# Q: ...` for questions, `# A: ...` once the question's been worked through. Once consolidated into the journal, leaving `# Q/A` in the copywork file as a personal trace is fine — or delete them, your call.

```python
class NanoChatConfig(PretrainedConfig):
    model_type = "nanochat"   # Q: why does HF need a model_type string? what registers with what?
                              # A: required so AutoConfig.from_pretrained() knows which class to instantiate.
                              #    Consolidated → journal/02_model_and_config.md "Auto class registration".

    def __init__(
        self,
        sequence_len=2048,
        ...
        **kwargs,             # Q: what does HF actually pass via **kwargs?
                              # A: things like torch_dtype, transformers_version, _name_or_path.
                              #    Forward via super().__init__(**kwargs); don't try to enumerate.
    ):
```

The point isn't to make the copywork file a textbook — it's to leave a personal trail of "what tripped me up here." Future-you will thank present-you.

---

## Adapting answer files to use your copywork model

The answer files (`hf_pipeline/0X_*.py`) import `NanoChatConfig` / `NanoChatModel` from `hf_nanochat/model.py` (the canonical source). To make your training script use *your* copywork code instead, replace the import block at the top of `copywork/0X_*/copywork_*.py`:

```python
import sys
from pathlib import Path

# Make the copywork dir importable even though its name starts with "02_"
COPYWORK_DIR = Path(__file__).resolve().parent.parent / "02_config_and_model"
sys.path.insert(0, str(COPYWORK_DIR))

from copywork_config import NanoChatConfig
from copywork_model import NanoChatModel

# Rebind so saved config.json reflects your code path
NanoChatModel.config_class = NanoChatConfig
```

Everything else in the answer file is reusable as-is.

---

## When a section is "done"

A section is internalised when:

- [ ] You can rewrite the **high-value** parts from a blank file, close to verbatim. Diffs are minor (variable naming, whitespace).
- [ ] You can paraphrase the **medium-value** parts cleanly in your own words.
- [ ] You've read the **skip** parts once and can name the concept without recalling the values.
- [ ] The matching `journal/0X_*.md` "Topics covered" section feels complete to you.

You don't need all four to start the next section — start when boxes 2 and 4 feel solid. Come back to box 1 over multiple passes.

---

## Diff tooling

Plain diff:

```bash
diff copywork/02_config_and_model/copywork_model.py hf_nanochat/model.py
```

VS Code visual diff:

```bash
code --diff copywork/02_config_and_model/copywork_model.py hf_nanochat/model.py
```

Obsidian: install the **Diff** community plugin to compare two notes side by side.

---

## Where things live

Inside this repo:

| Path | Purpose |
|---|---|
| `copywork/` (this dir) | Your hand-typed code + Q/A traces |
| `hf_nanochat/`, `hf_pipeline/` | The "answer" code |
| `sections/` | Short conceptual notes (read-first reference) |
| `journal/` | Consolidated reference book — every concept in detail |

Pattern: copywork is the *practice*, journal is the *reference*. Practice generates questions; answers refine the reference.

For the staged plan with checklists and what-comes-next, see [`ROADMAP.md`](ROADMAP.md).
