# Roadmap — the end-to-end copywork journey

Five sections, ~4–6 hours of typing across a few sittings, one trained LLM at the end. This document is the plan.

If you haven't yet, read [`README.md`](README.md) for the *method* (tier list, the per-section loop, why typing beats reading). This file is the *journey*.

---

## Where you are vs where you'll be

```
START                                                               END
─────                                                               ─────
raw text on Wikipedia    →   tokenizer    →   trained transformer   →   Hub-loadable directory
(2.1M tokens)                (vocab 4096)     (3M params, 500 steps)    (config + weights + tokenizer)
```

You'll do five sections. Each maps to one journal entry, one or two answer files, and one or two copywork targets.

| # | Section | Status |
|---|---|---|
| 01 | Data + BPE tokenizer | ⬜ |
| 02 | Config + Model | ⬜ |
| 03 | Training loop | ⬜ |
| 04 | Evaluation + generation | ⬜ |
| 05 | Save + Hub-ready | ⬜ |

---

## Section 01 — Data and BPE tokenizer

| | |
|---|---|
| Goal | Download WikiText-2, drop empty lines, train a byte-level BPE tokenizer on the train split |
| Answer files | `hf_pipeline/01_data.py` (~45 lines), `hf_pipeline/02_tokenizer.py` (~60 lines) |
| Copywork targets | `copywork/01_data_and_tokenizer/copywork_data.py`, `copywork_tokenizer.py` |
| Output | `~/.cache/hf_pipeline/{train.txt, val.txt, tokenizer.json}` |
| Time | ~2 min for download + ~30 sec for tokenizer training |
| Journal | [`journal/01_data_and_tokenizer.md`](../journal/01_data_and_tokenizer.md) |

**What you'll learn**

From `copywork_data.py`:
- `datasets.load_dataset(name, config)` — the canonical entry point. Caches downloads, handles checksum/version, returns a dict-like object with train/val/test splits.
- A WikiText "row" is a line of text in the `text` column. For language modelling you concatenate non-empty lines into one big string per split — the tokenizer doesn't care about row boundaries.
- Why save split files instead of one concat: stages 03 and 04 read `train.txt` and `val.txt` separately to compute train vs val loss.

From `copywork_tokenizer.py`:
- The four-component `tokenizers` pipeline: **pre_tokenizer → model → trainer → decoder**.
- `models.BPE()` is the algorithm; `BpeTrainer` is the training procedure that iterates over the corpus to learn merges.
- **Byte-level pre-tokenization** (vs whitespace pre-tokenization) — handles any Unicode codepoint without UNK tokens; encodes leading spaces as `Ġ` (U+0120).
- Streaming the corpus in 10K-character chunks via `train_from_iterator` — memory-efficient for large datasets.
- Special tokens (`<|pad|>`, `<|bos|>`, `<|eos|>`) get assigned the lowest IDs (0, 1, 2) so they're guaranteed not to collide with learned BPE merges.

**How to run**

```bash
python copywork/01_data_and_tokenizer/copywork_data.py
python copywork/01_data_and_tokenizer/copywork_tokenizer.py
```

Section 01 only needs to run once — its outputs are stable and every later stage reads them from disk.

---

## Section 02 — Config and Model

| | |
|---|---|
| Goal | Write `NanoChatConfig` (a `PretrainedConfig` subclass) and `NanoChatModel` (a `PreTrainedModel` subclass) with custom attention, RoPE, RMSNorm, and ReLU² MLP |
| Answer file | `hf_nanochat/model.py` (~300 lines, the most architecture-rich source file) |
| Copywork targets | `copywork/02_config_and_model/copywork_config.py`, `copywork_model.py` |
| Output | Two files you can `import` and instantiate; verify with the smoke test below |
| Time | ~30 min for config + ~90 min for model (split across two sittings) |
| Journal | [`journal/02_model_and_config.md`](../journal/02_model_and_config.md) |

**What you'll learn**

- `PretrainedConfig` inheritance and the `model_type` registration mechanic
- `PreTrainedModel` inheritance — what you get free (`save_pretrained`, `from_pretrained`, `post_init`, `_init_weights`, `num_parameters`)
- The four attention projections (`c_q`, `c_k`, `c_v`, `c_proj`) and how they shape `(B, T, n_embd)` into `(B, n_head, T, head_dim)`
- RoPE precomputation and the `(cos, sin)` rotation formula
- The pre-norm residual block: `x = x + attn(norm(x))` then `x = x + mlp(norm(x))`
- Why `forward()` returns `CausalLMOutput` (HF compatibility for `Trainer` and `generate()`)

**Smoke test** (verify your code works before moving on):

```python
import torch, sys
sys.path.insert(0, "copywork/02_config_and_model")
from copywork_config import NanoChatConfig
from copywork_model import NanoChatModel

cfg = NanoChatConfig(sequence_len=128, vocab_size=256, n_layer=2, n_head=4, n_kv_head=4, n_embd=128)
model = NanoChatModel(cfg)
ids = torch.randint(0, 256, (2, 16))
out = model(ids, labels=ids)
print(f"params       = {model.num_parameters():,}")
print(f"loss         = {out.loss.item():.4f}   (expect ~{torch.tensor(256.0).log().item():.2f})")
print(f"logits shape = {tuple(out.logits.shape)}   (expect (2, 16, 256))")
```

If those three lines print without error, your model is wired correctly. The loss will be near `ln(vocab_size)` — that's random-baseline cross-entropy, exactly what you should see for an untrained model.

**File split** (since `hf_nanochat/model.py` is one file but copywork splits into two):

| Copywork file | Source lines | What |
|---|---|---|
| `copywork_config.py` (~40 lines) | imports + `class NanoChatConfig(PretrainedConfig)` | the config class only |
| `copywork_model.py` (~200 lines) | imports + helpers + `CausalSelfAttention` + `MLP` + `Block` + `NanoChatModel` | everything else |
| (skip — read once) | the `if __name__ == "__main__":` demo block | usage examples; tier-3 plumbing |

---

## Section 03 — Training loop

| | |
|---|---|
| Goal | Tokenize the corpus, build your model, run AdamW for 500 steps, save the trained weights |
| Answer file | `hf_pipeline/03_train.py` (~125 lines) |
| Copywork target | `copywork/03_training_loop/copywork_train.py` |
| Output | `~/.cache/hf_pipeline/model_copywork/` (HF `save_pretrained` directory) |
| Time | 5–10 min on M-series Mac (training is most of this) |
| Journal | [`journal/03_training_loop.md`](../journal/03_training_loop.md) |

**What you'll learn**

- The minimal training loop: `for step in range(N): forward → backward → clip → step → zero_grad`
- Random batch sampling from a token stream (`get_batch`)
- AdamW setup in three lines (vs. nanochat's 30-line multi-param-group `setup_optimizer`)
- Gradient clipping (`torch.nn.utils.clip_grad_norm_`)
- Periodic eval (`model.eval()` → `no_grad` → loss → `model.train()`)
- `model.save_pretrained()` produces a Hub-format directory in one call

**Adapting the answer file to use YOUR copywork model**

The answer file `hf_pipeline/03_train.py` imports from the source `hf_nanochat/model.py`. To use *your* copywork code instead, replace the import block at the top of `copywork_train.py`:

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

Everything else in `03_train.py` is reusable as-is.

**How to run**

```bash
python copywork/03_training_loop/copywork_train.py
```

You should see something like:

```
Train tokens: 2,099,000  |  Val tokens: 217,000
Parameters:   3,234,304  |  Device: mps

Step    0/500  |  train: 8.31  |  val: 8.30  |  ppl: 4054.5  |  410 tok/s
Step   50/500  |  train: 5.92  |  val: 5.97  |  ppl: 391.8   |  3,200 tok/s
Step  100/500  |  train: 5.41  |  val: 5.59  |  ppl: 268.4   |  3,500 tok/s
...
Step  499/500  |  train: 4.98  |  val: 5.27  |  ppl: 194.2   |  3,600 tok/s

Training complete in 215.3s
Final val loss: 5.27  |  Perplexity: 194.2
Model saved to: ~/.cache/hf_pipeline/model_copywork
```

Loss should drop from ~`ln(vocab_size)` ≈ 8.3 (random baseline) to ~5–6 over 500 steps. If loss doesn't move, something's wrong (most likely a bug in your copywork model — re-do the section 02 smoke test).

---

## Section 04 — Evaluation and generation

| | |
|---|---|
| Goal | Load the trained model, compute perplexity on val set, generate text samples |
| Answer file | `hf_pipeline/04_eval_and_generate.py` (~115 lines) |
| Copywork target | `copywork/04_evaluation_and_generation/copywork_eval.py` |
| Output | stdout — a perplexity number + sample continuations |
| Time | <1 min |
| Journal | [`journal/04_evaluation_and_generation.md`](../journal/04_evaluation_and_generation.md) |

**What you'll learn**

- `NanoChatModel.from_pretrained(path)` — round-trip the saved model from disk
- Perplexity from cross-entropy loss (`exp(loss)`) and bits-per-token (`loss / ln(2)`)
- HF's `model.generate()` — `do_sample`, `temperature`, `top_k`, `max_new_tokens`
- Tokenizer round-trip: `tokenizer.encode(text).ids` → model → `tokenizer.decode(ids)`

The output samples will be **gibberish** — your tiny model trained 500 steps on 2M tokens can't write coherent English. That's fine. You're verifying the *pipeline* works (loaded model → ran inference → produced text), not model quality.

---

## Section 05 — Save and share (Hub-ready)

| | |
|---|---|
| Goal | Wrap the raw `tokenizers` BPE in HF's `PreTrainedTokenizerFast`, save model + tokenizer in one HF directory, demonstrate `push_to_hub` |
| Answer file | `hf_pipeline/05_save_share.py` (~80 lines) |
| Copywork target | `copywork/05_save_load_hub/copywork_save.py` |
| Output | `~/.cache/hf_pipeline/my-first-llm/` (Hub-loadable directory) |
| Time | <1 min |
| Journal | [`journal/05_save_load_hub.md`](../journal/05_save_load_hub.md) |

**What you'll learn**

- The HF directory layout: `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`
- `PreTrainedTokenizerFast` — wrapping a raw `tokenizers.Tokenizer` with HF metadata (special tokens)
- safetensors format vs. pickle-based `.bin` — security and lazy-loading wins
- Round-trip verification — load the saved dir back, confirm parameter count matches
- The `push_to_hub()` calling pattern (we don't actually push; the code shows the API)

After section 05, the directory is genuinely shippable — anyone with `transformers` installed could `from_pretrained` it and run inference.

---

## Optional: live monitoring during training

For a 5-min run, console output is enough. If you want graphed curves in a browser:

### TensorBoard (local, no account)

```bash
pip install tensorboard
```

Add to your `copywork_train.py`:

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("~/.cache/hf_pipeline/runs/copywork")

# inside the training loop, every step:
writer.add_scalar("train/loss", out.loss.item(), step)

# at eval time:
writer.add_scalar("val/loss", val_loss, step)
writer.add_scalar("val/perplexity", math.exp(val_loss), step)

# at the end:
writer.close()
```

In a separate terminal:

```bash
tensorboard --logdir=~/.cache/hf_pipeline/runs --port=6006
```

Open http://localhost:6006 — auto-refreshes; you can watch loss come down live.

### Weights & Biases (cloud, free, account needed)

The industry standard for experiment tracking.

```bash
pip install wandb
wandb login   # one-time, opens browser
```

Add to script:

```python
import wandb
wandb.init(project="my-first-llm", config={
    "n_layer": N_LAYER, "n_embd": N_EMBD, "lr": LR, "num_steps": NUM_STEPS,
})

# inside the loop:
wandb.log({"train_loss": out.loss.item()}, step=step)
wandb.log({"val_loss": val_loss, "val_perplexity": math.exp(val_loss)}, step=step)
```

Cloud dashboard, automatic run comparison across experiments, no `localhost`. Worth setting up once you train more than one model.

---

## Order of operations

```
section 01    →    section 02    →    section 03    →    section 04    →    section 05
(data+tok)         (config+model)     (training)         (eval+gen)         (save+share)
```

After all five, you've gone end-to-end: raw corpus → tokenizer → trained model → generated samples → reusable artifact. **That's "training an LLM by yourself."**

You can do section 02 first if you'd rather get the architectural thinking out of the way before plumbing — both orders work. The dependency is just: section 01's outputs need to exist before section 03 runs.

---

## "Done" checklist

- [ ] **Section 01** — `copywork_data.py` matches `hf_pipeline/01_data.py`; `train.txt` + `val.txt` exist in cache
- [ ] **Section 01** — `copywork_tokenizer.py` matches `hf_pipeline/02_tokenizer.py`; `tokenizer.json` exists; "The quick brown fox" sample encodes/decodes round-trip
- [ ] **Section 02** — `copywork_config.py` + `copywork_model.py` instantiate cleanly; smoke test passes
- [ ] **Section 03** — `copywork_train.py` matches `hf_pipeline/03_train.py`; you ran it; loss decreased from ~8 to ~5–6; `model_copywork/` saved to cache
- [ ] **Section 04** — `copywork_eval.py` matches `hf_pipeline/04_eval_and_generate.py`; you got a perplexity number; samples were gibberish (expected)
- [ ] **Section 05** — `copywork_save.py` matches `hf_pipeline/05_save_share.py`; `~/.cache/hf_pipeline/my-first-llm/` exists with `config.json`, `model.safetensors`, `tokenizer.json`

When all six rows are checked, you've completed the smoke-test loop end-to-end.

---

## What comes next (after smoke test)

Once your tiny model trains and saves successfully, the natural upgrades:

| Upgrade | What changes | Why |
|---|---|---|
| **Bigger model** | bump `N_LAYER`, `N_EMBD`, `SEQ_LEN`, `NUM_STEPS` | More capacity → better samples |
| **Better corpus** | switch to TinyStories (~700M tokens of simple English) | Better signal at small scale than WikiText's encyclopedic prose |
| **Real GPU** | rent an H100 hour | ~50× faster training; lets you go from 500 steps to 100K+ |
| **`bf16` mixed precision** | add `torch.amp.autocast` or `Trainer(bf16=True)` | Half the memory, ~2× throughput on H100 |
| **Gradient checkpointing** | `model.gradient_checkpointing_enable()` | Trade compute for memory; lets you train bigger models on the same GPU |
| **FlashAttention 2** | `attn_implementation="flash_attention_2"` in config | ~2–4× attention speedup at long context |
| **KV cache for inference** | flip `use_cache=True` and add `prepare_inputs_for_generation` cache plumbing | Linear-time generation instead of quadratic |
| **HF `Trainer`** | replace your hand-rolled loop with `trainer.train()` | Get DDP, checkpointing, eval cadence, logging for free |
| **TRL + PEFT** | `SFTTrainer(peft_config=LoraConfig(...))` | Fine-tune a real base model into a chat model with QLoRA |

All of these are documented in `journal/06_deep_dive_internals.md`. None are required for the smoke-test loop — they're "what real production training looks like."

---

## Files in this copywork directory

```
copywork/
├── README.md                                   ← method, tier list, conventions
├── ROADMAP.md                                  ← THIS FILE — the journey
├── 01_data_and_tokenizer/
│   ├── copywork_data.py
│   └── copywork_tokenizer.py
├── 02_config_and_model/
│   ├── copywork_config.py
│   └── copywork_model.py
├── 03_training_loop/
│   └── copywork_train.py
├── 04_evaluation_and_generation/
│   └── copywork_eval.py
└── 05_save_load_hub/
    └── copywork_save.py
```

Each `copywork_*.py` is a stub. Open it next to the answer file in your IDE, type the source line by line, drop `# Q:` comments for anything unclear, and ask. Diff visually when done.
