# Roadmap — From Copywork-Model to Trained LLM

You finished section 02 (config + model). Loss is finite, shapes match, the architecture is sound. **But the model isn't trained yet** — those weights are random Gaussian. To turn this into a working LLM you still need three more sections: training, evaluation+generation, save+share.

This document is the **end-to-end journey** from where you are to "I trained an LLM by myself."

---

## Where you are

- ⬜ Section 01 — data download + BPE tokenizer
- ✅ Section 02 — `NanoChatConfig` + `NanoChatModel` written from scratch in `vault/copywork/hf/02_config_and_model/`
- ⬜ Section 03 — training loop
- ⬜ Section 04 — evaluation + generation
- ⬜ Section 05 — save + Hub-ready directory

You did section 02 first because it's the conceptual heart of the model. Now we go back and copywork section 01 (data + tokenizer) for the full picture, then forward through 03→05.

---

## Section 01 — Data and BPE tokenizer

| | |
|---|---|
| Goal | Download WikiText-2, drop empty lines, train a byte-level BPE tokenizer on the train split |
| Answer files | `hf_pipeline/01_data.py` (~45 lines), `hf_pipeline/02_tokenizer.py` (~60 lines) |
| Copywork targets | `vault/copywork/hf/01_data_and_tokenizer/copywork_data.py`, `copywork_tokenizer.py` |
| Output | `~/.cache/hf_pipeline/{train.txt, val.txt, tokenizer.json}` |
| Time | ~2 min for download + ~30 sec for tokenizer training |

### What you'll learn

**From `copywork_data.py`:**
- `datasets.load_dataset(name, config)` — the canonical entry point. Caches downloads, handles checksum/version, returns a dict-like object with train/val/test splits.
- A WikiText "row" is a line of text in the `text` column. For language modelling, you concatenate non-empty lines into one big string per split — the tokenizer doesn't care about row boundaries.
- Why split files instead of a single concat: stages 03 and 04 use `train.txt` and `val.txt` separately to compute train vs val loss.

**From `copywork_tokenizer.py`:**
- The four-component `tokenizers` pipeline: **pre_tokenizer → model → trainer → decoder**
- `models.BPE()` is the algorithm; `BpeTrainer` is the training procedure that iterates over the corpus to learn merges
- **Byte-level pre-tokenization** (vs whitespace pre-tokenization) — handles any Unicode codepoint without UNK tokens; encodes leading spaces as `Ġ` (U+0120)
- Streaming the corpus in 10K-character chunks via `train_from_iterator` — memory-efficient for large datasets
- Special tokens (`<|pad|>`, `<|bos|>`, `<|eos|>`) get assigned the lowest IDs (0, 1, 2) so they're guaranteed not to collide with learned BPE merges

### How to run

```bash
python vault/copywork/hf/01_data_and_tokenizer/copywork_data.py
python vault/copywork/hf/01_data_and_tokenizer/copywork_tokenizer.py
```

Section 01 only has to run once — its outputs are stable and every later stage reads them from disk.

---

## Section 03 — Training loop

| | |
|---|---|
| Goal | Tokenize the corpus, build YOUR copywork model, run AdamW for 500 steps, save the trained weights |
| Answer file | `hf_pipeline/03_train.py` (~125 lines) |
| Copywork target | `vault/copywork/hf/03_training_loop/copywork_train.py` |
| Output | `~/.cache/hf_pipeline/model_copywork/` (HF `save_pretrained` directory) |
| Time | 5–10 min on M-series Mac |

### What you'll learn

- The minimal training loop: `for step in range(N): forward → backward → optimizer.step → optimizer.zero_grad`
- Random batch sampling from a token stream (`get_batch`)
- AdamW setup in three lines (vs. nanochat's 30-line `setup_optimizer`)
- Gradient clipping (`torch.nn.utils.clip_grad_norm_`)
- Periodic eval (`model.eval()` → `no_grad` → loss → `model.train()`)
- `model.save_pretrained()` produces a Hub-format directory in one call

### Adapting the answer file to use YOUR copywork model

The answer file `03_train.py` imports from the source `hf_nanochat/model.py`. To use *your* copywork code instead, replace the import block at the top of your `copywork_train.py`:

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

### How to run

```bash
python vault/copywork/hf/03_training_loop/copywork_train.py
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

Loss should drop from ~`ln(vocab_size)` ≈ 8.3 (random baseline) to ~5–6 over 500 steps. If loss doesn't move, something's wrong (most likely a bug in your copywork model).

---

## Section 04 — Evaluation and generation

| | |
|---|---|
| Goal | Load the trained model, compute perplexity on val set, generate text samples |
| Answer file | `hf_pipeline/04_eval_and_generate.py` (~115 lines) |
| Copywork target | `vault/copywork/hf/04_evaluation_and_generation/copywork_eval.py` |
| Output | stdout — perplexity number + sample continuations |
| Time | <1 min |

### What you'll learn

- `NanoChatModel.from_pretrained(path)` — round-trip the saved model from disk
- Computing perplexity from cross-entropy loss (`exp(loss)`)
- Bits-per-token metric (`loss / ln(2)`)
- HF's `model.generate()` — `do_sample`, `temperature`, `top_k`, `max_new_tokens`
- Tokenizer round-trip: `tokenizer.encode(text).ids` → model → `tokenizer.decode(ids)`

The output samples will be **gibberish** — your tiny model trained 500 steps on 2M tokens can't write coherent English. That's fine. You're verifying the *pipeline* works (loaded model → ran inference → produced text), not model quality.

---

## Section 05 — Save + Hub-ready

| | |
|---|---|
| Goal | Wrap the raw `tokenizers` BPE in HF's `PreTrainedTokenizerFast`, save model + tokenizer in one HF directory, demonstrate `push_to_hub` |
| Answer file | `hf_pipeline/05_save_share.py` (~80 lines) |
| Copywork target | `vault/copywork/hf/05_save_load_hub/copywork_save.py` |
| Output | `~/.cache/hf_pipeline/my-first-llm/` (Hub-loadable directory) |
| Time | <1 min |

### What you'll learn

- The HF directory layout: `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`
- `PreTrainedTokenizerFast` — wrapping a raw `tokenizers.Tokenizer` with HF metadata (special tokens)
- safetensors format (vs. pickle-based `.bin`) — security and lazy-loading wins
- Round-trip verification — load the saved dir back, confirm parameter count matches
- The `push_to_hub` calling pattern (we don't actually push, but the code shows the API)

After section 05, the directory is genuinely shippable — anyone with `transformers` installed could `from_pretrained` it and run inference.

---

## Optional: monitoring during training (web UI)

For a 5-min run, console output is enough. If you want live curves in a browser:

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

The industry standard, what nanochat uses.

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

Cloud dashboard, automatic run comparison across experiments, no `localhost` needed. Worth setting up once you train more than one model.

---

## Order of operations

```
section 01    →    section 02    →    section 03    →    section 04    →    section 05
(data+tok)         (config+model)     (training)         (eval+gen)         (save+share)
```

1. **Section 01** — copywork data download + tokenizer training, run them, produce `train.txt`, `val.txt`, `tokenizer.json` in `~/.cache/hf_pipeline/`
2. **Section 02** — `NanoChatConfig` + `NanoChatModel` (already done ✓)
3. **Section 03** — copywork the training loop, run it, watch loss decrease, save model
4. **Section 04** — copywork eval+gen, load saved model, compute perplexity, generate samples
5. **Section 05** — copywork save+share, produce Hub-ready directory

After all five, you've gone end-to-end: raw corpus → tokenizer → trained model → generated samples → reusable artifact. **That's "training an LLM by yourself."**

---

## "Done" checklist

- [ ] Section 01 — `copywork_data.py` matches `hf_pipeline/01_data.py`; `train.txt` + `val.txt` exist in cache
- [ ] Section 01 — `copywork_tokenizer.py` matches `hf_pipeline/02_tokenizer.py`; `tokenizer.json` exists; "The quick brown fox" sample encodes/decodes round-trip
- [x] Section 02 — `copywork_config.py` + `copywork_model.py` instantiate cleanly; smoke test passes
- [ ] Section 03 — `copywork_train.py` matches `hf_pipeline/03_train.py`; you ran it; loss decreased from ~8 to ~5–6; `model_copywork/` saved to cache
- [ ] Section 04 — `copywork_eval.py` matches `hf_pipeline/04_eval_and_generate.py`; you got a perplexity number; samples were gibberish (expected)
- [ ] Section 05 — `copywork_save.py` matches `hf_pipeline/05_save_share.py`; `~/.cache/hf_pipeline/my-first-llm/` exists with `config.json`, `model.safetensors`, `tokenizer.json`

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

All of these are documented in `journal/hf/hf_*.md`. None are required for the smoke-test loop — they're "what real production training looks like."

---

## Files in this copywork directory

```
vault/copywork/hf/
├── README.md                                   ← method, tier list, conventions
├── ROADMAP.md                                  ← THIS FILE — the journey
├── 01_data_and_tokenizer/
│   ├── copywork_data.py                        ← do this first
│   └── copywork_tokenizer.py                   ← then this
├── 02_config_and_model/
│   ├── copywork_config.py                      ← done ✓
│   └── copywork_model.py                       ← done ✓
├── 03_training_loop/
│   └── copywork_train.py
├── 04_evaluation_and_generation/
│   └── copywork_eval.py
└── 05_save_load_hub/
    └── copywork_save.py
```

Each `copywork_*.py` is created as a stub. Open it next to the answer file in your IDE, type the source line by line, drop `# Q:` comments for anything unclear, and ask. Diff visually when done.
