# hf_pipeline — the answer files

Five small staged scripts (plus one all-in-one) that together build a tiny LLM from raw text. These are the **answer files** the copywork in `../copywork/` is written against.

---

## Quick start

```bash
# All five stages in one process (uses LlamaForCausalLM — the abstracted version):
python -m hf_pipeline.run_all
```

That's the smoke test. ~5–10 minutes on an M-series Mac. Loss should drop from ~8 to ~5 over 500 steps.

---

## Two ways to run

### A. One-shot (`run_all.py`)

`run_all.py` chains every stage in a single Python process — data → tokenizer → tokenize → build model → train → eval → generate → save. It uses HF's prebuilt `LlamaConfig` / `LlamaForCausalLM`, so all you see is the *pipeline* shape, not the model internals.

Best for: smoke-testing your environment, or as a comparison reference.

### B. Five staged scripts (`01_data.py` ... `05_save_share.py`)

Each script does one stage, reads its inputs from `~/.cache/hf_pipeline/`, writes its outputs there, then exits. The next stage picks up where the previous left off — disk-passing pattern.

These use the **custom** `NanoChatConfig` / `NanoChatModel` from `../hf_nanochat/model.py` instead of `LlamaForCausalLM`. That's the less-abstracted path: every line of the model is in repo code you can read.

Best for: copywork, debugging one stage at a time, swapping in your own model.

---

## The five stages

| # | Script | What it does | Reads | Writes | Time |
|---|---|---|---|---|---|
| 1 | `01_data.py` | Download WikiText-2 (~2M tokens), drop empty lines | (network) | `train.txt`, `val.txt` | ~30 s |
| 2 | `02_tokenizer.py` | Train byte-level BPE, vocab 4096, special tokens `<\|pad\|>` `<\|bos\|>` `<\|eos\|>` | `train.txt` | `tokenizer.json` | ~30 s |
| 3 | `03_train.py` | Tokenize corpus; build `NanoChatModel` (4 layers, 256 dim, ~3M params); 500 steps of AdamW; save | `train.txt`, `val.txt`, `tokenizer.json` | `model_trained/` (HF format) | 5–10 min |
| 4 | `04_eval_and_generate.py` | Load model; perplexity on val; sample text from prompts | `tokenizer.json`, `val.txt`, `model_trained/` | (stdout only) | <1 min |
| 5 | `05_save_share.py` | Wrap tokenizer in `PreTrainedTokenizerFast`; save model + tokenizer in one HF dir; round-trip test | `tokenizer.json`, `model_trained/` | `my-first-llm/` (Hub-loadable) | <1 min |

All paths are under `~/.cache/hf_pipeline/`.

Run them in order:

```bash
python hf_pipeline/01_data.py
python hf_pipeline/02_tokenizer.py
python hf_pipeline/03_train.py
python hf_pipeline/04_eval_and_generate.py
python hf_pipeline/05_save_share.py
```

You can rerun any stage independently as long as its inputs exist. E.g. once you have `model_trained/`, you can rerun `04_eval_and_generate.py` with different sampling parameters without retraining.

---

## File layout

```
hf_pipeline/
├── __init__.py
├── README.md                ← this file
├── run_all.py               ← all 8 stages in one process (LlamaForCausalLM)
├── 01_data.py               ← stage 1: download corpus
├── 02_tokenizer.py          ← stage 2: train BPE
├── 03_train.py              ← stage 3: tokenize + build NanoChatModel + train + save
├── 04_eval_and_generate.py  ← stage 4: load + perplexity + samples
└── 05_save_share.py         ← stage 5: HF-format save + Hub instructions
```

The staged scripts depend on `../hf_nanochat/model.py` for the custom model classes.

---

## Tunable knobs

Open any of the staged scripts; the constants at the top are what you'd typically change for a real run:

| Constant | Default (in `03_train.py`) | What changing it does |
|---|---|---|
| `N_LAYER` | 4 | Number of transformer blocks. Bigger → more capacity, more compute. |
| `N_HEAD` | 4 | Attention heads per layer. Must divide `N_EMBD`. |
| `N_EMBD` | 256 | Hidden dimension (per-token vector size). |
| `SEQ_LEN` | 256 | Context length. Bigger → quadratic attention cost. |
| `BATCH_SIZE` | 8 | Sequences per training step. |
| `LR` | 3e-4 | AdamW learning rate. |
| `NUM_STEPS` | 500 | Training iterations. |

For TinyStories or FineWeb-edu, swap `01_data.py`'s `load_dataset(...)` call for the new corpus. Everything downstream is corpus-agnostic.

---

## What you'll see when it works

End of `run_all.py`:

```
Step  499/500 | train loss: 4.98 | val loss: 5.27 | 3,612 tok/s
Training complete in 215.3s
Best val loss: 5.27 | Perplexity: 194.2
...
Generated: torch.Size([1, 10]) → torch.Size([1, 30])
...
Saved to: /Users/<you>/.cache/hf_pipeline/my-first-llm
```

The samples are gibberish-but-statistical — that's expected for a 3M-param model trained 500 steps on 2M tokens.

---

## Why two model paths (custom vs `LlamaForCausalLM`)

`run_all.py` uses `LlamaForCausalLM` (HF prebuilt) — the abstracted path. Fastest to run, least code to read.

`03_train.py` uses `NanoChatModel` from `../hf_nanochat/model.py` — the less-abstracted path. The model class has ~300 lines of readable code (attention, RoPE, MLP, block, full model). When you copywork section 02, you're typing this file.

Both produce models that train identically. The choice is about how much of the model internals you want visible.

---

## Cost at scale (for context)

| Scale | Compute cost | Output |
|---|---|---|
| This demo: 3M params, 500 steps | ~$0 (your laptop) | Gibberish (learning exercise) |
| nanochat speedrun: 300M params, 10B tokens | ~$50 (one H100 hour) | GPT-2-grade chatbot |
| LLaMA-7B equivalent: 7B params, ~1T tokens | ~$75K (cluster + days) | Production model |

The compute cost scales as roughly `6 × params × tokens` FLOPs regardless of library. See [`../journal/06_deep_dive_internals.md`](../journal/06_deep_dive_internals.md) for the derivation.

---

## Where to learn more

- [`../journal/`](../journal/) — consolidated reference book; every concept in detail (attention math, RoPE, KV cache, optimizer state, LoRA, RLHF/PPO/GRPO, etc.)
- [`../sections/`](../sections/) — short conceptual notes for each stage (read-first reference)
- [`../copywork/ROADMAP.md`](../copywork/ROADMAP.md) — the staged copywork journey with checklists
- [`../copywork/README.md`](../copywork/README.md) — copywork method, tier list, conventions
