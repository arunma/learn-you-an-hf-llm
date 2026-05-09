# learn-you-an-hf-llm

> Build a tiny language model from scratch using the Hugging Face stack — `transformers`, `tokenizers`, `datasets`, `safetensors` — in five small stages you can run on a laptop in under fifteen minutes.

This repo is a **learning project**, not a production toolkit. The model it produces is 3 million parameters trained for 500 steps on Wikipedia. It writes gibberish — but gibberish that statistically resembles its training corpus, with a wiring diagram that's identical to what production training pipelines look like at scale.

The point is that **you understand every line** of how a modern LLM is trained, from raw text to a Hub-shippable model directory.

---

## Who this is for

- **You've heard of `transformers` and `model.generate()`** but never built a model from scratch.
- **You can read Python** and have used PyTorch enough to know what a tensor is.
- **You don't need a GPU.** Apple Silicon (MPS) or CPU will do for the smoke run.
- **Math comfort: minimal.** Matrix multiplication and softmax intuition is plenty. The journal walks through any harder math when it appears.

If you want to go deeper into the *architecture* (RoPE math, attention internals, weight init), the companion repo [learn-you-a-nanochat](../learn-you-a-nanochat) implements the same model in plain PyTorch with no framework abstractions. Reading either teaches the same ideas through a different lens.

---

## What you'll build

By the end you'll have:

- A **3M-parameter GPT-style transformer** trained for 500 steps on WikiText-2
- A **byte-level BPE tokenizer** (vocab 4096, three special tokens) you trained yourself
- A `~/.cache/hf_pipeline/my-first-llm/` directory that any other machine could `from_pretrained()` and run
- Working understanding of the five-stage HF training pipeline:
  1. **Data** — `datasets.load_dataset()` → text files
  2. **Tokenizer** — `tokenizers` BPE → `tokenizer.json`
  3. **Train** — your `NanoChatConfig` + `NanoChatModel` + AdamW loop → trained weights
  4. **Evaluate / Generate** — perplexity on val set, sample text via `model.generate()`
  5. **Save / Share** — `save_pretrained()` directory ready for `push_to_hub()`

---

## Repo layout

```
learn-you-an-hf-llm/
├── README.md              ← you are here
├── pyproject.toml         ← dependencies (pip install -e .)
├── .gitignore
│
├── hf_nanochat/           ← the source model wrapper
│   ├── __init__.py
│   └── model.py           ← NanoChatConfig + NanoChatModel (HF-compatible)
│
├── hf_pipeline/           ← the answer files — five staged training scripts
│   ├── README.md
│   ├── 01_data.py         ← download WikiText-2
│   ├── 02_tokenizer.py    ← train BPE
│   ├── 03_train.py        ← build model + AdamW loop + save
│   ├── 04_eval_and_generate.py
│   ├── 05_save_share.py
│   └── run_all.py         ← chain all stages with LlamaForCausalLM (smoke test)
│
├── copywork/              ← type-it-yourself companion files
│   ├── README.md          ← method, tier list, conventions
│   ├── ROADMAP.md         ← end-to-end journey, "done" checklist
│   ├── 01_data_and_tokenizer/
│   ├── 02_config_and_model/
│   ├── 03_training_loop/
│   ├── 04_evaluation_and_generation/
│   └── 05_save_load_hub/
│
├── sections/              ← short conceptual notes for each stage
│   └── HF-01..04, 14..16
│
└── journal/               ← consolidated reference book
    └── 00..06_*.md        ← every concept in detail; read when stuck
```

---

## Setup

Requires Python 3.10 or newer.

```bash
git clone <this repo>
cd learn-you-an-hf-llm
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

That installs `torch`, `transformers`, `tokenizers`, `datasets`, `safetensors`, and `tensorboard`.

### Smoke test (~5 min on M-series Mac)

Verify your environment by running the abstracted end-to-end pipeline:

```bash
python -m hf_pipeline.run_all
```

Expected output (last few lines):

```
Step  499/500 | train loss: 4.98 | val loss: 5.27 | 3,612 tok/s
Training complete in 215.3s
Best val loss: 5.27 | Perplexity: 194.2
...
Generated: torch.Size([1, 10]) → torch.Size([1, 30])
...
Saved to: /Users/<you>/.cache/hf_pipeline/my-first-llm
```

If that completes without errors, you're ready to learn.

---

## Two paths through the repo

### Path 1 — read the answer files

Run the five staged scripts in order, read each one's docstring, dip into `journal/` when something is unclear.

```bash
python hf_pipeline/01_data.py
python hf_pipeline/02_tokenizer.py
python hf_pipeline/03_train.py
python hf_pipeline/04_eval_and_generate.py
python hf_pipeline/05_save_share.py
```

Total: ~15 minutes runtime plus reading. Good for a quick orientation.

### Path 2 — copywork

Open each `copywork/0X_*/copywork_*.py` file next to its matching `hf_pipeline/0X_*.py` answer file. **Type the source line by line.** Drop `# Q: <question>` comments on anything unclear. After each section, eyeball-diff your file against the answer.

This is the slow path — ~4–6 hours over a few sittings — but it's the one that actually builds intuition. Reading is passive; typing is active. When your hand pauses over a line, that's the gap you didn't know you had.

Start with `copywork/ROADMAP.md` for the staged plan and `copywork/README.md` for the method.

---

## What you'll learn

From the answer files:

- How a custom model class plugs into the HF ecosystem (`PretrainedConfig`, `PreTrainedModel`, `GenerationMixin`)
- BPE tokenizer training from scratch — pre-tokenizer, model, trainer, decoder
- A minimal AdamW training loop (no `Trainer`, no DDP — just the loop)
- Why `model.generate()`, `save_pretrained()`, and `push_to_hub()` work
- The shape of a real production training pipeline

From the journal entries:

- Attention mechanics, RoPE math, KV cache memory accounting
- Optimizer state breakdown (why Adam costs 18 bytes/param)
- LoRA / QLoRA / `BitsAndBytesConfig`
- RLHF: PPO, GRPO, value functions, reward models
- Generation knobs: beam search, top-k, top-p, repetition penalty

---

## Architectural ancestry

The model architecture comes from Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat) — a clean, framework-free GPT implementation. This repo wraps that architecture in HF abstractions so you can compare:

|                  | nanochat (plain PyTorch) | This repo (HF) |
|------------------|--------------------------|----------------|
| Config           | `@dataclass GPTConfig`   | `class NanoChatConfig(PretrainedConfig)` |
| Model            | `class GPT(nn.Module)`   | `class NanoChatModel(PreTrainedModel)` |
| Generation       | manual autoregressive loop | inherited from `GenerationMixin` |
| Save / load      | `torch.save(state_dict)` | `model.save_pretrained(path)` |
| Tokenizer        | hand-rolled BPE           | `tokenizers` + `PreTrainedTokenizerFast` |

Same architecture (RoPE, GQA-ready attention, RMSNorm, ReLU² MLP). Different abstraction level.

---

## Going beyond the smoke test

Once your pipeline runs end-to-end, the natural upgrades — all documented in `journal/06_deep_dive_internals.md`:

| Upgrade | What changes |
|---|---|
| Bigger model | bump `n_layer`, `n_embd`, `seq_len`, `num_steps` |
| Better corpus | swap WikiText-2 for TinyStories or FineWeb-edu |
| Real GPU | rent an H100 hour, ~50× faster |
| `bf16` mixed precision | half the memory, ~2× throughput |
| FlashAttention 2 | 2–4× attention speedup at long context |
| KV cache | linear-time inference instead of quadratic |
| HF `Trainer` | DDP, checkpointing, logging for free |
| TRL + PEFT | QLoRA fine-tuning of real base models |

---

## License

MIT. See [LICENSE](LICENSE).

## Credits

- [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy — architecture and design choices
- Hugging Face — `transformers`, `tokenizers`, `datasets`, `safetensors`
