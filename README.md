# learn-you-an-hf-llm

Build a tiny language model from scratch using the Hugging Face stack — `transformers`, `tokenizers`, `datasets`, `safetensors` — in five small stages you can run on a laptop in 15 minutes.

The point isn't model quality (it produces gibberish — it's 3M parameters trained for 500 steps). The point is that you understand every line of the pipeline.

## Companion repo

This is the **Hugging Face track** of a two-repo learning project. The companion repo, [`learn-you-a-nanochat`](../learn-you-a-nanochat), implements the same architecture in plain PyTorch with no framework abstractions. Either is self-contained; together they show the same model from both ends.

The model architecture (RoPE, GQA-ready attention, RMSNorm, ReLU² MLP) comes from Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat).

## Repo layout

```
learn-you-an-hf-llm/
├── hf_nanochat/        # source model wrapper (NanoChatConfig + NanoChatModel)
├── hf_pipeline/        # five staged scripts that chain into a full training run
├── copywork/           # type-it-yourself companion files for each stage
├── sections/           # conceptual section notes (read-first reference)
├── journal/            # consolidated reference book — every concept in detail
├── pyproject.toml
└── README.md
```

## Getting started

```bash
git clone <this repo>
cd learn-you-an-hf-llm
pip install -e .

# Smoke test the abstracted pipeline (~5 min on M-series Mac):
python -m hf_pipeline.run_all
```

If that completes successfully, your environment is ready.

## Two ways through the repo

**Path 1 — Just run it.** Run `hf_pipeline/01_data.py` through `05_save_share.py` in order, read the docstrings, dip into `journal/` when something is unclear. ~15 min runtime plus reading.

**Path 2 — Copywork.** Open each `copywork/0X_*/copywork_*.py` next to the matching `hf_pipeline/0X_*.py`. Type the source line by line. Drop `# Q: ...` comments on anything unclear. ~4–6 hours, builds real intuition.

See `copywork/ROADMAP.md` for the staged plan and `copywork/README.md` for the method.

## License

MIT. See LICENSE.
