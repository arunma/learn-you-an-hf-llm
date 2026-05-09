---

## HF Pipeline 5 — Save, Load, and the Hub
*Persistence, sharing, and version control for models, tokenizers, and datasets.*

> **Companion to:** [Architecture Overview](../00_architecture_overview.md)
> **Source code:** `hf_pipeline/08_save_and_share/`
> **Annotation legend:** `[PT]` PyTorch · `[NC]` nanochat custom · `[HF]` Hugging Face

---

### Scope

Everything that turns a trained model into something reusable:
- `save_pretrained()` / `from_pretrained()` `[HF]` — what's in the directory it produces (`config.json`, `model.safetensors`, `tokenizer.json`, `generation_config.json`, `special_tokens_map.json`)
- safetensors `[HF]` — why it replaced `pickle`-based `pytorch_model.bin`, the security and lazy-loading wins
- Sharded checkpoints (`model-00001-of-00003.safetensors` + index)
- `push_to_hub()` `[HF]` — repos, commits, branches, large file storage (LFS)
- `model_card.md` and the `huggingface_hub` `[HF]` API for metadata
- Private vs public, gated models, access tokens
- `from_pretrained(..., revision=..., token=...)` — pinning to a commit/branch
- `AutoModel` discovery and `trust_remote_code=True` for custom architectures

---

### Topics covered

*(empty — fills out as we work through saving and sharing)*

---

### Related sections

- Pipeline overview → [hf_00_overview.md](hf_00_overview.md)
- Evaluation and generation → [hf_04_evaluation_and_generation.md](hf_04_evaluation_and_generation.md)
- Internals deep dive → [hf_06_deep_dive_internals.md](hf_06_deep_dive_internals.md)
