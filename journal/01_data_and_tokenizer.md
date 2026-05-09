---

## HF Pipeline 1 — Data and Tokenizer
*Datasets, BPE training, and tokenization in the Hugging Face stack.*

> **Companion to:** [Tokenisation](../01_tokenisation.md), [Embeddings](../02_embeddings.md)
> **Source code:** `hf_pipeline/01_data/`, `hf_pipeline/02_tokenizer/`
> **Annotation legend:** `[PT]` PyTorch · `[NC]` nanochat custom · `[HF]` Hugging Face

---

### Scope

Everything that runs *before* the model sees a tensor:
- Acquiring a corpus with `datasets` `[HF]`
- Streaming vs in-memory loading
- Training a BPE tokenizer with `tokenizers` `[HF]`
- `AutoTokenizer` round-trips (encode, decode, batch padding, attention masks)
- How HF tokenization differs from nanochat's manual `tiktoken` path

---

### Topics covered

*(empty — fills out as we work through data loading and tokenization)*

---

### Related sections

- Pipeline overview → [00_overview.md](00_overview.md)
- Model and config → [02_model_and_config.md](02_model_and_config.md)
- nanochat counterpart → [01_tokenisation.md](../01_tokenisation.md)
