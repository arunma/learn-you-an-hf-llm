---

## HF Pipeline 6 — Deep Dive Internals
*Component-by-component map of every nanochat building block to its Hugging Face equivalent, with constraints and tradeoffs.*

> **Companion to:** [Appendix: Modern Techniques](../08_appendix_modern_techniques.md), [Transformer Block](../06_transformer_block.md)
> **Source code:** entire `hf_nanochat/` package
> **Annotation legend:** `[PT]` PyTorch · `[NC]` nanochat custom · `[HF]` Hugging Face

---

### Scope

The detail layer underneath the other HF notes. Cross-references everywhere.
- Architecture choices: where HF lets you swap in custom code vs where the framework forces a shape
- RoPE, GQA, sliding window attention — how they look in HF reference implementations (LLaMA, Mistral, Gemma)
- Weight tying (`lm_head.weight = wte.weight`) — when HF does it automatically, when you have to wire it yourself
- Custom modeling code on the Hub (`modeling_*.py`, `configuration_*.py`) and `trust_remote_code`
- Activation checkpointing, FSDP shard plans, tensor parallelism hooks
- Numerical precision pitfalls (RoPE in fp16, attention masks in bf16, softcap stability)
- Performance: `torch.compile`, FlashAttention 2, paged attention, speculative decoding
- Quantization (`bitsandbytes`, GPTQ, AWQ) — what changes in the model wrapper

---

### Topics covered

*(empty — fills out as deep questions arise during copywork)*

---

### Related sections

- Pipeline overview → [hf_00_overview.md](hf_00_overview.md)
- Model and config → [hf_02_model_and_config.md](hf_02_model_and_config.md)
- Save / load / Hub → [hf_05_save_load_hub.md](hf_05_save_load_hub.md)
- nanochat counterpart → [08_appendix_modern_techniques.md](../08_appendix_modern_techniques.md)
