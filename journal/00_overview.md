---

## HF Pipeline 0 — Overview & nanochat ↔ Hugging Face Mapping
*The same GPT architecture, wrapped in HF APIs. What's identical, what changes, and why.*

> **Companion to:** [Architecture Overview](../00_architecture_overview.md)
> **Source code:** `hf_nanochat/`, `hf_pipeline/`
> **Annotation legend:** `[PT]` PyTorch built-in · `[NC]` nanochat custom · `[HF]` Hugging Face built-in

---

### Why a Hugging Face track?

nanochat is deliberately framework-free — every line is yours to read. That's perfect for learning. But in production, almost everyone uses Hugging Face Transformers because it gives you:

1. `model.generate()` — beam search, top-k, top-p, temperature, repetition penalty, all built-in
2. `model.save_pretrained()` / `from_pretrained()` — one-line save/load with config + weights + tokenizer
3. `model.push_to_hub()` — share on the Hugging Face Hub
4. `Trainer` — fine-tuning with built-in logging, checkpointing, eval, distributed training, mixed precision
5. `pipeline("text-generation")` — zero-code inference

**The architecture is identical to nanochat.** Only the wrapping changes.

---

### nanochat → HF mapping cheatsheet

| nanochat | Hugging Face | What changes |
|----------|-------------|-------------|
| `@dataclass GPTConfig` | `class NanoChatConfig(PretrainedConfig)` `[HF]` | JSON serialisation, `from_pretrained()` support |
| `class GPT(nn.Module)` | `class NanoChatModel(PreTrainedModel)` `[HF]` | Save/load, generate(), Hub integration |
| `def forward(self, idx, targets)` | `def forward(self, input_ids, labels)` | HF naming convention, returns dict |
| `def generate()` (manual loop) | Inherited from `GenerationMixin` `[HF]` | All sampling strategies built-in |
| `def init_weights()` | `def _init_weights(module)` | HF calls it per-module automatically |
| `torch.save(state_dict)` | `model.save_pretrained("path")` `[HF]` | Saves config + weights + metadata |
| Manual tokenizer | `AutoTokenizer.from_pretrained()` `[HF]` | Hub-hosted tokenizer |
| Custom training loop | `Trainer` `[HF]` | Logging, checkpointing, eval, DDP for free |

---

### Topics covered

<!-- Conceptual sections grow here as understanding deepens. Each topic gets a single, refined explanation — not a Q&A log. -->

*(empty — fills out as we work through the HF pipeline)*

---

### Related sections

- Data and tokenizer details → [01_data_and_tokenizer.md](01_data_and_tokenizer.md)
- Model and config classes → [02_model_and_config.md](02_model_and_config.md)
- Training loop and `Trainer` → [03_training_loop.md](03_training_loop.md)
- Evaluation and generation → [04_evaluation_and_generation.md](04_evaluation_and_generation.md)
- Save/load/Hub → [05_save_load_hub.md](05_save_load_hub.md)
- Internals deep dive → [06_deep_dive_internals.md](06_deep_dive_internals.md)
