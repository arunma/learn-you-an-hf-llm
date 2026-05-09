---

## HF Pipeline 4 — Evaluation and Generation
*Perplexity, `model.generate()`, and the `GenerationConfig` knob set.*

> **Companion to:** [Training & Generation](../07_training_and_generation.md), [Appendix: Modern Techniques](../08_appendix_modern_techniques.md)
> **Source code:** `hf_pipeline/06_evaluate/`, `hf_pipeline/07_generate/`
> **Annotation legend:** `[PT]` PyTorch · `[NC]` nanochat custom · `[HF]` Hugging Face

---

### Scope

Measuring quality and producing text:
- Perplexity — definition, what it actually means, why it's `exp(loss)` and not loss itself
- `evaluate` `[HF]` library — perplexity, BLEU, ROUGE, accuracy on benchmark datasets
- `model.generate()` `[HF]` — greedy, sampling, top-k, top-p, beam search, contrastive
- `GenerationConfig` `[HF]` — every sampling parameter, what each does to the distribution
- Stopping criteria, repetition penalty, length penalty
- Streaming output (`TextStreamer`)
- Batched generation, padding-side semantics, attention masks during generation
- KV cache reuse for autoregressive decoding

---

### Topics covered

#### Beam search

Greedy decoding picks the single highest-probability token at every step — locally optimal, globally myopic. Beam search keeps the **top `k` partial sequences ("beams") alive** at every step.

The algorithm with beam width `k`:

1. Start with the prompt as the only beam.
2. Expand each beam by every possible next token, producing `k × V` candidates.
3. Score each by **cumulative log-probability** of the whole sequence.
4. Keep the top `k` overall, discard the rest.
5. Repeat until all `k` beams hit EOS or `max_length`.
6. Return the highest-scoring completed beam.

```text
prompt: "The cat"      k = 3

step 1 — expand "The cat":
  "The cat sat"   logp = -1.2  ← keep
  "The cat ran"   logp = -1.5  ← keep
  "The cat slept" logp = -1.8  ← keep
  "The cat blue"  logp = -7.0  ← drop
  …

step 2 — expand all 3 beams, keep top 3 overall:
  "The cat sat on"     logp = -1.9
  "The cat ran fast"   logp = -2.4
  "The cat sat down"   logp = -2.5
  …
```

`model.generate()` knobs `[HF]`:

| Parameter | Effect |
|---|---|
| `num_beams=1` | Greedy (the default) |
| `num_beams=4` | Beam search with width 4 |
| `length_penalty=α` | Score divided by `length^α`. Without it, every extra token adds a negative log-prob, so beam search collapses to short outputs. `α<1` favours shorter, `α>1` favours longer. |
| `early_stopping=True` | Stop as soon as `num_beams` beams have hit EOS, even if better beams might still emerge. |
| `no_repeat_ngram_size=3` | Forbid any beam from repeating a trigram. Cheap quality boost. |

**Use it for:** translation, summarisation, code completion — tasks with a "right answer".

**Avoid it for:** open-ended generation (chat, creative writing). Beam search produces bland, repetitive, "average" text because it gravitates to the highest-likelihood paths, which are often the most generic ones — the **likelihood trap**. For these, sample with top-p / top-k instead.

**Cost:** roughly `k×` greedy compute and memory; KV cache must be replicated per beam.

#### Repetition penalty

Pure language models love to loop. Once a token is in the context, its high probability tends to stay high, producing `"the the the"` or whole phrases on repeat. `repetition_penalty` (Keskar et al., CTRL paper, 2019) penalises every token already in context:

```text
for every token t already in (prompt + generated so far):
    logit[t] = logit[t] / r   if logit[t] > 0
    logit[t] = logit[t] * r   if logit[t] ≤ 0
```

The sign flip is essential — you can't just divide a negative logit by `r > 1`; that makes it *less* negative, i.e. *more* likely. The two-branch formula guarantees the penalty always reduces the value.

| `repetition_penalty` | Behaviour |
|---|---|
| `1.0` | Off (default) |
| `1.1` – `1.2` | Light, usually safe |
| `1.3` – `1.5` | Aggressive; can suppress legitimate repetition (names, articles, code keywords) |
| `> 1.5` | Text starts feeling unnatural and word-salady |

Related knobs in `GenerationConfig` `[HF]`:

- **`no_repeat_ngram_size=N`** — hard ban on repeating any n-gram of length `N`. More surgical than `repetition_penalty`: the penalty hits *every* repeated single token; the n-gram ban only hits *exact* n-gram repeats. Common pairing: `no_repeat_ngram_size=3` for summarisation.
- **`encoder_repetition_penalty`** — seq2seq variant: penalises tokens from the *input*, useful when you want abstractive (not extractive) output.
- **`frequency_penalty` / `presence_penalty`** (OpenAI-style; available in newer HF) — additive penalties scaled by how *often* a token appeared, not just whether it appeared. Finer-grained.

**Heuristic:** start with `repetition_penalty=1.1` + `no_repeat_ngram_size=3`. Raise only if loops persist; lower if names or keywords get suppressed.

#### Attention mask and pad tokens during generation

When calling `model.generate(input_ids, ...)` without `attention_mask` or `pad_token_id`, HF prints up to three warnings in sequence:

```text
The attention mask and the pad token id were not set. ...
Setting `pad_token_id` to `eos_token_id`:N for open-end generation.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. ...
```

These are **defensive warnings, not errors**. To know whether to care, you need to know what attention masks are for.

##### What `attention_mask` is for

When batching sequences of different lengths together, shorter ones need padding tokens to make the batch tensor rectangular:

```text
prompts:        ["The cat", "Once upon a"]
                (length 2)   (length 3)

batched (B=2, T=4):
  input_ids       = [[101, 234,   0,   0],     ← 0 = <|pad|> id
                     [567, 890,  12,   0]]
  attention_mask  = [[  1,   1,   0,   0],
                     [  1,   1,   1,   0]]
```

Without the mask, attention would mix padding tokens into the output as real content. With it, masked positions get `-∞` injected into the `(B, n_head, T, T)` attention scores before softmax → zeroed out after softmax.

##### Why HF warns even when batch=1

For a single unpadded prompt, the mask doesn't matter — no padding exists. But HF's `generate()` has no way to know your input is unpadded, so it warns defensively whenever `attention_mask` isn't provided. It then auto-sets `pad_token_id = eos_token_id` so generation can at least stop at EOS. The third warning fires because pad and EOS now share an ID — HF can no longer disambiguate "real EOS, stop here" from "this is padding, ignore it" by looking at `input_ids` alone.

##### When to care

| Situation | Warnings matter? |
|---|---|
| Single prompt, batch = 1 | **No** — outputs identical with or without mask |
| Batched generation, all prompts same length | No — no padding present |
| Batched generation, mixed lengths | **Yes** — without mask, attention attends to pad tokens; output degrades |
| Generation with explicit EOS stopping | **Yes if** `pad_token_id == eos_token_id` — model may stop on padding |

For smoke tests with single prompts, the warnings can be ignored. They don't change behaviour.

##### How to silence them cleanly

For a single prompt with no padding, pass an all-ones mask and explicit pad/eos IDs:

```python
input_ids = torch.tensor([tokenizer.encode(prompt).ids], device=DEVICE)
attention_mask = torch.ones_like(input_ids)  # all 1s — no padding

out = model.generate(
    input_ids,
    attention_mask=attention_mask,
    pad_token_id=tokenizer.token_to_id("<|pad|>"),  # real pad, NOT eos
    eos_token_id=tokenizer.token_to_id("<|eos|>"),
    max_new_tokens=60,
    do_sample=True,
    temperature=0.8,
    top_k=50,
)
```

For batched generation with mixed-length prompts, get both from the tokenizer's padded output:

```python
encoded = tokenizer(prompts, padding=True, return_tensors="pt")
input_ids       = encoded["input_ids"]
attention_mask  = encoded["attention_mask"]
```

---

### Related sections

- Pipeline overview → [hf_00_overview.md](hf_00_overview.md)
- Training loop → [hf_03_training_loop.md](hf_03_training_loop.md)
- Save / load / Hub → [hf_05_save_load_hub.md](hf_05_save_load_hub.md)
- nanochat counterpart → [07_training_and_generation.md](../07_training_and_generation.md)
