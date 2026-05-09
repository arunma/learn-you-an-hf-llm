---

## HF Pipeline 2 — Model and Config
*Wrapping a GPT in `PreTrainedModel` and `PretrainedConfig`.*

> **Companion to:** [Q/K/V Projections](../03_qkv_projections.md), [Scaled Attention](../04_scaled_attention.md), [Residuals, Multi-Head & MLP](../05_residuals_multihead_mlp.md), [Transformer Block](../06_transformer_block.md)
> **Source code:** `hf_nanochat/model.py`, `hf_nanochat/configuration.py`
> **Annotation legend:** `[PT]` PyTorch · `[NC]` nanochat custom · `[HF]` Hugging Face

---

### Scope

How the same architecture from `nanochat/gpt.py` becomes Hub-loadable:
- `PretrainedConfig` `[HF]` vs `@dataclass GPTConfig` `[NC]`
- `PreTrainedModel` `[HF]` inheritance, `_init_weights()` hook
- Forward signature: `input_ids` / `labels` / `attention_mask` / returning `CausalLMOutputWithPast`
- `AutoModelForCausalLM` registration, `model_type` strings, custom code on the Hub
- Attention implementations (`eager`, `sdpa`, `flash_attention_2`)
- KV cache contract (`past_key_values`, `use_cache`)

---

### Topics covered

#### Attention layer naming — what `q_proj`, `k_proj`, `v_proj`, `o_proj` actually are

A common mix-up: thinking `q_proj` is a `T × T` matrix. It's not — that's the *attention score matrix*, a runtime activation. The projections are **learned weight matrices of shape `(d_model, d_model)`**, independent of sequence length.

Inside one attention block, given input `x` of shape `(B, T, d_model)`:

```text
Q = x @ W_q.T    shape (B, T, d_model)        W_q from q_proj — (d, d)
K = x @ W_k.T    shape (B, T, d_model)        W_k from k_proj — (d, d) (or (d, d_kv) with GQA)
V = x @ W_v.T    shape (B, T, d_model)        W_v from v_proj — (d, d)

# now T × T appears, but as a runtime activation, not a weight:
scores = Q @ K.T / √d_h           shape (B, n_head, T, T)   ← THIS is the T×T thing
out    = softmax(scores) @ V      shape (B, T, d_model)
out    = out @ W_o.T              shape (B, T, d_model)     W_o from o_proj — (d, d)
```

Standard naming in HF LLaMA-style models:

| Layer | Role | Shape | Typical LoRA target? |
|---|---|---|---|
| `q_proj` | input → query | `(d, d)` | **Yes — almost always** |
| `k_proj` | input → key | `(d, d)` or `(d, d_kv)` with GQA | Sometimes |
| `v_proj` | input → value | `(d, d)` | **Yes — almost always** |
| `o_proj` | head outputs → residual stream | `(d, d)` | Sometimes |
| `gate_proj`, `up_proj`, `down_proj` | the SwiGLU MLP | `(d, d_ff)`, `(d, d_ff)`, `(d_ff, d)` | For harder tasks |

Why LoRA targets `q_proj` and `v_proj` first: the original LoRA paper found these carry most task-specific signal. `q_proj` shapes "what am I looking for"; `v_proj` shapes "what information do I extract." `k_proj` is more about content matching and is often less task-specific. Adapting just `q_proj` + `v_proj` gets ~80% of the benefit at 50% of the params.

#### What each projection does, intuitively

All four matrices have the same shape `(C, C)` and act on the same input `x`. What makes Q, K, V, and the output projection different is the values *learned* in their respective weight matrices.

- **`q_proj` — "what am I looking for?"** Projects the current token into a **query**. The query gets compared against every other token's key.
- **`k_proj` — "what do I have to offer?"** Each token advertises itself with a **key**. High dot product between query and key → high attention weight.
- **`v_proj` — "here's my actual content."** Once attention weights are computed (from `Q·Kᵀ`), they take a weighted sum of `V` vectors. **`V` is what actually gets passed forward.**
- **`o_proj` — "mix back into the residual stream."** After attention combines the `V` vectors, `o_proj` projects the result back into the residual stream's coordinate system before adding it to the residual. With multi-head attention it also mixes information *across* heads — each head produces a `(head_dim,)`-vector, the heads are concatenated to `C`, and `o_proj` linearly mixes them so heads can pool information.

At initialisation, `c_q`, `c_k`, `c_v` are random matrices, so `Q`, `K`, `V` are basically random projections of the same data. Through training they specialise — `c_q` learns "what is this token looking for", `c_k` learns "what does this token offer", `c_v` learns "what content should this token contribute".

In the nanochat code (`my_nanochat/gpt.py:67-74`):

```python
self.c_q = Linear(self.n_embd, self.n_head    * self.head_dim, bias=False)   # (768, 768)
self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)   # (768, 768)
self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)   # (768, 768)
self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)                   # (768, 768)
```

(In LLaMA naming these become `q_proj`, `k_proj`, `v_proj`, `o_proj`.)

For LLaMA-7B with `C=4096`: each projection is `4096 × 4096 = 16.8M` params. Four of them = 67M per layer. With 32 layers ≈ **2.1B params just for attention projections** — about 30% of total model parameters.

#### Projections are functions, not stores

A common confusion: thinking `c_k` "stores" keys. It doesn't.

`self.c_k(x)` is a *function call* — it computes `x @ c_k.weight.T` on the fly. Every forward pass produces fresh `K` from whatever `x` you feed in. There's no cache, no memory of past keys, no database lookup.

The clean mental model:

| Thing | What it is | Shape | Computed how |
|---|---|---|---|
| `c_k.weight` (parameter) | Learned matrix that *produces* keys | `(C, C)` | Set during training |
| K (activation) | Actual keys for this batch's tokens | `(B, T, C)` | `x @ c_k.weight.T` per forward pass |
| KV cache (inference only) | Stored K, V from past tokens | `(B, T, C)` per layer | Accumulated during generation |

The KV cache is the *only* place K/V values get stored — and only during inference, never during training. Phrased differently: **`c_k` is the recipe; `K` is the dish; the KV cache is yesterday's leftovers.**

#### Training vs inference: same projections, different behaviour

**During training**, the entire sequence flows through together:

```python
x.shape == (B, T, C)             # all T tokens at once
k = self.c_k(x)                  # all T keys, computed in parallel
v = self.c_v(x)                  # all T values, computed in parallel
```

`k` and `v` already contain keys/values for *every position simultaneously*. There's no notion of "added to context" — the whole sequence is the input. The output is `(B, T, vocab_size)` — predictions for every position at once. Loss is computed in parallel: position 1 predicts token 2, position 2 predicts token 3, …, position T-1 predicts token T. **One forward pass produces T-1 useful predictions.**

Causality is enforced by the **causal mask** inside `softmax(Q·Kᵀ + mask)`. The mask adds `-∞` to the upper triangle of the `(T, T)` score matrix, zeroing out attention to future positions after softmax. Position `i` can only attend to positions `1..i` even though `K` and `V` contain all positions. **The mask, not sequential processing, is what makes attention causal.**

**During inference (autoregressive generation)**, tokens are produced one at a time:

```text
1. Generate token 1.  Compute k_1 = x_1 @ c_k.weight.T.   Cache: K = [k_1].
2. Generate token 2.  Compute k_2 = x_2 @ c_k.weight.T.   Cache: K = [k_1, k_2].
3. For attention at step 2:
     q_2     = x_2 @ c_q.weight.T
     scores  = q_2 @ K.T            ← uses cached K, no recomputation
     out     = softmax(scores) @ V  ← uses cached V
4. Append k_3, v_3 to cache. Repeat.
```

This is the **KV cache**. It makes generation cost per token roughly *constant* in past-token compute (you only compute `q`, `k`, `v` for the new token), instead of `O(T)`. Without the cache, generating the 1000th token would require recomputing the previous 999 keys and values.

Why no cache during training: training has all `T` tokens up front, so it processes them in parallel — the cache offers no speedup, and storing it would waste memory.

This parallel-during-training, sequential-during-inference duality is *the* reason transformers train so much better than RNNs. RNNs had to process tokens one at a time even during training. Transformers parallelise the forward pass over `T` positions and use the mask to enforce causality.

#### Activations vs parameters — keep these separate

Two fundamentally different kinds of tensors live in a model:

| | Activations | Parameters |
|---|---|---|
| Shape | `(B, T, C)`, `(B, n_head, T, T)`, … | `(C, C)`, `(C, d_ff)`, … |
| Per-batch? | Yes — change every batch | No — same regardless of input |
| Examples | x, Q, K, V, attention scores, MLP intermediates | `c_q.weight`, layer-norm γ |
| Has a `.grad`? | No | Yes — same shape as parameter |
| Has Adam `m`, `v`? | No | Yes — same shape as parameter |

This separation matters for memory:

- Adam's `m` and `v` track **parameters**. They don't scale with batch size or sequence length. For `c_q.weight` of shape `(C, C)`, `m` and `v` are also `(C, C)` — fixed cost, independent of training data.
- Activations *do* scale with batch and sequence — `(B, T, C)`. They cost memory because the backward pass needs them, but they're not optimizer state.
- The KV cache is also activation-shaped `(B, T, C)`, but lives during inference only.

So when people say "Adam state costs 8 bytes per parameter", they mean per *parameter*, not per *activation*. Big relief — otherwise optimizer memory would explode with sequence length.

#### End-to-end attention forward + backward shape walkthrough

For one input batch `x` of shape `(B, T, C)`:

```text
Step 1 — Project (matmul against c_q, c_k, c_v):
  Q = x @ c_q.weight.T      → (B, T, C)
  K = x @ c_k.weight.T      → (B, T, C)
  V = x @ c_v.weight.T      → (B, T, C)

Step 2 — Reshape across heads:
  Q → (B, n_head,    T, head_dim)
  K → (B, n_kv_head, T, head_dim)        ← n_kv_head ≤ n_head with GQA
  V → (B, n_kv_head, T, head_dim)

Step 3 — Compute attention scores:
  scores = Q @ K.transpose(-2,-1) / √head_dim   → (B, n_head, T, T)

Step 4 — Apply causal mask + softmax:
  scores += causal_mask         ← -∞ in the upper triangle
  weights = softmax(scores)     → (B, n_head, T, T)

Step 5 — Weighted sum of values, project back:
  out = weights @ V             → (B, n_head, T, head_dim)
  out = out.transpose(1,2).reshape(B, T, C)
  out = out @ c_proj.weight.T   → (B, T, C)

Step 6 — Backward (loss.backward()):
  Autograd accumulates dL/dc_q, dL/dc_k, dL/dc_v, dL/dc_proj.
  Each gradient has the same shape as its weight: (C, C).
  Gradient contributions from every token position get
  accumulated into ONE (C, C) tensor per weight matrix.
  Adam reads the gradient, updates m and v ((C,C) each),
  and computes the new weight.
```

Two things worth noticing:

- **Output shape `(B, T, C)` matches input shape.** That's why blocks stack — block N+1 takes block N's output with no reshaping.
- **One training step on a sequence of length T is roughly T times more effective than one autoregressive step**, because the gradient accumulates contributions from `T` predictions simultaneously into the *same* `(C, C)` weight gradient.

---

### Related sections

- Pipeline overview → [hf_00_overview.md](hf_00_overview.md)
- Data and tokenizer → [hf_01_data_and_tokenizer.md](hf_01_data_and_tokenizer.md)
- Training loop → [hf_03_training_loop.md](hf_03_training_loop.md)
- nanochat counterpart → [06_transformer_block.md](../06_transformer_block.md)
