---

## HF Pipeline 3 — Training Loop
*From a hand-rolled loop to `Trainer`, `TrainingArguments`, and `accelerate`.*

> **Companion to:** [Training & Generation](../07_training_and_generation.md)
> **Source code:** `hf_pipeline/04_pretrain/`, `hf_pipeline/05_finetune/`
> **Annotation legend:** `[PT]` PyTorch · `[NC]` nanochat custom · `[HF]` Hugging Face

---

### Scope

Everything that turns a forward pass into a trained model:
- `Trainer` `[HF]` vs hand-rolled training loops `[NC]`
- `TrainingArguments` `[HF]` — every knob (batch size, grad accumulation, mixed precision, logging, eval cadence, save strategy)
- `DataCollatorForLanguageModeling` `[HF]` — causal LM masking, padding, label shifting
- Optimizer parameter groups, weight decay, learning-rate schedulers
- Mixed precision (`fp16`, `bf16`), gradient checkpointing
- Distributed training with `accelerate` `[HF]` — DDP, FSDP, DeepSpeed configs
- Logging hooks (`Weights & Biases`, `TensorBoard`, custom callbacks)
- Checkpointing semantics — what's saved, what resumes, what doesn't

---

### Topics covered

#### TRL — Transformer Reinforcement Learning

HF library for **post-training** an LLM — turning a base model into an instruction-following or aligned model. It assumes the base model already exists; it doesn't pretrain.

Trainers it ships:

| Trainer | What it does | Data shape | Memory cost |
|---|---|---|---|
| `SFTTrainer` `[HF/TRL]` | **Supervised fine-tuning** on instruction data — the first step of alignment. Next-token prediction with a chat template applied; optionally completion-only loss. | `(prompt, response)` or `messages=[…]` | 1× model |
| `DPOTrainer` `[HF/TRL]` | **Direct Preference Optimization.** Trains directly on preference pairs without a reward model. Default for preference tuning post-2024. | `(prompt, chosen, rejected)` | 2× model (policy + frozen reference) |
| `PPOTrainer` `[HF/TRL]` | **Classical RLHF.** Needs reward model + reference + policy + value head. Powerful but finicky. | `(prompt, generated, reward)` | 4× model |
| `GRPOTrainer` `[HF/TRL]` | **Group Relative Policy Optimization** (DeepSeek-R1 style). Multiple completions per prompt; advantages normalised within the group; no value head. | `(prompt, reward_fn)` | 2× model |
| `RewardTrainer` `[HF/TRL]` | Trains a scalar-output classifier on preference pairs. Needed only as a step before PPO. | `(prompt, chosen, rejected)` | 1× model |
| `KTOTrainer`, `ORPOTrainer`, `RLOOTrainer` `[HF/TRL]` | Alternative preference algorithms with different data shapes. | varies | varies |

**Use TRL when** you're doing instruction tuning (SFT), preference tuning (DPO/KTO/ORPO), or full RLHF (reward model → PPO/GRPO). It also handles chat templates, completion-only loss, and the bookkeeping that's painful in vanilla `Trainer`.

**Don't use TRL for** pretraining — that's vanilla `Trainer` `[HF]` with `DataCollatorForLanguageModeling` `[HF]`.

##### PPO and the RLHF setup — what each piece does

In SFT, you show the model "input → correct output" and it imitates. But for chat alignment there's often **no single correct output** — many responses are valid, just some better than others. RL teaches the model differently:

> Generate a response → get a *score* → nudge the weights so high-scoring patterns become more likely.

That's it. RL is "trial and error guided by a score." PPO and GRPO are bookkeeping to make this stable.

**The four models PPO juggles** (analogy: a writer being coached):

| Name | Plain meaning | Analogy |
|---|---|---|
| **Policy** | The LLM you're training. Produces tokens. | The writer being coached |
| **Reference (ref)** | Frozen snapshot of the LLM at the start of RL. Never updated. | The writer's "old self" — keeps the new writing recognisable |
| **Reward model** | Separate model trained on human preferences. `(prompt, response) → scalar quality score`. | The editor who grades the writing |
| **Value model** | Predicts *expected future reward* from a partial response. | A coach whispering "going well, expected ~7/10" |

All four are transformer-based. Three are LLM-shaped (policy, ref, reward); the value model usually shares a backbone with the reward model.

**Two side concepts you need first:**

- **Value function `V(state)`** — "given where I am right now, what total reward do I expect by the end?" Like a weather forecast: morning sky (state) → predicted rain by evening (future reward). For LLMs the "state" is the partial response generated so far.
- **Regression head** — the last layer of a transformer. The normal LM head outputs one number per vocab token (~32k logits → next-token distribution). A regression head is just `Linear(d_model, 1)` — outputs **one number total**. Reward and value models both use a regression head because they need to score, not generate.

**How PPO trains, step by step:**

```text
Setup:
  policy  ← copy of SFT model         (trainable)
  ref     ← frozen copy of SFT model  (never changes)
  reward  ← pre-trained reward model  (frozen)
  value   ← initialised from reward   (trainable)

Loop:
  1. Sample a prompt:               "Write a poem about cats"
  2. Policy generates a response:   "Whiskers drift..."   (token by token)

  3. For each token position t in the response:
       Ask value: "from this state, what reward do you expect?" → V_t

  4. After response is complete, reward model scores it:  r = 8.4

  5. Compute ADVANTAGE per step:
       A_t = (actual outcome r) − (predicted V_t)

       A_t > 0 → BETTER than expected → reinforce this token
       A_t < 0 → worse than expected  → discourage this token

  6. Compute KL PENALTY per token:
       penalty_t = β · KL(policy(t) || ref(t))
       "How different is the new policy from the original here?"
       Keeps policy from drifting away from coherent English.

  7. Update POLICY:
       - Increase probability of high-advantage tokens
       - Decrease probability of low-advantage tokens
       - CLIP the update — change policy by at most ε in one step
         (the "Proximal" in PPO — stay close to previous policy)

  8. Update VALUE:
       - Train V to predict actual rewards better (regression loss)

  9. Repeat.
```

**Why every piece exists** (remove one and training breaks in a specific way):

| Piece | If you remove it… |
|---|---|
| **Policy** | Nothing to train. |
| **Ref** | Policy drifts to gibberish that games the reward (`"the the the"` or empty flattery). KL leash gone. |
| **Reward** | No score → no signal → RL has nothing to optimise. |
| **Value** | Variance explodes. Raw rewards swing wildly across prompts; without a baseline, gradients are noisy and training is unstable. |
| **Clipping** | One update can change the policy massively → catastrophic forgetting. This is why pre-PPO methods (vanilla policy gradient) failed for LLMs. |

**The output you actually ship:** after PPO converges, the **policy** is your aligned model. Ref / reward / value are training-time scaffolding — throw them away when done. PPO is a 4-model dance *during training* producing a single fine-tuned LLM *at the end*.

##### GRPO in detail — why "drops the value head" matters

PPO needs a **value function** ("critic") — a separately trained model that predicts expected reward from any state. That's an extra model the size of your policy, with a regression head, eating GPU memory and training time. GRPO replaces it with a much simpler trick:

```text
1. Take one prompt:        "What is 12 × 13?"
2. Sample G completions:   G = 4–16
   ["156", "It's 156", "Hmm, 12*13=156", "144", "I think 156", …]
3. Score each:             r = [1.0, 1.0, 1.0, 0.0, 1.0, …]
4. Group-relative advantage:
       advantage_i = (r_i − mean(r)) / std(r)
5. PPO-style clipped policy update using these advantages.
```

The **group mean** acts as the baseline that the value function would normally provide. The **group std** normalises the advantage scale.

| | PPO | GRPO |
|---|---|---|
| Models in memory | policy, ref, reward, **value** | policy, ref, reward |
| Baseline | learned `V(s)` | group mean reward |
| Generations per prompt | 1 | G (typically 4–16) |
| Best for | open-ended (chat) | verifiable rewards (math, code, RLVR) |

DeepSeek-R1 paired GRPO with **rule-based rewards** (correctness + format compliance) and got chain-of-thought reasoning to emerge from a base model with no SFT. "Drops the value head" just means: no critic model, ~25–50% memory saved.

#### Memory accounting during training — bytes per parameter

The biggest memory cost during training is rarely the weights themselves — it's the **optimizer state** plus stored activations. This is the foundation that makes the LoRA story below click.

##### Adam mechanics — what `m` and `v` are

Adam (and AdamW) is a per-parameter adaptive optimizer. For every trainable parameter `θ`:

```text
g_t = ∇L(θ)                            # current gradient (from backprop)
m_t = β₁·m_{t-1} + (1-β₁)·g_t          # 1st moment — running mean of gradients
v_t = β₂·v_{t-1} + (1-β₂)·g_t²         # 2nd moment — running mean of squared gradients
θ_t = θ_{t-1} − lr · m_t / (√v_t + ε)  # update with adaptive step size
```

In plain English:

- **`m`** smooths the gradient direction over time — like momentum. If a parameter's gradient has been pointing the same way for many steps, `m` reflects that consistent direction so the update doesn't get knocked around by noise.
- **`v`** tracks how *volatile* a parameter's gradient has been. Dividing by `√v` means parameters with large noisy gradients take small steps, while parameters with consistently small gradients take larger ones. This is what makes Adam *adaptive*.

Both `m` and `v` are tensors **shaped exactly like the parameter** — for `c_q.weight` of shape `(C, C)`, `m` and `v` are also `(C, C)`. They persist across training steps because they're running averages.

Gradients (`.grad`) are a separate, *transient* memory cost — overwritten every backward pass. Optimizer state is what *survives across steps* and dominates persistent memory.

##### Bytes per trainable parameter (full mixed-precision Adam)

| What | Bytes (typical mixed-precision) |
|---|---|
| Weight (fp16/bf16 copy for forward + backward) | 2 |
| Weight (fp32 master copy, for optimizer step) | 4 |
| Gradient (fp32) | 4 |
| Adam `m` (fp32) | 4 |
| Adam `v` (fp32) | 4 |
| **Total** | **18 bytes / param** |

So a 7B-param model trained with full Adam needs ~126 GB just for weights + grads + optimizer state. A single A100-80GB can't fit that, even though the weights alone in fp16 are only 14 GB. **Optimizer state, not weights, is the bottleneck.**

For comparison:

- **Plain SGD:** weight + grad = 8 bytes/param. No `m`, no `v`.
- **SGD with momentum:** weight + grad + `m` = 12 bytes/param.
- **Adam:** 18 bytes/param.
- **Adam-8bit (`bitsandbytes`):** stores `m`, `v` in 8-bit → ~10 bytes/param.

##### Activation memory — separate from optimizer state

During the forward pass, intermediate tensors (input, Q, K, V, attention scores, MLP intermediates) get **kept alive** until the backward pass uses them to compute gradients. Rough estimate per transformer layer:

```text
~10–15 × B × T × C × bytes_per_activation
```

For B=4, T=2048, C=4096, fp16:
- Per layer: `~12 × 4 × 2048 × 4096 × 2 ≈ 800 MB`
- Across 32 layers (LLaMA-7B): `~25 GB` of activations during training.

Two techniques reduce this:

- **Gradient checkpointing** — store activations only every N layers, recompute the rest during backward. Trade compute for memory.
- **FlashAttention** — never materialise the `(T, T)` attention matrix; compute attention in tiles. Saves the quadratic-in-T term that dominates at long context.

##### Inference is a different game

No backward pass → no activations stored. No optimizer → no `m`, `v`. The dominant non-weight cost is the **KV cache**: K and V activations at every layer, for every cached token.

```text
KV cache size = 2 × L × B × T × C × bytes_per_act
```

For LLaMA-7B (`L=32`, `C=4096`) at B=1, T=2048, fp16: `2 × 32 × 2048 × 4096 × 2 ≈ 1 GB`. Scales linearly with `T` — at T=128K context that's 64 GB just for K and V, dwarfing the 14 GB of weights. This is why long-context serving becomes KV-cache-bound, not weight-bound.

##### Why LoRA dominates this picture

The frozen base weights need only the weight column (2 or 4 bytes), no gradient, no `m`, no `v`. Only the LoRA adapters carry the 18-bytes-per-param cost.

For a 7B model with LoRA rank 8 over `q_proj` + `v_proj`:

- Trainable params ≈ 8M
- Trainable budget: `8M × 18 = 144 MB`
- Frozen base: `7B × 2 = 14 GB` (fp16)
- **Total: ~14 GB instead of ~126 GB**

That ~9× reduction is the LoRA win. **QLoRA** quantises the frozen base further (4-bit NF4 → ~4 GB for the base) — total drops to ~5 GB, fitting comfortably on a consumer GPU.

#### PEFT — Parameter-Efficient Fine-Tuning

Fine-tunes a model by **updating only a tiny fraction of parameters** (typically 0.1–1%) while the base weights stay frozen. Started as a memory hack; now the default for almost all fine-tuning.

Methods it ships:

| Method | Idea | Trainable params | Notes |
|---|---|---|---|
| **LoRA** | Freeze `W`. Inject a low-rank update: `W'x = Wx + BAx` where `B ∈ ℝ^{d×r}`, `A ∈ ℝ^{r×d}`, `r` ≪ `d`. Train only `A` and `B`. | `r × (d_in + d_out)` per target layer — usually <1% of total | The default. Apply to `q_proj`, `v_proj` first; add `k_proj`/`o_proj`/MLP for harder tasks. |
| **QLoRA** | LoRA on top of a 4-bit-quantized base model (NF4 + double quant). | Same as LoRA | Lets you fine-tune a 70B model on a single 24GB GPU. The standard recipe. |
| **DoRA** | Decompose `W` into magnitude + direction; apply LoRA only on direction. | ~LoRA | Often beats LoRA at the same rank. |
| **AdaLoRA** | LoRA with adaptive rank per layer. | ~LoRA | Useful when you don't know which layers matter. |
| **IA³** | Multiplicative adapters — even smaller than LoRA. | ~0.01% | Very lightweight; less expressive. |
| **Prefix / Prompt tuning** | Trainable virtual tokens prepended to the input. | Tiny | Largely superseded by LoRA. |

**Why LoRA actually works (intuition):** the *change* in weights during fine-tuning has empirically low intrinsic rank. You don't need to update a full `d × d` matrix to teach a new task — a rank-8 or rank-16 perturbation is usually enough.

##### What `A` and `B` actually are

`W` is the original weight of some linear layer (e.g. `q_proj`, shape `(d, d)`). LoRA freezes `W` and adds a parallel path with two small matrices:

```text
       d                            d                       d
   ┌───────┐                   ┌─────────┐            ┌─────────┐
   │       │                 d │         │ ×        r │         │
 d │   W   │ ←frozen           │    B    │            │    A    │
   │       │                   │  (d,r)  │            │  (r,d)  │
   └───────┘                   └─────────┘            └─────────┘
                               init: zeros            init: random Gaussian

   forward:  output = W·x + (alpha/r) · B·(A·x)
```

- **`A` is `(r, d)`**, initialised **random Gaussian** (small, Kaiming-style).
- **`B` is `(d, r)`**, initialised **all zeros**.
- `r ≪ d`. Common: `r = 8, 16, 32`.

The init choices matter:
- `B = 0` at step 0 → `BA = 0` → the model behaves *exactly* like the frozen base. No surprise gradient kick at the start.
- `A` random Gaussian → gradients can flow into both `A` and `B` from the first backward pass. (If both were zero the product would be zero with zero gradient — dead path.)

After training:
- **Keep adapters separate** — swap multiple task adapters on the same base model. Each adapter is MB-sized.
- **Merge into base** — `W_merged = W + (alpha/r)·BA`. No inference overhead, but you lose swap-ability.

##### Rank `r` — adapter capacity

`A` is `(r, d)`, `B` is `(d, r)`. The product `BA` is at most rank `r`. So `r` is the dimension of the bottleneck the LoRA update has to squeeze through.

| `r` | When |
|---|---|
| 4 – 8 | Very simple tuning (style, format) |
| **16 – 32** | **Default for instruction tuning** |
| 64 – 128 | Big domain shift, code/math specialisation |
| 256+ | Rare; usually wastes memory |

Doubling `r` past 32 gives diminishing returns — the intrinsic rank of fine-tuning updates is genuinely low. The original LoRA paper showed `r=2` already captured most of the signal on GLUE tasks.

Memory: each LoRA layer adds `r × (d_in + d_out)` params. For LLaMA-7B with all attention layers wrapped, `r=16` adds ~4M params; `r=64` adds ~16M.

##### Where the dimensions come from

`d` in `r × (d_in + d_out)` is the model's **hidden dim `C`** — the size of every token's residual stream vector. For attention projections, `d_in = d_out = C`. Common values:

| Model | Hidden dim `C` |
|---|---|
| nanochat default / GPT-2 small | 768 |
| GPT-2 medium | 1024 |
| LLaMA-7B / LLaMA-2-7B | 4096 |
| LLaMA-13B | 5120 |
| LLaMA-65B / LLaMA-3-70B | 8192 |

So `q_proj` in LLaMA-7B is a `(4096, 4096)` matrix → 16.8M params. In nanochat (`C=768`) it's `(768, 768)` → 590K params.

`r` is a hyperparameter you choose — not derived from architecture. The LoRA paper (Hu et al., 2021) showed that very low ranks (1, 2, 4, 8) work surprisingly well even for full-rank weight matrices, because the *update* `ΔW` lies in a low-dimensional subspace. Most practitioners default to `r=8` or `r=16`; specialised tasks (code, math, large domain shift) may benefit from `r=64`. The original paper's sweet spot was `r=8`.

##### Alpha `lora_alpha` — adapter strength

The forward pass is:

```text
output = W·x + (alpha / r) · B·A·x
                ─────────────
                scaling factor
```

`alpha` multiplies the LoRA delta. Big alpha → adapter contributes more to the output relative to the base; small alpha → barely perturbs the base.

**Why divide by `r`:** without it, doubling `r` would also roughly double the magnitude of `BA`, forcing you to retune the LR. Scaling by `alpha/r` decouples capacity from magnitude — change `r` to tune capacity, the effective scale stays the same.

**Pin a ratio.** Most recipes use `alpha = 2r` (effective scale 2) or `alpha = r` (effective scale 1). Then `r` becomes a clean knob.

| `r` | `alpha` | `alpha/r` | Notes |
|---|---|---|---|
| 8 | 16 | 2.0 | "alpha = 2r" recipe |
| 16 | 32 | 2.0 | Same scale, more capacity |
| 32 | 64 | 2.0 | Same scale, more capacity |
| 8 | 32 | 4.0 | Stronger updates per step (used to "punch through" stubborn base models) |

**Common pitfall:** setting `alpha = 32`, `r = 8` (effective scale 4), then doubling `r` to 16 hoping for "more LoRA." But `alpha/r` halves to 2 — the model now trains *less* aggressively per step. Fix: pin the ratio.

**RSLoRA variant** uses `alpha / √r` instead. Helps when `r` is high (standard scaling decays too aggressively at large `r`).

##### The other LoRA hyperparameters

- `target_modules`: which linear layers to wrap. Start with `["q_proj", "v_proj"]`. For instruction tuning and DPO, modern recipes wrap *all* linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, plus the three MLP projections) for max quality. See [hf_02 attention naming](02_model_and_config.md#attention-layer-naming--what-q_proj-k_proj-v_proj-o_proj-actually-are).
- `lora_dropout`: 0.05–0.1 typical.

**Use PEFT when** GPU memory is the constraint (almost always), when you want to ship many task-specific adapters as MB-sized files, or when you want to avoid catastrophic forgetting on the base model.

**Don't use PEFT when** you have unlimited compute and need the absolute best quality — full fine-tuning still has a small edge on the hardest tasks. Almost no one is in that regime.

#### The standard recipe — TRL + PEFT together

`SFTTrainer` and `DPOTrainer` both take a `peft_config` directly. No glue code.

```python
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
model = AutoModelForCausalLM.from_pretrained("base-model", quantization_config=bnb)

lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(...),
    train_dataset=dataset,
    peft_config=lora,        # PEFT plugs straight into TRL
)
trainer.train()
```

This is the 2025 default for aligning a model: QLoRA quantises the base, LoRA adapters carry the new behaviour, `SFTTrainer` (or `DPOTrainer`) runs the loop.

#### `bitsandbytes` and `BitsAndBytesConfig`

**`bitsandbytes`** is a CUDA library by Tim Dettmers (the QLoRA author) that provides quantization kernels and 8-bit optimizers:

- **8-bit weight quantization** — half the memory of fp16 for inference.
- **4-bit weight quantization** — quarter memory; this is what QLoRA uses.
- **NF4** (NormalFloat 4-bit) — a quantization scheme designed to be information-theoretically optimal for normally distributed weights (which trained transformer weights tend to be).
- **Double quantization** — quantize the quantization constants themselves; saves another ~0.4 bit per parameter on average.
- **8-bit optimizers** (`Adam8bit`, `AdamW8bit`) — store Adam moments in 8-bit instead of 32-bit, cutting optimizer memory by 4×.

**`BitsAndBytesConfig`** `[HF]` is the wrapper that lets you pass quantization options to `from_pretrained`:

```python
from transformers import BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # vs "fp4"; nf4 is better
    bnb_4bit_use_double_quant=True,       # extra memory saving
    bnb_4bit_compute_dtype=torch.bfloat16 # matmuls done in bf16
)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-70B", quantization_config=bnb)
```

What happens at load:
1. Weights read from disk (fp16 / bf16).
2. Quantized to NF4 on the fly.
3. Stored in GPU memory in 4-bit (~17 GB for a 70B model).
4. **Forward pass:** for each linear layer, dequantize a block back to bf16, do the matmul in bf16, discard the bf16 block. **Compute stays high-precision; only storage is low-precision.**

Memory math (70B model):

| Format | Weights memory | Fits on … |
|---|---|---|
| fp32 | 280 GB | nothing typical |
| fp16 / bf16 | 140 GB | 2× A100-80GB |
| int8 | 70 GB | 1× A100-80GB |
| NF4 | 35 GB | 1× A100-40GB |
| NF4 + double quant | ~33 GB | 1× A100-40GB with room for adapters |

**Why this forces LoRA on top:** with 4-bit base weights you cannot directly fine-tune them — there's no precision left to absorb gradient updates. QLoRA pairs 4-bit base + LoRA adapters in bf16: the *adapters* carry the gradient, the *base* just provides the frozen forward signal.

#### TRL and PEFT are orthogonal axes

It's tempting to think "TRL is fine-tuning, PEFT is one method of fine-tuning, so PEFT is a kind of TRL." That's *almost* right but conflates two independent choices:

- **TRL = the training *objective*** — what loss are you optimising? Next-token (vanilla), SFT, DPO, PPO, GRPO, KTO, ORPO.
- **PEFT = the *parameterization*** — which parameters are actually trainable? Full model, LoRA adapters, QLoRA, IA³, prefix tokens.

Any cell in this 2×N grid is a valid recipe:

```text
                          OBJECTIVE (TRL or vanilla)
                ┌─────────────┬──────────────┬─────────────┬─────────────┐
                │ next-token  │     SFT      │     DPO     │     GRPO    │
                │  (vanilla   │   (TRL)      │   (TRL)     │    (TRL)    │
                │   Trainer)  │              │             │             │
   ─────────────┼─────────────┼──────────────┼─────────────┼─────────────┤
   Full FT      │ pretraining │ rare:        │ rare:       │ rare:       │
                │             │ huge GPU     │ huge GPU    │ huge GPU    │
   ─────────────┼─────────────┼──────────────┼─────────────┼─────────────┤
   LoRA (PEFT)  │ classifier  │ standard     │ standard    │ standard    │
                │ heads etc.  │ recipe       │ recipe      │ recipe      │
   ─────────────┼─────────────┼──────────────┼─────────────┼─────────────┤
   QLoRA (PEFT) │ rare        │ standard for │ standard    │ standard    │
                │             │ big models   │ for 70B+    │ for 70B+    │
                └─────────────┴──────────────┴─────────────┴─────────────┘
```

`peft_config=lora` is a parameter on `SFTTrainer` and `DPOTrainer` because TRL chose to integrate PEFT as a first-class wrapper — but you could equally use PEFT with vanilla `Trainer`, or use TRL with full fine-tuning. They just *combine* well, which is why the QLoRA + SFT/DPO pairing is everywhere.

#### Decision tree — when to reach for what

```text
Pretraining from scratch?
├── Yes → vanilla Trainer + DataCollatorForLanguageModeling. No TRL, no PEFT.
└── No → fine-tuning some base model
    │
    ├── Task-specific (classification, QA, NER)?
    │   → vanilla Trainer + LoRA if memory-tight.
    │
    ├── Instruction / chat fine-tuning (base → chat model)?
    │   → SFTTrainer (TRL) + LoRA/QLoRA (PEFT).
    │
    ├── Preference data available (chosen vs rejected)?
    │   → DPOTrainer (TRL) + LoRA/QLoRA (PEFT).
    │     Usually: SFT first, then DPO on top.
    │
    └── Full RLHF with a reward model?
        → RewardTrainer → PPOTrainer / GRPOTrainer (TRL) + PEFT for memory.
```

---

### Related sections

- Pipeline overview → [00_overview.md](00_overview.md)
- Model and config → [02_model_and_config.md](02_model_and_config.md)
- Evaluation and generation → [04_evaluation_and_generation.md](04_evaluation_and_generation.md)
- nanochat counterpart → [07_training_and_generation.md](../07_training_and_generation.md)
