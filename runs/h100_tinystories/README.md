# H100 training run on TinyStories

Your first real training run. A 28M-param `NanoChatModel` trained on the
TinyStories corpus on a single Lambda Labs H100 — about 30 minutes of
compute, ~$2 of spend.

This README walks through every step. You've never done this before; it
assumes nothing.

---

## What this run produces

| | Value |
|---|---|
| Hardware | 1× H100 80 GB SXM5 (Lambda Labs) |
| Pricing | $2.99/hr on-demand |
| Wall-clock training | ~30 min |
| Total session (provision + data + train + shutdown) | ~1 hr |
| Total cost | ~$2-3 |
| Output | A 28M-param model with val perplexity ~5, generates coherent short stories |

For comparison: the same model architecture trained on your laptop for
500 steps on WikiText-2 hits perplexity ~190 and generates gibberish.
Switching corpus + 30× more steps + 50× faster GPU = readable stories.

---

## Files in this folder

```
runs/h100_tinystories/
├── README.md           ← this file
├── prepare_data.py     ← runs once: download TinyStories + train BPE
├── train.py            ← the H100 training script
├── data_cache/         ← created by prepare_data.py (gitignored)
└── checkpoints/        ← created by train.py (gitignored)
```

`data_cache/` and `checkpoints/` will appear after you run the scripts.
The whole `runs/` tree is already in the repo's top-level `.gitignore`.

---

## Step-by-step

### 1. Sign up at Lambda Labs

Go to https://lambdalabs.com → **Sign up** → add a payment method. Without
a card on file you can't launch instances.

You also need an SSH key uploaded under **Account → SSH Keys**. If you
don't have one on this machine:

```bash
ssh-keygen -t ed25519 -C "lambda-labs"
# accept the default path; press enter twice to skip the passphrase
cat ~/.ssh/id_ed25519.pub
```

Copy that public key into Lambda's web UI.

### 2. Push your repo to GitHub

The Lambda VM needs to pull your code from somewhere. The simplest path
is GitHub.

If your repo isn't pushed yet:

```bash
# from this repo's root, on your laptop
gh repo create learn-you-an-hf-llm --private --source=. --push
```

If you keep the repo private, on the VM you'll need either a Personal
Access Token or to upload an SSH key for your GitHub account.

### 3. Provision a 1× H100 instance

In the Lambda dashboard → **Launch Instance** → pick **"1× H100 (80 GB
SXM5)"**. Choose any region listed as "Available" — H100 inventory shifts
hourly; pick whatever you can get.

Click **Launch**. Provisioning takes 1-3 minutes. When it's ready, the
dashboard shows a public IPv4 address — copy that.

### 4. SSH in

```bash
ssh ubuntu@<public-ip>
```

If SSH refuses with "Permission denied (publickey)", your local `~/.ssh/`
key isn't the one you uploaded to Lambda. Either upload the right
public key in step 1 or specify the right private key explicitly:

```bash
ssh -i ~/.ssh/id_ed25519 ubuntu@<public-ip>
```

The Lambda image already has CUDA 12.x, PyTorch, and the standard ML
stack pre-installed. You won't need to install drivers.

### 5. Pull your code and install

```bash
git clone https://github.com/<your-username>/learn-you-an-hf-llm.git
cd learn-you-an-hf-llm
pip install -e .
```

If the repo is private, GitHub will prompt for a username + token. Create
a token at https://github.com/settings/tokens (classic, scope `repo`) and
paste it as the password.

### 6. Verify GPU + PyTorch see each other

```bash
nvidia-smi
```

Should show one H100, ~80 GB free, CUDA 12.x. Then:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect: `True NVIDIA H100 80GB HBM3`.

If it prints `False`, the PyTorch on the VM doesn't have CUDA support
(rare on the Lambda image but possible). Reinstall with:

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
```

### 7. Run data preparation (one-time, ~5 min)

```bash
cd runs/h100_tinystories
python prepare_data.py
```

This downloads TinyStories from Hugging Face Hub (~1 GB), concatenates
non-empty stories into `data_cache/train.txt` and `data_cache/val.txt`,
and trains a BPE tokenizer saved to `data_cache/tokenizer.json`. You'll
see something like:

```
Train: 1,840,123,456 chars
Val:   18,456,789 chars
Vocab size:       4096
Sample text:      "Once upon a time, there was a little girl named Lily."
Encoded tokens:   ['Once', 'Ġupon', 'Ġa', 'Ġtime', ',', ...]
```

### 8. Run training (~30 min)

**Always wrap long-running training in `tmux`.** A network blip will kill
your SSH session, and without tmux that kills the training too — you'd
lose 30 minutes of compute.

```bash
tmux new -s train
python train.py | tee train.log
```

You'll see something like (numbers are approximate):

```
Tokenizing corpus...
Train tokens: 470,123,456  |  Val tokens: 4,891,234
Parameters: 28,341,248, Device: cuda
Compiling model (first step will be slow due to graph tracing)...
Step :     0/15000, Train: 8.32, val: 8.31, ppl: 4054.3, lr: 0.00e+00, 4,200 tok/s
Step :   500/15000, Train: 4.89, val: 4.92, ppl: 137.0, lr: 3.00e-04, 76,000 tok/s
Step :  1000/15000, Train: 3.71, val: 3.78, ppl: 43.8, lr: 2.99e-04, 78,000 tok/s
...
Step : 14999/15000, Train: 1.62, val: 1.71, ppl: 5.5, lr: 0.00e+00, 79,000 tok/s
Training completed in : 1834.21s
Final valuation loss : 1.7100, Perplexity: 5.5300
Model saved at /home/ubuntu/learn-you-an-hf-llm/runs/h100_tinystories/checkpoints/final
```

**Detach** from the tmux session with `Ctrl-b` then `d`. The training
keeps running. You can close your SSH connection — training survives.

**Reattach later** with `tmux attach -t train`.

To watch the loss curve from outside tmux:

```bash
tail -f runs/h100_tinystories/train.log
```

### 9. Pull the trained model back to your laptop

When training prints `Final model saved`, pull the checkpoint folder
back over scp:

```bash
# from your laptop, in the repo root
scp -r ubuntu@<public-ip>:~/learn-you-an-hf-llm/runs/h100_tinystories/checkpoints \
       runs/h100_tinystories/checkpoints
```

Verify before terminating:

```bash
ls runs/h100_tinystories/checkpoints/final/
# expect: config.json  model.safetensors
```

Also pull the tokenizer if you want to run inference locally:

```bash
scp -r ubuntu@<public-ip>:~/learn-you-an-hf-llm/runs/h100_tinystories/data_cache/tokenizer.json \
       runs/h100_tinystories/data_cache/tokenizer.json
```

### 10. Terminate the instance — most important step

Lambda dashboard → **Instances** → select yours → **Terminate**.

Instances bill by the second whether you're using them or not. Forgetting
to terminate is the single most expensive mistake. Set a calendar reminder.

```bash
# verify on the dashboard that the instance is gone — DO NOT trust your memory
```

### 11. Run inference locally

Back on your laptop, you can use your local
`copywork/04_evaluation_and_generation/copywork_eval.py` once you point
it at the new artifacts:

```python
# in copywork_eval.py main(), replace:
tokenizer = Tokenizer.from_file(str(CACHE_DIR / "tokenizer.json"))
model = NanoChatModel.from_pretrained(CACHE_DIR / "model_trained").to(DEVICE)

# with:
RUN_DIR = Path("runs/h100_tinystories")
tokenizer = Tokenizer.from_file(str(RUN_DIR / "data_cache" / "tokenizer.json"))
model = NanoChatModel.from_pretrained(RUN_DIR / "checkpoints" / "final").to(DEVICE)
```

Pick prompts that match the corpus — TinyStories is short stories, so
`"Once upon a time"` will produce more coherent output than `"The president of"`.

---

## What changed in `train.py` vs your `copywork_train.py`

This is the diff in plain English, by category. None of the changes
affect the *shape* of the loop — it's still get_batch → forward → backward
→ clip → step → zero_grad. Everything else is dialed for H100 throughput.

### Model architecture (capacity)

| Knob | copywork | train.py | Why |
|---|---|---|---|
| `N_LAYERS` | 4 | **8** | More residual blocks → can model multi-clause sentences |
| `N_HEADS` | 4 | **8** | More attention heads at each layer |
| `N_EMBD` | 256 | **512** | Hidden dim → model capacity per token |
| `SEQ_LEN` | 256 | **512** | Longer context → can attend across paragraph |

Total params: ~3M → ~28M. This is the smallest scale that the TinyStories
paper showed produces *story-level* coherence.

### Training schedule

| Knob | copywork | train.py | Why |
|---|---|---|---|
| `BATCH_SIZE` | 8 | **64** | H100 has 80 GB; small batch wastes the GPU |
| `NUM_STEPS` | 500 | **15,000** | Need ~Chinchilla-scale compute (20 tokens/param) |
| `EVAL_EVERY` | 50 | **500** | Less eval overhead at longer runs |
| `EVAL_BATCHES` | 10 (hardcoded) | **20** | More batches → less noise in val loss |
| `WARMUP_STEPS` | — | **200** | Linear warmup avoids early-step gradient explosions |
| `WEIGHT_DECAY` | — | **0.1** | Standard transformer regularisation |
| `SAVE_EVERY` | — | **2,500** | Periodic checkpoints — survive instance preemption |

### Optimizer

- `Adam` → **`AdamW`** with `weight_decay=0.1`. AdamW decouples weight
  decay from the gradient update; it's the modern default for transformers.
  Adam's `weight_decay=0` produced the same trajectory but offered no
  regularisation.
- Added `LambdaLR` scheduler running `cosine_with_warmup` — linear ramp
  from 0 → LR over 200 steps, then cosine decay to 0 over the remaining
  14,800 steps. Constant LR + 15K steps would oscillate.
- Added `scheduler.step()` inside the loop after `optimizer.step()`.

### Hardware acceleration

- `DEVICE` detection block (MPS / CUDA / CPU) → hardcoded `"cuda"`. This
  script only runs on H100; the fallbacks would mask configuration bugs.
- Wrapped the forward pass in `torch.autocast(device_type="cuda",
  dtype=torch.bfloat16)` — half-precision matmul, fp32 master weights.
  Roughly 2× throughput on H100 with negligible accuracy loss. bf16
  (not fp16) means no `GradScaler` needed; range is wide enough to
  rarely overflow.
- Added `model = torch.compile(model)` after `.to(DEVICE)`. PyTorch's
  graph compiler fuses kernels and skips overhead. First step takes
  ~60 s for tracing; subsequent steps drop to ~25 ms. ~1.5-2× free.
- `optimizer.zero_grad()` → `optimizer.zero_grad(set_to_none=True)`.
  Avoids a useless zero-fill on every step.
- `.to(device)` → `.to(device, non_blocking=True)` in `get_batch`. CPU→GPU
  copies overlap with compute when async.

### Bug fix from your existing code

- Line 75-76 of `copywork_train.py`:
  ```python
  loss = model(vx, labels=vy)
  losses.append(loss.item())   # AttributeError: CausalLMOutput has no .item()
  ```
  Fixed in `train.py`:
  ```python
  out_v = model(vx, labels=vy)
  losses.append(out_v.loss.item())
  ```
  This was bug #3 from the earlier review of your copywork file. It would
  have crashed at the first eval step. You'll want to fix it in your
  copywork file too.

### Paths

- `CACHE_DIR = Path(os.path.expanduser('~/.cache/hf_pipeline'))` →
  `RUN_DIR / "data_cache"` and `RUN_DIR / "checkpoints"`. Run-local paths
  isolate this run from your WikiText copywork cache. No risk of mixing
  TinyStories tokenizer with WikiText weights.
- Removed `import os` — `os.path.expanduser` no longer needed.

### Checkpointing

- Added `save_checkpoint()` helper that handles `torch.compile`'s
  `_orig_mod` wrapper. `model.save_pretrained()` directly on a compiled
  model would save the compiler wrapper, not the weights.
- Added periodic save every 2,500 steps to `checkpoints/step_NNNNN/`.
- Final save to `checkpoints/final/` at end of training.

### Logging

- Added `lr` and `ppl` to the eval print line so you can see warmup +
  cosine working in real time.

---

## Troubleshooting

**"CUDA out of memory" on first step.** Drop `BATCH_SIZE` from 64 to 32
in `train.py`. The 80 GB H100 should fit 64 easily but `torch.compile`'s
first-step memory spike is unpredictable — sometimes it spills.

**Step 0 takes 60+ seconds.** Expected. That's `torch.compile` tracing
the graph. Subsequent steps drop to ~25 ms.

**Loss is NaN after a few hundred steps.** Lower `LR` from `3e-4` to
`1e-4`, or remove the `torch.autocast` block to fall back to fp32. bf16
rarely NaNs but it can on bad init.

**Network died and tmux session is gone.** Reconnect SSH, then
`tmux ls` to find detached sessions. If the instance survived but tmux
didn't, your training is gone unless you have a recent checkpoint in
`./checkpoints/`. Always wrap in tmux *before* starting.

**Forgot to terminate and racked up $$.** Lambda doesn't refund. Always
verify on the dashboard before logging off, every session. Set a calendar
reminder for one hour from launch.

**`pip install -e .` fails on the VM.** The Lambda image's pip might be
old. Try `pip install --upgrade pip` first, then retry.

---

## What this run gives you

After ~30 min you should have:

- A 28M-param model with val loss ~1.7, perplexity ~5
- Generation samples that look like coherent short stories — full
  sentences, simple plots, sometimes a moral. Compare against your
  laptop's WikiText run (val loss ~5, perplexity ~190): same architecture
  family, very different output, because the corpus + compute scale matter
  far more than tweaking model design at this size.

## What comes next

Once this works, the natural next step (~$15-20, ~6 hr):

| Knob | This run | "Real-world knowledge" upgrade |
|---|---|---|
| Corpus | TinyStories (~470M tok, simple vocab) | FineWeb-edu (~10B tok, real text) |
| Params | 28M | 120M (n_layer=12, n_embd=768) |
| Tokens trained | ~250M | ~2B (~Chinchilla-optimal at 120M) |
| Steps | 15K | ~30K |
| Time on 1× H100 | 30 min | ~6 hr |
| Cost | ~$2 | ~$18 |

You'd reuse this `train.py` with bumped knobs and a different
`prepare_data.py`. Don't run the upgrade until this one prints stories.
