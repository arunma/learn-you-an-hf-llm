# GCP A100 training run

Same `prepare_data.py` and `train.py` as the Lambda walkthrough — different
provisioning path. This guide replaces steps 1–5 of `README.md` for GCP.
Steps 6–11 (verify GPU → run → scp back → stop instance → run inference
locally) are provider-agnostic; follow them from `README.md`.

---

## What this run produces

| | Value |
|---|---|
| Hardware | 1× A100 40GB (`a2-highgpu-1g`) |
| Region | `asia-southeast1` (the region with your existing A100 headroom) |
| Pricing | $3.85/hr on-demand, ~$1.20/hr spot |
| Wall-clock training | ~60–90 min |
| Total cost | ~$4–6 on-demand, ~$1–2 spot |
| With your GCP credits | effectively free |

A100 is roughly half the throughput of H100 for this workload. Same model,
same script, just longer wall-clock.

---

## Step-by-step

### 1. Verify training quota (NOT serving quota)

The "Custom model serving Nvidia A100 GPUs" quota you saw is for **Vertex
AI inference deployment**, not Compute Engine training. They're separate.

Check Compute Engine training quota:

```bash
gcloud compute regions describe asia-southeast1 \
  --format="table(quotas[].metric, quotas[].limit)" \
  | grep -i a100
```

You're looking for `NVIDIA_A100_GPUS` with a non-zero limit. If it shows
`0`, request the quota:

GCP Console → **IAM & Admin → Quotas** → filter `NVIDIA A100` →
find row `compute.googleapis.com/nvidia_a100_gpus` in `asia-southeast1`
→ **Edit Quotas** → request `1` → submit.

Justification for the form: `Single-instance training experiment, ~5 GPU-hours`.
Approval is usually within 1–3 days for small asks.

If you can't wait, fall back to L4 instead — see "Smaller GPU option" below.

### 2. Install + configure gcloud locally

```bash
# macOS
brew install --cask google-cloud-sdk

# initial setup
gcloud init
gcloud auth login
gcloud auth application-default login
```

Pick your project when prompted. Note the project ID.

### 3. Set CLI defaults so commands stay short

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud config set compute/region asia-southeast1
gcloud config set compute/zone asia-southeast1-a
```

If `asia-southeast1-a` doesn't have A100 capacity at provision time, try
`-b` and `-c` zones; they share the regional quota.

### 4. Provision the instance (spot pricing)

```bash
gcloud compute instances create nanochat-train \
  --machine-type=a2-highgpu-1g \
  --image-family=common-cu124-debian-11 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=DELETE \
  --metadata="install-nvidia-driver=True"
```

What each flag does:

| Flag | Why |
|---|---|
| `a2-highgpu-1g` | 1× A100 40GB + 12 vCPU + 85GB RAM |
| `common-cu124-debian-11` | GCP's Deep Learning VM image — PyTorch 2.x + CUDA 12.4 preinstalled |
| `boot-disk-size=200GB` | Enough for TinyStories (~2GB) + checkpoints (~150MB each × 6) |
| `pd-ssd` | Faster disk — matters for the initial dataset download |
| `--provisioning-model=SPOT` | ~70% off. Can be preempted; checkpoints every 2,500 steps protect you. |
| `--instance-termination-action=DELETE` | If preempted, instance is deleted (you stop paying for the disk too). For a one-shot experiment this is what you want. Use `STOP` if you need the disk to survive preemption. |
| `install-nvidia-driver=True` | Triggers the driver install on first boot |

Wait ~2–3 minutes for provisioning + driver install.

If spot capacity is exhausted in your zone, GCP will reject with a
"resources not available" error. Either retry in 5–10 min, try a different
zone, or drop `--provisioning-model=SPOT --instance-termination-action=DELETE`
to fall back to on-demand at $3.85/hr.

### 5. SSH in

```bash
gcloud compute ssh nanochat-train
```

gcloud handles SSH key setup automatically — no manual `ssh-keygen` needed.

The Deep Learning VM image takes ~30–60 s on first boot to finalize the
NVIDIA driver install. If `nvidia-smi` returns "command not found"
immediately, wait a minute and retry.

---

## Continue from `README.md` step 6

The remaining steps are identical regardless of provider:

- **Step 6** — verify GPU + PyTorch (`nvidia-smi`, `torch.cuda.is_available()`)
- **Step 7** — `python prepare_data.py` (~5 min)
- **Step 8** — `tmux new -s train; python train.py | tee train.log` (~60–90 min on A100)
- **Step 9** — scp checkpoints back to your laptop
- **Step 10** — **terminate the instance** (GCP equivalent below)
- **Step 11** — run inference locally

### GCP-specific equivalent of step 10 (TERMINATE)

In GCP, "terminate" = delete the instance. Stopping just powers it off but
keeps billing the persistent disk (~$0.04/GB/month).

```bash
# from your laptop, after scp-ing checkpoints back
gcloud compute instances delete nanochat-train
```

Confirm with `y`. Verify it's actually gone:

```bash
gcloud compute instances list | grep nanochat-train
# expect: no output
```

Don't trust your memory. Run that `list` command before logging off, every
session.

---

## Smaller GPU option — L4 (if A100 quota denied or you want minimum spend)

L4 is GCP's cheapest current-gen GPU with bf16 + Tensor Cores (Ada
Lovelace). 24GB VRAM, plenty for the 28M model.

Differences from A100 in step 4:

```bash
gcloud compute instances create nanochat-train \
  --machine-type=g2-standard-4 \
  --image-family=common-cu124-debian-11 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=DELETE \
  --metadata="install-nvidia-driver=True"
```

Just `g2-standard-4` instead of `a2-highgpu-1g`. The `g2` SKUs include
the L4 GPU; no separate accelerator flag needed (unlike `n1` which would
need `--accelerator=type=nvidia-l4,count=1`).

Trade-off:

| | A100 40GB | L4 24GB |
|---|---|---|
| Quota | 14 (existing serving quota — verify training quota separately) | needs separate request |
| Hourly (spot) | ~$1.20 | ~$0.30 |
| 15K-step train time | ~60–90 min | ~3–4 hr |
| `train.py` changes | none | none |
| Risk | low | network/preemption window is 4× longer |

L4 quota request is the same flow as A100 (Console → IAM → Quotas → filter
"L4"). Often approved instantly because L4 supply is plentiful.

---

## What if you want T4 or V100 (cheapest possible)

Don't, with the current `train.py`. T4 (Turing) and V100 (Volta) don't
support bf16 — the `torch.autocast(..., dtype=torch.bfloat16)` block in
`train.py` will throw or silently produce garbage.

To use them you'd:

1. Comment out the autocast blocks in `train.py` (forward pass and eval pass).
2. Either run in fp32 (slow, ~2× memory) or switch to fp16 + add a `GradScaler`.
3. Accept ~10+ hour training on T4, which is too long to reliably hold a
   spot instance.

For your first run, A100 spot is the right answer. Save T4/V100 economy
runs for a future iteration once `train.py` has fp16+GradScaler support.

---

## Troubleshooting (GCP-specific)

**`gcloud compute instances create` errors with "QUOTA_EXCEEDED"**.
Confirms training quota is 0. Either request quota (1–3 days) or fall
back to L4.

**"resources not available in zone"**. Spot capacity exhausted in that
zone. Try `asia-southeast1-b` or `-c`, or wait 5–10 min, or switch to
on-demand.

**`nvidia-smi` says "command not found" after SSH**. Driver still
installing. Wait 60 s and retry. If still missing after 3 min, the
metadata flag didn't take — re-run:

```bash
sudo /opt/deeplearning/install-driver.sh
```

**Spot instance preempted mid-training**. Your training is dead. Latest
checkpoint sits in `runs/h100_tinystories/checkpoints/step_NNNNN/` on
the (now-deleted) instance disk — gone too unless you used
`--instance-termination-action=STOP` instead of `DELETE`. For long runs
where preemption matters, switch to on-demand or write checkpoints to a
GCS bucket each save (not currently implemented in `train.py`; ~10 lines
to add if you want it).

**"Forgot to delete and racked up $$"**. GCP doesn't refund. Run
`gcloud compute instances list` at the end of every session. Set a
billing alert: Console → Billing → Budgets & alerts → create one at 50%
/ 80% / 100% of your credit balance.

---

## Cost expectation

| Run mode | A100 spot | A100 on-demand | L4 spot | L4 on-demand |
|---|---|---|---|---|
| 60 min | $1.20 | $3.85 | $0.30 | $0.71 |
| 90 min | $1.80 | $5.78 | n/a | n/a |
| 4 hours (L4) | n/a | n/a | $1.20 | $2.84 |

Add ~$0.20 for the boot disk (200GB × $0.04/GB/month × ~1 hour ≈ negligible
if you delete promptly). All within easy reach of even a small GCP credit
balance.
