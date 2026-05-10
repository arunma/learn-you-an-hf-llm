# RunPod training run

Same `prepare_data.py` and `train.py` as the Lambda walkthrough — different
provisioning path. This guide replaces steps 1–5 of `README.md` for RunPod.
Steps 6–11 (verify GPU → run → scp back → terminate → run inference
locally) are provider-agnostic; follow them from `README.md`.

RunPod has the lowest signup friction of the three providers and the
SSH/tmux workflow is essentially identical to Lambda.

---

## What this run produces

| | Value |
|---|---|
| Hardware | 1× H100 PCIe 80GB (Secure Cloud) |
| Pricing | ~$2.39/hr Secure Cloud, ~$1.99/hr Community Cloud |
| Wall-clock training | ~30 min |
| Total cost | ~$1.20 (Secure) or ~$1.00 (Community) |

Or pick A100 if you want to save ~30%; ~60–90 min wall-clock for ~$1.80.

---

## Secure Cloud vs Community Cloud (pick one)

| | Secure Cloud | Community Cloud |
|---|---|---|
| Hosts | RunPod's own datacenters | Individual operators worldwide |
| Reliability | Production-grade | Varies per host |
| Pricing | Higher | ~20–30% cheaper |
| Best for | First runs, anything you can't lose | Cost-sensitive throwaway experiments |

**For your first run, pick Secure Cloud.** Save Community Cloud for after
this works once.

---

## Step-by-step

### 1. Add an SSH key to your RunPod account

This is the easiest thing to skip and the most painful one to fix later.
RunPod injects your public key into pods *at boot* — adding it after
deploy means SSH won't work until you redeploy.

```bash
# from your laptop, if you don't have a key already
ssh-keygen -t ed25519 -C "runpod"
cat ~/.ssh/id_ed25519.pub
```

Then in the RunPod dashboard:
- **Account → Settings → SSH Public Keys** → paste the public key → save

### 2. Add credit to your RunPod account

RunPod is prepaid — you load credit, then spend it. **Account → Billing →
Add Credits**. $10 is more than enough for a few experiments.

### 3. Deploy a pod

In the dashboard:

1. **Pods → Deploy** (top-right)
2. **GPU type:** "H100 PCIe" (or "A100 PCIe" if you want cheaper)
3. **Cloud type:** Secure Cloud
4. **Template:** "RunPod PyTorch 2.4" (PyTorch + CUDA 12.4 + common ML libs preinstalled)
5. **Volume Disk:** 50 GB (default is 20 GB; bump it so TinyStories + checkpoints fit comfortably)
6. **Pod Name:** `nanochat-train` (anything; just a label)
7. **Click Deploy**

Provisioning takes ~1–2 minutes. The pod's status goes
`PROVISIONING → RUNNING`. When it's running, click into it and you'll see:

- **Connect** button (top of pod detail page) — gives you the SSH command
- **Logs / Web Terminal** — fallback if SSH won't work
- **Stop / Terminate** buttons — stop preserves the volume (still bills);
  terminate deletes everything

### 4. Connect via SSH

On the pod detail page, click **Connect → SSH over exposed TCP**. RunPod
shows a command like:

```bash
ssh root@123.45.67.89 -p 12345 -i ~/.ssh/id_ed25519
```

A few RunPod-specific quirks:
- User is `root`, not `ubuntu` (unlike Lambda or GCP).
- SSH port is **not 22** — it's a randomly assigned high port shown in the
  Connect dialog. Always copy the full command from the dashboard.
- If the Connect dialog doesn't show an SSH command at all, your pod
  template doesn't have SSH exposed by default. Use the **Web Terminal**
  in the dashboard instead — same shell, different transport.

If SSH refuses with "Permission denied (publickey)", your local key
doesn't match the one in your RunPod account settings. Fix the key in
**Account → Settings → SSH Public Keys**, then **terminate and redeploy
the pod** (changes to your account key only take effect for new pods).

### 5. Pull your code and install

```bash
git clone https://github.com/<your-username>/learn-you-an-hf-llm.git
cd learn-you-an-hf-llm
pip install -e .
```

If the repo is private, GitHub will prompt for username + token. Create a
token at https://github.com/settings/tokens (classic, scope `repo`) and
paste it as the password.

The "RunPod PyTorch 2.4" template already has PyTorch + CUDA + most ML
libs preinstalled, so `pip install -e .` mainly installs the repo's
extras (`tokenizers`, `datasets`, `safetensors`, `tensorboard`).

---

## Continue from `README.md` step 6

The remaining steps are identical regardless of provider:

- **Step 6** — verify GPU + PyTorch (`nvidia-smi`, `torch.cuda.is_available()`)
- **Step 7** — `python prepare_data.py` (~5 min)
- **Step 8** — `tmux new -s train; python train.py | tee train.log` (~30 min on H100, ~60–90 min on A100)
- **Step 9** — scp checkpoints back to your laptop
- **Step 10** — **terminate the pod** (RunPod equivalent below)
- **Step 11** — run inference locally

### RunPod-specific notes

**SCP back to your laptop (step 9).** Use the same SSH port RunPod
assigned:

```bash
# from your laptop
scp -P 12345 -r root@123.45.67.89:/workspace/learn-you-an-hf-llm/runs/h100_tinystories/checkpoints \
       runs/h100_tinystories/checkpoints
```

The `-P` (capital P) is mandatory; without it scp tries port 22 and
fails. Default working directory inside the pod is `/workspace`, not
`/root` — your `git clone` likely landed there.

**Stop vs Terminate (step 10).** In RunPod:

- **Stop** — pauses the pod, preserves your volume. Still bills
  $0.10/GB/month for the disk. Useful if you might want to come back and
  resume; bad if you'll forget about it.
- **Terminate** — deletes the pod and its volume. **Use this** for a
  one-shot experiment.

```text
RunPod dashboard → Pods → your pod → ⋯ menu → Terminate
```

Confirm with the pod name. Verify it's gone:

```text
RunPod dashboard → Pods → expect your pod no longer listed
```

Don't trust your memory. Eyeball the pods list at the end of every
session before logging off.

---

## Smaller GPU options on RunPod

If you want to spend even less:

| GPU | VRAM | Secure Cloud | Community Cloud | 15K-step train time | `train.py` changes |
|---|---|---|---|---|---|
| H100 PCIe 80GB | 80 | $2.39/hr | $1.99/hr | ~30 min | none |
| A100 PCIe 80GB | 80 | $1.79/hr | $1.49/hr | ~60–90 min | none |
| L40S 48GB | 48 | $0.99/hr | $0.79/hr | ~70–100 min | none |
| RTX A6000 48GB | 48 | $0.49/hr | $0.34/hr | ~2 hr | none (Ampere supports bf16) |
| RTX 4090 24GB | 24 | $0.74/hr | $0.34/hr | ~90 min | none (Ada supports bf16) |
| A40 48GB | 48 | $0.39/hr | n/a | ~2 hr | none |

Anything Ampere or newer (A100 / L40S / A6000 / 4090 / A40) supports bf16
and runs `train.py` unchanged.

**Avoid V100 and T4** on RunPod for this script — they pre-date bf16 and
the autocast block in `train.py` will throw or silently produce garbage.

For minimum spend with current `train.py`: **RTX A6000 Community Cloud,
~$0.34/hr × ~2 hr ≈ $0.70 total**. Slightly slower than A100 but
significantly cheaper, and 48GB VRAM is overkill for a 28M model so you
have headroom to bump batch size.

---

## Troubleshooting (RunPod-specific)

**"This pod cannot be deployed at the requested cost." Capacity gone**
in the GPU class you picked. Either pick a different GPU class or wait
5–10 min for inventory to refill.

**SSH "Permission denied (publickey)".** Your local key isn't the one
RunPod injected into the pod. Account-level key changes only apply to
**new** pods; redeploy after fixing the key.

**`nvidia-smi` works but `torch.cuda.is_available()` returns False.**
The pod template doesn't include CUDA-built PyTorch. Switch templates
on next deploy ("RunPod PyTorch 2.4" is the safest), or:

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
```

**Web terminal disconnects.** RunPod's web terminal drops after ~15 min
idle. Always use SSH + tmux for training; web terminal is a fallback for
quick commands only.

**Out-of-disk while downloading TinyStories.** Default container disk is
20 GB; TinyStories alone is ~2 GB and the tokenization step pushes peak
usage higher. Always set Volume Disk to **50 GB** at deploy time. Can't
expand after the fact — you'd need to redeploy.

**Forgot to terminate and racked up $$.** RunPod bills per-minute and
**doesn't refund** unused credit. Set a spend alert: **Account →
Billing → Spending Limits** → cap monthly spend at a sane number. Also
worth setting a per-pod spend cap at deploy time (the option's near
the bottom of the deploy form).

---

## Cost expectation

| GPU | Cloud | 30 min | 60 min | 2 hr |
|---|---|---|---|---|
| H100 PCIe | Secure | $1.20 | $2.39 | n/a |
| H100 PCIe | Community | $1.00 | $1.99 | n/a |
| A100 PCIe | Secure | n/a | $1.79 | $3.58 |
| A100 PCIe | Community | n/a | $1.49 | $2.98 |
| RTX A6000 | Community | n/a | n/a | ~$0.70 |
| RTX 4090 | Community | n/a | n/a | ~$0.50 |

For your first run: **H100 PCIe Secure, ~$1.20**. Cheap insurance against
flakiness; if it works, retry on Community Cloud or a smaller GPU for
future runs.
