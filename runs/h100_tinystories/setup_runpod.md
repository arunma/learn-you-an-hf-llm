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

On the pod detail page, click **Connect**. RunPod offers two SSH methods:

**A. SSH via runpod.io proxy** (default, recommended)

This is what you'll see at the top of the Connect dialog. The command
looks like:

```bash
ssh <pod-id>-<hash>@ssh.runpod.io -i ~/.ssh/<your-key>
# real example:
# ssh nrw0yf1b7pvg7a-644120c1@ssh.runpod.io -i ~/.ssh/id_ed25519_arunma
```

What's happening: `ssh.runpod.io` is RunPod's SSH bastion. You
authenticate against the proxy with your public key (uploaded to your
RunPod account in step 1), and the proxy forwards you into the pod as
root. No port management on your side — just `ssh ...@ssh.runpod.io`,
default port 22.

This works as long as your account-level SSH key is set up. It does
not depend on the pod template exposing TCP ports.

**B. SSH over exposed TCP** (fallback / direct)

If you also enabled "Expose TCP ports" on the pod template, RunPod
also shows a direct command like:

```bash
ssh root@123.45.67.89 -p 12345 -i ~/.ssh/id_ed25519
```

This bypasses the proxy and goes straight to the pod's IP on a
randomly assigned high port. Slightly faster for big file transfers
but requires the right port and IP, both of which change per-pod.

**Quirks for either method:**
- Inside the pod you're `root`, working directory is `/workspace`.
- If SSH refuses with "Permission denied (publickey)", your local key
  doesn't match the one in your RunPod account settings. Fix the key
  in **Account → Settings → SSH Public Keys**, then **terminate and
  redeploy the pod** — changes to your account key only take effect
  for new pods.
- If neither command appears in the Connect dialog at all, fall back
  to the **Web Terminal** in the dashboard (same shell, different
  transport).

The rest of this guide assumes the proxy method (option A) since it's
RunPod's default. If you used exposed TCP, swap `<pod-id>-<hash>@ssh.runpod.io`
for `root@<ip> -p <port>` in any later `ssh` / `scp` command.

### 5. Pull your code and install

**Fresh pod (first-time clone):**

```bash
cd /workspace
git clone https://github.com/<your-username>/learn-you-an-hf-llm.git
cd learn-you-an-hf-llm
pip install -e .
```

**Existing pod (already cloned, picking up new commits):**

```bash
cd /workspace/learn-you-an-hf-llm
git pull
# only re-run pip install if pyproject.toml changed
# (look for it in `git pull` output — if not mentioned, skip):
pip install -e .
```

If the repo is private, GitHub will prompt for username + token. Create a
token at https://github.com/settings/tokens (classic, scope `repo`) and
paste it as the password.

The "RunPod PyTorch 2.8.0" template already has PyTorch + CUDA + most
ML libs preinstalled, so `pip install -e .` mainly installs the repo's
extras (`tokenizers`, `datasets`, `safetensors`, `tensorboard`).

---

### 6. Verify GPU + PyTorch

```bash
nvidia-smi
# expect: 1× H100 (or whatever you picked), ~80 GB free, CUDA 12.x

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: True NVIDIA H100 80GB HBM3
```

If `torch.cuda.is_available()` returns `False` despite `nvidia-smi`
working, the pod's PyTorch wasn't built with CUDA. Reinstall:

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
```

### 7. Prepare the dataset and tokenizer (~5 min)

```bash
cd runs/h100_tinystories
python prepare_data.py
```

Downloads TinyStories from HF Hub (~1 GB), writes `data_cache/train.txt`
and `val.txt`, then trains a BPE tokenizer saved to
`data_cache/tokenizer.json`. Only needs to run once per pod.

### 8. Run training inside tmux (~30 min on H100)

**Always wrap training in tmux.** A network blip kills your SSH session,
and without tmux it kills the training too — losing 30 minutes of paid
compute. Skipping this step is the most common expensive mistake.

```bash
tmux new -s train
python train.py | tee train.log
```

`tee` writes the output to both your terminal and `train.log` so you
can scp the log out later if you want to inspect the run after the
fact.

You'll see something like:

```
Tokenizing corpus...
Train tokens: 470,123,456  |  Val tokens: 4,891,234
Parameters: 28,341,248, Device: cuda
Compiling model (first step will be slow due to graph tracing)...
Step :     0/15000, Train: 8.32, val: 8.31, ppl: 4054.3, ...
Step :   500/15000, Train: 4.89, val: 4.92, ppl: 137.0, ...
...
Step : 14999/15000, Train: 1.62, val: 1.71, ppl: 5.5, ...
Training completed in : 1834.21s
Final valuation loss : 1.7100, Perplexity: 5.5300
```

Wait until at least the second eval line prints (~step 500) so you
confirm loss is dropping from ~8 toward ~5. Then **detach with
`Ctrl-b` then `d`** — training keeps running in the background. You
can close your SSH connection without losing it.

To reattach later: `tmux attach -t train`.

To watch the log without reattaching to tmux:

```bash
tail -f /workspace/learn-you-an-hf-llm/runs/h100_tinystories/train.log
```

Total time including setup, tokenization, compile warm-up, and 15K
training steps: ~30–35 min on H100 SXM.

### 8a. (Optional) Live monitoring with TensorBoard

`train.py` writes scalars to `runs/h100_tinystories/tb_logs/` as
training progresses (`train/loss`, `train/lr`, `val/loss`,
`val/perplexity`, `perf/tokens_per_sec`). To view them live in your
laptop's browser, you need two things: a TensorBoard server running on
the pod, and an SSH tunnel from your laptop to that server.

**On the pod**, in a second tmux window (`Ctrl-b c` to create one,
`Ctrl-b 0` / `Ctrl-b 1` to switch):

```bash
cd /workspace/learn-you-an-hf-llm/runs/h100_tinystories
tensorboard --logdir=tb_logs --port=6006 --bind_all
```

Leave it running. `--bind_all` makes it listen on all interfaces, which
is needed for the SSH tunnel below.

**From your laptop**, open a new terminal and tunnel port 6006 through
the same SSH connection method you used in step 4:

```bash
# proxy method (option A)
ssh -L 6006:localhost:6006 <pod-id>-<hash>@ssh.runpod.io -i ~/.ssh/<your-key>

# or exposed-TCP method (option B)
ssh -L 6006:localhost:6006 root@<pod-ip> -p <pod-port> -i ~/.ssh/<your-key>

# leave this connection open; it's just the tunnel
```

Then open **http://localhost:6006** in your browser. You'll see five
scalar tags. The `train/loss` curve is noisy — drag the **Smoothing
slider** (top of the dashboard) to ~0.95 to make the trend visible.

When training finishes you can kill the TensorBoard process with
`Ctrl-c` in its tmux window. Or leave it running until you terminate
the pod.

### 9. Pull checkpoints back to your laptop

When training prints `Final model saved`, scp the checkpoint folder
home using the same SSH method you used in step 4.

**Proxy method (option A — what most people use):**

```bash
# from your laptop, in the repo root
scp -i ~/.ssh/<your-key> -r \
    <pod-id>-<hash>@ssh.runpod.io:/workspace/learn-you-an-hf-llm/runs/h100_tinystories/checkpoints \
    runs/h100_tinystories/checkpoints

# also pull the tokenizer if you want local inference
scp -i ~/.ssh/<your-key> \
    <pod-id>-<hash>@ssh.runpod.io:/workspace/learn-you-an-hf-llm/runs/h100_tinystories/data_cache/tokenizer.json \
    runs/h100_tinystories/data_cache/tokenizer.json
```

No `-P` flag — the proxy listens on default port 22.

**Exposed TCP method (option B):**

```bash
scp -P <pod-port> -i ~/.ssh/<your-key> -r \
    root@<pod-ip>:/workspace/learn-you-an-hf-llm/runs/h100_tinystories/checkpoints \
    runs/h100_tinystories/checkpoints
```

The `-P` (capital P, not `-p`) is mandatory; without it scp tries port
22 and fails.

RunPod's working directory inside the pod is `/workspace`, not `/root`
— the `git clone` lives there.

**Verify the checkpoint actually downloaded *before* terminating
anything:**

```bash
ls runs/h100_tinystories/checkpoints/final/
# expect: config.json  model.safetensors
```

### 10. Terminate the pod AND delete the network volume

Two separate cleanup steps. Skipping either keeps billing.

**10a. Terminate the pod**

RunPod dashboard → **Pods** → your pod → **⋯ menu → Terminate** →
confirm by typing the pod name. Verify the pod no longer appears in
the list.

**Stop ≠ Terminate.** Stop pauses the pod but keeps billing its disk
(~$0.10/GB/month). Always pick Terminate for a one-shot experiment.

**10b. Delete the network volume — separately**

When you set "Persistent storage" to Network volume during deploy,
RunPod auto-created a 50 GB volume named something like
`nanochat-hf-train-<random>`. **Terminating the pod does NOT delete
this volume.** It keeps billing ~$5/month forever until you remove it
manually.

RunPod dashboard → **Storage** → find the auto-created volume → **⋯
menu → Delete** → confirm.

Eyeball the Storage page before logging off; if a volume is still
listed, it's still costing you.

### 11. Run inference locally

Back on your laptop, point your existing
`copywork/04_evaluation_and_generation/copywork_eval.py` at the new
artifacts:

```python
RUN_DIR = Path("runs/h100_tinystories")
tokenizer = Tokenizer.from_file(str(RUN_DIR / "data_cache" / "tokenizer.json"))
model = NanoChatModel.from_pretrained(RUN_DIR / "checkpoints" / "final").to(DEVICE)
```

TinyStories is a children's-stories corpus, so prompts that match the
domain produce the most coherent output — `"Once upon a time"` works
better than `"The president of"`.

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
