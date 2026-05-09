"""Stage 3 — Tokenize the corpus, build a tiny NanoChatModel, train it.

Loads tokenizer (stage 2) and raw text (stage 1), tokenizes the whole
corpus into in-memory tensors, builds a tiny `NanoChatModel` with your
own `NanoChatConfig` (NOT `LlamaConfig` / `LlamaForCausalLM`), and runs
a simple AdamW training loop for 500 steps.

The model class lives in `hf_nanochat/model.py` — copywork that as
section 02. This stage is the training-loop shell only.

The optimizer setup here is deliberately minimal — three lines, not the
six-param-group / Muon dance from `my_nanochat/gpt.py:setup_optimizer`.
You can complicate it later if you have a measured reason.

Run (after stages 1 and 2):
    python hf_pipeline/03_train.py

Output:
    ~/.cache/hf_pipeline/model_trained/   (HF save_pretrained directory)
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from hf_nanochat.model import NanoChatConfig, NanoChatModel

CACHE_DIR = Path(os.path.expanduser("~/.cache/hf_pipeline"))

# Model — tiny enough for laptop training.
N_LAYER = 4
N_HEAD = 4
N_EMBD = 256
SEQ_LEN = 256

# Training hyperparameters.
BATCH_SIZE = 8
LR = 3e-4
NUM_STEPS = 500
EVAL_EVERY = 50
GRAD_CLIP = 1.0

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def get_batch(
    data: torch.Tensor, batch_size: int, seq_len: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (input, target) sequences from a token stream.

    Targets are inputs shifted right by one — standard next-token prediction.
    """
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix]).to(device)
    return x, y


def main() -> None:
    # --- Load tokenizer + corpus ---
    tokenizer = Tokenizer.from_file(str(CACHE_DIR / "tokenizer.json"))
    train_text = (CACHE_DIR / "train.txt").read_text()
    val_text = (CACHE_DIR / "val.txt").read_text()
    train_ids = torch.tensor(tokenizer.encode(train_text).ids, dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(val_text).ids, dtype=torch.long)
    print(f"Train tokens: {len(train_ids):,}  |  Val tokens: {len(val_ids):,}")

    # --- Build model with your custom config ---
    config = NanoChatConfig(
        sequence_len=SEQ_LEN,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_kv_head=N_HEAD,
        n_embd=N_EMBD,
    )
    model = NanoChatModel(config).to(DEVICE)
    print(f"Parameters:   {model.num_parameters():,}  |  Device: {DEVICE}")

    # --- Optimizer — minimal version ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95))

    # --- Training loop ---
    model.train()
    t0 = time.time()
    val_loss = float("inf")
    for step in range(NUM_STEPS):
        x, y = get_batch(train_ids, BATCH_SIZE, SEQ_LEN, DEVICE)
        out = model(x, labels=y)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad()

        if step % EVAL_EVERY == 0 or step == NUM_STEPS - 1:
            model.eval()
            with torch.no_grad():
                losses = []
                for _ in range(10):
                    vx, vy = get_batch(val_ids, BATCH_SIZE, SEQ_LEN, DEVICE)
                    losses.append(model(vx, labels=vy).loss.item())
                val_loss = sum(losses) / len(losses)
            model.train()

            elapsed = time.time() - t0
            tok_per_sec = (
                (step + 1) * BATCH_SIZE * SEQ_LEN / elapsed if elapsed > 0 else 0
            )
            print(
                f"Step {step:4d}/{NUM_STEPS}  |  "
                f"train: {out.loss.item():.4f}  |  val: {val_loss:.4f}  |  "
                f"{tok_per_sec:,.0f} tok/s"
            )

    # --- Save trained model ---
    save_path = CACHE_DIR / "model_trained"
    model.save_pretrained(save_path)
    print(f"\nTraining complete in {time.time() - t0:.1f}s")
    print(f"Final val loss: {val_loss:.4f}  |  Perplexity: {math.exp(val_loss):.1f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
