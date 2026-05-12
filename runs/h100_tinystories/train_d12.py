import math
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.utils.tensorboard import SummaryWriter

from hf_nanochat.model import NanoChatConfig, NanoChatModel

RUN_DIR = Path(__file__).resolve().parent
DATA_DIR = RUN_DIR / "data_cache"
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
TB_LOG_DIR = RUN_DIR / "tb_logs"

N_LAYERS = 12
N_HEADS = 12
N_KV_HEAD = 12
N_EMBD = 768
SEQ_LEN = 512

BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 0.1
NUM_STEPS = 30_000
WARMUP_STEPS = 500
EVAL_EVERY = 1_000
EVAL_BATCHES = 20
SAVE_EVERY = 5_000
GRAD_CLIP = 1.0

DEVICE = "cuda"


def get_batch(data: torch.Tensor, batch_size, seq_len, device):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix]).to(device, non_blocking=True)
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix]).to(device, non_blocking=True)
    return x, y


def cosine_with_warmup(step):
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, NUM_STEPS - WARMUP_STEPS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(model, path):
    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    base.save_pretrained(path)


def main() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TB_LOG_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(TB_LOG_DIR))
    print(f"TensorBoard log dir: {TB_LOG_DIR}")

    tokenizer = Tokenizer.from_file(str(DATA_DIR / "tokenizer.json"))

    print("Loading pre-tokenized corpus...")
    train_ids = torch.load(DATA_DIR / "train_ids.pt")
    val_ids = torch.load(DATA_DIR / "val_ids.pt")
    print(f"Train tokens: {len(train_ids):,}  |  Val tokens: {len(val_ids):,}")

    config = NanoChatConfig(
        sequence_len=SEQ_LEN,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=N_LAYERS,
        n_head=N_HEADS,
        n_kv_head=N_KV_HEAD,
        n_embd=N_EMBD,
    )

    model = NanoChatModel(config).to(DEVICE)
    print(f"Parameters: {model.num_parameters():,}, Device: {DEVICE}")

    print("Compiling model...")
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup)

    model.train()
    t0 = time.time()
    val_loss = float("inf")

    for step in range(NUM_STEPS):
        x, y = get_batch(train_ids, BATCH_SIZE, SEQ_LEN, DEVICE)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x, labels=y)

        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        writer.add_scalar("train/loss", out.loss.item(), step)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)

        if step % EVAL_EVERY == 0 or step == NUM_STEPS - 1:
            model.eval()
            with torch.no_grad():
                losses = []
                for _ in range(EVAL_BATCHES):
                    vx, vy = get_batch(val_ids, BATCH_SIZE, SEQ_LEN, DEVICE)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out_v = model(vx, labels=vy)
                    losses.append(out_v.loss.item())
                val_loss = sum(losses) / len(losses)
            model.train()

            elapsed = time.time() - t0
            tok_per_sec = (step + 1) * BATCH_SIZE * SEQ_LEN / elapsed if elapsed > 0 else 0
            current_lr = scheduler.get_last_lr()[0]
            print(f"Step : {step:5d}/{NUM_STEPS}, Train: {out.loss.item():.4f}, val: {val_loss:.4f}, ppl: {math.exp(val_loss):.1f}, lr: {current_lr:.2e}, {tok_per_sec:,.0f} tok/s")

            writer.add_scalar("val/loss", val_loss, step)
            writer.add_scalar("val/perplexity", math.exp(val_loss), step)
            writer.add_scalar("perf/tokens_per_sec", tok_per_sec, step)

        if (step + 1) % SAVE_EVERY == 0 and step > 0:
            ckpt_path = CHECKPOINT_DIR / f"step_{step + 1:05d}"
            save_checkpoint(model, ckpt_path)
            print(f"  -> checkpoint saved: {ckpt_path}")

    save_path = CHECKPOINT_DIR / "final"
    save_checkpoint(model, save_path)
    writer.close()
    print(f"Training completed in : {time.time() - t0:.2f}s")
    print(f"Final valuation loss : {val_loss:.4f}, Perplexity: {math.exp(val_loss):.4f}")
    print(f"Model saved at {save_path}")


if __name__ == "__main__":
    main()
