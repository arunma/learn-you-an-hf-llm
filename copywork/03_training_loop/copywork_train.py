import math
import os
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from hf_nanochat.model import NanoChatConfig, NanoChatModel

CACHE_DIR = Path(os.path.expanduser('~/.cache/hf_pipeline'))

N_LAYERS = 4
N_HEADS = 4
N_EMBD = 256
SEQ_LEN = 256

BATCH_SIZE = 8
LR = 3e-4
NUM_STEPS = 500
EVAL_EVERY = 50
GRAD_CLIP = 1.0

DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else 'cpu')


def get_batch(data: torch.Tensor, batch_size, seq_len, device):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix]).to(device)
    return x, y


def main() -> None:
    tokenizer = Tokenizer.from_file(str(CACHE_DIR / "tokenizer.json"))
    train_text = (CACHE_DIR / "train.txt").read_text()
    val_text = (CACHE_DIR / "val.txt").read_text()

    train_ids = torch.tensor(tokenizer.encode(train_text).ids, dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(val_text).ids, dtype=torch.long)

    config = NanoChatConfig(
        sequence_len=SEQ_LEN,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=N_LAYERS,
        n_head=N_HEADS,
        n_kv_head=N_HEADS,
        n_embd=N_EMBD,
    )

    model = NanoChatModel(config).to(DEVICE)
    model=torch.compile(model)

    print (f"Parameters: {model.num_parameters():,}, Device: {DEVICE}" )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95))

    model.train()
    t0=time.time()

    for step in range(NUM_STEPS):
        x, y = get_batch(train_ids, BATCH_SIZE, SEQ_LEN, DEVICE)
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            out = model(x, labels=y)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad()

        if step % EVAL_EVERY == 0 or step == NUM_STEPS-1:
            model.eval() #switch to eval mode
            with torch.no_grad():
                losses=[]
                for _ in range(10):
                    vx, vy = get_batch(val_ids, BATCH_SIZE, SEQ_LEN, DEVICE)
                    out_v = model(vx, labels=vy)
                    losses.append(out_v.loss.item())
                val_loss = sum(losses)/len(losses)
            model.train() #switch to train mode

            elapsed = time.time() - t0
            tok_per_sec = (
                (step+1) * BATCH_SIZE * SEQ_LEN / elapsed if elapsed > 0 else 0
            )
            print (f"Step : {step:4d}/{NUM_STEPS}, Train: {out.loss.item():.4f}, val: {val_loss:.4f}, took: {tok_per_sec:.2f}s")

    save_path = CACHE_DIR / "model_trained"
    model.save_pretrained(save_path)
    print(f"Training completed in : {time.time() - t0:.2f}s")
    print(f"Final valuation loss : {val_loss:.4f}, Perplexity: {math.exp(val_loss):.4f}")
    print(f"Model saved at {save_path}")



if __name__ == "__main__":
    main()