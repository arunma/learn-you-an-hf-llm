"""Stage 4 — Load the trained model, compute perplexity, generate samples.

Loads the trained `NanoChatModel` from stage 3 and the tokenizer from
stage 2. Reports validation perplexity and bits/token, then generates
continuations for a few prompts using HF's `model.generate()`.

This stage produces no disk outputs — its results are the printed
metrics and sample text.

Run (after stages 1-3):
    python hf_pipeline/04_eval_and_generate.py
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer

from hf_nanochat.model import NanoChatModel

CACHE_DIR = Path(os.path.expanduser("~/.cache/hf_pipeline"))

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 8
SEQ_LEN = 256
EVAL_BATCHES = 50


def get_batch(
    data: torch.Tensor, batch_size: int, seq_len: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same sampler as stage 3 — random (x, y) batches from a token stream."""
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix]).to(device)
    return x, y


def evaluate(model: NanoChatModel, val_ids: torch.Tensor) -> float:
    """Average cross-entropy loss over EVAL_BATCHES validation batches."""
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x, y = get_batch(val_ids, BATCH_SIZE, SEQ_LEN, DEVICE)
            losses.append(model(x, labels=y).loss.item())
    return sum(losses) / len(losses)


def generate_samples(
    model: NanoChatModel, tokenizer: Tokenizer, prompts: list[str]
) -> None:
    """Generate continuations using HF's built-in `model.generate()`."""
    model.eval()
    for prompt in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt).ids], device=DEVICE)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.8,
                top_k=50,
            )
        text = tokenizer.decode(out[0].tolist())
        print(f"  Prompt: {prompt!r}")
        print(f"  Output: {text}")
        print()


def main() -> None:
    # --- Load tokenizer + model ---
    tokenizer = Tokenizer.from_file(str(CACHE_DIR / "tokenizer.json"))
    model = NanoChatModel.from_pretrained(CACHE_DIR / "model_trained").to(DEVICE)
    print(f"Loaded model: {model.num_parameters():,} params on {DEVICE}")

    # --- Tokenize val text for perplexity ---
    val_text = (CACHE_DIR / "val.txt").read_text()
    val_ids = torch.tensor(tokenizer.encode(val_text).ids, dtype=torch.long)

    # --- Evaluate ---
    avg_loss = evaluate(model, val_ids)
    perplexity = math.exp(avg_loss)
    bits_per_token = avg_loss / math.log(2)
    print("\n--- Evaluation ---")
    print(f"Val loss:        {avg_loss:.4f}")
    print(f"Perplexity:      {perplexity:.1f}")
    print(f"Bits per token:  {bits_per_token:.4f}")
    print(f"(Random baseline perplexity = vocab size = {tokenizer.get_vocab_size()})")

    # --- Generate samples ---
    print("\n--- Generation ---")
    generate_samples(
        model,
        tokenizer,
        prompts=[
            "The meaning of life is",
            "In the year 2025,",
            "The president of",
        ],
    )


if __name__ == "__main__":
    main()
