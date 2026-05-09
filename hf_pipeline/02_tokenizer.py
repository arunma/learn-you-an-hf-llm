"""Stage 2 — Train a BPE tokenizer on the corpus and save it.

Reads `train.txt` from stage 1, trains a byte-level BPE tokenizer with
vocab size 4096 (small for laptop training; production models use
32K-128K), and saves it to disk in HF's standard tokenizer.json format.

The tokenizer training has nothing to do with the model — it's pure
data preprocessing. The output is a deterministic mapping from text to
integer token IDs that every later stage uses.

Run (after stage 1):
    python hf_pipeline/02_tokenizer.py

Output:
    ~/.cache/hf_pipeline/tokenizer.json
"""

from __future__ import annotations

import os
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

CACHE_DIR = Path(os.path.expanduser("~/.cache/hf_pipeline"))

VOCAB_SIZE = 4096
SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>"]


def main() -> None:
    train_text = (CACHE_DIR / "train.txt").read_text()

    # Byte-level BPE — same family as GPT-2's tokenizer.
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    # Stream the corpus in chunks (memory-efficient for large datasets).
    chunks = (train_text[i : i + 10_000] for i in range(0, len(train_text), 10_000))
    tokenizer.train_from_iterator(chunks, trainer=trainer)

    out_path = CACHE_DIR / "tokenizer.json"
    tokenizer.save(str(out_path))

    # Quick sanity check.
    sample = "The quick brown fox jumps over the lazy dog"
    encoded = tokenizer.encode(sample)
    avg_chars_per_token = len(train_text) / len(tokenizer.encode(train_text).ids)
    print(f"Vocab size:       {tokenizer.get_vocab_size()}")
    print(f"Sample text:      {sample!r}")
    print(f"Encoded tokens:   {encoded.tokens[:8]}...")
    print(f"Encoded IDs:      {encoded.ids[:8]}...")
    print(f"Avg chars/token:  {avg_chars_per_token:.1f}")
    print(f"Saved to:         {out_path}")


if __name__ == "__main__":
    main()
