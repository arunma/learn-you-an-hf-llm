"""Stage 1 — Download data and save raw text to disk.

Downloads WikiText-2 (~2M tokens of clean Wikipedia text) using HF's
`datasets` library and writes train / val text files to a cache directory.

Each subsequent stage reads from this cache, so you can rerun any stage
in isolation without re-downloading.

Run:
    python hf_pipeline/01_data.py

Output:
    ~/.cache/hf_pipeline/train.txt
    ~/.cache/hf_pipeline/val.txt
"""

from __future__ import annotations

import os
from pathlib import Path

from datasets import load_dataset

CACHE_DIR = Path(os.path.expanduser("~/.cache/hf_pipeline"))


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Drop empty / whitespace-only lines, then join with newlines.
    train_text = "\n".join(t for t in dataset["train"]["text"] if t.strip())
    val_text = "\n".join(t for t in dataset["validation"]["text"] if t.strip())

    (CACHE_DIR / "train.txt").write_text(train_text)
    (CACHE_DIR / "val.txt").write_text(val_text)

    print(f"Train: {len(train_text):,} chars ({len(train_text.split()):,} words)")
    print(f"Val:   {len(val_text):,} chars ({len(val_text.split()):,} words)")
    print(f"Saved to: {CACHE_DIR}")


if __name__ == "__main__":
    main()
