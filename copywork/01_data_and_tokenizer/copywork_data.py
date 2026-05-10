"""Copywork target — Section 01a — Data download.

Answer file:  hf_pipeline/01_data.py   (~45 lines, the smallest stage)
Journal:      journal/01_data_and_tokenizer.md

What you're building here:
  - Download WikiText-2 using HF's `datasets` library
  - Concatenate non-empty lines into one big string per split
  - Write train.txt and val.txt to ~/.cache/hf_pipeline/

This is the smallest, most plumbing-heavy stage in the pipeline. The
*ideas* worth taking away:
  - `datasets.load_dataset(name, config)` is the canonical entry point.
    It caches downloads, handles checksum/version, and returns a dict-like
    object with train/val/test splits.
  - Each split has columns; for WikiText that's just `text`.
  - For language modelling you typically concatenate all rows into one
    long string — the tokenizer doesn't care about row boundaries.

Method:
  1. Open hf_pipeline/01_data.py side by side with this file.
  2. Type each line. Pause on `load_dataset(...)` and the list-comprehension
     filter that drops empty lines.
  3. Run:
         python copywork/01_data_and_tokenizer/copywork_data.py
  4. You should see:
         Train: 10,918,134 chars (1,801,350 words)
         Val:     1,143,184 chars (  187,605 words)
         Saved to: /Users/.../.cache/hf_pipeline

Tier-3 skip:  none — this stage is short enough that everything is
              worth typing once for muscle memory.

----------------------------------------------------------------------------
Type your code below. Delete this docstring once you've internalised the
structure.
----------------------------------------------------------------------------
"""
import os.path
from pathlib import Path

from datasets import load_dataset

CACHE_DIR = Path(os.path.expanduser("~/.cache/hf_pipeline/"))

def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    dataset= load_dataset(path ="Salesforce/wikitext", name ="wikitext-2-raw-v1")

    train_text = "\n".join(t for t in dataset["train"]["text"] if t.strip())
    val_text = "\n".join(t for t in dataset["validation"]["text"] if t.strip())

    (CACHE_DIR /"train.txt").write_text(train_text)
    (CACHE_DIR /"val.txt").write_text(val_text)

    print(f"Train data: {len(train_text)}, words: {len(train_text.split())}")
    print(f"Validation data: {len(val_text)}, words: {len(val_text.split())}")


if __name__ == "__main__":
    main()

