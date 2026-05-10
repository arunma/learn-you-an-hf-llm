"""Copywork target — Section 01b — BPE tokenizer training.

Answer file:  hf_pipeline/02_tokenizer.py   (~60 lines)
Journal:      journal/01_data_and_tokenizer.md

What you're building here:
  - Read train.txt produced by copywork_data.py
  - Configure a byte-level BPE tokenizer (same family as GPT-2's)
  - Train it on the corpus to produce a 4096-token vocabulary
  - Register three special tokens: <|pad|>, <|bos|>, <|eos|>
  - Save the trained tokenizer as ~/.cache/hf_pipeline/tokenizer.json

This stage is more conceptually rich than data download. The patterns
worth internalising:
  - The four-component tokenizer pipeline: pre_tokenizer → model → trainer → decoder
  - `models.BPE()` is the algorithm; `BpeTrainer` is the training procedure
  - Byte-level pre-tokenization (vs whitespace pre-tokenization) — handles
    any Unicode without UNK tokens
  - Streaming the corpus in chunks via `train_from_iterator` — memory-
    efficient for large datasets
  - Special tokens get assigned the lowest IDs (pad=0, bos=1, eos=2)

Method:
  1. Open hf_pipeline/02_tokenizer.py side by side with this file.
  2. Type each line. Pause on the four assignments to `tokenizer.pre_tokenizer`,
     `.decoder`, the `BpeTrainer(...)` config, and the chunk generator.
  3. Run (after copywork_data.py has produced train.txt):
         python copywork/01_data_and_tokenizer/copywork_tokenizer.py
  4. You should see:
         Vocab size:       4096
         Sample text:      'The quick brown fox jumps over the lazy dog'
         Encoded tokens:   ['The', 'Ġquick', 'Ġbrown', ...]   (Ġ = leading space)
         Encoded IDs:      [..., ..., ...]
         Avg chars/token:  ~3.5

Tier-3 skip:  the print-block at the bottom (sanity check) is optional;
              the four configuration lines (`models.BPE`, pre_tokenizer,
              decoder, BpeTrainer) and the train_from_iterator call are
              the architectural meat.

Why "Ġ"?  Byte-level BPE encodes leading spaces as the character "Ġ"
          (U+0120). It's how the tokenizer represents "this token came
          after a space". Lets the tokenizer roundtrip any text including
          whitespace without ambiguity.

----------------------------------------------------------------------------
Type your code below. Delete this docstring once you've internalised the
structure.
----------------------------------------------------------------------------
"""
import os
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

CACHE_DIR=Path(os.path.expanduser('~/.cache/hf_pipeline'))
VOCAB_SIZE=4096
SPECIAL_TOKENS=["<|pad|>", "<|bos|>", "<|eos|>"]

def main() -> None:
    train_text = (CACHE_DIR/"train.txt").read_text()

    tokenizer = Tokenizer(model=models.BPE())
    tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder=decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True
    )

    chunks = (train_text[i : i + 10_000] for i in range(0, len(train_text), 10_000))
    tokenizer.train_from_iterator(chunks, trainer=trainer)
    out_path = CACHE_DIR/"tokenizer.json"

    tokenizer.save(str(out_path))

    avg_chars_per_token = len(train_text) / len(tokenizer.encode(train_text).ids)
    sample ="The quick brown fox jumps over the lazy dog"
    encoded = tokenizer.encode(sample)

    print("Average chars per token:", avg_chars_per_token)
    print("Sample:", sample)
    print("Encoded tokens:", encoded.tokens[:8])
    print("Encoded IDs:", encoded.ids[:8])






if __name__ == "__main__":
    main()
