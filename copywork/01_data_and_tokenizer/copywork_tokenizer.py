"""Copywork target — Section 01b — BPE tokenizer training.

Answer file:  hf_pipeline/02_tokenizer.py   (~60 lines)
Journal:      ~/projects/ai/learn-you-an-llm/journal/hf/hf_01_data_and_tokenizer.md

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
         python vault/copywork/hf/01_data_and_tokenizer/copywork_tokenizer.py
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
