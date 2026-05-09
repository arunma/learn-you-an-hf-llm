"""Copywork target — Section 05 — Save and share.

Answer file:  hf_pipeline/05_save_share.py   (~80 lines)
Journal:      journal/05_save_load_hub.md

What you're building here:
  - Load the trained model from ~/.cache/hf_pipeline/model_copywork/
  - Wrap the raw `tokenizers` BPE in HF's `PreTrainedTokenizerFast`
  - Save model + tokenizer together in a single Hub-loadable directory
  - Round-trip test: load it back, verify parameter count

Method:
  1. Open hf_pipeline/05_save_share.py side by side with this file.
  2. Type each section into this file, line by line.
  3. Replace the import at the top so the script uses YOUR copywork model:

         import sys
         from pathlib import Path
         COPYWORK_DIR = Path(__file__).resolve().parent.parent / "02_config_and_model"
         sys.path.insert(0, str(COPYWORK_DIR))

         from copywork_model import NanoChatModel

  4. After section 03 has produced ~/.cache/hf_pipeline/model_copywork/, run:
         python copywork/05_save_load_hub/copywork_save.py

  5. Output: ~/.cache/hf_pipeline/my-first-llm/ with five files:
         config.json
         model.safetensors
         tokenizer.json
         tokenizer_config.json
         special_tokens_map.json

     This directory is genuinely shippable — anyone with `transformers`
     installed could `from_pretrained` it.

Tier-3 skip:  the print-to-stdout block of push_to_hub instructions at the
              bottom is reference material — read once, don't re-type. The
              key patterns are PreTrainedTokenizerFast wrapping +
              save_pretrained on both model and tokenizer to the same dir.

----------------------------------------------------------------------------
Type your code below. Delete this docstring once you've internalised the
structure.
----------------------------------------------------------------------------
"""
