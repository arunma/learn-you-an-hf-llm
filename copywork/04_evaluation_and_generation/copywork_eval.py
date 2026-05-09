"""Copywork target — Section 04 — Evaluation and generation.

Answer file:  hf_pipeline/04_eval_and_generate.py   (~115 lines)
Journal:      ~/projects/ai/learn-you-an-llm/journal/hf/hf_04_evaluation_and_generation.md

What you're building here:
  - Load the trained NanoChatModel from ~/.cache/hf_pipeline/model_copywork/
  - Compute validation perplexity (exp of cross-entropy loss)
  - Generate text continuations from a few prompts using model.generate()

Method:
  1. Open hf_pipeline/04_eval_and_generate.py side by side with this file.
  2. Type each section into this file, line by line.
  3. Replace the import at the top so the script uses YOUR copywork model:

         import sys
         from pathlib import Path
         COPYWORK_DIR = Path(__file__).resolve().parent.parent / "02_config_and_model"
         sys.path.insert(0, str(COPYWORK_DIR))

         from copywork_model import NanoChatModel

  4. After section 03 has produced ~/.cache/hf_pipeline/model_copywork/, run:
         python vault/copywork/hf/04_evaluation_and_generation/copywork_eval.py

  5. Expect: a perplexity number (probably 100-300 for a 500-step run on WikiText-2)
     and gibberish-but-statistical text samples. That's the correct outcome —
     proves inference works end-to-end.

Tier-3 skip:  the prompts list at the bottom is illustrative — change the
              strings to anything. The pattern (encode → generate → decode)
              is what matters, not the specific prompts.

----------------------------------------------------------------------------
Type your code below. Delete this docstring once you've internalised the
structure.
----------------------------------------------------------------------------
"""
