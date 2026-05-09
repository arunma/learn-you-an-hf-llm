"""Stage 5 — Save model + tokenizer in HF format and demo Hub upload.

Wraps the raw `tokenizers` BPE in HF's `PreTrainedTokenizerFast`, saves
both the trained `NanoChatModel` (from stage 3) and the wrapped tokenizer
into a single directory with HF's standard `save_pretrained` layout:

    my-first-llm/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── special_tokens_map.json

That directory is `from_pretrained`-loadable and `push_to_hub`-ready.

Run (after stages 1-4):
    python hf_pipeline/05_save_share.py

Output:
    ~/.cache/hf_pipeline/my-first-llm/   (HF-format directory)
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from hf_nanochat.model import NanoChatModel

CACHE_DIR = Path(os.path.expanduser("~/.cache/hf_pipeline"))

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def main() -> None:
    save_path = CACHE_DIR / "my-first-llm"

    # --- Load trained model from stage 3 ---
    model = NanoChatModel.from_pretrained(CACHE_DIR / "model_trained").to(DEVICE)

    # --- Wrap raw `tokenizers` BPE in HF's PreTrainedTokenizerFast ---
    raw_tokenizer = Tokenizer.from_file(str(CACHE_DIR / "tokenizer.json"))
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        pad_token="<|pad|>",
    )

    # --- Save both in HF format ---
    model.save_pretrained(save_path)
    hf_tokenizer.save_pretrained(save_path)

    print(f"Saved to: {save_path}\n")
    print("Files produced:")
    for f in sorted(save_path.iterdir()):
        size = f.stat().st_size
        print(f"  {f.name:32s}  {size:>12,} bytes")

    # --- Round-trip test ---
    loaded = NanoChatModel.from_pretrained(save_path).to(DEVICE)
    print(f"\nRound-trip OK: {loaded.num_parameters():,} params loaded back.")

    # --- Hub upload instructions ---
    print("\nTo push to Hugging Face Hub:")
    print("  huggingface-cli login")
    print("  python -c \"")
    print("      from hf_nanochat.model import NanoChatModel")
    print("      from transformers import AutoTokenizer")
    print(f"      m = NanoChatModel.from_pretrained('{save_path}')")
    print(f"      t = AutoTokenizer.from_pretrained('{save_path}')")
    print("      m.push_to_hub('your-username/my-first-llm')")
    print("      t.push_to_hub('your-username/my-first-llm')")
    print("  \"")


if __name__ == "__main__":
    main()
