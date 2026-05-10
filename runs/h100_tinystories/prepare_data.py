"""Stage 0 — Download TinyStories, train BPE tokenizer, pre-tokenize the corpus.

Combines what copywork sections 01a + 01b do for WikiText, but pointed at
TinyStories instead. One script because this is a one-shot setup before the
training run on Lambda Labs — no need for the disk-passing-between-stages
discipline of the copywork pipeline.

Each step is idempotent: skipped if its output file already exists. Safe to
re-run on the same pod without re-downloading or re-tokenizing.

Output:
    ./data_cache/train.txt        (~1.8 GB of concatenated stories)
    ./data_cache/val.txt          (~18 MB)
    ./data_cache/tokenizer.json   (~120 KB BPE)
    ./data_cache/train_ids.pt     (~3.7 GB int64 tensor — pre-tokenized train)
    ./data_cache/val_ids.pt       (~38 MB int64 tensor — pre-tokenized val)
"""
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

DATA_DIR = Path(__file__).resolve().parent / "data_cache"

VOCAB_SIZE = 4096
SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>"]

# Chunk size for parallel encode_batch — bounds peak memory during tokenization
TOKENIZE_CHUNK_CHARS = 1_000_000


def tokenize_corpus(text: str, tokenizer: Tokenizer) -> torch.Tensor:
    """Tokenize a large string using parallel chunked encoding.

    encode_batch is multi-threaded in Rust; chunks are processed in parallel
    across all available CPU cores. Memory bounded by chunk size * batch
    rather than by the full corpus.
    """
    chunks = [
        text[i : i + TOKENIZE_CHUNK_CHARS]
        for i in range(0, len(text), TOKENIZE_CHUNK_CHARS)
    ]
    encodings = tokenizer.encode_batch(chunks)
    pieces = [torch.tensor(enc.ids, dtype=torch.long) for enc in encodings]
    return torch.cat(pieces)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_txt = DATA_DIR / "train.txt"
    val_txt = DATA_DIR / "val.txt"
    tokenizer_json = DATA_DIR / "tokenizer.json"
    train_ids_pt = DATA_DIR / "train_ids.pt"
    val_ids_pt = DATA_DIR / "val_ids.pt"

    # --- Step 1: text splits -------------------------------------------------
    if train_txt.exists() and val_txt.exists():
        print(f"[1/3] Reusing existing {train_txt.name} and {val_txt.name}")
        train_text = train_txt.read_text()
        val_text = val_txt.read_text()
    else:
        print("[1/3] Downloading TinyStories from Hugging Face Hub...")
        print("      (~1 GB; takes ~3 min the first time)")
        dataset = load_dataset("roneneldan/TinyStories")

        train_text = "\n".join(t for t in dataset["train"]["text"] if t and t.strip())
        val_text = "\n".join(t for t in dataset["validation"]["text"] if t and t.strip())

        train_txt.write_text(train_text)
        val_txt.write_text(val_text)
        print(f"      Train: {len(train_text):,} chars")
        print(f"      Val:   {len(val_text):,} chars")

    # --- Step 2: BPE tokenizer ----------------------------------------------
    if tokenizer_json.exists():
        print(f"[2/3] Reusing existing {tokenizer_json.name}")
        tokenizer = Tokenizer.from_file(str(tokenizer_json))
    else:
        print("\n[2/3] Training BPE tokenizer...")
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=VOCAB_SIZE,
            special_tokens=SPECIAL_TOKENS,
            min_frequency=2,
            show_progress=True,
        )

        chunks = (
            train_text[i : i + 10_000]
            for i in range(0, len(train_text), 10_000)
        )
        tokenizer.train_from_iterator(chunks, trainer=trainer)
        tokenizer.save(str(tokenizer_json))

        sample = "Once upon a time, there was a little girl named Lily."
        encoded = tokenizer.encode(sample)
        print(f"      Vocab size:     {tokenizer.get_vocab_size()}")
        print(f"      Sample text:    {sample!r}")
        print(f"      Encoded tokens: {encoded.tokens[:10]}...")
        print(f"      Encoded IDs:    {encoded.ids[:10]}...")

    # --- Step 3: pre-tokenize the corpus to tensors --------------------------
    if train_ids_pt.exists() and val_ids_pt.exists():
        print(f"[3/3] Reusing existing {train_ids_pt.name} and {val_ids_pt.name}")
    else:
        print("\n[3/3] Tokenizing corpus into tensors (chunked + parallel)...")
        print("      Train (~5–10 min on a typical pod CPU)...")
        train_ids = tokenize_corpus(train_text, tokenizer)
        torch.save(train_ids, train_ids_pt)
        print(f"      Train tokens: {len(train_ids):,}  ->  {train_ids_pt}")

        print("      Val (~10 sec)...")
        val_ids = tokenize_corpus(val_text, tokenizer)
        torch.save(val_ids, val_ids_pt)
        print(f"      Val tokens:   {len(val_ids):,}  ->  {val_ids_pt}")

    print("\nDone. train.py will load these tensors directly via torch.load().")


if __name__ == "__main__":
    main()
