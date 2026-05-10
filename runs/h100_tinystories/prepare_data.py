"""Stage 0 — Download TinyStories and train a BPE tokenizer.

Combines what copywork sections 01a + 01b do for WikiText, but pointed at
TinyStories instead. One script because this is a one-shot setup before the
training run on Lambda Labs — no need for the disk-passing-between-stages
discipline of the copywork pipeline.

Run once on the Lambda VM before train.py.

Output:
    ./data_cache/train.txt        (~1.8 GB of concatenated stories)
    ./data_cache/val.txt          (~18 MB)
    ./data_cache/tokenizer.json   (~120 KB BPE)
"""
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

DATA_DIR = Path(__file__).resolve().parent / "data_cache"

VOCAB_SIZE = 4096
SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>"]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading TinyStories from Hugging Face Hub...")
    print("(~1 GB; takes ~3 min on a Lambda H100 instance the first time.)")
    dataset = load_dataset("roneneldan/TinyStories")

    train_text = "\n".join(t for t in dataset["train"]["text"] if t and t.strip())
    val_text = "\n".join(t for t in dataset["validation"]["text"] if t and t.strip())

    (DATA_DIR / "train.txt").write_text(train_text)
    (DATA_DIR / "val.txt").write_text(val_text)
    print(f"Train: {len(train_text):,} chars")
    print(f"Val:   {len(val_text):,} chars")

    print("\nTraining BPE tokenizer...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    chunks = (train_text[i:i + 10_000] for i in range(0, len(train_text), 10_000))
    tokenizer.train_from_iterator(chunks, trainer=trainer)

    out_path = DATA_DIR / "tokenizer.json"
    tokenizer.save(str(out_path))

    sample = "Once upon a time, there was a little girl named Lily."
    encoded = tokenizer.encode(sample)
    print(f"\nVocab size:       {tokenizer.get_vocab_size()}")
    print(f"Sample text:      {sample!r}")
    print(f"Encoded tokens:   {encoded.tokens[:10]}...")
    print(f"Encoded IDs:      {encoded.ids[:10]}...")
    print(f"Saved to:         {out_path}")


if __name__ == "__main__":
    main()
