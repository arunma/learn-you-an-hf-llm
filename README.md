# learn-you-an-hf-llm

Code behind [this blog post](https://www.arunma.com/2026/05/11/small-dog-small-language-model-training-a-transformer-for-5/) — training a 91M-parameter transformer from scratch on TinyStories. Around $1.50 of rented H100 time, ~20 minutes of training, writes coherent short children's stories.

The model is a small GPT-style architecture (RoPE, RMSNorm, ReLU² MLP) wrapped in the Hugging Face `transformers` interface. It lives in [`hf_nanochat/model.py`](hf_nanochat/model.py). Training and inference scripts are in [`runs/h100_tinystories/`](runs/h100_tinystories/).

## Run inference

```bash
pip install -e .
python runs/h100_tinystories/infer.py
```

Opens a Gradio UI at `http://127.0.0.1:7860`. You'll need `runs/h100_tinystories/checkpoints/final/` and `runs/h100_tinystories/data_cache/tokenizer.json` present locally — either train your own (below) or grab the release.

## Train it yourself

Rent an H100 on RunPod (~$3/hr), then follow [`runs/h100_tinystories/setup_runpod.md`](runs/h100_tinystories/setup_runpod.md). The 91M-param run is ~20 minutes.

Long write-up of the whole journey: [`runs/h100_tinystories/BLOG.md`](runs/h100_tinystories/BLOG.md).

## License

MIT.
