"""Local inference UI for the TinyStories-trained NanoChatModel.

A small Gradio web UI that loads the trained model + tokenizer and lets you
type prompts in a browser. Runs locally on your laptop — no cloud, no LM
Studio. Roughly an LM-Studio-shaped experience for a custom 28M-param model
that LM Studio can't load directly (because the architecture isn't in
llama.cpp's supported set).

Setup (one-time, on your laptop):
    pip install gradio
    # then scp the artifacts back from your pod (see setup_runpod.md step 9)

Run:
    python runs/h100_tinystories/infer.py

Browser opens automatically at http://127.0.0.1:7860
"""
from pathlib import Path

import gradio as gr
import torch
from tokenizers import Tokenizer
from torch.nn.attention import SDPBackend, sdpa_kernel

from hf_nanochat.model import NanoChatModel

RUN_DIR = Path(__file__).resolve().parent
TOKENIZER_PATH = RUN_DIR / "data_cache" / "tokenizer.json"
MODEL_PATH = RUN_DIR / "checkpoints" / "final"

# DEVICE = (
#     "mps" if torch.backends.mps.is_available()
#     else "cuda" if torch.cuda.is_available()
#     else "cpu"
# )
DEVICE="cpu"

# --- Load once at import time --------------------------------------------
print(f"Loading tokenizer from {TOKENIZER_PATH}...")
tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

print(f"Loading model from {MODEL_PATH} (device={DEVICE})...")
model = NanoChatModel.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

# HF's from_pretrained loading path can leave the RoPE cos/sin buffers
# (registered with persistent=False, so not in the state_dict) partially
# uninitialised — sometimes a few entries come back as NaN, which then
# poisons every attention layer's q/k via RoPE multiplication and crashes
# torch.multinomial during sampling. Re-running _init_rope after load
# overwrites the buffers with the correct deterministic values.
head_dim = model.config.n_embd // model.config.n_head
model._init_rope(model.config.sequence_len, head_dim)
model.cos = model.cos.to(DEVICE)
model.sin = model.sin.to(DEVICE)

print(f"Loaded {model.num_parameters():,} params.")


def generate(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    seed: int,
) -> str:
    """Generate a continuation for `prompt` using the loaded model."""
    if not prompt.strip():
        return "(enter a prompt)"

    if seed >= 0:
        torch.manual_seed(int(seed))

    input_ids = torch.tensor([tokenizer.encode(prompt).ids], device=DEVICE)
    # Force the math SDPA backend: optimized backends (flash / mem-efficient)
    # on MPS sometimes produce NaN logits with is_causal=True, which then
    # crash torch.multinomial during sampling. The math backend is bug-free.
    # Speed loss is negligible at 28M params during single-stream inference.
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        output = model.generate(
            input_ids,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
        )
    return tokenizer.decode(output[0].tolist())


# --- UI -------------------------------------------------------------------
with gr.Blocks(title="TinyStories nanochat") as demo:
    gr.Markdown(
        f"# TinyStories nanochat\n\n"
        f"A {model.num_parameters():,}-param transformer trained on "
        f"TinyStories. Try children's-story prompts — that's what the "
        f"model has seen.\n\n"
        f"**Device:** `{DEVICE}` · **Vocab:** {tokenizer.get_vocab_size()} · "
        f"**Context length:** {model.config.sequence_len}"
    )

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Prompt",
                lines=3,
                placeholder="Once upon a time, there was a little girl named Lily.",
            )
            output = gr.Textbox(label="Generated", lines=12, interactive=False)
            run = gr.Button("Generate", variant="primary")

            gr.Examples(
                examples=[
                    ["Once upon a time, there was a little girl named Lily."],
                    ["The dog and the cat were best friends."],
                    ["In the forest there lived a wise old owl."],
                    ["Tom found a shiny key in the garden."],
                    ["The little boy was very happy because"],
                    ["One day, the rabbit decided to"],
                ],
                inputs=prompt,
                label="Example prompts (click to fill)",
            )

        with gr.Column(scale=1):
            max_new_tokens = gr.Slider(
                10, 500, value=200, step=10, label="Max new tokens"
            )
            temperature = gr.Slider(
                0.1, 1.5, value=0.7, step=0.05,
                label="Temperature",
                info="Lower = more deterministic; higher = more random",
            )
            top_k = gr.Slider(
                1, 200, value=40, step=1,
                label="Top-k",
                info="Sample only from the K most-likely next tokens",
            )
            top_p = gr.Slider(
                0.5, 1.0, value=0.95, step=0.05,
                label="Top-p (nucleus)",
                info="Keep tokens whose cumulative prob ≤ p; 1.0 = disabled",
            )
            repetition_penalty = gr.Slider(
                1.0, 2.0, value=1.3, step=0.05,
                label="Repetition penalty",
                info="Divides logits of already-generated tokens; 1.0 = none, 1.2-1.5 helps break loops",
            )
            no_repeat_ngram_size = gr.Slider(
                0, 5, value=0, step=1,
                label="No-repeat n-gram size",
                info="Forbid any n-gram of this size from repeating; 0 = disabled",
            )
            seed = gr.Number(
                value=-1, precision=0,
                label="Seed",
                info="-1 = random each run; any non-negative int = reproducible",
            )

    inputs = [
        prompt,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        seed,
    ]
    run.click(generate, inputs=inputs, outputs=output)
    prompt.submit(generate, inputs=inputs, outputs=output)


if __name__ == "__main__":
    demo.launch(inbrowser=True)
