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

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# --- Load once at import time --------------------------------------------
print(f"Loading tokenizer from {TOKENIZER_PATH}...")
tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

print(f"Loading model from {MODEL_PATH} (device={DEVICE})...")
model = NanoChatModel.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()
print(f"Loaded {model.num_parameters():,} params.")


def generate(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
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
                0.1, 1.5, value=0.8, step=0.05,
                label="Temperature",
                info="Lower = more deterministic; higher = more random",
            )
            top_k = gr.Slider(
                1, 200, value=40, step=1,
                label="Top-k",
                info="Sample only from the K most-likely next tokens",
            )
            seed = gr.Number(
                value=-1, precision=0,
                label="Seed",
                info="-1 = random each run; any non-negative int = reproducible",
            )

    inputs = [prompt, max_new_tokens, temperature, top_k, seed]
    run.click(generate, inputs=inputs, outputs=output)
    prompt.submit(generate, inputs=inputs, outputs=output)


if __name__ == "__main__":
    demo.launch(inbrowser=True)
