import torch
import torch.nn.functional as F
from sympy.physics.units import temperature
from torch import nn
from torch.nn import Linear, Embedding, ModuleList
from torch.nn.functional import relu
from torch.testing._internal import generated
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

from hf_nanochat.model import NanoChatConfig


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotate_emb(x, cos, sin):
    d = x.size(3) // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        q = apply_rotate_emb(q, cos, sin)
        k = apply_rotate_emb(k, cos, sin)

        q = norm(q)
        k = norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, config.n_embd * 4, bias=False)
        self.c_proj = Linear(config.n_embd * 4, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(norm(x), cos, sin)
        x = x + self.mlp(norm(x))
        return x


class NanoChatModel(PreTrainedModel, GenerationMixin):
    config_class = NanoChatConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.wte = Embedding(config.vocab_size, config.n_embd)
        self.blocks = ModuleList([Block(config) for _ in range(config.n_layer)])
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)

        head_dim = config.n_embd // config.n_head
        self.register_buffer('cos', None, persistent=False)
        self.register_buffer('sin', None, persistent=False)
        self._init_rope(config.sequence_len, head_dim)
        self.post_init()

    def _init_rope(self, sequence_len, head_dim, base=100000):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(sequence_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        self.cos = cos[None, :, None, :]
        self.sin = sin[None, :, None, :]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, input_ids, labels=None, **kwargs):
        B, T = input_ids.size()

        cos = self.cos[:, :T].to(input_ids.device)
        sin = self.sin[:, :T].to(input_ids.device)

        x = self.wte(input_ids)
        x = norm(x)

        for block in self.blocks:
            x = block(x, cos, sin)
        x = norm(x)

        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)

        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


if __name__ == "__main__":
    # config = NanoChatConfig(sequence_len=128, vocab_size=256, n_layer=2, n_head=4, n_kv_head=4, n_embd=128)
    # model = NanoChatModel(config)
    # ids = torch.randint(0,256, (2,16))
    # out = model(ids, labels=ids)
    # print(f"loss = {out.loss.item():.4f}")
    # print(f"logits shape = {tuple(out.logits.shape)}")

    config = NanoChatConfig(sequence_len=256, vocab_size=32768, n_layer=4, n_head=4, n_kv_head=4, n_embd=256)
    model = NanoChatModel(config)

    ids= torch.randint(0, config.vocab_size, (2, 64)) #2 batches, 64 token sequence
    output=model(ids)
    print("Forward pass: logits shape =", output.logits.shape)

    labels = torch.randint(0, config.vocab_size, (2, 64))
    output=model(ids, labels=labels)
    print(f"Loss: {output.loss.item():.4f}")

    prompt=torch.randint(0, config.vocab_size, (1, 10))
    generated=model.generate(prompt, max_new_tokens=20, do_sample=True, temperature=0.8, top_k=50)
    print(f"Generated: {prompt.shape} -> {generated.shape}")

    model.save_pretrained("/tmp/hf-nanochat")
    loaded=NanoChatModel.from_pretrained("/tmp/hf-nanochat")
    print(f"Saved and reloaded: {loaded.num_parameters():,}")

