from transformers import PretrainedConfig


class NanoChatConfig(PretrainedConfig):
    model_type = "nanochat"

    def __init__(self, sequence_len=2048, vocab_size=32768, n_layer=12, n_embd=768, n_kv_head=6, n_head=6,
                 window_pattern='SSSL', **kwargs):

        self.n_layer = n_layer
        self.num_hidden_layers = n_layer
        self.n_embd = n_embd
        self.n_kv_head = n_kv_head
        self.n_head = n_head
        self.window_pattern = window_pattern
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size

        super().__init__(**kwargs)



