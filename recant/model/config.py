from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class TextConfig:
    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    rms_norm_eps: float = 1e-06
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    tie_word_embeddings: bool = False
    layer_types: list = field(default_factory=list)

    def __post_init__(self):
        if not self.layer_types:
            self.layer_types = ['linear_attention' if (i + 1) % 4 else 'full_attention' for i in range(self.num_hidden_layers)]

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def key_dim(self) -> int:
        return self.linear_key_head_dim * self.linear_num_key_heads

    @property
    def value_dim(self) -> int:
        return self.linear_value_head_dim * self.linear_num_value_heads

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    @classmethod
    def from_hf(cls, path_or_dict) -> 'TextConfig':
        d = path_or_dict
        if not isinstance(d, dict):
            p = Path(d)
            d = json.loads((p / 'config.json' if p.is_dir() else p).read_text())
        tie = d.get('tie_word_embeddings')
        d = d.get('text_config', d)
        if tie is None:
            tie = d.get('tie_word_embeddings', False)
        rope = d.get('rope_parameters') or {}
        names = set(cls.__dataclass_fields__) - {'rope_theta', 'partial_rotary_factor', 'tie_word_embeddings'}
        kw = {k: d[k] for k in names if k in d}
        kw['rope_theta'] = float(rope.get('rope_theta', d.get('rope_theta', 10000000.0)))
        kw['tie_word_embeddings'] = bool(tie)
        kw['partial_rotary_factor'] = float(rope.get('partial_rotary_factor', d.get('partial_rotary_factor', 0.25)))
        return cls(**kw)

    @classmethod
    def tiny(cls, vocab_size: int=300, layers: int=4, tie: bool=False, like_2b: bool=False) -> 'TextConfig':
        if like_2b:
            return cls(vocab_size=vocab_size, hidden_size=64, intermediate_size=192, num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=1, head_dim=32, linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=4, linear_num_value_heads=4, rope_theta=10000.0, tie_word_embeddings=True)
        return cls(vocab_size=vocab_size, tie_word_embeddings=tie, hidden_size=64, intermediate_size=128, num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, rope_theta=10000.0)