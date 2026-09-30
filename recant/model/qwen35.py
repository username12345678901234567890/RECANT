"""Qwen3.5 text model with a frozen (optionally NVFP4) backbone, LoRA, hidden-state taps.

The model exposes the pieces the RECANT trainer needs instead of a monolithic forward:
`embed`, `layers[i](x, cache, pos_offset)`, `final_norm`, `logits`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TextConfig
from .layers import DecoderLayer, QLinear, RMSNorm, SeqCache

LORA_EXCLUDE = ("in_proj_a", "in_proj_b")


class Qwen35(nn.Module):
    def __init__(self, c: TextConfig, device="cpu", dtype=torch.float32):
        super().__init__()
        self.c = c
        self.dtype = dtype
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size, device=device, dtype=dtype)
        self.embed_tokens.weight.requires_grad_(False)
        self.layers = nn.ModuleList()
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False, device=device, dtype=dtype)
        self.lm_head.weight.requires_grad_(False)
        if c.tie_word_embeddings:            # 0.8B / 2B / 4B share the embedding matrix with the LM head
            self.lm_head.weight = self.embed_tokens.weight
        self.device = device

    # ---- construction --------------------------------------------------------------
    def _new_layer(self, i: int) -> DecoderLayer:
        layer = DecoderLayer(self.c, i)
        layer.to(device=self.device, dtype=self.dtype)
        return layer

    @torch.no_grad()
    def init_random(self, std=0.02, seed=0):
        """Random weights for tests."""
        g = torch.Generator().manual_seed(seed)
        self.layers = nn.ModuleList(self._new_layer(i) for i in range(self.c.num_hidden_layers))
        for p in self.parameters():
            p.copy_((torch.randn(p.shape, generator=g) * std).to(p))
        for layer in self.layers:
            for m in layer.modules():
                if hasattr(m, "A_log"):
                    m.A_log.copy_(torch.log(torch.empty(m.A_log.shape).uniform_(0.5, 4.0, generator=g)))
                    m.dt_bias.copy_(torch.randn(m.dt_bias.shape, generator=g) * 0.5)

    def quantize_(self, use_rht=True):
        for layer in self.layers:
            for m in layer.modules():
                if isinstance(m, QLinear):
                    m.quantize_(use_rht)

    def add_lora(self, r=128, alpha=128.0):
        n = 0
        for name, m in self.named_modules():
            if isinstance(m, QLinear) and m.lora_ok and not name.endswith(LORA_EXCLUDE):
                m.add_lora(r, alpha)
                n += 1
        return n

    def lora_parameters(self):
        return [p for n, p in self.named_parameters() if "lora_" in n]

    def lora_state_dict(self) -> dict:
        return {n: p.detach() for n, p in self.named_parameters() if "lora_" in n}

    def load_lora_state_dict(self, sd: dict):
        own = dict(self.named_parameters())
        for n, t in sd.items():
            own[n].data.copy_(t.to(own[n]))

    # ---- forward pieces -------------------------------------------------------------
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(ids)

    def final_norm(self, x):
        return self.norm(x)

    def logits(self, h):
        return F.linear(h, self.lm_head.weight)

    def new_cache(self) -> SeqCache:
        return SeqCache.new(len(self.layers))

    def run_layers(self, x, cache: SeqCache | None = None, pos_offset: int = 0, record=None):
        """Run every layer on x [B,T,d]. If `record` is a list of preallocated [T_total,d]
        buffers, layer inputs are written at rows [pos_offset : pos_offset+T]."""
        t = x.shape[1]
        for i, layer in enumerate(self.layers):
            if record is not None:
                record[i][pos_offset:pos_offset + t] = x[0]
            x = layer(x, cache.layers[i] if cache is not None else None, pos_offset)
        if record is not None:
            record[len(self.layers)][pos_offset:pos_offset + t] = x[0]
        return x

    @torch.no_grad()
    def forward_logits(self, ids: torch.Tensor) -> torch.Tensor:
        """Convenience full forward (tests): ids [B,T] -> logits [B,T,V]."""
        x = self.run_layers(self.embed(ids))
        return self.logits(self.final_norm(x))


# --------------------------------------------------------------------------- HF loading
_PREFIXES = ("model.language_model.", "model.")


def _strip(name: str) -> str | None:
    for p in _PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    if name == "lm_head.weight":
        return name
    return None


def _shard_map(path: Path) -> dict[str, str]:
    idx = path / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    from safetensors import safe_open

    out = {}
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(f, "pt") as sf:
            out.update({k: f.name for k in sf.keys()})
    return out


def _assign(layer: DecoderLayer, tensors: dict[str, torch.Tensor]):
    own = dict(layer.named_parameters())
    for name, t in tensors.items():
        if name not in own:
            raise KeyError(f"layer {layer.idx}: unexpected checkpoint tensor {name}")
        keep_fp32 = name.endswith(("A_log", "dt_bias"))     # SSM decay params stay fp32 (precision-critical)
        own[name].data = t.to(own[name].device, torch.float32 if keep_fp32 else own[name].dtype).contiguous()
    missing = [n for n, p in own.items() if p.data.numel() and n not in tensors and "lora_" not in n]
    if missing:
        raise KeyError(f"layer {layer.idx}: checkpoint is missing {missing}")


@torch.no_grad()
def load_hf(path: str | Path, device="cuda", dtype=torch.bfloat16, quantize=True, use_rht=True,
            cfg: TextConfig | None = None, log=print) -> Qwen35:
    """Stream a HF Qwen3.5 checkpoint layer by layer: load -> (NVFP4-quantize) -> free bf16."""
    from safetensors import safe_open

    path = Path(path)
    cfg = cfg or TextConfig.from_hf(path)
    model = Qwen35(cfg, device=device, dtype=dtype)
    wmap = _shard_map(path)
    by_layer: dict[int, dict[str, str]] = {}
    other: dict[str, str] = {}
    for full, shard in wmap.items():
        s = _strip(full)
        if s is None or full.startswith(("mtp.", "model.visual")):
            continue
        m = re.match(r"layers\.(\d+)\.(.+)", s)
        if m:
            by_layer.setdefault(int(m.group(1)), {})[m.group(2)] = full
        else:
            other[s] = full
    handles: dict[str, object] = {}

    def get(full):
        shard = wmap[full]
        if shard not in handles:
            handles[shard] = safe_open(path / shard, "pt", device="cpu")
        return handles[shard].get_tensor(full)

    layers = []
    for i in range(cfg.num_hidden_layers):
        layer = model._new_layer(i)
        _assign(layer, {n: get(f) for n, f in by_layer[i].items()})
        if quantize:
            for m in layer.modules():
                if isinstance(m, QLinear):
                    m.quantize_(use_rht)
        layers.append(layer)
        if i % 8 == 0:
            log(f"[load] layer {i}/{cfg.num_hidden_layers}")
    model.layers = nn.ModuleList(layers)
    model.embed_tokens.weight.data = get(other["embed_tokens.weight"]).to(device, dtype)
    model.norm.weight.data = get(other["norm.weight"]).to(device, dtype)
    if "lm_head.weight" in other and not cfg.tie_word_embeddings:
        model.lm_head.weight.data = get(other["lm_head.weight"]).to(device, dtype)
    elif "lm_head.weight" in other:
        log("[load] tie_word_embeddings=true: ignoring the checkpoint's separate lm_head.weight")
    else:
        if not cfg.tie_word_embeddings:
            raise KeyError("checkpoint has no lm_head.weight but config.tie_word_embeddings is false")
        log("[load] tied embeddings: LM head shares the embedding matrix")
    model.lm_head.weight = model.embed_tokens.weight if cfg.tie_word_embeddings else model.lm_head.weight
    model.lm_head.weight.requires_grad_(False)
    return model


def save_hf_layout(model: Qwen35, path: str | Path):
    """Write a (dense) model in the HF Qwen3.5 checkpoint layout - used to build tiny test checkpoints."""
    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    sd = {}
    for n, p in model.named_parameters():
        if "lora_" in n:
            continue
        if n.startswith("layers."):
            sd["model.language_model." + n] = p.detach().cpu().contiguous()
    sd["model.language_model.embed_tokens.weight"] = model.embed_tokens.weight.detach().cpu().contiguous()
    sd["model.language_model.norm.weight"] = model.norm.weight.detach().cpu().contiguous()
    if not model.c.tie_word_embeddings:
        sd["lm_head.weight"] = model.lm_head.weight.detach().cpu().contiguous()
    save_file(sd, str(path / "model.safetensors"))
    c = model.c
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": {
        **{k: getattr(c, k) for k in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
           "num_attention_heads", "num_key_value_heads", "head_dim", "rms_norm_eps", "linear_conv_kernel_dim",
           "linear_key_head_dim", "linear_value_head_dim", "linear_num_key_heads", "linear_num_value_heads",
           "layer_types")},
        "tie_word_embeddings": c.tie_word_embeddings,
        "rope_parameters": {"rope_theta": c.rope_theta, "partial_rotary_factor": c.partial_rotary_factor}}}))
