import pytest
import torch

from recant.model.config import TextConfig
from recant.model.qwen35 import Qwen35, load_hf, save_hf_layout


def tiny_model(seed=0, layers=4, like_2b=False):
    torch.manual_seed(seed)
    cfg = TextConfig.tiny(layers=layers, like_2b=like_2b)
    m = Qwen35(cfg, device="cpu", dtype=torch.float32)
    m.init_random(std=0.05, seed=seed)
    return m


def _hf_match(m):
    pytest.importorskip("transformers")
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    c = m.c
    hf_cfg = Qwen3_5TextConfig(
        vocab_size=c.vocab_size, hidden_size=c.hidden_size, intermediate_size=c.intermediate_size,
        num_hidden_layers=c.num_hidden_layers, num_attention_heads=c.num_attention_heads,
        num_key_value_heads=c.num_key_value_heads, head_dim=c.head_dim, rms_norm_eps=c.rms_norm_eps,
        linear_conv_kernel_dim=c.linear_conv_kernel_dim, linear_key_head_dim=c.linear_key_head_dim,
        linear_value_head_dim=c.linear_value_head_dim, linear_num_key_heads=c.linear_num_key_heads,
        linear_num_value_heads=c.linear_num_value_heads, layer_types=c.layer_types,
        rope_parameters={"rope_type": "default", "rope_theta": c.rope_theta,
                         "partial_rotary_factor": c.partial_rotary_factor, "mrope_section": [3, 2, 2],
                         "mrope_interleaved": True},
        max_position_embeddings=4096, attention_bias=False, tie_word_embeddings=c.tie_word_embeddings)
    hf = Qwen3_5ForCausalLM(hf_cfg).eval().float()
    sd = {}
    for n, p in m.named_parameters():
        if n.startswith("layers."):
            sd["model." + n] = p.detach().clone()
    sd["model.embed_tokens.weight"] = m.embed_tokens.weight.detach().clone()
    sd["model.norm.weight"] = m.norm.weight.detach().clone()
    if not c.tie_word_embeddings:
        sd["lm_head.weight"] = m.lm_head.weight.detach().clone()
    missing, unexpected = hf.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert not [k for k in missing if "rotary" not in k and not (c.tie_word_embeddings and k == "lm_head.weight")], missing
    ids = torch.randint(0, c.vocab_size, (1, 37))
    with torch.no_grad():
        want = hf(input_ids=ids).logits
        got = m.forward_logits(ids)
    assert torch.allclose(got, want, atol=2e-4, rtol=1e-3), (got - want).abs().max()


def test_matches_hf_reference():
    """Logits equal transformers' Qwen3_5ForCausalLM with the same random weights (9B-style layout)."""
    _hf_match(tiny_model())


def test_matches_hf_reference_2b_layout():
    """Same, for the 2B/0.8B layout: value heads == key heads (no repeat), GQA 4:1, tied embeddings."""
    m = tiny_model(like_2b=True)
    assert m.lm_head.weight is m.embed_tokens.weight
    _hf_match(m)


def test_chunked_cached_forward_equals_full():
    m = tiny_model()
    ids = torch.randint(0, m.c.vocab_size, (1, 61))
    with torch.no_grad():
        full = m.run_layers(m.embed(ids))
        cache = m.new_cache()
        outs, pos = [], 0
        for n in (7, 20, 1, 33):
            outs.append(m.run_layers(m.embed(ids[:, pos:pos + n]), cache, pos))
            pos += n
    assert torch.allclose(torch.cat(outs, 1), full, atol=1e-4, rtol=1e-3), (torch.cat(outs, 1) - full).abs().max()


def test_fork_leaves_base_cache_untouched():
    m = tiny_model()
    ids = torch.randint(0, m.c.vocab_size, (1, 40))
    with torch.no_grad():
        cache = m.new_cache()
        m.run_layers(m.embed(ids[:, :25]), cache, 0)
        fork = cache.fork()
        m.run_layers(m.embed(torch.randint(0, m.c.vocab_size, (1, 9))), fork, 25)   # side branch
        rest = m.run_layers(m.embed(ids[:, 25:]), cache, 25)                          # main continues
        full = m.run_layers(m.embed(ids))[:, 25:]
    assert torch.allclose(rest, full, atol=1e-4, rtol=1e-3)


def test_checkpoint_roundtrip_and_fp4_backbone(tmp_path):
    m = tiny_model()
    save_hf_layout(m, tmp_path / "ck")
    m2 = load_hf(tmp_path / "ck", device="cpu", dtype=torch.float32, quantize=False)
    ids = torch.randint(0, m.c.vocab_size, (1, 24))
    assert torch.allclose(m.forward_logits(ids), m2.forward_logits(ids), atol=1e-6)
    # NVFP4 backbone: close but not equal (FP4 noise); LoRA at init (B=0) changes nothing
    q = load_hf(tmp_path / "ck", device="cpu", dtype=torch.float32, quantize=True)
    n_fp4 = sum(1 for mod in q.modules() if getattr(mod, "fp4", None) is not None)
    assert n_fp4 > 0
    a, b = m.forward_logits(ids), q.forward_logits(ids)
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    assert cos > 0.9, cos


def test_lora_only_trainable_and_starts_as_identity():
    m = tiny_model()
    ids = torch.randint(0, m.c.vocab_size, (1, 16))
    before = m.forward_logits(ids)
    n = m.add_lora(r=8, alpha=8)
    assert n > 0 and all(p.requires_grad for p in m.lora_parameters())
    assert all(not p.requires_grad for nme, p in m.named_parameters() if "lora_" not in nme)
    assert torch.allclose(m.forward_logits(ids), before, atol=1e-6)
    assert not any("in_proj_a" in k or "in_proj_b" in k for k in m.lora_state_dict())


def test_quantize_chunking_is_exact_and_bounded():
    from recant.model import nvfp4 as q

    torch.manual_seed(0)
    x = torch.randn(1000, 96) * 2
    whole = q.quantize(x, rows_per_chunk=1 << 30)
    parts = q.quantize(x, rows_per_chunk=96 * 7)          # 7 rows per chunk
    assert torch.equal(whole.codes, parts.codes) and torch.equal(whole.scales.view(torch.uint8), parts.scales.view(torch.uint8))
    assert torch.equal(whole.dequant(rows_per_chunk=1 << 30), parts.dequant(rows_per_chunk=96 * 5))


def test_stochastic_rounding_is_unbiased():
    from recant.model import nvfp4 as q

    v = torch.full((1, 16), 0.3)
    v[0, 0] = 6.0                                           # pins the block scale to 1
    acc = sum(q.quantize(v, stochastic=True).dequant() for _ in range(20000)) / 20000
    assert abs(acc[0, 1].item() - 0.3) < 0.01               # RTN would give 0.5


def test_ssm_params_keep_fp32(tmp_path):
    m = tiny_model()
    m.to(torch.bfloat16) if False else None
    save_hf_layout(m, tmp_path / "ck")
    q = load_hf(tmp_path / "ck", device="cpu", dtype=torch.bfloat16, quantize=False)
    for layer in q.layers:
        if layer.kind == "linear_attention":
            assert layer.linear_attn.A_log.dtype == torch.float32 and layer.linear_attn.dt_bias.dtype == torch.float32
            assert layer.linear_attn.in_proj_qkv.weight.dtype == torch.bfloat16


def test_cached_attention_blocking_matches_unblocked():
    from recant.model import ops

    torch.manual_seed(0)
    q = torch.randn(1, 4, 30, 16)
    k = torch.randn(1, 2, 80, 16)
    v = torch.randn(1, 2, 80, 16)
    a = ops.attn_cached(q, k, v, block=7)
    b = ops.attn_cached(q, k, v, block=1000)
    assert torch.allclose(a, b, atol=1e-5)
    old = ops.SCORE_BUDGET_BYTES
    ops.SCORE_BUDGET_BYTES = 4 * 4 * 80 * 64          # forces the 64-row minimum block
    try:
        assert torch.allclose(ops.attn_cached(q, k, v), b, atol=1e-5)
    finally:
        ops.SCORE_BUDGET_BYTES = old


def test_tied_checkpoint_roundtrip_and_fp4(tmp_path):
    m = tiny_model(like_2b=True)
    save_hf_layout(m, tmp_path / "ck")
    from safetensors import safe_open

    with safe_open(tmp_path / "ck" / "model.safetensors", "pt") as f:
        assert "lm_head.weight" not in set(f.keys())          # like the real 2B checkpoint
    for quant in (False, True):
        q = load_hf(tmp_path / "ck", device="cpu", dtype=torch.float32, quantize=quant)
        assert q.c.tie_word_embeddings and q.lm_head.weight is q.embed_tokens.weight
        ids = torch.randint(0, q.c.vocab_size, (1, 24))
        a, b = m.forward_logits(ids), q.forward_logits(ids)
        if not quant:
            assert torch.allclose(a, b, atol=1e-6)
        else:
            assert torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0) > 0.9


def test_real_2b_config_structure():
    """The actual Qwen3.5-2B config: 18 GDN + 6 attention layers, and every large linear is FP4-quantizable."""
    import json
    from pathlib import Path

    from recant.model.layers import DecoderLayer, QLinear

    cfg = TextConfig.from_hf(Path(__file__).parent / "qwen35_2b_config.json")
    assert (cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers) == (2048, 6144, 24)
    assert cfg.layer_types.count("full_attention") == 6 and cfg.layer_types.count("linear_attention") == 18
    assert cfg.tie_word_embeddings and cfg.rotary_dim == 64
    assert cfg.linear_num_key_heads == cfg.linear_num_value_heads == 16
    for i in (0, 3):                                           # one GDN layer, one attention layer
        with torch.device("meta"):
            layer = DecoderLayer(cfg, i)
        for name, mod in layer.named_modules():
            if isinstance(mod, QLinear):
                big = mod.out_features >= 256
                assert mod.quantizable == big, (i, name, mod.out_features)
                assert mod.in_features % 16 == 0 and mod.out_features % 16 == 0, name
