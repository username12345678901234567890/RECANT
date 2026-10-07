import random
import torch
import triton
import triton.language as tl
from .nvfp4 import BLOCK, E4M3_MAX, FP4_MAX, FP4Tensor, _SIGNS
_EXPECTED_SIGNS = [1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1]
assert [int(v) for v in _SIGNS.tolist()] == _EXPECTED_SIGNS, 'nvfp4._SIGNS changed; regenerate fp4_kernels.py'

@triton.jit
def _to_in_dtype(v, IN_DT: tl.constexpr):
    if IN_DT == tl.bfloat16:
        bits = v.to(tl.int32, bitcast=True)
        bits = bits + 32767 + (bits >> 16 & 1) & -65536
        return bits.to(tl.float32, bitcast=True)
    elif IN_DT == tl.float16:
        return v.to(tl.float16).to(tl.float32)
    else:
        return v

@triton.jit
def _grid(i):
    return tl.where(i == 0, 0.0, tl.where(i == 1, 0.5, tl.where(i == 2, 1.0, tl.where(i == 3, 1.5, tl.where(i == 4, 2.0, tl.where(i == 5, 3.0, tl.where(i == 6, 4.0, 6.0)))))))

@triton.jit
def _round_code(q, u, SR: tl.constexpr):
    neg = q < 0.0
    a = tl.minimum(tl.abs(q), 6.0)
    if SR:
        hi = (a > 0.0).to(tl.int32) + (a > 0.5).to(tl.int32) + (a > 1.0).to(tl.int32) + (a > 1.5).to(tl.int32) + (a > 2.0).to(tl.int32) + (a > 3.0).to(tl.int32) + (a > 4.0).to(tl.int32)
        hi = tl.maximum(hi, 1)
        lo = hi - 1
        glo = _grid(lo)
        ghi = _grid(hi)
        p = (a - glo) / (ghi - glo)
        idx = tl.where(u < p, hi, lo)
    else:
        idx = (a > 0.25).to(tl.int32) + (a > 0.75).to(tl.int32) + (a > 1.25).to(tl.int32) + (a > 1.75).to(tl.int32) + (a > 2.5).to(tl.int32) + (a > 3.5).to(tl.int32) + (a > 5.0).to(tl.int32)
    return idx + 8 * neg.to(tl.int32)

@triton.jit
def _rht_blockmax_kernel(x_ptr, bmax_ptr, M, K, G, ROT: tl.constexpr, IN_DT: tl.constexpr, BM: tl.constexpr, BG: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)[:, None]
    grp = tl.program_id(1) * BG + tl.arange(0, BG)[None, :]
    mask = (rows < M) & (grp < G)
    base = x_ptr + rows.to(tl.int64) * K + grp * 16
    v0 = tl.load(base + 0, mask=mask, other=0.0).to(tl.float32)
    v1 = tl.load(base + 1, mask=mask, other=0.0).to(tl.float32)
    v2 = tl.load(base + 2, mask=mask, other=0.0).to(tl.float32)
    v3 = tl.load(base + 3, mask=mask, other=0.0).to(tl.float32)
    v4 = tl.load(base + 4, mask=mask, other=0.0).to(tl.float32)
    v5 = tl.load(base + 5, mask=mask, other=0.0).to(tl.float32)
    v6 = tl.load(base + 6, mask=mask, other=0.0).to(tl.float32)
    v7 = tl.load(base + 7, mask=mask, other=0.0).to(tl.float32)
    v8 = tl.load(base + 8, mask=mask, other=0.0).to(tl.float32)
    v9 = tl.load(base + 9, mask=mask, other=0.0).to(tl.float32)
    v10 = tl.load(base + 10, mask=mask, other=0.0).to(tl.float32)
    v11 = tl.load(base + 11, mask=mask, other=0.0).to(tl.float32)
    v12 = tl.load(base + 12, mask=mask, other=0.0).to(tl.float32)
    v13 = tl.load(base + 13, mask=mask, other=0.0).to(tl.float32)
    v14 = tl.load(base + 14, mask=mask, other=0.0).to(tl.float32)
    v15 = tl.load(base + 15, mask=mask, other=0.0).to(tl.float32)
    if ROT:
        v2 = -v2
        v4 = -v4
        v5 = -v5
        v6 = -v6
        v12 = -v12
        v13 = -v13
        v15 = -v15
        ta = v0 + v1
        tb = v0 - v1
        v0 = ta
        v1 = tb
        ta = v2 + v3
        tb = v2 - v3
        v2 = ta
        v3 = tb
        ta = v4 + v5
        tb = v4 - v5
        v4 = ta
        v5 = tb
        ta = v6 + v7
        tb = v6 - v7
        v6 = ta
        v7 = tb
        ta = v8 + v9
        tb = v8 - v9
        v8 = ta
        v9 = tb
        ta = v10 + v11
        tb = v10 - v11
        v10 = ta
        v11 = tb
        ta = v12 + v13
        tb = v12 - v13
        v12 = ta
        v13 = tb
        ta = v14 + v15
        tb = v14 - v15
        v14 = ta
        v15 = tb
        ta = v0 + v2
        tb = v0 - v2
        v0 = ta
        v2 = tb
        ta = v1 + v3
        tb = v1 - v3
        v1 = ta
        v3 = tb
        ta = v4 + v6
        tb = v4 - v6
        v4 = ta
        v6 = tb
        ta = v5 + v7
        tb = v5 - v7
        v5 = ta
        v7 = tb
        ta = v8 + v10
        tb = v8 - v10
        v8 = ta
        v10 = tb
        ta = v9 + v11
        tb = v9 - v11
        v9 = ta
        v11 = tb
        ta = v12 + v14
        tb = v12 - v14
        v12 = ta
        v14 = tb
        ta = v13 + v15
        tb = v13 - v15
        v13 = ta
        v15 = tb
        ta = v0 + v4
        tb = v0 - v4
        v0 = ta
        v4 = tb
        ta = v1 + v5
        tb = v1 - v5
        v1 = ta
        v5 = tb
        ta = v2 + v6
        tb = v2 - v6
        v2 = ta
        v6 = tb
        ta = v3 + v7
        tb = v3 - v7
        v3 = ta
        v7 = tb
        ta = v8 + v12
        tb = v8 - v12
        v8 = ta
        v12 = tb
        ta = v9 + v13
        tb = v9 - v13
        v9 = ta
        v13 = tb
        ta = v10 + v14
        tb = v10 - v14
        v10 = ta
        v14 = tb
        ta = v11 + v15
        tb = v11 - v15
        v11 = ta
        v15 = tb
        ta = v0 + v8
        tb = v0 - v8
        v0 = ta
        v8 = tb
        ta = v1 + v9
        tb = v1 - v9
        v1 = ta
        v9 = tb
        ta = v2 + v10
        tb = v2 - v10
        v2 = ta
        v10 = tb
        ta = v3 + v11
        tb = v3 - v11
        v3 = ta
        v11 = tb
        ta = v4 + v12
        tb = v4 - v12
        v4 = ta
        v12 = tb
        ta = v5 + v13
        tb = v5 - v13
        v5 = ta
        v13 = tb
        ta = v6 + v14
        tb = v6 - v14
        v6 = ta
        v14 = tb
        ta = v7 + v15
        tb = v7 - v15
        v7 = ta
        v15 = tb
        v0 = v0 * 0.25
        v1 = v1 * 0.25
        v2 = v2 * 0.25
        v3 = v3 * 0.25
        v4 = v4 * 0.25
        v5 = v5 * 0.25
        v6 = v6 * 0.25
        v7 = v7 * 0.25
        v8 = v8 * 0.25
        v9 = v9 * 0.25
        v10 = v10 * 0.25
        v11 = v11 * 0.25
        v12 = v12 * 0.25
        v13 = v13 * 0.25
        v14 = v14 * 0.25
        v15 = v15 * 0.25
    v0 = _to_in_dtype(v0, IN_DT)
    v1 = _to_in_dtype(v1, IN_DT)
    v2 = _to_in_dtype(v2, IN_DT)
    v3 = _to_in_dtype(v3, IN_DT)
    v4 = _to_in_dtype(v4, IN_DT)
    v5 = _to_in_dtype(v5, IN_DT)
    v6 = _to_in_dtype(v6, IN_DT)
    v7 = _to_in_dtype(v7, IN_DT)
    v8 = _to_in_dtype(v8, IN_DT)
    v9 = _to_in_dtype(v9, IN_DT)
    v10 = _to_in_dtype(v10, IN_DT)
    v11 = _to_in_dtype(v11, IN_DT)
    v12 = _to_in_dtype(v12, IN_DT)
    v13 = _to_in_dtype(v13, IN_DT)
    v14 = _to_in_dtype(v14, IN_DT)
    v15 = _to_in_dtype(v15, IN_DT)
    amax = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.abs(v0), tl.abs(v1)), tl.abs(v2)), tl.abs(v3)), tl.abs(v4)), tl.abs(v5)), tl.abs(v6)), tl.abs(v7)), tl.abs(v8)), tl.abs(v9)), tl.abs(v10)), tl.abs(v11)), tl.abs(v12)), tl.abs(v13)), tl.abs(v14)), tl.abs(v15))
    tl.store(bmax_ptr + rows * G + grp, amax, mask=mask)

@triton.jit
def _quant_kernel(x_ptr, denom_ptr, codes_ptr, seed, M, K, G, ROT: tl.constexpr, SR: tl.constexpr, IN_DT: tl.constexpr, BM: tl.constexpr, BG: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)[:, None]
    grp = tl.program_id(1) * BG + tl.arange(0, BG)[None, :]
    mask = (rows < M) & (grp < G)
    base = x_ptr + rows.to(tl.int64) * K + grp * 16
    v0 = tl.load(base + 0, mask=mask, other=0.0).to(tl.float32)
    v1 = tl.load(base + 1, mask=mask, other=0.0).to(tl.float32)
    v2 = tl.load(base + 2, mask=mask, other=0.0).to(tl.float32)
    v3 = tl.load(base + 3, mask=mask, other=0.0).to(tl.float32)
    v4 = tl.load(base + 4, mask=mask, other=0.0).to(tl.float32)
    v5 = tl.load(base + 5, mask=mask, other=0.0).to(tl.float32)
    v6 = tl.load(base + 6, mask=mask, other=0.0).to(tl.float32)
    v7 = tl.load(base + 7, mask=mask, other=0.0).to(tl.float32)
    v8 = tl.load(base + 8, mask=mask, other=0.0).to(tl.float32)
    v9 = tl.load(base + 9, mask=mask, other=0.0).to(tl.float32)
    v10 = tl.load(base + 10, mask=mask, other=0.0).to(tl.float32)
    v11 = tl.load(base + 11, mask=mask, other=0.0).to(tl.float32)
    v12 = tl.load(base + 12, mask=mask, other=0.0).to(tl.float32)
    v13 = tl.load(base + 13, mask=mask, other=0.0).to(tl.float32)
    v14 = tl.load(base + 14, mask=mask, other=0.0).to(tl.float32)
    v15 = tl.load(base + 15, mask=mask, other=0.0).to(tl.float32)
    if ROT:
        v2 = -v2
        v4 = -v4
        v5 = -v5
        v6 = -v6
        v12 = -v12
        v13 = -v13
        v15 = -v15
        ta = v0 + v1
        tb = v0 - v1
        v0 = ta
        v1 = tb
        ta = v2 + v3
        tb = v2 - v3
        v2 = ta
        v3 = tb
        ta = v4 + v5
        tb = v4 - v5
        v4 = ta
        v5 = tb
        ta = v6 + v7
        tb = v6 - v7
        v6 = ta
        v7 = tb
        ta = v8 + v9
        tb = v8 - v9
        v8 = ta
        v9 = tb
        ta = v10 + v11
        tb = v10 - v11
        v10 = ta
        v11 = tb
        ta = v12 + v13
        tb = v12 - v13
        v12 = ta
        v13 = tb
        ta = v14 + v15
        tb = v14 - v15
        v14 = ta
        v15 = tb
        ta = v0 + v2
        tb = v0 - v2
        v0 = ta
        v2 = tb
        ta = v1 + v3
        tb = v1 - v3
        v1 = ta
        v3 = tb
        ta = v4 + v6
        tb = v4 - v6
        v4 = ta
        v6 = tb
        ta = v5 + v7
        tb = v5 - v7
        v5 = ta
        v7 = tb
        ta = v8 + v10
        tb = v8 - v10
        v8 = ta
        v10 = tb
        ta = v9 + v11
        tb = v9 - v11
        v9 = ta
        v11 = tb
        ta = v12 + v14
        tb = v12 - v14
        v12 = ta
        v14 = tb
        ta = v13 + v15
        tb = v13 - v15
        v13 = ta
        v15 = tb
        ta = v0 + v4
        tb = v0 - v4
        v0 = ta
        v4 = tb
        ta = v1 + v5
        tb = v1 - v5
        v1 = ta
        v5 = tb
        ta = v2 + v6
        tb = v2 - v6
        v2 = ta
        v6 = tb
        ta = v3 + v7
        tb = v3 - v7
        v3 = ta
        v7 = tb
        ta = v8 + v12
        tb = v8 - v12
        v8 = ta
        v12 = tb
        ta = v9 + v13
        tb = v9 - v13
        v9 = ta
        v13 = tb
        ta = v10 + v14
        tb = v10 - v14
        v10 = ta
        v14 = tb
        ta = v11 + v15
        tb = v11 - v15
        v11 = ta
        v15 = tb
        ta = v0 + v8
        tb = v0 - v8
        v0 = ta
        v8 = tb
        ta = v1 + v9
        tb = v1 - v9
        v1 = ta
        v9 = tb
        ta = v2 + v10
        tb = v2 - v10
        v2 = ta
        v10 = tb
        ta = v3 + v11
        tb = v3 - v11
        v3 = ta
        v11 = tb
        ta = v4 + v12
        tb = v4 - v12
        v4 = ta
        v12 = tb
        ta = v5 + v13
        tb = v5 - v13
        v5 = ta
        v13 = tb
        ta = v6 + v14
        tb = v6 - v14
        v6 = ta
        v14 = tb
        ta = v7 + v15
        tb = v7 - v15
        v7 = ta
        v15 = tb
        v0 = v0 * 0.25
        v1 = v1 * 0.25
        v2 = v2 * 0.25
        v3 = v3 * 0.25
        v4 = v4 * 0.25
        v5 = v5 * 0.25
        v6 = v6 * 0.25
        v7 = v7 * 0.25
        v8 = v8 * 0.25
        v9 = v9 * 0.25
        v10 = v10 * 0.25
        v11 = v11 * 0.25
        v12 = v12 * 0.25
        v13 = v13 * 0.25
        v14 = v14 * 0.25
        v15 = v15 * 0.25
    v0 = _to_in_dtype(v0, IN_DT)
    v1 = _to_in_dtype(v1, IN_DT)
    v2 = _to_in_dtype(v2, IN_DT)
    v3 = _to_in_dtype(v3, IN_DT)
    v4 = _to_in_dtype(v4, IN_DT)
    v5 = _to_in_dtype(v5, IN_DT)
    v6 = _to_in_dtype(v6, IN_DT)
    v7 = _to_in_dtype(v7, IN_DT)
    v8 = _to_in_dtype(v8, IN_DT)
    v9 = _to_in_dtype(v9, IN_DT)
    v10 = _to_in_dtype(v10, IN_DT)
    v11 = _to_in_dtype(v11, IN_DT)
    v12 = _to_in_dtype(v12, IN_DT)
    v13 = _to_in_dtype(v13, IN_DT)
    v14 = _to_in_dtype(v14, IN_DT)
    v15 = _to_in_dtype(v15, IN_DT)
    denom = tl.load(denom_ptr + rows * G + grp, mask=mask, other=1.0)
    eoff = rows * K + grp * 16
    if True:
        c0 = _round_code(tl.math.div_rn(v0, denom), tl.rand(seed, eoff + 0), SR)
        c1 = _round_code(tl.math.div_rn(v1, denom), tl.rand(seed, eoff + 1), SR)
        c2 = _round_code(tl.math.div_rn(v2, denom), tl.rand(seed, eoff + 2), SR)
        c3 = _round_code(tl.math.div_rn(v3, denom), tl.rand(seed, eoff + 3), SR)
        c4 = _round_code(tl.math.div_rn(v4, denom), tl.rand(seed, eoff + 4), SR)
        c5 = _round_code(tl.math.div_rn(v5, denom), tl.rand(seed, eoff + 5), SR)
        c6 = _round_code(tl.math.div_rn(v6, denom), tl.rand(seed, eoff + 6), SR)
        c7 = _round_code(tl.math.div_rn(v7, denom), tl.rand(seed, eoff + 7), SR)
        c8 = _round_code(tl.math.div_rn(v8, denom), tl.rand(seed, eoff + 8), SR)
        c9 = _round_code(tl.math.div_rn(v9, denom), tl.rand(seed, eoff + 9), SR)
        c10 = _round_code(tl.math.div_rn(v10, denom), tl.rand(seed, eoff + 10), SR)
        c11 = _round_code(tl.math.div_rn(v11, denom), tl.rand(seed, eoff + 11), SR)
        c12 = _round_code(tl.math.div_rn(v12, denom), tl.rand(seed, eoff + 12), SR)
        c13 = _round_code(tl.math.div_rn(v13, denom), tl.rand(seed, eoff + 13), SR)
        c14 = _round_code(tl.math.div_rn(v14, denom), tl.rand(seed, eoff + 14), SR)
        c15 = _round_code(tl.math.div_rn(v15, denom), tl.rand(seed, eoff + 15), SR)
        tl.store(codes_ptr + rows * (K // 2) + grp * 8 + 0, (c0 | c1 << 4).to(tl.uint8), mask=mask)
        tl.store(codes_ptr + rows * (K // 2) + grp * 8 + 1, (c2 | c3 << 4).to(tl.uint8), mask=mask)
        tl.store(codes_ptr + rows * (K // 2) + grp * 8 + 2, (c4 | c5 << 4).to(tl.uint8), mask=mask)
        tl.store(codes_ptr + rows * (K // 2) + grp * 8 + 3, (c6 | c7 << 4).to(tl.uint8), mask=mask)
        tl.store(codes_ptr + rows * (K // 2) + grp * 8 + 4, (c8 | c9 << 4).to(tl.uint8), mask=mask)
        tl.store(codes_ptr + rows * (K // 2) + grp * 8 + 5, (c10 | c11 << 4).to(tl.uint8), mask=mask)
        tl.store(codes_ptr + rows * (K // 2) + grp * 8 + 6, (c12 | c13 << 4).to(tl.uint8), mask=mask)
        tl.store(codes_ptr + rows * (K // 2) + grp * 8 + 7, (c14 | c15 << 4).to(tl.uint8), mask=mask)
_TL_DTYPE = {torch.float32: tl.float32, torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}

def quantize_fast(x: torch.Tensor, use_rht: bool=True, stochastic: bool=False, BM: int=8, BG: int=64) -> FP4Tensor:
    m, k = x.shape
    g = k // BLOCK
    bmax = torch.empty(m, g, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(m, BM), triton.cdiv(g, BG))
    dt = _TL_DTYPE[x.dtype]
    _rht_blockmax_kernel[grid](x, bmax, m, k, g, ROT=use_rht, IN_DT=dt, BM=BM, BG=BG, num_warps=4)
    sg = (bmax.max().clamp(min=1e-30) / (FP4_MAX * E4M3_MAX)).reshape(1)
    s_b = (bmax / FP4_MAX / sg).clamp(max=E4M3_MAX)
    scales = s_b.to(torch.float8_e4m3fn)
    sq = scales.float()
    denom = (torch.where(sq == 0, torch.ones_like(sq), sq) * sg).contiguous()
    codes = torch.empty(m, k // 2, dtype=torch.uint8, device=x.device)
    seed = random.getrandbits(30) if stochastic else 0
    _quant_kernel[grid](x, denom, codes, seed, m, k, g, ROT=use_rht, SR=stochastic, IN_DT=dt, BM=BM, BG=BG, num_warps=4)
    return FP4Tensor(codes, scales, sg.reshape(()).clone(), (m, k))