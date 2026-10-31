// FAST lever ds41-attn-f16wmma-v1 (STRIX_DS41_FAST_ATTN=1, default off; prefill only). NOT bit-exact: a new declared
// arithmetic, qualified behind the FAST profile. Included by src/hip_ds41_attention.hip inside its anonymous namespace.
//
// One block per (key chunk of 64, 16 heads, row), 8 waves, in two tiles of 32 keys:
// - Q: the row's 16 heads after RoPE (FP32), rounded once to f16. Keys: the dequantised K/V values in f16, compressed
//   entries times 2^-7 (exact), a window key times 2^-ek (ek > 0 only when its largest E8M0 scale is above 134); the
//   factor is undone in FP32 on its scores and folded into its probabilities, so nothing overflows.
// - QK^T on WMMA f16 x f16 -> f32 (the 512 dims split into four quarters over the waves, quarters summed in a fixed tree).
// - Softmax statistics in FP32 (scale, max, exp_f, row sums in a fixed lane tree), online over the two tiles with the
//   ordered rescale of acc; the probabilities rounded once to f16.
// - PV on WMMA f16 x f16 -> f32. The chunk's (m, l, acc) partial and the combine are today's (FP32, chunk order, sink,
//   1 / L, inverse RoPE, A8 of o).
// Keys sit in LDS as pairs (dword = key 2p and 2p + 1 at one dim), 16-byte vectors XOR-swizzled by the pair, so the QK
// operand (16 dims of one key per lane) and the PV operand (16 keys of one dim per lane) both read without conflicts.
constexpr int kFastPScale = 15;   // f16 probabilities carry 2^15 (see the PV comment in the kernel)
constexpr unsigned kFastHeads = 16, kFastKeys = 32, kFastPairs = kFastKeys / 2, kFastQStride = kHeadDim / 2 + 4;
using fhalf16 = _Float16 __attribute__((ext_vector_type(16)));
using ffloat8 = float __attribute__((ext_vector_type(8)));
using fhalf2 = _Float16 __attribute__((ext_vector_type(2)));
using fu8 = unsigned __attribute__((ext_vector_type(8)));

// dword index of (pair p, dim d) in the key tile
__device__ __forceinline__ unsigned fast_kslot(unsigned p, unsigned d) { return p * kHeadDim + (((d >> 2U) ^ (p & 15U)) << 2U) + (d & 3U); }
__device__ __forceinline__ unsigned fast_pk_mul(unsigned x, unsigned f) {
    return __builtin_bit_cast(unsigned, __builtin_bit_cast(fhalf2, x) * __builtin_bit_cast(fhalf2, f));
}
// f16 bits of 2^e for e in [-14, 15], in both halves
__device__ __forceinline__ unsigned fast_pow2x2(int e) { const unsigned b = static_cast<unsigned>(e + 15) << 10U; return b | (b << 16U); }
// (byte d of a, byte d of b) as E4M3 -> f16 pair, times 2^-8 (exact, subnormals included)
__device__ __forceinline__ unsigned fast_e4m3x2(unsigned a, unsigned b, unsigned d) {
    const unsigned x = __builtin_amdgcn_perm(b, a, 0x0c000c00U | ((4U + d) << 16U) | d);
    return ((x & 0x007f007fU) << 7U) | ((x & 0x00800080U) << 8U);
}
// (nibble i of a, nibble i of b) as E2M1 -> f16 pair, times 2^-14 (exact)
__device__ __forceinline__ unsigned fast_e2m1x2(unsigned a, unsigned b, unsigned i) {
    const unsigned x = ((a >> (4U * i)) & 15U) | (((b >> (4U * i)) & 15U) << 16U);
    return ((x & 0x00070007U) << 9U) | ((x & 0x00080008U) << 12U);
}
// (byte k of a, byte k of b) -> their low (j = 0) or high (j = 1) E2M1 nibbles as an f16 pair times 2^-14 (exact)
__device__ __forceinline__ unsigned fast_e2m1x2_byte(unsigned a, unsigned b, unsigned k, unsigned j) {
    const unsigned x = __builtin_amdgcn_perm(b, a, 0x0c000c00U | ((4U + k) << 16U) | k);   // [a.k, 0, b.k, 0]
    return j == 0U ? ((x & 0x00070007U) << 9U) | ((x & 0x00080008U) << 12U) : ((x & 0x00700070U) << 5U) | ((x & 0x00800080U) << 8U);
}
// The thread's 32 dims (32 part ..) of keys a = 2p, b = 2p + 1 into the tile as pairs. ek: a window key's exponent
// shift. Stored values: window keys times 2^-ek, compressed keys times 2^-7 (kfac undoes it on the scores; the PV
// passes go by exponent band, see the kernel).
// The three key-pair kinds take separate branches (uniform across a pair's 16 threads; a wave holds two pairs).
__device__ __forceinline__ void fast_store_pair(unsigned* kt, unsigned p, unsigned part, const KeyRaw& ka, const KeyRaw& kb, bool wa,
                                                bool wb, bool va, bool vb, int eka, int ekb) {
    unsigned out[32];
    const std::uint32_t ca[8] = {ka.c0.x, ka.c0.y, ka.c0.z, ka.c0.w, ka.c1.x, ka.c1.y, ka.c1.z, ka.c1.w};
    const std::uint32_t cb[8] = {kb.c0.x, kb.c0.y, kb.c0.z, kb.c0.w, kb.c1.x, kb.c1.y, kb.c1.z, kb.c1.w};
    // compressed: per half of 16 dims, the E4M3 scale times 2^7 as f16 (exact: at most 448 x 128 = 57344)
    const auto cscale = [](const KeyRaw& k, unsigned hh) {
        const unsigned s = (k.sc >> (8U * hh)) & 0xFFU;
        return fast_pk_mul(((s & 0x7FU) << 7U) | ((s & 0x80U) << 8U), 0x7800U) & 0xFFFFU;   // (s x 2^-8) x 2^15
    };
    // window: 2^(b - 119 - ek) in one or two f16 powers of two (the E4M3 bit form is the value times 2^-8)
    const auto wscale = [](const KeyRaw& k, int ek, unsigned& g1, unsigned& g2) {
        const int e = static_cast<int>(k.sc & 0xFFU) - 119 - ek, e1 = e < -14 ? -14 : e > 15 ? 15 : e;
        const int r = e - e1, e2 = r < -14 ? -14 : r > 15 ? 15 : r;
        g1 = fast_pow2x2(e1) & 0xFFFFU; g2 = fast_pow2x2(e2) & 0xFFFFU;
    };
    if (!wa && !wb) {
#pragma unroll
        for (unsigned hh = 0; hh < 2; ++hh) {
            const unsigned f = cscale(ka, hh) | (cscale(kb, hh) << 16U);
#pragma unroll
            for (unsigned i = 16 * hh; i < 16 * hh + 16; i += 2) {   // dims i, i + 1: byte (i % 8) / 2 of word i / 8
                out[i] = fast_pk_mul(fast_e2m1x2_byte(ca[i / 8], cb[i / 8], (i % 8) / 2, 0U), f);
                out[i + 1] = fast_pk_mul(fast_e2m1x2_byte(ca[i / 8], cb[i / 8], (i % 8) / 2, 1U), f);
            }
        }
    } else if (wa && wb) {
        unsigned a1, a2, b1, b2;
        wscale(ka, eka, a1, a2);
        wscale(kb, ekb, b1, b2);
        const unsigned f1 = a1 | (b1 << 16U), f2 = a2 | (b2 << 16U);
#pragma unroll
        for (unsigned i = 0; i < 32; ++i) out[i] = fast_pk_mul(fast_pk_mul(fast_e4m3x2(ca[i / 4], cb[i / 4], i % 4), f1), f2);
    } else {   // a window key next to a compressed one (the window / selected boundary)
        unsigned a1 = 0, a2 = 0, b1 = 0, b2 = 0;
        if (wa) wscale(ka, eka, a1, a2);
        if (wb) wscale(kb, ekb, b1, b2);
#pragma unroll
        for (unsigned i = 0; i < 32; ++i) {
            const unsigned ga1 = wa ? a1 : cscale(ka, i / 16), ga2 = wa ? a2 : 0x3C00U, gb1 = wb ? b1 : cscale(kb, i / 16), gb2 = wb ? b2 : 0x3C00U;
            const unsigned xa = wa ? fast_e4m3x2(ca[i / 4], 0U, i % 4) : fast_e2m1x2(ca[i / 8], 0U, i % 8);
            const unsigned xb = wb ? fast_e4m3x2(cb[i / 4], 0U, i % 4) : fast_e2m1x2(cb[i / 8], 0U, i % 8);
            out[i] = fast_pk_mul(fast_pk_mul((xa & 0xFFFFU) | (xb << 16U), ga1 | (gb1 << 16U)), ga2 | (gb2 << 16U));
        }
    }
#pragma unroll
    for (unsigned i = 0; i < 32; ++i) {
        if (!va) out[i] &= 0xFFFF0000U;
        if (!vb) out[i] &= 0x0000FFFFU;
    }
#pragma unroll
    for (unsigned v4 = 0; v4 < 8; ++v4)
        *reinterpret_cast<uint4*>(kt + fast_kslot(p, 32U * part + 4U * v4)) = make_uint4(out[4 * v4], out[4 * v4 + 1], out[4 * v4 + 2], out[4 * v4 + 3]);
}

// Test-only ablations (benchmarks of where the time goes; results are then wrong): bit 0 no key loads, 1 no key
// dequantisation, 2 no QK products, 3 no PV products, 4 no Q loads.
__device__ unsigned g_fast_attn_ablate = 0U;

// Cpb chunks per block, one after the other: the next tile's key bytes are always in flight (also across a chunk
// boundary) and Q is staged once for all of them. The block's online softmax runs over all of its keys and writes ONE
// (m, l, acc) partial (slot blockIdx.x): the per-chunk partials and their combine reads were the largest cost (an
// ablation with no loads, dequantisation or products kept 2/3 of the time). The combine merges the blocks in order.
// H heads per block (16, or 32: each key is loaded and dequantised once for 32 heads; dequantisation was the next cost).
template <unsigned Cpb, unsigned H>
__global__ __launch_bounds__(256) void attention_f16wmma_kernel(AttentionView v) {
    DS41_TIMED(5U);
    static_assert(H == 16U || H == 32U);
    constexpr unsigned MT = H / 16U, HW = H / 8U;   // head row tiles; heads per wave in the softmax and the combine
    // the key tile (32 KB), or before the first tile the Q staging (H rows of kFastQStride dwords)
    constexpr unsigned kt_words = kFastPairs * kHeadDim > H * kFastQStride ? kFastPairs * kHeadDim : H * kFastQStride;
    __shared__ __attribute__((aligned(16))) unsigned kt[kt_words];
    __shared__ float spart[4][H][kFastKeys];
    __shared__ __attribute__((aligned(16))) float pf[H][kFastKeys];   // the tile's probabilities (FP32; banded tiles)
    __shared__ __attribute__((aligned(16))) unsigned pt[H][kFastPairs]; // the same as f16 pairs times 2^(x_k - hi) (one band)
    __shared__ int band_s[2];                                            // the tile's largest shift, and whether it needs bands
    __shared__ float scl[H], kfac[kFastKeys], qfac[H];
    __shared__ int kexp[kFastKeys];
    __shared__ std::uint32_t cidx[Cpb * kChunk];
    __shared__ unsigned last_s;
    const unsigned ablate = g_fast_attn_ablate;
    const unsigned r = blockIdx.z, tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U, lo = lane & 15U, half = lane >> 4U;
    const unsigned hb = blockIdx.y * H;
    unsigned nw;
    const unsigned n = visible_keys(v, r, nw), chunks = (n + kChunk - 1) / kChunk, blocks = (chunks + Cpb - 1) / Cpb;
    if (v.max_visible != 0U && n > v.max_visible) __builtin_trap();
    const unsigned c0 = blockIdx.x * Cpb;
    if (c0 >= chunks) return;
    const unsigned j0 = c0 * kChunk, end = n - j0 < Cpb * kChunk ? n : j0 + Cpb * kChunk;   // the block's keys [j0, end)
    const std::uint64_t p = v.pos[r];
    const unsigned first_slot = static_cast<unsigned>((p + 1U - nw) % v.capacity);
    for (unsigned t = tid; t < Cpb * kChunk; t += 256U)
        if (j0 + t < end && j0 + t >= nw) cidx[t] = v.idx[static_cast<std::size_t>(r) * v.topk + j0 + t - nw];
    // Q: wave w stages heads HW w .. (lane: 16 dims, RoPE in FP32) as f16 pairs, rows padded by 16 bytes
#pragma unroll
    for (unsigned hh = 0; hh < HW; ++hh) {
        const unsigned h = HW * wave + hh;
        float q[16];
        const float* src = v.q + (static_cast<std::size_t>(r) * v.heads + hb + h) * kHeadDim + 16 * lane;
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) q[i] = (ablate & 16U) != 0U ? 0.01F : src[i];
        rope512(q, v.rope[r], lane, false);
        // a head whose largest |q| is above 2^14 is stored times 2^-k (qfac = 2^k undoes it on its scores in FP32)
        float amax = 0.0F;
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(q[i]));
        amax = wave_max(amax);
        const int k = amax > 16384.0F && __builtin_isfinite(amax) ? log2_ceil(amax * (1.0F / 16384.0F)) : 0;
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) q[i] = ldexpf(q[i], -k);
        if (lane == 0U) qfac[h] = ldexpf(1.0F, k);
        unsigned w[8];
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) w[i] = __builtin_bit_cast(unsigned, fhalf2{static_cast<_Float16>(q[2 * i]), static_cast<_Float16>(q[2 * i + 1])});
        uint4* dst = reinterpret_cast<uint4*>(kt + h * kFastQStride + 8 * lane);
        dst[0] = make_uint4(w[0], w[1], w[2], w[3]);
        dst[1] = make_uint4(w[4], w[5], w[6], w[7]);
    }
    __syncthreads();
    // QK roles. H 16: key tile kq = wave & 1, dim quarter dq = wave >> 1. H 32: head tile mq = wave & 1 (both key tiles),
    // dim quarter dq = wave >> 1. A operands (the wave's head tile x dim quarter) in registers.
    const unsigned kq = wave & 1U, dq = wave >> 1U, mq = H == 32U ? (wave & 1U) : 0U;
    fhalf16 qa[8];
#pragma unroll
    for (unsigned s = 0; s < 8; ++s) {
        const uint4* src = reinterpret_cast<const uint4*>(kt + (16U * mq + lo) * kFastQStride + (128U * dq + 16U * s) / 2U);
        const uint4 x0 = src[0], x1 = src[1];
        qa[s] = __builtin_bit_cast(fhalf16, fu8{x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w});
    }
    // loads of a tile: thread -> pair tid / 16 (keys t0 + 2 pair, + 1), dims 32 part ..
    const unsigned pr = tid >> 4U, part = tid & 15U;
    const auto load = [&](unsigned j, KeyRaw& k) {
        k = KeyRaw{};
        if (j >= end || (ablate & 1U) != 0U) return;
        if (j < nw) {
            unsigned slot = first_slot + j;
            if (slot >= v.capacity) slot -= v.capacity;
            const std::uint8_t* e = v.ring + static_cast<std::size_t>(slot) * kWindowEntryBytes;
            const uint4* cp = reinterpret_cast<const uint4*>(e + 32 * part);
            k.c0 = cp[0]; k.c1 = cp[1];
            k.sc = e[512 + part];
        } else {
            const std::uint8_t* e = v.kv_cache + static_cast<std::size_t>(cidx[j - j0]) * kKvEntryBytes;
            k.c0 = *reinterpret_cast<const uint4*>(e + 16 * part);
            k.sc = *reinterpret_cast<const std::uint16_t*>(e + 256 + 2 * part);
        }
    };
    __syncthreads();   // cidx visible; every wave holds its Q operands (the tile may be overwritten)
    KeyRaw ra{}, rb{};
    load(j0 + 2U * pr, ra);
    load(j0 + 2U * pr + 1U, rb);
    float m[MT], l[MT];   // softmax roles: heads HW wave + 2 i + half
    ffloat8 acc[MT][4];
#pragma unroll
    for (unsigned i = 0; i < MT; ++i) {
        m[i] = -__builtin_inff(); l[i] = 0.0F;
#pragma unroll
        for (unsigned t = 0; t < 4; ++t) acc[i][t] = ffloat8{0, 0, 0, 0, 0, 0, 0, 0};
    }
#pragma unroll 1
    for (unsigned t0 = j0; t0 < end; t0 += kFastKeys) {
        const unsigned cnt = end - t0 < kFastKeys ? end - t0 : kFastKeys;
        {   // the tile: exponent shift of window keys whose largest scale is above 134 (max over the key's 16 threads), and
            // NaN encodings (E4M3 0x7f / 0xff codes or scales, E8M0 0xff): the key's factor becomes NaN, so its scores,
            // probabilities and the output propagate NaN as the exact kernel does
            const unsigned ka = t0 + 2U * pr, kb = ka + 1U;
            const bool wa = ka < nw, wb = kb < nw, va = 2U * pr < cnt, vb = 2U * pr + 1U < cnt;
            int ba = wa && va ? static_cast<int>(ra.sc & 0xFFU) : 0, bb = wb && vb ? static_cast<int>(rb.sc & 0xFFU) : 0;
            const auto nan_of = [](const KeyRaw& k, bool win) {
                if (!win) return ((k.sc & 0x7FU) == 0x7FU) || (((k.sc >> 8U) & 0x7FU) == 0x7FU);
                const std::uint32_t c[8] = {k.c0.x, k.c0.y, k.c0.z, k.c0.w, k.c1.x, k.c1.y, k.c1.z, k.c1.w};
                std::uint32_t any = 0;
#pragma unroll
                for (unsigned i = 0; i < 8; ++i) any |= ((c[i] & 0x7F7F7F7FU) + 0x01010101U) & 0x80808080U;
                return any != 0U || (k.sc & 0xFFU) == 0xFFU;
            };
            int na = va && nan_of(ra, wa) ? 1 : 0, nb = vb && nan_of(rb, wb) ? 1 : 0;
#pragma unroll
            for (unsigned d = 1; d < 16; d *= 2) {
                ba = max(ba, __shfl_xor(ba, static_cast<int>(d), 32)); bb = max(bb, __shfl_xor(bb, static_cast<int>(d), 32));
                na |= __shfl_xor(na, static_cast<int>(d), 32); nb |= __shfl_xor(nb, static_cast<int>(d), 32);
            }
            const int eka = ba > 134 ? ba - 134 : 0, ekb = bb > 134 ? bb - 134 : 0;
            if ((ablate & 2U) == 0U) fast_store_pair(kt, pr, part, ra, rb, wa, wb, va, vb, eka, ekb);
            if (part == 0U) {
                const int xa = wa ? eka : 7, xb = wb ? ekb : 7;   // the stored value is the key times 2^-x
                kexp[2U * pr] = va ? xa : 0; kexp[2U * pr + 1U] = vb ? xb : 0;
                kfac[2U * pr] = na != 0 ? __builtin_nanf("") : ldexpf(kAttnScale, xa);
                kfac[2U * pr + 1U] = nb != 0 ? __builtin_nanf("") : ldexpf(kAttnScale, xb);
            }
        }
        __syncthreads();
        if (t0 + kFastKeys < end) { load(t0 + kFastKeys + 2U * pr, ra); load(t0 + kFastKeys + 2U * pr + 1U, rb); }
        // QK: the wave's key tiles (H 16: kq; H 32: both), its head tile mq, dims 128 dq + 16 s ..
#pragma unroll
        for (unsigned kk = 0; kk < MT; ++kk) {
            const unsigned ktile = H == 32U ? kk : kq;
            const unsigned key = 16U * ktile + lo, kp = key >> 1U, sel = (key & 1U) != 0U ? 0x07060302U : 0x05040100U;
            ffloat8 d = ffloat8{0, 0, 0, 0, 0, 0, 0, 0};
            if ((ablate & 4U) == 0U) {
#pragma unroll
                for (unsigned s = 0; s < 8; ++s) {
                    const unsigned d0 = 128U * dq + 16U * s;
                    unsigned w[16];
#pragma unroll
                    for (unsigned v4 = 0; v4 < 4; ++v4) {
                        const uint4 x = *reinterpret_cast<const uint4*>(kt + fast_kslot(kp, d0 + 4U * v4));
                        w[4 * v4] = x.x; w[4 * v4 + 1] = x.y; w[4 * v4 + 2] = x.z; w[4 * v4 + 3] = x.w;
                    }
                    fu8 b;
#pragma unroll
                    for (unsigned j = 0; j < 8; ++j) b[j] = __builtin_amdgcn_perm(w[2 * j + 1], w[2 * j], sel);
                    d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[s], __builtin_bit_cast(fhalf16, b), d);
                }
            }
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) spart[dq][16U * mq + 2 * e + half][16U * ktile + lo] = d[e];
        }
        __syncthreads();
#pragma unroll
        for (unsigned i = 0; i < MT; ++i) {   // softmax of head HW wave + 2 i + half: lane lo holds keys 2 lo, 2 lo + 1
            const unsigned h = HW * wave + 2U * i + half, k0 = 2U * lo;
            float s[2];
#pragma unroll
            for (unsigned j = 0; j < 2; ++j) {
                const unsigned k = k0 + j;
                const float x = (spart[0][h][k] + spart[1][h][k]) + (spart[2][h][k] + spart[3][h][k]);
                s[j] = k < cnt ? (x * kfac[k]) * qfac[h] : -__builtin_inff();
            }
            float tm = fmaxf(s[0], s[1]);
#pragma unroll
            for (unsigned d = 1; d < 16; d *= 2) tm = fmaxf(tm, __shfl_xor(tm, static_cast<int>(d), 32));
            const float mn = fmaxf(m[i], tm), sc = exp_f(m[i] - mn);
            const float p0 = exp_f(s[0] - mn), p1 = exp_f(s[1] - mn);
            float ps = p0 + p1;
#pragma unroll
            for (unsigned d = 1; d < 16; d *= 2) ps = ps + __shfl_xor(ps, static_cast<int>(d), 32);
            l[i] = l[i] * sc + ps;
            m[i] = mn;
            // the tile's key shifts x (7 compressed, ek window): one band (largest - smallest <= 8, the usual tile) packs
            // f16 probabilities times 2^(x_k - hi) here; wider tiles keep FP32 probabilities for the banded passes below
            const bool v0 = k0 < cnt, v1 = k0 + 1U < cnt;
            int xh = max(v0 ? kexp[k0] : -1, v1 ? kexp[k0 + 1U] : -1), xl = min(v0 ? kexp[k0] : 1 << 20, v1 ? kexp[k0 + 1U] : 1 << 20);
#pragma unroll
            for (unsigned d = 1; d < 16; d *= 2) { xh = max(xh, __shfl_xor(xh, static_cast<int>(d), 32)); xl = min(xl, __shfl_xor(xl, static_cast<int>(d), 32)); }
            const bool banded = xh - xl > 8;
            if (banded) { pf[h][k0] = p0; pf[h][k0 + 1U] = p1; }
            else pt[h][lo] = __builtin_bit_cast(unsigned, fhalf2{static_cast<_Float16>(v0 ? ldexpf(p0, kexp[k0] - xh + kFastPScale) : 0.0F),
                                                                 static_cast<_Float16>(v1 ? ldexpf(p1, kexp[k0 + 1U] - xh + kFastPScale) : 0.0F)});
            if (lo == 0U) scl[h] = sc;
            if (tid == 0U) { band_s[0] = xh; band_s[1] = banded ? 1 : 0; }
        }
        __syncthreads();
        // PV: wave w owns dims 64 w + 16 t + lo of every head tile; rows = heads 16 i + 2 e + half. The stored keys carry
        // 2^-x_k (x = 7 compressed, ek window). Every f16 probability also carries 2^15: p * 2^(x_k - S + 15) <= 32768 for
        // the pass's top shift S, and f16 rounding takes a probability to zero only when p <= 2^(S - x_k - 40) of the
        // running softmax maximum (2^-40 at the top of a band, 2^-32 at its bottom) - the declared limit of this
        // approximation. Each product tile is formed from zero and added in FP32: acc = acc * sc + pv * 2^(S - 15) (the
        // accumulator is never scaled up, so it cannot overflow where the exact sum does not). One band (the usual tile,
        // largest - smallest shift <= 8): one pass with S = hi on the packed probabilities. Wider tiles: one pass per
        // band (S - 8 <= x_k <= S) on FP32 probabilities. Bands are uniform across the block.
        if (band_s[1] == 0) {
            const float up = ldexpf(1.0F, band_s[0] - kFastPScale);
#pragma unroll
            for (unsigned i = 0; i < MT; ++i)
#pragma unroll
                for (unsigned t = 0; t < 4; ++t) {   // one product tile at a time (one extra accumulator live)
                    ffloat8 pv = ffloat8{0, 0, 0, 0, 0, 0, 0, 0};
                    if ((ablate & 8U) == 0U) {
                        const unsigned dim = 64U * wave + 16U * t + lo;
#pragma unroll
                        for (unsigned ks = 0; ks < 2; ++ks) {
                            const uint4* ps = reinterpret_cast<const uint4*>(&pt[16U * i + lo][8U * ks]);
                            const uint4 x0 = ps[0], x1 = ps[1];
                            const fhalf16 a = __builtin_bit_cast(fhalf16, fu8{x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w});
                            fu8 b;
#pragma unroll
                            for (unsigned j = 0; j < 8; ++j) b[j] = kt[fast_kslot(8U * ks + j, dim)];
                            pv = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, __builtin_bit_cast(fhalf16, b), pv);
                        }
                    }
#pragma unroll
                    for (unsigned e = 0; e < 8; ++e) acc[i][t][e] = __builtin_fmaf(pv[e], up, acc[i][t][e] * scl[16U * i + 2 * e + half]);
                }
            __syncthreads();
            continue;
        }
        int hi = -1;
        for (unsigned k = 0; k < cnt; ++k) hi = max(hi, kexp[k]);
#pragma unroll
        for (unsigned i = 0; i < MT; ++i)
#pragma unroll
            for (unsigned t = 0; t < 4; ++t)
#pragma unroll
                for (unsigned e = 0; e < 8; ++e) acc[i][t][e] *= scl[16U * i + 2 * e + half];
#pragma unroll 1
        while (hi >= 0 && (ablate & 8U) == 0U) {
            const int band = hi, floor = hi - 8;
            const float up = ldexpf(1.0F, band - kFastPScale);
#pragma unroll
            for (unsigned i = 0; i < MT; ++i)
#pragma unroll
                for (unsigned t = 0; t < 4; ++t) {   // one product tile at a time, k-steps rolled: few live values (rare path)
                    ffloat8 pv = ffloat8{0, 0, 0, 0, 0, 0, 0, 0};
                    const unsigned dim = 64U * wave + 16U * t + lo;
#pragma unroll 1
                    for (unsigned ks = 0; ks < 2; ++ks) {
                        fhalf16 a;
#pragma unroll
                        for (unsigned j = 0; j < 16; ++j) {
                            const unsigned k = 16U * ks + j;
                            const int x = kexp[k];
                            a[j] = static_cast<_Float16>(k < cnt && x <= band && x >= floor ? ldexpf(pf[16U * i + lo][k], x - band + kFastPScale) : 0.0F);
                        }
                        fu8 b;
#pragma unroll
                        for (unsigned j = 0; j < 8; ++j) b[j] = kt[fast_kslot(8U * ks + j, dim)];
                        pv = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, __builtin_bit_cast(fhalf16, b), pv);
                    }
#pragma unroll
                    for (unsigned e = 0; e < 8; ++e) acc[i][t][e] = __builtin_fmaf(pv[e], up, acc[i][t][e]);
                }
            int next = -1;
            for (unsigned k = 0; k < cnt; ++k) if (kexp[k] < floor) next = max(next, kexp[k]);
            hi = next;
        }
        __syncthreads();
    }
    // the block's partial (slot blockIdx.x): (m, l) of the softmax heads; acc[i][t][e] = head 16 i + 2 e + half, dim 64 wave + 16 t + lo
#pragma unroll
    for (unsigned i = 0; i < MT; ++i) {
        if (lo == 0U) {
            float* pp = v.partial + ((static_cast<std::size_t>(r) * v.heads + hb + HW * wave + 2U * i + half) * kMaxChunks + blockIdx.x) * (2 + kHeadDim);
            pp[0] = m[i]; pp[1] = l[i];
        }
#pragma unroll
        for (unsigned e = 0; e < 8; ++e) {
            float* pp = v.partial + ((static_cast<std::size_t>(r) * v.heads + hb + 16U * i + 2U * e + half) * kMaxChunks + blockIdx.x) * (2 + kHeadDim) + 2U + 64U * wave + lo;
#pragma unroll
            for (unsigned t = 0; t < 4; ++t) pp[16U * t] = acc[i][t][e];
        }
    }
    if (v.tickets == nullptr) return;
    __threadfence();
    __syncthreads();
    // tickets: one counter per (row, 8-head group); a block of H heads counts on its first group's
    std::uint32_t* ticket = v.tickets + r * kAttnGroups + blockIdx.y * MT;
    if (tid == 0U) last_s = atomicAdd(ticket, 1U) == blocks - 1U ? 1U : 0U;
    __syncthreads();
    if (last_s == 0U) return;
    __threadfence();
#pragma unroll 1
    for (unsigned hh = 0; hh < HW; ++hh) attention_combine_head(v, r, hb + HW * wave + hh, lane, blocks);
    if (tid == 0U) *ticket = 0U;
}

// Combine without tickets: the row's blocks (ceil(chunks / Cpb) partials) in order.
template <unsigned Cpb>
__global__ __launch_bounds__(32) void attention_f16wmma_combine_kernel(AttentionView v) {
    unsigned nw;
    const unsigned n = visible_keys(v, blockIdx.y, nw), chunks = (n + kChunk - 1) / kChunk;
    attention_combine_head(v, blockIdx.y, blockIdx.x, threadIdx.x, (chunks + Cpb - 1) / Cpb);
}
