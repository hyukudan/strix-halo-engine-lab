// Fast prefill profile, qsa component (included by hip_indexed_attention_launcher.hip; needs round_bf16).
//
// Sparse attention of a prefill group (Rows > 8 rows, head_dim 256) fused over the union of the
// selections of TR consecutive rows. union_kernel turns the rows' selected token lists into the
// ascending list of 4-token blocks any of them selected, each with a TR-bit row mask (and the
// multiplicity of tokens 0..3, which the selection may repeat). attend_kernel runs one tile of TR
// rows, one KV head and W waves of 16 query columns (a column is a (row, head) pair: TR rows times
// 16 / TR heads of that KV head) over a slice of the union: per 16-token chunk, scores
// S^T = K x Q^T and O^T += V^T x P^T with v_wmma_f32_16x16x16_f16 (FP32 accumulation), K and V
// converted exactly from BF16 to FP16 (zero past the tile's visible prefix), Q rounded once to FP16
// after a power-of-two scale per (row, head) that puts its largest |q| in [2^13, 2^14), undone with
// 1 / sqrt(256) on the FP32 scores. A (row, token) pair outside the row's selection or causal prefix
// gets score -inf; the softmax is the online one in FP32 (running maximum and sum per column, P
// rounded once to FP16 for the product, O rescaled only when a column's maximum grows).
// combine_kernel merges the slices (splits) of each (row, head), divides by the sum and applies the
// BF16 roundings and the BF16 sigmoid gate of attention_output_body (attend_kernel does it itself when
// there is one slice).
// Scratch (per row: its attention_scores buffer, heads x maximum_selected floats): slice partials
// [split][head][m, l, O[256]] of the row, then for the first row of each tile, at union_at, the
// tile's union: u32 [padded block count, 0, 0, 0, TR packed token-0..3 counts], uint2 (block, mask)[].
namespace qsa_fast {
typedef _Float16 f16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
constexpr unsigned kDim = 256U, kPitch = kDim + 8U, kChunk = 16U, kRangeBlocks = 8192U, kPartial = kDim + 2U;

template <unsigned Rows> struct Views {
    const float* q_gate[Rows];
    const std::uint32_t* selected[Rows];
    float* scratch[Rows];
    float* output[Rows];
    std::uint32_t count[Rows];
    std::uint32_t visible[Rows];
};
// Floats of a row's scratch the fast path needs (union_at + union) for TR-row tiles.
inline std::uint64_t scratch_floats(unsigned tile_rows, unsigned splits, unsigned heads, unsigned maximum_selected) {
    const auto blocks = static_cast<std::uint64_t>(tile_rows) * (maximum_selected / 4U + 2U) + 4U;
    return static_cast<std::uint64_t>(splits) * heads * kPartial + 4U + tile_rows + 2U * blocks;
}

__device__ __forceinline__ void lds_barrier() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }
// 16-byte global (not flat) load, so the loads in flight do not end at lds_barrier.
__device__ __forceinline__ u32x4 load16(const void* p) {
    return *reinterpret_cast<const __attribute__((address_space(1))) u32x4*>(reinterpret_cast<std::uintptr_t>(p));
}
__device__ __forceinline__ unsigned pack(float a, float b) {
    const _Float16 x = static_cast<_Float16>(a), y = static_cast<_Float16>(b);
    return static_cast<unsigned>(__builtin_bit_cast(unsigned short, x)) | static_cast<unsigned>(__builtin_bit_cast(unsigned short, y)) << 16U;
}
__device__ __forceinline__ float bf_lo(unsigned w) { return __uint_as_float(w << 16U); }
// Two BF16 (one word) as two FP16: exact for values in the FP16 normal range (BF16 has fewer significand
// bits); the conversion truncates, so magnitudes beyond the FP16 range saturate at 65504.
__device__ __forceinline__ unsigned bf16x2_to_f16x2(unsigned w) {
    return __builtin_bit_cast(unsigned, __builtin_amdgcn_cvt_pkrtz(__uint_as_float(w << 16U), __uint_as_float(w & 0xffff0000U)));
}
// LDS row of feature d in the transposed V tile: rows 8p + j of one store instruction (p = 0..7) and rows
// 16n + c of one fragment read fall in different bank groups.
__device__ __forceinline__ unsigned v_row(unsigned d) { return d ^ ((d >> 3U) & 7U); }
__device__ __forceinline__ float bf_hi(unsigned w) { return __uint_as_float(w & 0xffff0000U); }
__device__ __forceinline__ float swap16(float v) { // lane L ^ 16
    return __uint_as_float(__builtin_amdgcn_permlanex16(__float_as_uint(v), __float_as_uint(v), 0x76543210U, 0xfedcba98U, false, false));
}
// Exponent s with peak * 2^s in [2^13, 2^14) (0 for a zero or non-finite peak), 2^+-s normal.
__device__ __forceinline__ int q_exponent(float peak) {
    if (!(peak > 0.0F) || !isfinite(peak)) return 0;
    int e; (void)frexpf(peak, &e);
    return min(120, max(-120, 14 - e));
}
__device__ unsigned block_exclusive(unsigned v, unsigned* waves, unsigned& total) {
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    unsigned x = v;
    for (unsigned d = 1U; d < 32U; d <<= 1U) { const unsigned y = __shfl_up(x, d, 32); if (lane >= d) x += y; }
    if (lane == 31U) waves[wave] = x;
    __syncthreads();
    unsigned before = 0U;
    total = 0U;
    for (unsigned w = 0; w < blockDim.x / 32U; ++w) { if (w < wave) before += waves[w]; total += waves[w]; }
    __syncthreads();
    return before + x - v;
}

template <unsigned Rows, unsigned TR>
__global__ __launch_bounds__(256) void union_kernel(Views<Rows> v, std::uint32_t union_at) {
    __shared__ std::uint32_t mask[kRangeBlocks];
    __shared__ std::uint32_t weight[TR * 4U];
    __shared__ std::uint32_t waves[8];
    const unsigned t = threadIdx.x, r0 = blockIdx.x * TR;
    unsigned blocks = 0U;
    for (unsigned r = 0; r < TR; ++r) blocks = max(blocks, (v.visible[r0 + r] + 3U) / 4U);
    if (t < TR * 4U) weight[t] = 0U;
    auto* header = reinterpret_cast<std::uint32_t*>(v.scratch[r0] + union_at);
    auto* pairs = reinterpret_cast<uint2*>(header + 4U + TR);
    unsigned written = 0U;
    for (unsigned base = 0; base < blocks; base += kRangeBlocks) {
        const unsigned span = min(kRangeBlocks, blocks - base);
        for (unsigned i = t; i < span; i += 256U) mask[i] = 0U;
        __syncthreads();
        for (unsigned r = 0; r < TR; ++r) {
            const auto* sel = v.selected[r0 + r];
            const unsigned n = v.count[r0 + r];
            for (unsigned i = t; i < n; i += 256U) {
                const unsigned token = sel[i], b = token / 4U;
                if (base == 0U && token < 4U) atomicAdd(&weight[r * 4U + token], 1U);
                if (b >= base && b - base < span) atomicOr(&mask[b - base], 1U << r);
            }
        }
        __syncthreads();
        const unsigned per = (span + 255U) / 256U, first = min(t * per, span), last = min(first + per, span);
        unsigned c = 0U;
        for (unsigned j = first; j < last; ++j) c += mask[j] != 0U;
        unsigned total;
        unsigned at = written + block_exclusive(c, waves, total);
        for (unsigned j = first; j < last; ++j) if (mask[j]) pairs[at++] = make_uint2(base + j, mask[j]);
        written += total;
        __syncthreads();
    }
    const unsigned padded = (written + 3U) / 4U * 4U;
    if (t < padded - written) pairs[written + t] = make_uint2(0U, 0U);
    if (t == 0U) header[0] = padded;
    if (t < TR) {
        unsigned packed = 0U;
        for (unsigned k = 0; k < 4U; ++k) packed |= min(weight[t * 4U + k], 255U) << (8U * k);
        header[4U + t] = packed;
    }
}

// One workgroup: G column groups of 16 query columns, each run by a pair of compute waves that split the 256
// features (each keeps its half of Q as FP16 WMMA fragments in registers, forms the partial scores over that
// half, and accumulates that half of O), and four loader waves. The loaders keep D chunks in flight in
// registers and stage the oldest as FP16 into one of two LDS buffers (with its blocks and masks) while the compute
// waves run the other. Per chunk: partial scores, barrier, exchange (each wave adds the pair's halves in
// the same order, so both hold the same scores and run the same softmax), values, barrier. debug (fixture
// ablations): bit 0 loads token 0 for every slot, bit 1 skips the compute.
template <unsigned Rows, unsigned TR, unsigned G, unsigned D = 3U>
__global__ __launch_bounds__((2U * G + 4U) * 32U) void attend_kernel(Views<Rows> v, const std::uint16_t* kv, std::uint32_t kv_heads,
    std::uint32_t heads, std::uint32_t union_at, std::uint32_t splits, std::uint32_t debug = 0U) {
    constexpr unsigned HPW = 16U / TR, HG = G * HPW, Loaders = 128U, Compute = 2U * G;
    constexpr unsigned KT = kChunk * 32U / Loaders; // V: one task per loader thread
    __shared__ __align__(16) _Float16 k_lds[2][kChunk][kPitch];
    __shared__ __align__(16) _Float16 v_lds[2][kDim][kChunk];
    __shared__ __align__(16) float xch[G][2][32][8];
    __shared__ uint2 meta_lds[2][4];
    const unsigned t = threadIdx.x, lane = t % 32U, w = t / 32U, col = lane % 16U, half = lane / 16U;
    const bool loader = w >= Compute;
    const unsigned r0 = blockIdx.x * TR;
    const unsigned per_kv = heads / kv_heads, groups = per_kv / HG;
    const unsigned kvh = blockIdx.y / groups, hg = blockIdx.y % groups, split = blockIdx.z;
    unsigned tile_visible = 0U;
    for (unsigned i = 0; i < TR; ++i) tile_visible = max(tile_visible, v.visible[r0 + i]);
    const auto* header = reinterpret_cast<const std::uint32_t*>(v.scratch[r0] + union_at);
    const auto* pairs = reinterpret_cast<const uint2*>(header + 4U + TR);
    const unsigned chunks = header[0] / 4U;
    const unsigned c0 = static_cast<unsigned>(static_cast<std::uint64_t>(chunks) * split / splits);
    const unsigned c1 = static_cast<unsigned>(static_cast<std::uint64_t>(chunks) * (split + 1U) / splits);

    if (loader) {
        struct Regs { u32x4 k[KT]; u32x4 v[4]; unsigned valid; };
        // Lane i < 4 of each loader wave holds the chunk's (block, mask) i; a slot's block comes by lane
        // shuffle (not an indexed array, which would live in scratch, whose loads wait for every load in flight).
        const auto meta_at = [&](unsigned c) { return lane < 4U ? pairs[4U * c + lane] : make_uint2(0U, 0U); };
        const auto token_of = [](uint2 m, unsigned slot) {
            return static_cast<unsigned>(__shfl(static_cast<int>(m.x), static_cast<int>(slot / 4U), 32)) * 4U + slot % 4U;
        };
        const auto row_elements = static_cast<std::uint64_t>(kv_heads) * kDim * 2U;
        const auto* k_base = kv + static_cast<std::uint64_t>(kvh) * kDim;
        const auto* v_base = kv + static_cast<std::uint64_t>(kv_heads + kvh) * kDim;
        const unsigned lt = t - Compute * 32U;
        // K task i = (token i / 32, 8 features 8 (i % 32)); V task = (tokens 4 (lt % 4) .. + 3, 8 features
        // 8 (lt / 4)). Every load is issued (a token past the tile's visible prefix reads the last visible one
        // and is staged as zero), so no load waits on a branch.
        const auto fetch = [&](uint2 m, Regs& g) {
            g.valid = 0U;
#pragma unroll
            for (unsigned i = 0; i < KT; ++i) {
                const unsigned task = lt + i * Loaders;
                const unsigned token = (debug & 1U) ? 0U : token_of(m, task / 32U);
                g.valid |= (token < tile_visible ? 1U : 0U) << i;
                g.k[i] = load16(k_base + (token < tile_visible ? token : tile_visible - 1U) * row_elements + 8U * (task % 32U));
            }
#pragma unroll
            for (unsigned h = 0; h < 4U; ++h) {
                const unsigned token = (debug & 1U) ? 0U : token_of(m, 4U * (lt % 4U) + h);
                g.valid |= (token < tile_visible ? 1U : 0U) << (KT + h);
                g.v[h] = load16(v_base + (token < tile_visible ? token : tile_visible - 1U) * row_elements + 8U * (lt / 4U));
            }
        };
        const auto stage = [&](const Regs& g, uint2 m, unsigned buf) {
            const auto keep = [&](u32x4 x, unsigned b) { return ((g.valid >> b) & 1U) ? x : u32x4{0U, 0U, 0U, 0U}; };
#pragma unroll
            for (unsigned i = 0; i < KT; ++i) {
                const unsigned task = lt + i * Loaders;
                const u32x4 x = keep(g.k[i], i);
                *reinterpret_cast<uint4*>(&k_lds[buf][task / 32U][8U * (task % 32U)]) =
                    uint4{bf16x2_to_f16x2(x.x), bf16x2_to_f16x2(x.y), bf16x2_to_f16x2(x.z), bf16x2_to_f16x2(x.w)};
            }
            unsigned f[4][4]; // token h, FP16 pair k = features 8p + 2k, + 1
#pragma unroll
            for (unsigned h = 0; h < 4U; ++h) {
                const u32x4 x = keep(g.v[h], KT + h);
                f[h][0] = bf16x2_to_f16x2(x.x); f[h][1] = bf16x2_to_f16x2(x.y);
                f[h][2] = bf16x2_to_f16x2(x.z); f[h][3] = bf16x2_to_f16x2(x.w);
            }
            const unsigned feature = 8U * (lt / 4U), slot = 4U * (lt % 4U);
#pragma unroll
            for (unsigned k = 0; k < 4U; ++k) {
                *reinterpret_cast<uint2*>(&v_lds[buf][v_row(feature + 2U * k)][slot]) =
                    make_uint2((f[0][k] & 0xffffU) | (f[1][k] << 16U), (f[2][k] & 0xffffU) | (f[3][k] << 16U));
                *reinterpret_cast<uint2*>(&v_lds[buf][v_row(feature + 2U * k + 1U)][slot]) =
                    make_uint2((f[0][k] >> 16U) | (f[1][k] & 0xffff0000U), (f[2][k] >> 16U) | (f[3][k] & 0xffff0000U));
            }
            if (lt < 4U) meta_lds[buf][lt] = m;
        };
        // D chunks in flight, slot k of the loop unrolled by D: before step c, r[k] holds chunk c + 1 (fetched
        // with meta held[k]) and raw[k] the meta of chunk c + 1 + D, loaded D chunks before its fetch (so a wait
        // for it covers only loads at least as old as the chunk staged next). Fixed slots (moving a register
        // whose load is in flight would wait for it) and no branch around a load (chunk indices clamp to the
        // last chunk; what the tail stages goes to the buffer the compute waves no longer read).
        if (c0 < c1) {
            const auto chunk = [&](unsigned c) { return c < c1 ? c : c1 - 1U; };
            Regs r[D];
            uint2 held[D], raw[D];
            {
                Regs first;
                const uint2 m = meta_at(c0);
                fetch(m, first);
                stage(first, m, 0U);
            }
#pragma unroll
            for (unsigned d = 0; d < D; ++d) { held[d] = meta_at(chunk(c0 + 1U + d)); fetch(held[d], r[d]); }
#pragma unroll
            for (unsigned d = 0; d < D; ++d) raw[d] = meta_at(chunk(c0 + 1U + D + d));
            lds_barrier();
            for (unsigned c = c0; c < c1; c += D) {
#pragma unroll
                for (unsigned k = 0; k < D; ++k) {
                    const unsigned step = c + k;
                    if (step >= c1) break;
                    stage(r[k], held[k], (step - c0 + 1U) & 1U);
                    held[k] = raw[k];
                    fetch(held[k], r[k]);
                    raw[k] = meta_at(chunk(step + 1U + 2U * D));
                    lds_barrier();
                    lds_barrier();
                }
            }
        }
        return;
    }

    const unsigned grp = w / 2U, fh = w % 2U; // column group, feature half
    const unsigned row = col % TR, head = kvh * per_kv + hg * HG + grp * HPW + col / TR, r = r0 + row;
    // Q of this lane's column: a power-of-two scale from the largest |q| of the 256 features (the lane pair
    // (col, col + 16) reads half each), then this wave's 128 features as FP16 B fragments.
    float score_scale;
    f16x16 qf[8];
    {
        const float* q = v.q_gate[r] + static_cast<std::uint64_t>(head) * 2U * kDim;
        float peak = 0.0F;
        for (unsigned i = half * (kDim / 2U); i < (half + 1U) * (kDim / 2U); i += 4U) {
            const float4 x = *reinterpret_cast<const float4*>(q + i);
            peak = fmaxf(peak, fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w))));
        }
        peak = fmaxf(peak, swap16(peak));
        const int e = q_exponent(peak);
        const float scale = ldexpf(1.0F, e);
        // Scores in base 2: the softmax below uses exp2 (v_exp_f32).
        score_scale = ldexpf(1.0F, -e) / sqrtf(static_cast<float>(kDim)) * 1.4426950408889634F;
#pragma unroll
        for (unsigned k = 0; k < 8U; ++k)
#pragma unroll
            for (unsigned i = 0; i < 16U; i += 4U) {
                const float4 x = *reinterpret_cast<const float4*>(q + fh * (kDim / 2U) + 16U * k + i);
                qf[k][i] = static_cast<_Float16>(x.x * scale); qf[k][i + 1U] = static_cast<_Float16>(x.y * scale);
                qf[k][i + 2U] = static_cast<_Float16>(x.z * scale); qf[k][i + 3U] = static_cast<_Float16>(x.w * scale);
            }
    }
    const unsigned visible = v.visible[r], bit = 1U << row, weights = header[4U + row];
    f32x8 acc[8];
#pragma unroll
    for (unsigned n = 0; n < 8U; ++n) acc[n] = f32x8{};
    float m_run = -INFINITY, l_run = 0.0F;
    if (c0 < c1) lds_barrier();
    for (unsigned c = c0; c < c1; ++c) {
        const unsigned buf = (c - c0) & 1U;
        if (debug & 2U) { lds_barrier(); lds_barrier(); continue; }
        uint2 meta[4];
#pragma unroll
        for (unsigned i = 0; i < 4U; ++i) meta[i] = meta_lds[buf][i];
        f32x8 part[2] = {};
#pragma unroll
        for (unsigned k = 0; k < 8U; ++k) {
            const f16x16 a = *reinterpret_cast<const f16x16*>(&k_lds[buf][col][fh * (kDim / 2U) + 16U * k]);
            part[k % 2U] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, qf[k], part[k % 2U]);
        }
        const f32x8 mine = part[0] + part[1];
        *reinterpret_cast<f32x8*>(&xch[grp][fh][lane][0]) = mine;
        lds_barrier();
        const f32x8 other = *reinterpret_cast<const f32x8*>(&xch[grp][fh ^ 1U][lane][0]);
        const f32x8 s = fh == 0U ? mine + other : other + mine;
        // Element e: token slot 2e + half of the chunk (block e / 2), column col.
        float p[8], top = -INFINITY;
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) {
            const uint2 blk = meta[e / 2U];
            const unsigned token = blk.x * 4U + 2U * (e % 2U) + half;
            const bool live = (blk.y & bit) != 0U && token < visible;
            p[e] = live ? s[e] * score_scale : -INFINITY;
            top = fmaxf(top, p[e]);
        }
        top = fmaxf(top, swap16(top));
        const float m_new = fmaxf(m_run, top);
        const float ref = m_new == -INFINITY ? 0.0F : m_new;
        const float alpha = __builtin_amdgcn_exp2f(m_run - ref);
        float sum = 0.0F;
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) p[e] = __builtin_amdgcn_exp2f(p[e] - ref);
        if (meta[0].x == 0U) // tokens 0..3: their multiplicity in the row's selection
#pragma unroll
            for (unsigned e = 0; e < 2U; ++e) p[e] *= static_cast<float>((weights >> (8U * (2U * e + half))) & 255U);
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) sum += p[e];
        sum += swap16(sum);
        l_run = l_run * alpha + sum;
        m_run = m_new;
        if (__builtin_amdgcn_ballot_w32(alpha != 1.0F))
#pragma unroll
            for (unsigned n = 0; n < 8U; ++n) acc[n] *= alpha;
        f16x16 pb;
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) {
            const float o = swap16(p[e]);
            pb[2U * e] = static_cast<_Float16>(half ? o : p[e]);
            pb[2U * e + 1U] = static_cast<_Float16>(half ? p[e] : o);
        }
#pragma unroll
        for (unsigned n = 0; n < 8U; ++n) {
            const f16x16 a = *reinterpret_cast<const f16x16*>(&v_lds[buf][v_row(fh * (kDim / 2U) + 16U * n + col)][0]);
            acc[n] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, pb, acc[n]);
        }
        lds_barrier();
    }
    if (splits == 1U) {
        // One slice: combine_kernel's arithmetic for one split, in place.
        const float* gate_in = v.q_gate[r] + static_cast<std::uint64_t>(head) * 2U * kDim + kDim;
        float* out = v.output[r] + static_cast<std::uint64_t>(head) * kDim;
#pragma unroll
        for (unsigned n = 0; n < 8U; ++n)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const unsigned d = fh * (kDim / 2U) + 16U * n + 2U * e + half;
                const float attn = round_bf16(l_run > 0.0F ? acc[n][e] / l_run : 0.0F);
                const float gate = round_bf16(1.0F / (1.0F + expf(-gate_in[d])));
                out[d] = round_bf16(attn * gate);
            }
        return;
    }
    float* partial = v.scratch[r] + (static_cast<std::uint64_t>(split) * heads + head) * kPartial;
    if (fh == 0U && half == 0U) { partial[0] = m_run; partial[1] = l_run; }
#pragma unroll
    for (unsigned n = 0; n < 8U; ++n)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) partial[2U + fh * (kDim / 2U) + 16U * n + 2U * e + half] = acc[n][e];
}

template <unsigned Rows>
__global__ __launch_bounds__(256) void combine_kernel(Views<Rows> v, std::uint32_t heads, std::uint32_t splits) {
    const unsigned head = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
    const float* partial = v.scratch[r] + static_cast<std::uint64_t>(head) * kPartial;
    const auto stride = static_cast<std::uint64_t>(heads) * kPartial;
    float top = -INFINITY;
    for (unsigned s = 0; s < splits; ++s) top = fmaxf(top, partial[s * stride]); // base-2 maxima
    float l = 0.0F, o = 0.0F;
    if (top != -INFINITY)
        for (unsigned s = 0; s < splits; ++s) {
            const float m = partial[s * stride];
            if (m == -INFINITY) continue;
            const float f = exp2f(m - top);
            l += partial[s * stride + 1U] * f;
            o += partial[s * stride + 2U + d] * f;
        }
    const float attn = round_bf16(l > 0.0F ? o / l : 0.0F);
    const float gate = round_bf16(1.0F / (1.0F + expf(-v.q_gate[r][static_cast<std::uint64_t>(head) * 2U * kDim + kDim + d])));
    v.output[r][static_cast<std::uint64_t>(head) * kDim + d] = round_bf16(attn * gate);
}
} // namespace qsa_fast
