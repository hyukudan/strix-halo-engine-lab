// qpf2k/attn: attend_shpipe_multi_kernel (bench candidate: pipe schedule + F loaders). GENERATED from
// hip_qsa_fast_prefill_pipe.inc (generator schedule "pipe"): that schedule with F's loaders over the FP16 KV shadow
// (no per-chunk conversion; zeroing only when a slot of the wave's tasks is past the visible prefix). kv is kv_to_f16_kernel's output.
namespace qsa_fast {
template <unsigned Rows, unsigned TR, unsigned G, unsigned D>
__device__ __forceinline__ void attend_shpipe_body(const Views<Rows>& v, const std::uint16_t* kv, std::uint32_t kv_heads,
    std::uint32_t heads, std::uint32_t union_at, unsigned tile) {
    constexpr unsigned HPW = 16U / TR, HG = G * HPW, Loaders = 128U, Compute = 2U * G, NB = 3U;
    constexpr unsigned KT = kChunk * 32U / Loaders;
    __shared__ __align__(16) _Float16 k_lds[NB][kChunk][kPitch];
    __shared__ __align__(16) _Float16 v_lds[NB][kDim][kChunk];
    __shared__ __align__(16) float xch[2][G][2][32][8];
    __shared__ uint2 meta_lds[NB][4];
    const unsigned t = threadIdx.x, lane = t % 32U, w = t / 32U, col = lane % 16U, half = lane / 16U;
    const bool loader = w >= Compute;
    const unsigned r0 = tile * TR;
    const unsigned per_kv = heads / kv_heads, groups = per_kv / HG;
    const unsigned kvh = blockIdx.y / groups, hg = blockIdx.y % groups;
    unsigned tile_visible = 0U;
    for (unsigned i = 0; i < TR; ++i) tile_visible = max(tile_visible, v.visible[r0 + i]);
    const auto* header = reinterpret_cast<const std::uint32_t*>(v.scratch[r0] + union_at);
    const auto* pairs = reinterpret_cast<const uint2*>(header + 4U + TR);
    const unsigned chunks = header[0] / 4U;
    const unsigned c0 = 0U, c1 = chunks; // one slice
    const auto buf_of = [](unsigned c) { return c % NB; };

    if (loader) {
        struct R { u32x4 k[KT]; u32x4 v[4]; unsigned valid; };
        const auto meta_at = [&](unsigned c) { return lane < 4U ? pairs[4U * c + lane] : make_uint2(0U, 0U); };
        const auto token_of = [](uint2 m, unsigned slot) {
            return static_cast<unsigned>(__shfl(static_cast<int>(m.x), static_cast<int>(slot / 4U), 32)) * 4U + slot % 4U;
        };
        const auto row_elements = static_cast<std::uint64_t>(kv_heads) * kDim * 2U;
        const auto* k_base = kv + static_cast<std::uint64_t>(kvh) * kDim;
        const auto* v_base = kv + static_cast<std::uint64_t>(kv_heads + kvh) * kDim;
        const unsigned lt = t - Compute * 32U;
        const auto fetch = [&](uint2 m, R& g) {
            g.valid = 0U;
#pragma unroll
            for (unsigned i = 0; i < KT; ++i) {
                const unsigned task = lt + i * Loaders;
                const unsigned token = token_of(m, task / 32U);
                g.valid |= (token < tile_visible ? 1U : 0U) << i;
                const unsigned at = token < tile_visible ? token : tile_visible - 1U;
                g.k[i] = load16(k_base + at * row_elements + 8U * (task % 32U));
            }
#pragma unroll
            for (unsigned h = 0; h < 4U; ++h) {
                const unsigned token = token_of(m, 4U * (lt % 4U) + h);
                g.valid |= (token < tile_visible ? 1U : 0U) << (KT + h);
                const unsigned at = token < tile_visible ? token : tile_visible - 1U;
                g.v[h] = load16(v_base + at * row_elements + 8U * (lt / 4U));
            }
        };
        constexpr unsigned all_valid = (1U << (KT + 4U)) - 1U;
        const auto stage_with = [&]<bool Mask>(const R& g, uint2 m, unsigned buf) {
            const auto keep = [&](u32x4 x, unsigned b) {
                if constexpr (Mask) return ((g.valid >> b) & 1U) ? x : u32x4{0U, 0U, 0U, 0U};
                else return x;
            };
#pragma unroll
            for (unsigned i = 0; i < KT; ++i) {
                const unsigned task = lt + i * Loaders;
                const u32x4 x = keep(g.k[i], i);
                *reinterpret_cast<uint4*>(&k_lds[buf][task / 32U][8U * (task % 32U)]) = uint4{x.x, x.y, x.z, x.w};
            }
            unsigned f[4][4];
#pragma unroll
            for (unsigned h = 0; h < 4U; ++h) {
                const u32x4 x = keep(g.v[h], KT + h);
                f[h][0] = x.x; f[h][1] = x.y; f[h][2] = x.z; f[h][3] = x.w;
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
        const auto stage = [&](const R& g, uint2 m, unsigned buf) {
            if (__builtin_amdgcn_ballot_w32(g.valid != all_valid) == 0U) stage_with.template operator()<false>(g, m, buf);
            else stage_with.template operator()<true>(g, m, buf);
        };
        if (c0 < c1) {
            const auto chunk = [&](unsigned c) { return c < c1 ? c : c1 - 1U; };
            R r[D];
            uint2 held[D], raw[D];
            {
                R first;
                const uint2 m = meta_at(c0);
                fetch(m, first);
                stage(first, m, buf_of(c0));
            }
#pragma unroll
            for (unsigned d = 0; d < D; ++d) { held[d] = meta_at(chunk(c0 + 1U + d)); fetch(held[d], r[d]); }
#pragma unroll
            for (unsigned d = 0; d < D; ++d) raw[d] = meta_at(chunk(c0 + 1U + D + d));
            lds_barrier(); // B0
            // stage chunk c0 + 1 + s (s = 0, 1, ...) before barrier B(1 + s)
            for (unsigned c = c0; c <= c1; c += D) {
#pragma unroll
                for (unsigned k = 0; k < D; ++k) {
                    const unsigned step = c + k;   // stages chunk step + 1, then barrier B(step - c0 + 1)
                    if (step > c1) break;
                    stage(r[k], held[k], buf_of(step + 1U));
                    held[k] = raw[k];
                    fetch(held[k], r[k]);
                    raw[k] = meta_at(chunk(step + 1U + 2U * D));
                    lds_barrier();
                }
            }
        }
        return;
    }

    const unsigned grp = w / 2U, fh = w % 2U;
    const unsigned row = col % TR, head = kvh * per_kv + hg * HG + grp * HPW + col / TR, r = r0 + row;
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
    // Partial scores of chunk c over this wave's half and its exchange write (attend_body's statements).
    const auto partial = [&](unsigned c) {
        const unsigned buf = buf_of(c);
        f32x8 part[2] = {};
#pragma unroll
        for (unsigned k = 0; k < 8U; ++k) {
            const f16x16 a = *reinterpret_cast<const f16x16*>(&k_lds[buf][col][fh * (kDim / 2U) + 16U * k]);
            part[k % 2U] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, qf[k], part[k % 2U]);
        }
        const f32x8 mine = part[0] + part[1];
        *reinterpret_cast<f32x8*>(&xch[c % 2U][grp][fh][lane][0]) = mine;
        return mine;
    };
    const auto pair_sum = [&](unsigned c, const f32x8& mine) {
        const f32x8 other = *reinterpret_cast<const f32x8*>(&xch[c % 2U][grp][fh ^ 1U][lane][0]);
        return fh == 0U ? mine + other : other + mine;
    };
    if (c0 < c1) {
        lds_barrier(); // B0
        f32x8 s = pair_sum(c0, [&] { const f32x8 mine = partial(c0); lds_barrier(); return mine; }()); // B1
        for (unsigned c = c0; c < c1; ++c) {
            const unsigned buf = buf_of(c);
            uint2 meta[4];
#pragma unroll
            for (unsigned i = 0; i < 4U; ++i) meta[i] = meta_lds[buf][i];
            float p[8], top = -INFINITY;
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const uint2 blk = meta[e / 2U];
                const unsigned token = blk.x * 4U + 2U * (e % 2U) + half;
                const bool live = (blk.y & bit) != 0U && token < visible;
                const float score = s[e];
                p[e] = live ? score * score_scale : -INFINITY;
                top = fmaxf(top, p[e]);
            }
            top = fmaxf(top, swap16(top));
            const float m_new = fmaxf(m_run, top);
            const float ref = m_new == -INFINITY ? 0.0F : m_new;
            const float alpha = __builtin_amdgcn_exp2f(m_run - ref);
            float sum = 0.0F;
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) p[e] = __builtin_amdgcn_exp2f(p[e] - ref);
            if (meta[0].x == 0U)
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
                const float lo = half ? o : p[e], hi = half ? p[e] : o;
                pb[2U * e] = static_cast<_Float16>(lo);
                pb[2U * e + 1U] = static_cast<_Float16>(hi);
            }
#pragma unroll
            for (unsigned n = 0; n < 8U; ++n) {
                const f16x16 a = *reinterpret_cast<const f16x16*>(&v_lds[buf][v_row(fh * (kDim / 2U) + 16U * n + col)][0]);
                acc[n] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, pb, acc[n]);
            }
            if (c + 1U < c1) {
                const f32x8 mine = partial(c + 1U);
                lds_barrier();
                s = pair_sum(c + 1U, mine);
            } else lds_barrier();
        }
    }
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
}
template <unsigned Rows, unsigned TR, unsigned G, unsigned D = 3U>
__global__ __launch_bounds__((2U * G + 4U) * 32U) void attend_shpipe_multi_kernel(const Views<Rows>* groups, const std::uint16_t* kv,
    std::uint32_t kv_heads, std::uint32_t heads, std::uint32_t union_at) {
    attend_shpipe_body<Rows, TR, G, D>(groups[blockIdx.x / (Rows / TR)], kv, kv_heads, heads, union_at, blockIdx.x % (Rows / TR));
}
} // namespace qsa_fast
