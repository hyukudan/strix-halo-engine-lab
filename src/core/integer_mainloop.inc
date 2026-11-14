if constexpr (Operand::Plan::structure == IntegerStructure::grouped_chunk) {
#include "integer_grouped_mainloop.inc"
} else {
// Integer specialization inside tiled_core: same Adapter tile/raster interface.
static_assert(AD && !Fused && Probe == 0);
static_assert((Contract::int_flush == 32 || Contract::int_flush == 1) && Contract::out_transform == OutputTransform::none);
using Ops = typename Operand::Ops;
using V4 = typename Operand::Int4;
using V8 = typename Operand::Int8;
using V2 = int __attribute__((ext_vector_type(2)));
constexpr bool Fast = Operand::fast_i4, Affine = Operand::affine;
const int q = wave / G::waves, lo = lane & 15, h = lane >> 4;
auto* exchange = reinterpret_cast<float*>(lds);
// Eight class waves per spatial tile, c=q+8*k. No cross-class reassociation.
float leaf[4][MT][NT][8] = {};
constexpr bool Relaxed = Contract::relaxed;
const int groups = inputs / 32;
for (int chunk = 0; chunk < groups; chunk += Relaxed ? 1 : 32) {
    #pragma unroll
    for (int k = 0; k < (Relaxed ? 1 : 4); ++k) {
        const int group = Relaxed ? (q == 0 ? chunk : groups) : chunk + q + 8 * k;
        V4 a0[MT] = {}, a1[MT] = {};
        V2 aq0[MT] = {}, aq1[MT] = {};
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int r = first_row + wave_m * WM + mt * 16 + lo;
            float delta = 0.0f;
            int sums[2] = {};
            if (group < groups && r < rows) {
                const auto* codes = ad.code_row(at, r) + group * 32;
                int values[32], amax = 0;
                #pragma unroll
                for (int j = 0; j < 32; ++j) {
                    values[j] = int(codes[j]);
                    if constexpr (Fast) amax = max(amax, abs(values[j]));
                }
                delta = ad.delta(at, r, group);
                if constexpr (Fast) {
                    const float step = integer_math::a4_step(amax);
                    delta = delta * step;
                    #pragma unroll
                    for (int j = 0; j < 32; ++j)
                        values[j] = integer_math::a4_code(values[j], step);
                }
                #pragma unroll
                for (int j = 0; j < 32; ++j) {
                    sums[j / 16] += values[j];
                    if constexpr (Fast) {
                        auto& v = j < 16 ? aq0[mt] : aq1[mt];
                        v[(j % 16) / 8] |= int(integer_math::nibble(values[j]) << (4 * (j % 8)));
                    } else {
                        auto& v = j < 16 ? a0[mt] : a1[mt];
                        v[(j % 16) / 4] |= int((unsigned(values[j]) & 255U) << (8 * (j % 4)));
                    }
                }
            }
            if (h == 0) {
                const int slot = ((wave * MT + mt) * 16 + lo) * 3;
                exchange[slot] = delta;
                exchange[slot + 1] = float(sums[0]);
                exchange[slot + 2] = float(sums[1]);
            }
        }
        __syncthreads();
        if (group < groups) {
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int col = first_col + wave_n * WN + nt * 16 + lo;
                const auto raw = Operand::load_raw(ad.weight_row(at, col), unsigned(group));
                auto weight = Ops::decode(raw, ad.grid());
                V2 bq0 = {}, bq1 = {};
                if constexpr (Fast) {
                    #pragma unroll
                    for (int j = 0; j < 32; ++j) {
                        int value;
                        if constexpr (Affine) value = int((raw.q[j / 4] >> (8U * unsigned(j % 4) + raw.shift)) & 3U);
                        else {
                            const auto v = j < 16 ? weight.b0 : weight.b1;
                            value = integer_math::iq4(int(static_cast<signed char>(unsigned(v[(j % 16) / 4]) >> (8 * (j % 4)))));
                        }
                        auto& v = j < 16 ? bq0 : bq1;
                        v[(j % 16) / 8] |= int(integer_math::nibble(value) << (4 * (j % 8)));
                    }
                    if constexpr (!Affine) weight.scale = weight.scale * (43.0f / 7.0f);
                }
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    V8 dot = {}, dot1 = {};
                    if constexpr (Fast) {
                        dot = integer_wmma<4>(aq0[mt], bq0, V8{});
                        // Q2_K's K16 factors differ: retain both half dots until scaling.
                        dot1 = integer_wmma<4>(aq1[mt], bq1, Affine ? V8{} : dot);
                    } else {
                        const V8 magic{Ops::dot_magic, Ops::dot_magic, Ops::dot_magic, Ops::dot_magic,
                                       Ops::dot_magic, Ops::dot_magic, Ops::dot_magic, Ops::dot_magic};
                        dot = integer_wmma<8>(a0[mt], weight.b0, magic);
                        dot = integer_wmma<8>(a1[mt], weight.b1, dot);
                    }
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const int slot = ((wave * MT + mt) * 16 + 2 * e + h) * 3;
                        const float sa = exchange[slot];
                        float term;
                        if constexpr (Affine) {
                            const int minimum = weight.m0 * int(exchange[slot + 1]) + weight.m1 * int(exchange[slot + 2]);
                            if constexpr (Fast) {
                                const int scaled_dot = int(raw.scales & 15U) * dot[e] + int((raw.scales >> 8) & 15U) * dot1[e];
                                term = integer_math::affine(scaled_dot, minimum, weight.d, weight.dmin, sa);
                            } else {
                                const float scaled = weight.d * Ops::magic_float(dot[e]);
                                const float offset = weight.dmin * float(minimum);
                                term = Relaxed ? __builtin_fmaf(weight.d, Ops::magic_float(dot[e]), -offset) : (scaled - offset) * sa;
                            }
                        } else {
                            const float value = Fast ? float(dot1[e]) : Ops::magic_float(dot[e]);
                            term = value * (weight.scale * sa);
                        }
                        if constexpr (Relaxed) {
                            if constexpr (Affine) leaf[k][mt][nt][e] = __builtin_fmaf(term, sa, leaf[k][mt][nt][e]);
                            else leaf[k][mt][nt][e] = __builtin_fmaf(Ops::magic_float(dot[e]), weight.scale * sa, leaf[k][mt][nt][e]);
                        } else leaf[k][mt][nt][e] = leaf[k][mt][nt][e] + term;
                    }
                }
            }
        }
        // Missing classes also participate; no out-of-bounds final-chunk reads.
        __syncthreads();
    }
}
#pragma unroll
for (int mt = 0; mt < MT; ++mt)
    #pragma unroll
    for (int nt = 0; nt < NT; ++nt)
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int r = wave_m * WM + mt * 16 + 2 * e + h;
            const int c = wave_n * WN + nt * 16 + lo;
            exchange[(q * BM + r) * BN + c] = Relaxed ? leaf[0][mt][nt][e] : (leaf[0][mt][nt][e] + leaf[2][mt][nt][e]) +
                                                             (leaf[1][mt][nt][e] + leaf[3][mt][nt][e]);
        }
__syncthreads();
for (int i = thread; i < BM * BN; i += G::threads) {
    const float s0 = exchange[i] + exchange[4 * BM * BN + i];
    const float s1 = exchange[BM * BN + i] + exchange[5 * BM * BN + i];
    const float s2 = exchange[2 * BM * BN + i] + exchange[6 * BM * BN + i];
    const float s3 = exchange[3 * BM * BN + i] + exchange[7 * BM * BN + i];
    const float v = Relaxed ? exchange[i] : (s0 + s2) + (s1 + s3);
    exchange[i] = v;
    const int r = first_row + i / BN, c = first_col + i % BN;
    if (r < rows) ad.store_raw(at, r, c, v);
}
__syncthreads();
ad.finish_tile(at, exchange, BM, BN);
__syncthreads();

}
