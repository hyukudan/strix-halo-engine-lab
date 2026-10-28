#pragma once
#include "glm/moe/glm_trellis_device.hpp"

namespace strix::glm::pf::exact_x {
namespace d = moe::dev;
// Pair pitches are multiples of four so operand reads can merge to b128. The
// epilogue reuses staging storage; static LDS is 23040 bytes per CTA.
union alignas(16) Storage {
    struct { std::uint32_t w[128][36], x[32][36]; } stage;
    float total[32][128];
};
static_assert(sizeof(Storage) == 23040);

// Raw next-stage operands stay in registers across the current stage's dot2s.
// Decode only when publishing to LDS, keeping the live prefetch to 16 dwords.
struct PrefetchStage {
    std::uint32_t raw[4][3], x[4];
    template<int Bits, class Rows>
    __device__ __forceinline__ void load(const unsigned char* packed, const d::LaneGeom<Bits>& geo,
                                        Rows rows, unsigned nr, unsigned output, unsigned col0,
                                        unsigned base, unsigned nk) {
        const unsigned tid = threadIdx.x, wave = tid / 32U;
#pragma unroll
        for (unsigned kk = 0; kk < 4U; ++kk) if (kk < nk) {
            const auto* tile = packed + (std::size_t{base + kk} * (output / 16U) + col0 / 16U + wave) * Bits * 32U;
            raw[kk][0] = d::load_u32(tile, geo.oa);
            raw[kk][1] = d::load_u32(tile, geo.ob);
            raw[kk][2] = d::load_u32(tile, geo.oc);
        }
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
            const unsigned q = tid + j * 256U, r = q / 32U, p = q % 32U;
            x[j] = 0;
            // Guard input() itself: MoE resolves a.pairs there.
            if (r < nr && p < nk * 8U)
                __builtin_memcpy(&x[j], rows.input(r) + base * 16U + p * 2U, 4);
        }
        // Keep global reads ahead of the compute loop without consuming their
        // results here; their first use is after the retiring stage's barrier.
        asm volatile("" ::: "memory");
    }
    template<int Bits>
    __device__ __forceinline__ void store(const d::LaneGeom<Bits>& geo, unsigned nk, Storage& lds) const {
        const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid / 32U;
#pragma unroll
        for (unsigned kk = 0; kk < 4U; ++kk) if (kk < nk) {
            d::TileStates<Bits> st;
            st.high = __builtin_amdgcn_alignbit(raw[kk][0], raw[kk][1], geo.align);
            st.low = __builtin_amdgcn_alignbit(raw[kk][1], raw[kk][2], geo.align);
            st.tail = raw[kk][2] << (32U - geo.align);
            const unsigned o = wave * 16U + lane / 4U, p = kk * 8U + (lane & 3U);
            lds.stage.w[o][p] = __builtin_bit_cast(std::uint32_t, d::mul1_pair(st.template state<0>(), st.template state<1>()));
            lds.stage.w[o][p + 4U] = __builtin_bit_cast(std::uint32_t, d::mul1_pair(st.template state<2>(), st.template state<3>()));
            lds.stage.w[o + 8U][p] = __builtin_bit_cast(std::uint32_t, d::mul1_pair(st.template state<4>(), st.template state<5>()));
            lds.stage.w[o + 8U][p + 4U] = __builtin_bit_cast(std::uint32_t, d::mul1_pair(st.template state<6>(), st.template state<7>()));
        }
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
            const unsigned q = tid + j * 256U;
            lds.stage.x[q / 32U][q % 32U] = x[j];
        }
    }
};

// Rows supplies already transformed FP16 input and the final row destination.
template<int Bits, bool Prefetch = false, class Mat, class Rows>
__device__ __forceinline__ void core(Mat m, Rows rows, unsigned nr, unsigned splits,
                                    bool hp, Storage& lds) {
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid / 32U;
    // Adjacent lanes own adjacent outputs; each wave broadcasts four input rows.
    const unsigned col0 = blockIdx.x * 128U, og = lane, rg = wave;
    const unsigned units = hp ? m.input / 128U : m.input / 16U;
    const d::LaneGeom<Bits> geo(lane);
    const auto* packed = reinterpret_cast<const unsigned char*>(d::uniform(m.trellis));
    PrefetchStage next;
    bool ready = false;
    float total[4][4] = {};
    for (unsigned sp = 0; sp < splits; ++sp) {
        const unsigned begin = static_cast<unsigned>(std::uint64_t{units} * sp / splits) * (hp ? 8U : 1U);
        const unsigned end = static_cast<unsigned>(std::uint64_t{units} * (sp + 1U) / splits) * (hp ? 8U : 1U);
        float chain[4][4][4] = {}; // four independent +0 chains per (output, row)
        for (unsigned base = begin; base < end; base += 4U) {
            const unsigned nk = end - base < 4U ? end - base : 4U;
            if constexpr (Prefetch) {
                if (!ready) next.template load<Bits>(packed, geo, rows, nr, m.output, col0, base, nk);
                next.template store<Bits>(geo, nk, lds);
            } else {
                for (unsigned kk = 0; kk < nk; ++kk) {
                    const auto* raw = packed + (std::size_t{base + kk} * (m.output / 16U) + col0 / 16U + wave) * Bits * 32U;
                    const d::TileStates<Bits> st(raw, geo);
                    // Same mapping as core_kernel/core_v3: L/4 is output 0..7,
                    // L&3 is pair q; w1 adds four pairs, w2/w3 add eight outputs.
                    // Only FP16 bits move through LDS; no additional rounding.
                    const unsigned o = wave * 16U + lane / 4U, p = kk * 8U + (lane & 3U);
                    lds.stage.w[o][p] = __builtin_bit_cast(std::uint32_t, d::mul1_pair(st.template state<0>(), st.template state<1>()));
                    lds.stage.w[o][p + 4U] = __builtin_bit_cast(std::uint32_t, d::mul1_pair(st.template state<2>(), st.template state<3>()));
                    lds.stage.w[o + 8U][p] = __builtin_bit_cast(std::uint32_t, d::mul1_pair(st.template state<4>(), st.template state<5>()));
                    lds.stage.w[o + 8U][p + 4U] = __builtin_bit_cast(std::uint32_t, d::mul1_pair(st.template state<6>(), st.template state<7>()));
                }
                for (unsigned q = tid; q < 32U * 32U; q += 256U) {
                    const unsigned r = q / 32U, p = q % 32U;
                    std::uint32_t v = 0;
                    if (r < nr && p < nk * 8U)
                        __builtin_memcpy(&v, rows.input(r) + base * 16U + p * 2U, 4);
                    lds.stage.x[r][p] = v;
                }
            }
            __syncthreads();
            if constexpr (Prefetch) {
                unsigned next_base = base + nk, next_end = end;
                // Splits can be clipped or empty (including H128 partitions).
                // Advance to the next nonempty split without changing its fold.
                for (unsigned next_sp = sp + 1U; next_base == next_end && next_sp < splits; ++next_sp) {
                    next_base = static_cast<unsigned>(std::uint64_t{units} * next_sp / splits) * (hp ? 8U : 1U);
                    next_end = static_cast<unsigned>(std::uint64_t{units} * (next_sp + 1U) / splits) * (hp ? 8U : 1U);
                }
                const unsigned next_nk = next_end - next_base < 4U ? next_end - next_base : 4U;
                ready = next_nk != 0;
                if (ready) next.template load<Bits>(packed, geo, rows, nr, m.output, col0, next_base, next_nk);
            }
            for (unsigned kk = 0; kk < nk; ++kk) {
#pragma unroll 1
                for (unsigned h = 0; h < 2U; ++h) {
                    // Plain DWORD operands (aligned, so the compiler may still merge them): an ext_vector
                    // LDS read here was compiled to one DWORD per row, feeding every quarter chain.
                    std::uint32_t w[4][4], x[4][4];
#pragma unroll
                    for (unsigned i = 0; i < 4U; ++i)
#pragma unroll
                        for (unsigned q = 0; q < 4U; ++q) {
                            w[i][q] = lds.stage.w[og + 32U * i][kk * 8U + h * 4U + q];
                            x[i][q] = lds.stage.x[rg + 8U * i][kk * 8U + h * 4U + q];
                        }
#pragma unroll
                    for (unsigned q = 0; q < 4U; ++q)
#pragma unroll
                        for (unsigned i = 0; i < 4U; ++i)
#pragma unroll
                            for (unsigned j = 0; j < 4U; ++j)
                                chain[i][j][q] = __builtin_amdgcn_fdot2(__builtin_bit_cast(d::Half2, w[i][q]),
                                    __builtin_bit_cast(d::Half2, x[j][q]), chain[i][j][q], false);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (unsigned i = 0; i < 4U; ++i)
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) {
                // reduce4 lane zero: (c0+c2) + (c1+c3), then ascending split addition.
                const float v = (chain[i][j][0] + chain[i][j][2]) + (chain[i][j][1] + chain[i][j][3]);
                total[i][j] = total[i][j] + v;
            }
    }
#pragma unroll
    for (unsigned i = 0; i < 4U; ++i)
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) lds.total[rg + 8U * j][og + 32U * i] = total[i][j];
    __syncthreads();
    for (unsigned r = wave; r < nr; r += 8U) {
        float v[4];
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) v[j] = lds.total[r][lane + 32U * j];
        d::hadamard128(v, lane);
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
            const unsigned o = col0 + lane + 32U * j;
            rows.output(r)[o] = v[j] * d::half_bits(m.svh[o]);
        }
    }
    __syncthreads(); // finish epilogue reads before another row block stages operands
}
} // namespace strix::glm::pf::exact_x
