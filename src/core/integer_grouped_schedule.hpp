#pragma once
#include "integer_operand.hpp"

namespace strix::core {
// Register/LDS schedule for the chunked integer plan. Keep tensor bases outside
// the loop and array-bearing arguments by reference. No routing or epilogue here.
template<class Operand> struct IntegerGroupedSchedule {
    using Plan = typename Operand::Plan;
    using Ops = typename Operand::Ops;
    using V4 = typename Operand::Int4;
    using V8 = typename Operand::Int8;
    using Raw = typename Operand::Raw;
    static constexpr bool Q2 = Operand::affine;
    static constexpr bool Direct = Operand::direct_weights || !Plan::weight_lds;
    static constexpr unsigned CodeBytes = 16384, DeltaBytes = 2048;
    static constexpr unsigned StageBytes = Operand::activation_bytes + Operand::sum_bytes;
    static constexpr unsigned RowWords = Operand::chunk_row_bytes / 4;
    static constexpr unsigned RunWords = 32 * RowWords, Loads = (RunWords + 511) / 512;
    struct WeightRows { const unsigned char* base[2]; unsigned row_bytes; };
    struct TileRegs { unsigned w[2][Loads]; };
    struct Items { V4 v[2]; float d[2]; };

    static __device__ __forceinline__ void sync() {
        if constexpr (Plan::lds_only_barriers) {
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
        } else __syncthreads();
    }
    static __device__ __forceinline__ int other(int v) {
        int old;
        asm volatile("" : "=v"(old));
        return int(__builtin_amdgcn_permlanex16(unsigned(old), unsigned(v), 0x76543210U, 0xfedcba98U, false, false));
    }
    static __device__ __forceinline__ V4 other4(V4 v) {
        return V4{other(v[0]), other(v[1]), other(v[2]), other(v[3])};
    }
    static __device__ __forceinline__ TileRegs load_tile(const WeightRows& rows, unsigned chunk, unsigned groups) {
        TileRegs r{};
        const unsigned first = chunk * 32;
        if (first >= groups) return r;
        const unsigned left = groups - first;
        const auto* b0 = rows.base[0] + std::uint64_t(chunk) * Operand::chunk_row_bytes;
        const auto* b1 = rows.base[1] + std::uint64_t(chunk) * Operand::chunk_row_bytes;
        const unsigned rb = rows.row_bytes;
        if (!Plan::partial_blocks || left >= 32) {
            integer_unroll<Loads>([&](auto I) __attribute__((always_inline)) {
                constexpr unsigned i = decltype(I)::value;
                const unsigned word = threadIdx.x + 512 * i;
                if (word < RunWords) {
                    const unsigned off = integer_weight_offset(word, RowWords, rb);
                    r.w[0][i] = *reinterpret_cast<const unsigned*>(b0 + off);
                    r.w[1][i] = *reinterpret_cast<const unsigned*>(b1 + off);
                }
            });
        } else {
            const unsigned used = ((left + 7) / 8 * Operand::block_bytes + 3) / 4;
            integer_unroll<Loads>([&](auto I) __attribute__((always_inline)) {
                constexpr unsigned i = decltype(I)::value;
                const unsigned word = threadIdx.x + 512 * i;
                if (word < RunWords && word % RowWords < used) {
                    const unsigned off = integer_weight_offset(word, RowWords, rb);
                    r.w[0][i] = *reinterpret_cast<const unsigned*>(b0 + off);
                    r.w[1][i] = *reinterpret_cast<const unsigned*>(b1 + off);
                }
            });
        }
        return r;
    }
    static __device__ __forceinline__ void store_tile(unsigned char* tile, const TileRegs& r) {
        integer_unroll<Loads>([&](auto I) __attribute__((always_inline)) {
            constexpr unsigned i = decltype(I)::value;
            const unsigned word = threadIdx.x + 512 * i;
            if (word < RunWords) {
                reinterpret_cast<unsigned*>(tile)[word] = r.w[0][i];
                reinterpret_cast<unsigned*>(tile)[RunWords + word] = r.w[1][i];
            }
        });
    }
    static __device__ __forceinline__ Items load_items(unsigned chunk, unsigned groups, const unsigned* sources,
        unsigned count, const signed char* codes, const float* deltas) {
        Items x{};
        integer_unroll<2>([&](auto I) __attribute__((always_inline)) {
            constexpr unsigned i = decltype(I)::value;
            const auto item = integer_stage_item(threadIdx.x, i);
            const unsigned r = item.row, kh = item.half, g = chunk * 32 + item.group;
            if (g < groups && r < count) {
                x.v[i] = *reinterpret_cast<const V4*>(codes + std::uint64_t(sources[r]) * (groups * 32U) + g * 32 + kh * 16);
                if (!kh) x.d[i] = deltas[std::uint64_t(sources[r]) * groups + g];
            }
        });
        return x;
    }
    static __device__ __forceinline__ void store_items(unsigned char* lds, const Items& x, unsigned chunk, unsigned groups) {
        integer_unroll<2>([&](auto I) __attribute__((always_inline)) {
            constexpr unsigned i = decltype(I)::value;
            const auto item = integer_stage_item(threadIdx.x, i);
            const unsigned r = item.row, kh = item.half, gg = item.group;
            if (chunk * 32 + gg >= groups) return;
            const V4 v = x.v[i];
            reinterpret_cast<V4*>(lds)[(2 * gg + kh) * 16 + r] = v;
            const unsigned at_row = integer_row_position(r);
            if (!kh) reinterpret_cast<float*>(lds + CodeBytes)[gg * 16 + at_row] = x.d[i];
            if constexpr (Q2) {
                int sum = 0;
                sum = __builtin_amdgcn_sudot4(false, 0x01010101, true, v[0], sum, false);
                sum = __builtin_amdgcn_sudot4(false, 0x01010101, true, v[1], sum, false);
                sum = __builtin_amdgcn_sudot4(false, 0x01010101, true, v[2], sum, false);
                sum = __builtin_amdgcn_sudot4(false, 0x01010101, true, v[3], sum, false);
                if constexpr (Plan::float_half_sums)
                    reinterpret_cast<float*>(lds + CodeBytes + DeltaBytes)[kh * 512 + gg * 16 + at_row] = float(sum);
                else reinterpret_cast<short*>(lds + CodeBytes + DeltaBytes)[2 * (gg * 16 + at_row) + kh] = short(sum);
            }
        });
    }

    // One class group of a chunk. run() calls it from an unrolled loop that
    // breaks at the first missing group, as the split-v2 kernel does, so each
    // class stays in its own block; k is a constant after unrolling.
    static __device__ __forceinline__ void group(unsigned k, unsigned char* lds, unsigned q, unsigned chunk, unsigned groups,
        const unsigned long long* grid, float (&leaf)[4][2][8], const unsigned char* tile,
        const unsigned char* own_row, Raw& ahead, const V8& magic) {
        const unsigned lane = threadIdx.x % 32U, lo = lane & 15U, h = lane >> 4U;
        const unsigned gg = q + 8 * k;
        Raw raw;
        if constexpr (Direct) {
            raw = ahead;
            const unsigned next = k + 1 < 4 ? chunk * 32 + gg + 8 : (chunk + 1) * 32 + q;
            if (next < groups) ahead = Operand::load_raw(own_row, next);
        } else raw = Operand::load_raw(tile, gg);
        typename Operand::Group own;
        if constexpr (Direct) own = Ops::decode(raw, grid);
        else own = Ops::decode_lean(raw, grid);
        const V4 x0 = other4(own.b0), x1 = other4(own.b1);
        const unsigned xdm = Q2 ? unsigned(other(int(own.dm_bits))) : 0;
        const unsigned xsc = Q2 ? unsigned(other(int(own.sc_bits))) : 0;
        const float xs = Q2 ? 0.0f : __int_as_float(other(__float_as_int(own.scale)));
        const V4 a0 = reinterpret_cast<const V4*>(lds)[2 * gg * 16 + lo];
        const V4 a1 = reinterpret_cast<const V4*>(lds)[(2 * gg + 1) * 16 + lo];
        const auto* dt = reinterpret_cast<const float*>(lds + CodeBytes) + gg * 16 + 8 * h;
        integer_unroll<2>([&](auto T) __attribute__((always_inline)) {
            constexpr unsigned t = decltype(T)::value;
            const bool mine = (t == 1) == (h != 0);
            const V4 b0 = mine ? own.b0 : x0, b1 = mine ? own.b1 : x1;
            V8 dot = integer_wmma<8>(a0, b0, magic);
            dot = integer_wmma<8>(a1, b1, dot);
            if constexpr (Q2) {
                const unsigned dm = mine ? own.dm_bits : xdm, sc = mine ? own.sc_bits : xsc;
                const float d = Ops::half_bits(static_cast<unsigned short>(dm));
                const float dmin = Ops::half_bits(static_cast<unsigned short>(dm >> 16));
                const int m0 = int((sc & 255U) >> 4U), m1 = int((sc >> 8U) >> 4U);
                if constexpr (Plan::float_half_sums) {
                    const auto* s0 = reinterpret_cast<const float*>(lds + CodeBytes + DeltaBytes) + gg * 16 + 8 * h;
                    const float* s1 = s0 + 512;
                    const float m0f = float(m0), m1f = float(m1);
                    integer_unroll<8>([&](auto E) __attribute__((always_inline)) {
                        constexpr unsigned e = decltype(E)::value;
                        const float minimum = __builtin_fmaf(m1f, s1[e], __builtin_fmaf(m0f, s0[e], 0.0f));
                        const float scaled = d * Ops::magic_float(dot[e]);
                        const float offset = dmin * minimum;
                        leaf[k][t][e] = leaf[k][t][e] + (scaled - offset) * dt[e];
                    });
                } else {
                    const auto* sp = reinterpret_cast<const int*>(lds + CodeBytes + DeltaBytes) + gg * 16 + 8 * h;
                    integer_unroll<8>([&](auto E) __attribute__((always_inline)) {
                        constexpr unsigned e = decltype(E)::value;
                        const int packed = sp[e];
                        const int minimum = __mul24(m0, int(short(packed & 0xffff))) + __mul24(m1, packed >> 16);
                        const float scaled = d * Ops::magic_float(dot[e]);
                        const float offset = dmin * float(minimum);
                        leaf[k][t][e] = leaf[k][t][e] + (scaled - offset) * dt[e];
                    });
                }
            } else {
                const float scale = mine ? own.scale : xs;
                integer_unroll<8>([&](auto E) __attribute__((always_inline)) {
                    constexpr unsigned e = decltype(E)::value;
                    leaf[k][t][e] = leaf[k][t][e] + Ops::magic_float(dot[e]) * (scale * dt[e]);
                });
            }
        });
    }

    // The chunk loop is defined once, in the shared schedule rather than in an
    // engine kernel. Leaves are an output reference, never a by-value array.
    static __device__ __forceinline__ void run(unsigned char* lds, unsigned q, unsigned width,
        const unsigned* sources, unsigned count, const signed char* codes, const float* deltas,
        const unsigned long long* grid, float (&leaf)[4][2][8], const WeightRows& weights, unsigned tile_row);
};


template<class Operand>
__device__ __forceinline__ void IntegerGroupedSchedule<Operand>::run(unsigned char* lds, unsigned q, unsigned width,
    const unsigned* sources, unsigned count, const signed char* codes, const float* deltas,
    const unsigned long long* grid, float (&leaf)[4][2][8], const WeightRows& weights, unsigned tile_row) {
    const unsigned groups = width / 32U, chunks = (groups + 31U) / 32U;
    integer_unroll<4>([&](auto K) __attribute__((always_inline)) {
        integer_unroll<2>([&](auto T) __attribute__((always_inline)) {
            integer_unroll<8>([&](auto E) __attribute__((always_inline)) {
                leaf[decltype(K)::value][decltype(T)::value][decltype(E)::value] = 0.0f;
            });
        });
    });
    auto* tile = lds + StageBytes;
    const auto* own_row = (tile_row < 32U ? weights.base[0] : weights.base[1]) + std::uint64_t(tile_row % 32U) * weights.row_bytes;
    Raw ahead{};
    TileRegs tile_next{};
    if constexpr (Direct) ahead = Operand::load_raw(own_row, q);
    else tile_next = load_tile(weights, 0, groups);
    V8 magic{Ops::dot_magic, Ops::dot_magic, Ops::dot_magic, Ops::dot_magic,
             Ops::dot_magic, Ops::dot_magic, Ops::dot_magic, Ops::dot_magic};
    if constexpr (Plan::opaque_magic) asm volatile("" : "+v"(magic));
    #pragma unroll 1
    for (unsigned chunk = 0; chunk < chunks; ++chunk) {
        sync();
        const Items items = load_items(chunk, groups, sources, count, codes, deltas);
        if constexpr (!Direct) {
            store_tile(tile, tile_next);
            if (chunk + 1 < chunks) tile_next = load_tile(weights, chunk + 1, groups);
        }
        store_items(lds, items, chunk, groups);
        sync();
        #pragma unroll
        for (unsigned k = 0; k < 4; ++k) {
            if (chunk * 32 + q + 8 * k >= groups) break;
            group(k, lds, q, chunk, groups, grid, leaf, tile + tile_row * Operand::chunk_row_bytes, own_row, ahead, magic);
        }
    }
}
}
