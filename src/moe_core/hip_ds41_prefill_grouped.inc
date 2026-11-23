// DS41 grouped integer prefill; the adapter supplies Ds41PrefillTraits.
#ifndef STRIX_DS41_PREFILL_STAGE_CLASSES
#define STRIX_DS41_PREFILL_STAGE_CLASSES 8
#endif
constexpr unsigned stage_classes = STRIX_DS41_PREFILL_STAGE_CLASSES;
constexpr unsigned max_groups_per_class = 5U; // widths up to 5120
// The 32-class reduction order (visiting walk, carry-stack fold) is the shared contract moe_core/class_tree.hpp.
using ::strix::moe_core::class_tree::class_of;
__device__ __forceinline__ unsigned tile_position(unsigned r) { return (r & ~15U) | ((r & 1U) << 3U) | ((r & 15U) >> 1U); }

struct StageLayout {
    unsigned rows, groups_per_class, slots; // rows = 16 * MT, slots = 8 * J
    __device__ __host__ unsigned code_bytes() const { return slots * 2U * rows * 16U; }
    __device__ __host__ unsigned delta_bytes() const { return slots * rows * 4U; }
    __device__ __host__ unsigned sum_bytes(bool q2) const { return q2 ? slots * 2U * rows * 4U : 0U; }
    __device__ __host__ unsigned bytes(bool q2) const { return code_bytes() + delta_bytes() + sum_bytes(q2); }
};

template <bool Q2>
__device__ __forceinline__ void stage_activations(unsigned char* lds, const StageLayout& s, unsigned stage, unsigned width,
                                                  const unsigned* row_of, unsigned count, const signed char* codes,
                                                  const float* deltas) {
    const unsigned groups = width / 32U;
    auto* a = reinterpret_cast<v4i*>(lds);
    auto* d = reinterpret_cast<float*>(lds + s.code_bytes());
    auto* sums = reinterpret_cast<int*>(lds + s.code_bytes() + s.delta_bytes());
    const unsigned items = s.slots * 2U * s.rows;
    for (unsigned item = threadIdx.x; item < items; item += blockDim.x) {
        const unsigned r = item % s.rows, kh = (item / s.rows) % 2U, q = item / (2U * s.rows);
        const unsigned u = q / s.groups_per_class, j = q % s.groups_per_class;
        const unsigned g = class_of(stage * stage_classes + u) + 32U * j;
        if (g >= groups) continue;
        v4i v{0, 0, 0, 0};
        const bool live = r < count;
        if (live) v = *reinterpret_cast<const v4i*>(codes + static_cast<unsigned long long>(row_of[r]) * width + 32U * g + 16U * kh);
        a[(2U * q + kh) * s.rows + r] = v;
        const unsigned at = tile_position(r);
        if (kh == 0U) d[q * s.rows + at] = live ? deltas[static_cast<unsigned long long>(row_of[r]) * groups + g] : 0.0F;
        if constexpr (Q2) {
            const int ones = 0x01010101;
            int sum = 0;
            sum = STRIX_HIP_SUDOT4(false, ones, true, v.x, sum, false);
            sum = STRIX_HIP_SUDOT4(false, ones, true, v.y, sum, false);
            sum = STRIX_HIP_SUDOT4(false, ones, true, v.z, sum, false);
            sum = STRIX_HIP_SUDOT4(false, ones, true, v.w, sum, false);
            sums[(2U * q + kh) * s.rows + at] = sum;
        }
    }
}

// The wave's 16 outputs x MT row tiles of one weight matrix: v[t][e] for (row 16t + 2e + h, output
// lane % 16). Every workgroup thread must call it (it stages and synchronizes).
template <WeightFormat Format, unsigned MT, unsigned J>
__device__ __forceinline__ void wave_products(unsigned char* lds, const StageLayout& s, const unsigned char* weights_row,
                                              unsigned width, const unsigned* row_of, unsigned count, const signed char* codes,
                                              const float* deltas, const unsigned long long* grid, float (&v)[MT][8]) {
    constexpr bool q2 = Format == WeightFormat::q2_k;
    const unsigned lane = threadIdx.x % 32U, i16 = lane & 15U, h = lane >> 4U, groups = width / 32U;
    const auto* a = reinterpret_cast<const v4i*>(lds);
    const auto* d = reinterpret_cast<const float*>(lds + s.code_bytes());
    const auto* sums = reinterpret_cast<const int*>(lds + s.code_bytes() + s.delta_bytes());
    // Carry stack of the butterfly tree: stack[k] holds a finished subtree of 2^k classes.
    float stack[5][MT][8];
#pragma unroll
    for (unsigned k = 0; k < 5U; ++k)
#pragma unroll
        for (unsigned t = 0; t < MT; ++t)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) stack[k][t][e] = 0.0F;
    // Weight pipeline: raws[j] holds group j of the next class (in visiting order) that has a
    // group j; once it is decoded, the same group of the following such class is requested into the
    // slot, so up to J loads stay in flight.
    const auto next_with = [&](unsigned i, unsigned j) {
        while (i < 32U && class_of(i) + 32U * j >= groups) ++i;
        return i;
    };
    Raw<Format> raws[J];
#pragma unroll
    for (unsigned j = 0; j < J; ++j) {
        const unsigned first = next_with(0U, j);
        if (first < 32U) raws[j] = load_raw<Format>(weights_row, class_of(first) + 32U * j);
    }
#pragma unroll 1
    for (unsigned stage = 0; stage < 32U / stage_classes; ++stage) {
        __syncthreads();
        stage_activations<q2>(lds, s, stage, width, row_of, count, codes, deltas);
        __syncthreads();
#pragma unroll 1
        for (unsigned u = 0; u < stage_classes; ++u) {
            const unsigned i = stage * stage_classes + u, l = class_of(i);
            float p[MT][8];
#pragma unroll
            for (unsigned t = 0; t < MT; ++t)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) p[t][e] = 0.0F;
            if (l < groups) {
#pragma unroll
                for (unsigned j = 0; j < J; ++j) {
                    if (l + 32U * j >= groups) continue;
                    const Operands w = decode(raws[j], grid);
                    const unsigned n = next_with(i + 1U, j);
                    if (n < 32U) raws[j] = load_raw<Format>(weights_row, class_of(n) + 32U * j);
                    const unsigned q = u * J + j;
#pragma unroll
                    for (unsigned t = 0; t < MT; ++t) {
                        if (16U * t >= count) continue;
                        const v4i a0 = a[(2U * q) * s.rows + 16U * t + i16];
                        const v4i a1 = a[(2U * q + 1U) * s.rows + 16U * t + i16];
                        const v8i r = group_dot(a0, w.b0, a1, w.b1);
                        const float* dt = d + q * s.rows + 16U * t + 8U * h;
                        if constexpr (q2) {
                            const int* s0 = sums + (2U * q) * s.rows + 16U * t + 8U * h;
                            const int* s1 = sums + (2U * q + 1U) * s.rows + 16U * t + 8U * h;
#pragma unroll
                            for (unsigned e = 0; e < 8U; ++e) {
                                const int b = __mul24(w.m0, s0[e]) + __mul24(w.m1, s1[e]);
                                const float scaled = w.d * magic_float(r[e]);
                                const float offset = w.dmin * static_cast<float>(b);
                                p[t][e] = p[t][e] + (scaled - offset) * dt[e];
                            }
                        } else {
#pragma unroll
                            for (unsigned e = 0; e < 8U; ++e) p[t][e] = p[t][e] + magic_float(r[e]) * (w.scale * dt[e]);
                        }
                    }
                }
            }
            // Fold class sum i into the tree (shared carry-stack fold).
            ::strix::moe_core::class_tree::fold(i, p, stack, v);
        }
    }
}

struct Batch { unsigned expert, begin, count; };
__device__ __forceinline__ bool load_batch(const PrefillRoutesView& routes, Batch& b) {
    if (blockIdx.x >= *routes.batch_total) return false;
    b.expert = routes.batch_expert[blockIdx.x];
    b.begin = routes.batch_begin[blockIdx.x];
    b.count = routes.batch_count[blockIdx.x];
    return true;
}
__device__ __forceinline__ void load_grid(unsigned long long* grid) {
    for (unsigned t = threadIdx.x; t < 256U; t += blockDim.x) grid[t] = prefill_iq2_grid[t];
}

constexpr unsigned prefill_waves = 8U;

// Gate/up: wave w runs matrix w % 2 (gate, up) for outputs 16 (w / 2) .. of the block's 64.
template <WeightFormat Format, unsigned MT, unsigned J>
__global__ __launch_bounds__(32U * prefill_waves) void prefill_gate_up_kernel(PrefillGateUpView view, StageLayout layout) {
    extern __shared__ int4 dynamic_lds[];
    auto* lds = reinterpret_cast<unsigned char*>(dynamic_lds);
    __shared__ unsigned long long grid[Format == WeightFormat::iq2_xxs ? 256 : 1];
    __shared__ unsigned row_of[32], slot_of[32];
    Batch batch;
    if (!load_batch(view.routes, batch)) return;
    if constexpr (Format == WeightFormat::iq2_xxs) load_grid(grid);
    if (threadIdx.x < batch.count) {
        row_of[threadIdx.x] = view.routes.entry_rows[batch.begin + threadIdx.x];
        slot_of[threadIdx.x] = view.routes.entry_slots[batch.begin + threadIdx.x];
    }
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U, matrix = wave & 1U, tile = wave >> 1U;
    const unsigned out = blockIdx.y * 64U + 16U * tile + (lane & 15U), h = lane >> 4U;
    const auto* weights = static_cast<const unsigned char*>(matrix == 0U ? view.gate : view.up) +
        batch.expert * view.expert_stride_bytes + static_cast<unsigned long long>(out) * view.row_bytes;
    float v[MT][8];
    wave_products<Format, MT, J>(lds, layout, weights, view.hidden, row_of, batch.count, view.codes, view.deltas, grid, v);
    __syncthreads(); // every wave is done with the stage: the tile reuses it
    auto* values = reinterpret_cast<float*>(lds); // [matrix][row][64]
#pragma unroll
    for (unsigned t = 0; t < MT; ++t)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) {
            const unsigned r = 16U * t + 2U * e + h;
            if (r < batch.count) values[(matrix * 32U + r) * 64U + 16U * tile + (lane & 15U)] = v[t][e];
        }
    __syncthreads();
    // The decode window's epilogue for each (entry, 32-output group), one wave per pair.
    const unsigned groups = view.inter / 32U;
    for (unsigned job = wave; job < batch.count * 2U; job += prefill_waves) {
        const unsigned e = job / 2U, quad = job % 2U, at = blockIdx.y * 64U + 32U * quad + lane;
        const unsigned long long slot = slot_of[e];
        const float g = values[e * 64U + 32U * quad + lane], u = values[(32U + e) * 64U + 32U * quad + lane];
        if (view.gate_out) { view.gate_out[slot * view.inter + at] = g; view.up_out[slot * view.inter + at] = u; }
        const float act = swiglu_clamped(g, u, view.limit);
        float amax = fabsf(act);
        for (int stride = 16; stride; stride /= 2) amax = fmaxf(amax, __shfl_xor(amax, stride, 32));
        const bool tiny = amax > 0.0F && amax < 0x1p-119F;
        const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
        const float rounded = tiny ? 0.0F : rintf(act / delta);
        const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
        int sum = code;
        for (int stride = 16; stride; stride /= 2) sum += __shfl_xor(sum, stride, 32);
        view.act_codes[slot * view.inter + at] = static_cast<signed char>(code);
        const unsigned group = at / 32U;
        if (lane == 0U) { view.act_deltas[slot * groups + group] = delta; view.act_sums[slot * groups + group] = sum; }
    }
}

// Down: wave w runs outputs 16 w .. of the block's 128; raw slot vectors straight from the tile.
template <WeightFormat Format, unsigned MT, unsigned J>
__global__ __launch_bounds__(32U * prefill_waves) void prefill_down_kernel(PrefillDownView view, StageLayout layout) {
    extern __shared__ int4 dynamic_lds[];
    auto* lds = reinterpret_cast<unsigned char*>(dynamic_lds);
    __shared__ unsigned slot_of[32];
    Batch batch;
    if (!load_batch(view.routes, batch)) return;
    if (threadIdx.x < batch.count) slot_of[threadIdx.x] = view.routes.entry_slots[batch.begin + threadIdx.x];
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U, h = lane >> 4U;
    const unsigned out = blockIdx.y * 128U + 16U * wave + (lane & 15U);
    const auto* weights = static_cast<const unsigned char*>(view.weights) + batch.expert * view.expert_stride_bytes +
        static_cast<unsigned long long>(out) * view.row_bytes;
    float v[MT][8];
    wave_products<Format, MT, J>(lds, layout, weights, view.width, slot_of, batch.count, view.codes, view.deltas, nullptr, v);
#pragma unroll
    for (unsigned t = 0; t < MT; ++t)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) {
            const unsigned r = 16U * t + 2U * e + h;
            if (r < batch.count) view.out[static_cast<unsigned long long>(slot_of[r]) * view.outputs + out] = v[t][e];
        }
}

// ---- Class-split workgroups (default for batches of 16 rows) ----
//
// The stage kernels above walk every weight row in the lane-class order, so each cache line of a row
// is touched in eight separate passes; with the working set of all resident waves beyond the L2, the
// weights come from the last-level cache about eight times. Here a workgroup is 2 x 8 waves:
// class-wave q (0..7) owns lane classes 4q .. 4q + 3, i.e. one half of block q / 2 + 4j of every
// weight row (blocks of 256 weights), and all class-waves walk the rows block by block in address
// order. The activations of the batch (16 rows) are staged one chunk of 32 groups (1024 values) at a
// time; chunk j feeds group 4q + k + 32j of class-wave q, so each class still accumulates its groups
// in ascending order. A wave covers 32 outputs (two 16-output tiles): lane i of half h decodes output
// 16h + i once and the halves trade their operands with v_permlanex16, so no group is decoded twice.
// At the end the class sums meet in the butterfly tree: levels 1-3 (class bits 4, 3, 2 = wave bits
// 2, 1, 0) pair waves q and q + 4, q + 2, q + 1 through LDS, and wave 0 finishes levels 4-5 (class
// bits 1, 0) in registers.
// Waves per SIMD the split kernels are register-allocated for (8: at most 192 VGPRs, two workgroups
// per WGP).
#ifndef STRIX_DS41_SPLIT_WAVES
#define STRIX_DS41_SPLIT_WAVES 8
#endif
#ifndef STRIX_DS41_SPLIT_DOWN_WAVES
#define STRIX_DS41_SPLIT_DOWN_WAVES 8
#endif
constexpr unsigned split_rows = 16U;
constexpr unsigned split_class_waves = 8U;
constexpr unsigned split_classes = 32U / split_class_waves;                            // 4 per wave
constexpr unsigned split_chunk_groups = 32U;
constexpr unsigned split_code_bytes = split_chunk_groups * 2U * split_rows * 16U;     // 16 KB
constexpr unsigned split_delta_bytes = split_chunk_groups * split_rows * 4U;          // 2 KB
constexpr unsigned split_sum_bytes = split_chunk_groups * split_rows * 4U;            // 2 KB
constexpr unsigned split_leaf_floats = split_classes * 8U * 32U;                      // one wave, one tile
constexpr unsigned split_exchange_bytes = 2U * 4U * split_leaf_floats * 4U;           // 32 KB
constexpr unsigned split_stage_bytes = split_code_bytes + split_delta_bytes + split_sum_bytes;      // 20 KB
// Activations double-buffered: chunk c + 1 is read from memory while chunk c is multiplied.
#ifndef STRIX_DS41_SPLIT_DOUBLE
#define STRIX_DS41_SPLIT_DOUBLE 0
#endif
// Weight chunks in flight ahead of the one being multiplied (1 or 2).
#ifndef STRIX_DS41_SPLIT_WEIGHT_DEPTH
#define STRIX_DS41_SPLIT_WEIGHT_DEPTH 2
#endif
constexpr unsigned split_weight_depth = STRIX_DS41_SPLIT_WEIGHT_DEPTH;
constexpr unsigned split_stage_buffers = STRIX_DS41_SPLIT_DOUBLE ? 2U : 1U;
constexpr unsigned split_threads = 2U * split_class_waves * 32U;

template <bool Q2>
__device__ __forceinline__ void stage_chunk(unsigned char* lds, unsigned chunk, unsigned width, const unsigned* row_of, unsigned count,
                                            const signed char* codes, const float* deltas) {
    const unsigned groups = width / 32U;
    auto* a = reinterpret_cast<v4i*>(lds);
    auto* d = reinterpret_cast<float*>(lds + split_code_bytes);
    // Q2_K half sums, packed per (group, row) as two int16: low = codes 0-15, high = codes 16-31.
    auto* sums = reinterpret_cast<short*>(lds + split_code_bytes + split_delta_bytes);
    for (unsigned item = threadIdx.x; item < split_chunk_groups * 2U * split_rows; item += blockDim.x) {
        const unsigned r = item % split_rows, kh = (item / split_rows) % 2U, gg = item / (2U * split_rows);
        const unsigned g = split_chunk_groups * chunk + gg;
        if (g >= groups) continue;
        v4i v{0, 0, 0, 0};
        const bool live = r < count;
        if (live) v = *reinterpret_cast<const v4i*>(codes + static_cast<unsigned long long>(row_of[r]) * width + 32U * g + 16U * kh);
        a[(2U * gg + kh) * split_rows + r] = v;
        const unsigned at = tile_position(r);
        if (kh == 0U) d[gg * split_rows + at] = live ? deltas[static_cast<unsigned long long>(row_of[r]) * groups + g] : 0.0F;
        if constexpr (Q2) {
            const int ones = 0x01010101;
            int sum = 0;
            sum = STRIX_HIP_SUDOT4(false, ones, true, v.x, sum, false);
            sum = STRIX_HIP_SUDOT4(false, ones, true, v.y, sum, false);
            sum = STRIX_HIP_SUDOT4(false, ones, true, v.z, sum, false);
            sum = STRIX_HIP_SUDOT4(false, ones, true, v.w, sum, false);
            sums[2U * (gg * split_rows + at) + kh] = static_cast<short>(sum); // |sum| <= 16 * 127
        }
    }
}

// The staging of stage_chunk in two steps: the global reads of a thread's two items into registers,
// then their LDS writes, so the reads of chunk c + 1 are in flight while chunk c is multiplied.
struct StagedItems { v4i v[2]; float d[2]; };
[[maybe_unused]] __device__ __forceinline__ StagedItems load_chunk_items(unsigned chunk, unsigned width, const unsigned* row_of, unsigned count,
                                                        const signed char* codes, const float* deltas) {
    const unsigned groups = width / 32U;
    StagedItems x{};
#pragma unroll
    for (unsigned i = 0; i < 2U; ++i) {
        const unsigned item = threadIdx.x + split_threads * i;
        const unsigned r = item % split_rows, kh = (item / split_rows) % 2U, gg = item / (2U * split_rows);
        const unsigned g = split_chunk_groups * chunk + gg;
        x.v[i] = v4i{0, 0, 0, 0};
        x.d[i] = 0.0F;
        if (g < groups && r < count) {
            x.v[i] = *reinterpret_cast<const v4i*>(codes + static_cast<unsigned long long>(row_of[r]) * width + 32U * g + 16U * kh);
            if (kh == 0U) x.d[i] = deltas[static_cast<unsigned long long>(row_of[r]) * groups + g];
        }
    }
    return x;
}
template <bool Q2>
__device__ __forceinline__ void store_chunk_items(unsigned char* lds, const StagedItems& x, unsigned chunk, unsigned width) {
    const unsigned groups = width / 32U;
    auto* a = reinterpret_cast<v4i*>(lds);
    auto* d = reinterpret_cast<float*>(lds + split_code_bytes);
    auto* sums = reinterpret_cast<short*>(lds + split_code_bytes + split_delta_bytes);
#pragma unroll
    for (unsigned i = 0; i < 2U; ++i) {
        const unsigned item = threadIdx.x + split_threads * i;
        const unsigned r = item % split_rows, kh = (item / split_rows) % 2U, gg = item / (2U * split_rows);
        if (split_chunk_groups * chunk + gg >= groups) continue;
        const v4i v = x.v[i];
        a[(2U * gg + kh) * split_rows + r] = v;
        const unsigned at = tile_position(r);
        if (kh == 0U) d[gg * split_rows + at] = x.d[i];
        if constexpr (Q2) {
            const int ones = 0x01010101;
            int sum = 0;
            sum = STRIX_HIP_SUDOT4(false, ones, true, v.x, sum, false);
            sum = STRIX_HIP_SUDOT4(false, ones, true, v.y, sum, false);
            sum = STRIX_HIP_SUDOT4(false, ones, true, v.z, sum, false);
            sum = STRIX_HIP_SUDOT4(false, ones, true, v.w, sum, false);
            sums[2U * (gg * split_rows + at) + kh] = static_cast<short>(sum);
        }
    }
}

// Value of lane l ^ 16 (the same lane of the other half-wave). A cross-lane read: call it only
// where every lane of the wave is active (never inside a lane-dependent branch or select).
__device__ __forceinline__ int other_half(int v) {
    const auto u = static_cast<unsigned>(v);
    return static_cast<int>(STRIX_HIP_EXCHANGE16(u, u, false, false));
}
__device__ __forceinline__ float other_half(float v) { return __int_as_float(other_half(__float_as_int(v))); }
__device__ __forceinline__ v4i other_half(v4i v) { return v4i{other_half(v.x), other_half(v.y), other_half(v.z), other_half(v.w)}; }
__device__ __forceinline__ v4i pick(bool first, v4i a, v4i b) { return first ? a : b; }

// The four groups 4q .. 4q + 3 of one chunk (one half of a 256-weight block, or four consecutive
// MXFP4 blocks), loaded with one contiguous read per lane a whole chunk ahead of their use.
template <WeightFormat Format> struct ChunkRaw;
template <> struct ChunkRaw<WeightFormat::iq2_xxs> { unsigned short d; unsigned long long aux[4]; };
template <> struct ChunkRaw<WeightFormat::q2_k> { unsigned q[8]; unsigned scales[2]; unsigned dm; };
template <> struct ChunkRaw<WeightFormat::mxfp4> { unsigned bytes[17]; };
template <WeightFormat Format> inline constexpr bool chunk_prefetch = Format != WeightFormat::q8_0;
template <> struct ChunkRaw<WeightFormat::q8_0> { unsigned unused; };

// first = the chunk's first group of this wave (a multiple of 4).
template <WeightFormat Format>
__device__ __forceinline__ ChunkRaw<Format> load_chunk(const unsigned char* row, unsigned first) {
    ChunkRaw<Format> c{};
    if constexpr (Format == WeightFormat::iq2_xxs) {
        const auto* block = row + first / 8U * iq2_xxs_block_bytes;
        c.d = load_u16(block);
        __builtin_memcpy(c.aux, block + 2U + 8U * (first % 8U), 32);
    } else if constexpr (Format == WeightFormat::q2_k) {
        const auto* block = row + first / 8U * q2_k_block_bytes;
        const unsigned half = (first % 8U) / 4U;
        __builtin_memcpy(c.q, block + 16U + 32U * half, 32);
        __builtin_memcpy(c.scales, block + 8U * half, 8);
        __builtin_memcpy(&c.dm, block + 80U, 4);
    } else if constexpr (Format == WeightFormat::mxfp4) {
        __builtin_memcpy(c.bytes, row + first * mxfp4_block_bytes, 68);
    }
    return c;
}
[[maybe_unused]] __device__ __forceinline__ unsigned byte_at(const unsigned (&words)[17], unsigned i) { return (words[i / 4U] >> (8U * (i % 4U))) & 0xffU; }
[[maybe_unused]] __device__ __forceinline__ unsigned word_at(const unsigned (&words)[17], unsigned i) {
    // Bytes i .. i + 3 (i constant after unrolling).
    if (i % 4U == 0U) return words[i / 4U];
    return __builtin_amdgcn_alignbyte(words[i / 4U + 1U], words[i / 4U], i % 4U);
}
// Group k of the chunk as the per-group raw form the decoders take.
template <WeightFormat Format>
__device__ __forceinline__ Raw<Format> chunk_group(const ChunkRaw<Format>& c, unsigned k) {
    Raw<Format> r{};
    if constexpr (Format == WeightFormat::iq2_xxs) {
        r.d = c.d;
        r.aux = c.aux[k];
    } else if constexpr (Format == WeightFormat::q2_k) {
#pragma unroll
        for (unsigned i = 0; i < 8U; ++i) r.q[i] = c.q[i];
        r.scales = static_cast<unsigned short>((c.scales[k / 2U] >> (16U * (k % 2U))) & 0xffffU);
        r.dm = c.dm;
        r.shift = 2U * k;
    } else if constexpr (Format == WeightFormat::mxfp4) {
        r.e = byte_at(c.bytes, 17U * k);
#pragma unroll
        for (unsigned i = 0; i < 4U; ++i) r.q[i] = word_at(c.bytes, 17U * k + 1U + 4U * i);
    }
    return r;
}

// Weight tiles in LDS (default for IQ2_XXS, Q2_K, MXFP4): the workgroup's 64 weight rows (two runs of
// 32 consecutive rows: gate and up, or the two output pairs of down) are read chunk by chunk with
// coalesced dword loads (consecutive lanes, consecutive bytes of one row) into an LDS tile, and each
// lane decodes its own row's groups from there. Read per lane from global memory, every lane of a
// load touched its own cache line (32 rows per instruction). A chunk is 32 groups: 4 blocks of 256
// weights (IQ2_XXS 264 bytes, Q2_K 336) or 32 MXFP4 blocks (544), always dword-aligned in a row.
#ifndef STRIX_DS41_SPLIT_WEIGHT_LDS
#define STRIX_DS41_SPLIT_WEIGHT_LDS 1
#endif
template <WeightFormat Format> inline constexpr unsigned chunk_row_bytes =
    Format == WeightFormat::iq2_xxs ? 4U * iq2_xxs_block_bytes
    : Format == WeightFormat::q2_k  ? 4U * q2_k_block_bytes
    : Format == WeightFormat::mxfp4 ? 32U * mxfp4_block_bytes
                                    : 32U * q8_0_block_bytes;
template <WeightFormat Format> inline constexpr bool weight_lds = STRIX_DS41_SPLIT_WEIGHT_LDS != 0 && Format != WeightFormat::q8_0;
constexpr unsigned tile_rows = 64U;
template <WeightFormat Format> inline constexpr unsigned tile_words = tile_rows * chunk_row_bytes<Format> / 4U;
template <WeightFormat Format> inline constexpr unsigned tile_loads = (tile_words<Format> + split_threads - 1U) / split_threads;
// Activation stage buffers of a form: 2 (double-buffered) from form 3 on, else the build default.
template <int Form> inline constexpr unsigned stage_buffers_of = Form == 3 ? 2U : split_stage_buffers;
// Forms 6 (= 1) and 7 (= 6 with a batched activation stage): the chunk loop's barriers are LDS-only (fence
// workgroup/local + s_barrier), so the next weight tile stays in flight across them (__syncthreads drains vmcnt; barrier measurements
// thread trace: 2 x 7 % of the gate/up kernel's latency). The same loads, stores and products: the same bits.
template <int Form> __device__ __forceinline__ void split_sync() {
    if constexpr (Form >= 5) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
    } else {
        __syncthreads();
    }
}
// Form 4: one barrier before the tile store and the activation stage, the stage's reads before the next tile's (so
// the in-order load counter never makes the stage wait for the prefetch), then one barrier: two per chunk, not three.
template <WeightFormat Format, int Form> inline constexpr bool merged_stage = Form == 4 && weight_lds<Format> && stage_buffers_of<Form> == 1U;
template <WeightFormat Format, unsigned StageBuffers = split_stage_buffers>
__host__ __device__ constexpr unsigned split_lds_for() {
    const unsigned tile = weight_lds<Format> ? tile_rows * chunk_row_bytes<Format> : 0U;
    const unsigned work = StageBuffers * split_stage_bytes + tile;
    return work > split_exchange_bytes ? work : split_exchange_bytes;
}
struct TileRows { const unsigned char* base[2]; std::uint64_t row_bytes; };
template <WeightFormat Format> struct TileRegs { unsigned w[tile_loads<Format>]; };
template <WeightFormat Format>
__device__ __forceinline__ TileRegs<Format> load_tile(const TileRows& t, unsigned chunk, unsigned groups) {
    constexpr unsigned row_words = chunk_row_bytes<Format> / 4U;
    TileRegs<Format> r{};
    if (split_chunk_groups * chunk >= groups) return r;
    const std::uint64_t at = static_cast<std::uint64_t>(chunk) * chunk_row_bytes<Format>;
#pragma unroll
    for (unsigned i = 0; i < tile_loads<Format>; ++i) {
        const unsigned word = threadIdx.x + split_threads * i;
        if (word < tile_words<Format>) {
            const unsigned row = word / row_words, off = word % row_words;
            const auto* src = t.base[row / 32U] + static_cast<std::uint64_t>(row % 32U) * t.row_bytes + at;
            r.w[i] = reinterpret_cast<const unsigned*>(src)[off];
        }
    }
    return r;
}
template <WeightFormat Format>
__device__ __forceinline__ void store_tile(unsigned char* tile, const TileRegs<Format>& r) {
#pragma unroll
    for (unsigned i = 0; i < tile_loads<Format>; ++i) {
        const unsigned word = threadIdx.x + split_threads * i;
        if (word < tile_words<Format>) reinterpret_cast<unsigned*>(tile)[word] = r.w[i];
    }
}

// Leaves leaf[k][t][e]: class 4q + k, output tile t, D element e (row 2e + h, output 16t + i).
template <WeightFormat Format, int Form>
__device__ __forceinline__ void class_wave(unsigned char* lds, unsigned q, const unsigned char* weights_row, unsigned width,
                                           const unsigned* row_of, unsigned count, const signed char* codes, const float* deltas,
                                           const unsigned long long* grid, const SignEntry* signs, float (&leaf)[split_classes][2][8], const TileRows& tile_rows_src,
                                           unsigned tile_row) {
    constexpr bool q2 = Format == WeightFormat::q2_k;
    const unsigned lane = threadIdx.x % 32U, i16 = lane & 15U, h = lane >> 4U, groups = width / 32U;
    const bool upper = h != 0U;
#pragma unroll
    for (unsigned k = 0; k < split_classes; ++k)
#pragma unroll
        for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) leaf[k][t][e] = 0.0F;
    const unsigned chunks = (groups + split_chunk_groups - 1U) / split_chunk_groups;
    // Weights: the chunk's four groups arrive one chunk ahead (Q8_0, whose shared-expert rows stay
    // in the L2, keeps a one-group lookahead).
    ChunkRaw<Format> chunk_now{}, chunk_next{}, chunk_after{};
    Raw<Format> cur{};
    constexpr bool wl = weight_lds<Format>;
    constexpr unsigned SB = stage_buffers_of<Form>;
    unsigned char* tile = lds + SB * split_stage_bytes;
    TileRegs<Format> tile_next{};
    if constexpr (wl) {
        (void)chunk_now; (void)chunk_next; (void)chunk_after;
        tile_next = load_tile<Format>(tile_rows_src, 0U, groups);
    } else if constexpr (chunk_prefetch<Format>) {
        if (split_classes * q < groups) chunk_next = load_chunk<Format>(weights_row, split_classes * q);
        if constexpr (split_weight_depth > 1U)
            if (split_classes * q + split_chunk_groups < groups) chunk_after = load_chunk<Format>(weights_row, split_classes * q + split_chunk_groups);
    } else {
        if (split_classes * q < groups) cur = load_raw<Format>(weights_row, split_classes * q);
    }
#pragma unroll 1
    for (unsigned chunk = 0; chunk < chunks; ++chunk) {
        const unsigned base = split_chunk_groups * chunk + split_classes * q;
        if constexpr (merged_stage<Format, Form>) {
            split_sync<Form>();
            store_tile<Format>(tile, tile_next);
            stage_chunk<q2>(lds, chunk, width, row_of, count, codes, deltas);
            if (chunk + 1U < chunks) tile_next = load_tile<Format>(tile_rows_src, chunk + 1U, groups);
        } else if constexpr (wl) {
            // The tile of chunk c: previous readers are done (barrier), then the stage below is
            // written and the next tile's loads go out before the products.
            split_sync<Form>();
            store_tile<Format>(tile, tile_next);
            if (chunk + 1U < chunks) tile_next = load_tile<Format>(tile_rows_src, chunk + 1U, groups);
        } else if constexpr (chunk_prefetch<Format>) {
            chunk_now = chunk_next;
            if constexpr (split_weight_depth > 1U) {
                chunk_next = chunk_after;
                if (base + 2U * split_chunk_groups < groups) chunk_after = load_chunk<Format>(weights_row, base + 2U * split_chunk_groups);
            } else if (base + split_chunk_groups < groups) {
                chunk_next = load_chunk<Format>(weights_row, base + split_chunk_groups);
            }
        }
        unsigned char* buffer = lds + (chunk % SB) * split_stage_bytes;
        StagedItems ahead_items{};
        if constexpr (SB == 2U) {
            if (chunk == 0U) {
                split_sync<Form>(); // the batch's row indices (written by other threads) are visible
                store_chunk_items<q2>(buffer, load_chunk_items(0U, width, row_of, count, codes, deltas), 0U, width);
            }
            if (chunk + 1U < chunks) ahead_items = load_chunk_items(chunk + 1U, width, row_of, count, codes, deltas);
            split_sync<Form>(); // chunk's buffer written; the other buffer's readers (chunk - 1) are done
        } else if constexpr (merged_stage<Format, Form>) {
            split_sync<Form>();
        } else if constexpr (Form == 7) {
            // Form 7 (= 6): both of the thread's stage items are read before either is stored (one wait, not two).
            split_sync<Form>();
            store_chunk_items<q2>(lds, load_chunk_items(chunk, width, row_of, count, codes, deltas), chunk, width);
            split_sync<Form>();
        } else {
            split_sync<Form>();
            stage_chunk<q2>(lds, chunk, width, row_of, count, codes, deltas);
            split_sync<Form>();
        }
        const auto* a = reinterpret_cast<const v4i*>(buffer);
        const auto* d = reinterpret_cast<const float*>(buffer + split_code_bytes);
        const auto* sums = reinterpret_cast<const int*>(buffer + split_code_bytes + split_delta_bytes);
#pragma unroll
        for (unsigned k = 0; k < split_classes; ++k) {
            if (base >= groups) break; // no groups of this wave in the chunk (it still joins the barriers)
            Raw<Format> ahead{};
            (void)ahead;
            if constexpr (wl) {
                cur = load_raw<Format>(tile + tile_row * chunk_row_bytes<Format>, split_classes * q + k);
            } else if constexpr (chunk_prefetch<Format>) {
                cur = chunk_group<Format>(chunk_now, k);
            } else {
                // Next group of this wave: the following one, else the first of its next chunk.
                const unsigned next = k + 1U < split_classes ? base + k + 1U : base + split_chunk_groups;
                if (next < groups) ahead = load_raw<Format>(weights_row, next);
            }
            const Operands own = decode_form<Form>(cur, grid, signs);
            const v4i x0 = other_half(own.b0), x1 = other_half(own.b1);
            // The other half's scale terms (Q2_K: the packed d / dmin and scale bytes).
            const unsigned x_dm = q2 ? static_cast<unsigned>(other_half(static_cast<int>(own.dm_bits))) : 0U;
            const unsigned x_sc = q2 ? static_cast<unsigned>(other_half(static_cast<int>(own.sc_bits))) : 0U;
            const float x_scale = q2 ? 0.0F : other_half(own.scale);
            const unsigned gg = split_classes * q + k;
            const v4i a0 = a[(2U * gg) * split_rows + i16];
            const v4i a1 = a[(2U * gg + 1U) * split_rows + i16];
            const float* dt = d + gg * split_rows + 8U * h;
#pragma unroll
            for (unsigned t = 0; t < 2U; ++t) {
                const bool mine = (t == 1U) == upper; // this half decoded tile t
                const v4i b0 = pick(mine, own.b0, x0), b1 = pick(mine, own.b1, x1);
                const v8i r = group_dot(a0, b0, a1, b1);
                if constexpr (q2) {
                    const unsigned dmb = mine ? own.dm_bits : x_dm, scb = mine ? own.sc_bits : x_sc;
                    const float dd = half_bits(static_cast<unsigned short>(dmb & 0xffffU));
                    const float dm = half_bits(static_cast<unsigned short>(dmb >> 16U));
                    // B = (sc0 >> 4) * S0 + (sc1 >> 4) * S1 from the packed half sums (exact integers).
                    const int m0 = static_cast<int>((scb & 0xffU) >> 4U), m1 = static_cast<int>((scb >> 8U) >> 4U);
                    const int* sp = sums + gg * split_rows + 8U * h;
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) {
                        const int packed = sp[e];
                        const int s0 = static_cast<int>(static_cast<short>(packed & 0xffff)), s1 = packed >> 16;
                        const int b = __mul24(m0, s0) + __mul24(m1, s1);
                        const float scaled = dd * magic_float(r[e]);
                        const float offset = dm * static_cast<float>(b);
                        leaf[k][t][e] = leaf[k][t][e] + (scaled - offset) * dt[e];
                    }
                } else {
                    const float scale = mine ? own.scale : x_scale;
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) leaf[k][t][e] = leaf[k][t][e] + magic_float(r[e]) * (scale * dt[e]);
                }
            }
            if constexpr (!chunk_prefetch<Format>) cur = ahead;
        }
        if constexpr (SB == 2U)
            if (chunk + 1U < chunks)
                store_chunk_items<q2>(lds + ((chunk + 1U) % 2U) * split_stage_bytes, ahead_items, chunk + 1U, width);
    }
}

// Combines the leaves of the eight class-waves of one pair into the butterfly result, held by
// class-wave 0 in v[t][e]. Every thread of the workgroup calls it.
__device__ __forceinline__ void split_tree(unsigned char* lds, unsigned q, unsigned pair, float (&leaf)[split_classes][2][8],
                                           float (&v)[2][8]) {
    auto* x = reinterpret_cast<float*>(lds);
    const unsigned lane = threadIdx.x % 32U;
#pragma unroll
    for (unsigned t = 0; t < 2U; ++t) {
        // Levels 1-3: wave q + s into wave q for s = 4, 2, 1 (class bits 4, 3, 2).
#pragma unroll
        for (unsigned s = 4U; s != 0U; s /= 2U) {
            __syncthreads();
            if (q >= s && q < 2U * s) {
                float* slot = x + (4U * pair + (q - s)) * split_leaf_floats;
#pragma unroll
                for (unsigned k = 0; k < split_classes; ++k)
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) slot[(8U * k + e) * 32U + lane] = leaf[k][t][e];
            }
            __syncthreads();
            if (q < s) {
                const float* slot = x + (4U * pair + q) * split_leaf_floats;
#pragma unroll
                for (unsigned k = 0; k < split_classes; ++k)
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) leaf[k][t][e] = leaf[k][t][e] + slot[(8U * k + e) * 32U + lane];
            }
        }
        // Levels 4-5 over the low two bits of the class.
        if (q == 0U)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const float c0 = leaf[0][t][e] + leaf[2][t][e], c1 = leaf[1][t][e] + leaf[3][t][e];
                v[t][e] = c0 + c1;
            }
    }
}

// Gate/up: wave w = 2q + m (matrix m: gate, up; class-wave q); 32 outputs of both matrices.
template <WeightFormat Format, int Form>
__global__ __launch_bounds__(split_threads) __attribute__((amdgpu_waves_per_eu(STRIX_DS41_SPLIT_WAVES))) void split_gate_up_kernel(PrefillGateUpView view) {
    extern __shared__ int4 dynamic_lds[];
    auto* lds = reinterpret_cast<unsigned char*>(dynamic_lds);
    __shared__ unsigned long long grid[Format == WeightFormat::iq2_xxs ? 256 : 1];
    __shared__ SignEntry signs[Form == 2 && Format == WeightFormat::iq2_xxs ? 128 : 1];
    __shared__ unsigned row_of[split_rows], slot_of[split_rows];
    __shared__ float values[2][split_rows][32];
    Batch batch;
    if (!load_batch(view.routes, batch)) return;
    if constexpr (Format == WeightFormat::iq2_xxs) load_grid(grid);
    if constexpr (Form == 2 && Format == WeightFormat::iq2_xxs) load_signs(signs);
    if (threadIdx.x < batch.count) {
        row_of[threadIdx.x] = view.routes.entry_rows[batch.begin + threadIdx.x];
        slot_of[threadIdx.x] = view.routes.entry_slots[batch.begin + threadIdx.x];
    }
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U, m = wave & 1U, q = wave >> 1U, h = lane >> 4U;
    const unsigned own_out = blockIdx.y * 32U + lane; // lane i of half h: output 16h + i
    const auto* weights = static_cast<const unsigned char*>(m == 0U ? view.gate : view.up) + batch.expert * view.expert_stride_bytes +
        static_cast<unsigned long long>(own_out) * view.row_bytes;
    const std::uint64_t expert_at = batch.expert * view.expert_stride_bytes + static_cast<std::uint64_t>(blockIdx.y) * 32U * view.row_bytes;
    const TileRows rows{{static_cast<const unsigned char*>(view.gate) + expert_at, static_cast<const unsigned char*>(view.up) + expert_at},
                        view.row_bytes};
    float leaf[split_classes][2][8];
    class_wave<Format, Form>(lds, q, weights, view.hidden, row_of, batch.count, view.codes, view.deltas, grid, signs, leaf, rows, 32U * m + lane);
    float v[2][8];
    split_tree(lds, q, m, leaf, v);
    if (q == 0U)
#pragma unroll
        for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const unsigned r = 2U * e + h;
                if (r < batch.count) values[m][r][16U * t + (lane & 15U)] = v[t][e];
            }
    __syncthreads();
    const unsigned groups = view.inter / 32U, at = blockIdx.y * 32U + lane;
    for (unsigned e = wave; e < batch.count; e += split_threads / 32U) {
        const unsigned long long slot = slot_of[e];
        const float g = values[0][e][lane], u = values[1][e][lane];
        if (view.gate_out) { view.gate_out[slot * view.inter + at] = g; view.up_out[slot * view.inter + at] = u; }
        const float act = swiglu_clamped(g, u, view.limit);
        float amax = fabsf(act);
        for (int stride = 16; stride; stride /= 2) amax = fmaxf(amax, __shfl_xor(amax, stride, 32));
        const bool tiny = amax > 0.0F && amax < 0x1p-119F;
        const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
        const float rounded = tiny ? 0.0F : rintf(act / delta);
        const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
        int sum = code;
        for (int stride = 16; stride; stride /= 2) sum += __shfl_xor(sum, stride, 32);
        view.act_codes[slot * view.inter + at] = static_cast<signed char>(code);
        if (lane == 0U) { view.act_deltas[slot * groups + blockIdx.y] = delta; view.act_sums[slot * groups + blockIdx.y] = sum; }
    }
}

// Down: wave w = 2q + p (output pair p: outputs 32p .. of the block's 64; class-wave q).
template <WeightFormat Format, int Form>
__global__ __launch_bounds__(split_threads) __attribute__((amdgpu_waves_per_eu(STRIX_DS41_SPLIT_DOWN_WAVES))) void split_down_kernel(PrefillDownView view) {
    extern __shared__ int4 dynamic_lds[];
    auto* lds = reinterpret_cast<unsigned char*>(dynamic_lds);
    __shared__ unsigned slot_of[split_rows];
    Batch batch;
    if (!load_batch(view.routes, batch)) return;
    if (threadIdx.x < batch.count) slot_of[threadIdx.x] = view.routes.entry_slots[batch.begin + threadIdx.x];
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U, p = wave & 1U, q = wave >> 1U, h = lane >> 4U;
    const unsigned first = blockIdx.y * 64U + 32U * p, own_out = first + lane;
    const auto* weights = static_cast<const unsigned char*>(view.weights) + batch.expert * view.expert_stride_bytes +
        static_cast<unsigned long long>(own_out) * view.row_bytes;
    const auto* tile_base = static_cast<const unsigned char*>(view.weights) + batch.expert * view.expert_stride_bytes +
        static_cast<std::uint64_t>(blockIdx.y) * 64U * view.row_bytes;
    const TileRows rows{{tile_base, tile_base + 32U * view.row_bytes}, view.row_bytes};
    float leaf[split_classes][2][8];
    class_wave<Format, Form>(lds, q, weights, view.width, slot_of, batch.count, view.codes, view.deltas, nullptr, nullptr, leaf, rows, 32U * p + lane);
    float v[2][8];
    split_tree(lds, q, p, leaf, v);
    if (q != 0U) return;
#pragma unroll
    for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) {
            const unsigned r = 2U * e + h;
            if (r < batch.count) view.out[static_cast<unsigned long long>(slot_of[r]) * view.outputs + first + 16U * t + (lane & 15U)] = v[t][e];
        }
}
