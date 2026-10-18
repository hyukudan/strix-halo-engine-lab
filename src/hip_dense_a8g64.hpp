#pragma once
// Included by hip_q8_rows_gemv.hpp inside namespace strix::q8_rows (after the dense A8 and FP16 forms).
//
// Dense A8-G64 projections of fast-prefill-f16wmma-v1, component `dense8` (prompt launches of 64..256 rows, a multiple of
// 64; not the dense A8 contract): int8 x int8 products accumulated in int32 over 64 features per scale step.
//   - Activations: per (row, 64-feature group) delta = amax / 127 (IEEE division), code = rint(x / delta) clamped to
//     [-127, 127]; zero/tiny/non-finite groups as the dense A8 quantizer (dense_a8_quantize8 rules, 64-wide groups).
//   - Weights: each Q8_0 block pair (d0, q0[32]; d1, q1[32]) requantized to one scale dp = max(|d0|, |d1|):
//     q' = clamp(rint((q * d_b) / dp), -128, 127) (exact when d0 == d1; the block with the smaller scale keeps
//     log2(dp / |d_b|) fewer bits). Stored as 68 bytes per pair, [FP32 dp][64 int8], so the requantized rows have the
//     Q8_0 rows' size and addressing. Built once per weight matrix and reused by the launches that follow
//     (the 256-row launches of one prefill step).
//   - y = sum over pairs of (dp * delta) * float(int32 dot of the pair), FP32 FMA in pair order; NaN canonical.
// The projection is project_rows_dense_a8_wmma_lds2_kernel's structure with one epilogue per pair instead of per block.

constexpr std::uint64_t kDenseG64ScratchBytes = 64ULL << 20;  // per family: the largest dense matrix (GDN in, 44.7 MB)

// Requantized weights: one scratch per family with the weight pointers it currently holds.
struct DenseG64Weights {
    unsigned char* base = nullptr;
    const unsigned char* source[kMaxSegments]{};
    std::uint32_t outputs = 0, width = 0;
};
inline DenseG64Weights& dense_g64_weights(unsigned family) {
    static DenseG64Weights slots[kDenseA8Families];
    auto& s = slots[family % kDenseA8Families];
    if (!s.base) {
        void* p = nullptr;
        if (hipMalloc(&p, kDenseG64ScratchBytes) != hipSuccess) throw std::runtime_error("q8_rows: dense A8-G64 weight scratch allocation failed");
        s.base = static_cast<unsigned char*>(p);
    }
    return s;
}
// Forgets every family's requantized weights (tests: a freed weight buffer can come back at the same address with
// other contents; engine weights are immutable while resident).
inline void dense_g64_invalidate() {
    for (unsigned f = 0; f < kDenseA8Families; ++f) {
        auto& s = dense_g64_weights(f);
        for (auto& p : s.source) p = nullptr;
        s.outputs = s.width = 0;
    }
}
// Pitch-256 activation slots: codes [row][width], deltas [pair * 256 + row].
inline DenseA8Slot dense_g64_slot(unsigned family) {
    constexpr std::uint64_t code_bytes = std::uint64_t{kDenseA8WideRows} * kDenseA8MaxWidth;
    constexpr std::uint64_t slot_bytes = code_bytes + std::uint64_t{kDenseA8MaxWidth / 64U} * kDenseA8WideRows * 4U;
    static unsigned char* const pool = [] {
        void* p = nullptr;
        if (hipMalloc(&p, slot_bytes * 2U * kDenseA8Families) != hipSuccess)
            throw std::runtime_error("q8_rows: dense A8-G64 activation scratch allocation failed");
        return static_cast<unsigned char*>(p);
    }();
    static std::atomic<unsigned> turn[kDenseA8Families]{};
    family %= kDenseA8Families;
    const unsigned slot = family * 2U + (turn[family].fetch_add(1U, std::memory_order_relaxed) & 1U);
    auto* base = pool + slot * slot_bytes;
    return {reinterpret_cast<std::int8_t*>(base), reinterpret_cast<float*>(base + code_bytes)};
}

// One thread per (output row, pair) of one matrix of `rows` rows x `pairs` pairs (a template: the header is
// included by several translation units).
template <int = 0>
__global__ __launch_bounds__(256) void dense_g64_requant_kernel(const unsigned char* __restrict__ q8, unsigned char* __restrict__ out,
    std::uint32_t rows, std::uint32_t pairs) {
#pragma clang fp contract(off)
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * 256U + threadIdx.x;
    if (item >= static_cast<std::uint64_t>(rows) * pairs) return;
    const auto* src = reinterpret_cast<const unsigned*>(q8 + item * 68U);
    unsigned q[17];
#pragma unroll
    for (unsigned k = 0; k < 17U; ++k) q[k] = src[k];
    const float d0 = static_cast<float>(__ushort_as_half(static_cast<unsigned short>(q[0] & 0xffffU)));
    const float d1 = static_cast<float>(__ushort_as_half(static_cast<unsigned short>(q[8] >> 16U)));
    const bool finite = isfinite(d0) && isfinite(d1);
    const float dp = finite ? fmaxf(fabsf(d0), fabsf(d1)) : __uint_as_float(kDenseA8CanonicalNan);
    unsigned word[17];
    word[0] = __float_as_uint(dp);
#pragma unroll
    for (unsigned k = 0; k < 16U; ++k) {
        // Codes 4 k .. 4 k + 3 of the pair: block 0 bytes 2 + c of dword-aligned q, block 1 dwords 9..16.
        const unsigned raw = k < 8U ? __builtin_amdgcn_alignbyte(q[k + 1U], q[k], 2U) : q[k + 1U];
        const float d = k < 8U ? d0 : d1;
        unsigned packed = 0U;
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
            const int code = static_cast<signed char>((raw >> (8U * j)) & 255U);
            float v = 0.0F;
            if (finite && dp > 0.0F) v = fminf(127.0F, fmaxf(-128.0F, rintf(__fdiv_rn(static_cast<float>(code) * d, dp))));
            packed |= (static_cast<unsigned>(static_cast<int>(v)) & 0xffU) << (8U * j);
        }
        word[k + 1U] = packed;
    }
    auto* dst = reinterpret_cast<unsigned*>(out + item * 68U);
#pragma unroll
    for (unsigned k = 0; k < 17U; ++k) dst[k] = word[k];
}

// Eight features per thread, eight threads (one DPP row-of-8) per 64-feature group.
template <class Site>
__global__ __launch_bounds__(kThreads) void dense_g64_quantize_kernel(Site site, std::int8_t* codes, float* deltas) {
#pragma clang fp contract(off)
    const std::uint32_t width = site.width, groups = width / 8U;
    const std::uint32_t item = blockIdx.x * kThreads + threadIdx.x;
    if (item >= Site::rows * groups) return;  // rows * width / 8 is a multiple of 8: whole groups drop together
    const unsigned r = item / groups;
    const std::uint32_t g = item % groups;
    float x[8];
    load8(row_input(site, r) + 8U * g, x);
    int bad = 0;
    float amax = 0.0F;
#pragma unroll
    for (unsigned l = 0; l < 8U; ++l) { bad |= !isfinite(x[l]); amax = fmaxf(amax, fabsf(x[l])); }
    bad |= lane_xor_int<1U>(bad); bad |= lane_xor_int<2U>(bad); bad |= lane_xor_int<4U>(bad);
    amax = fmaxf(amax, lane_xor<1U>(amax)); amax = fmaxf(amax, lane_xor<2U>(amax)); amax = fmaxf(amax, lane_xor<4U>(amax));
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    const float delta = bad ? __uint_as_float(kDenseA8CanonicalNan) : amax == 0.0F || tiny ? 1.0F : __fdiv_rn(amax, 127.0F);
    unsigned word[2] = {0U, 0U};
#pragma unroll
    for (unsigned l = 0; l < 8U; ++l) {
        const float rounded = bad || tiny ? 0.0F : rintf(__fdiv_rn(x[l], delta));
        const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
        word[l / 4U] |= (static_cast<unsigned>(code) & 0xffU) << (8U * (l % 4U));
    }
    *reinterpret_cast<uint2*>(codes + static_cast<std::uint64_t>(r) * width + 8U * g) = make_uint2(word[0], word[1]);
    if (!(g & 7U)) deltas[(g / 8U) * kDenseA8WideRows + r] = delta;
}

struct DenseG64Stage {
    dense_a8_v4i codes[2][4][64];
    float deltas[2][64];
};
// 64 rows x 32 outputs per wave (4 x 2 tiles), rows staged in LDS as in project_rows_dense_a8_wmma_lds2_kernel; per pair
// four v_wmma_i32_16x16x16_iu8 into one accumulator, then one FP32 step acc += (dp * delta) * float(dot).
template <class Site, unsigned Waves>
__global__ __launch_bounds__(32U * Waves) void project_rows_dense_g64_wmma_kernel(Site site, Segments weights, const std::int8_t* codes,
    const float* deltas) {
#pragma clang fp contract(off)
    constexpr unsigned Rows = Site::rows, TM = 4U, TN = 2U, Threads = 32U * Waves;
    static_assert(Rows % 64U == 0U && Rows <= kDenseA8WideRows && Threads >= 128U);
    constexpr std::uint32_t pitch = kDenseA8WideRows;
    __shared__ DenseG64Stage stage;
    const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x / 32U, i = lane & 15U, h = lane >> 4U;
    constexpr std::uint32_t row_tiles = Rows / 64U;
    const std::uint32_t width = site.width, outputs = site.outputs, pairs = width / 64U;
    const std::uint32_t m0 = blockIdx.x % row_tiles * 64U, n0 = (blockIdx.x / row_tiles * Waves + wave) * 16U * TN;
    const bool active = n0 < outputs;
    const unsigned char* w[TN];
#pragma unroll
    for (unsigned t = 0; t < TN; ++t) {
        unsigned m; std::uint32_t row;
        w[t] = segment_row(weights, min(min(n0, outputs - 1U) + 16U * t + i, outputs - 1U), width, m, row);
    }
    const unsigned slot = (threadIdx.x / 2U) % 64U, half = threadIdx.x % 2U;
    const std::uint32_t slot_row = m0 + 16U * (slot / 16U) + 8U * (slot % 2U) + (slot % 16U) / 2U;
    const auto* code_src = codes + static_cast<std::uint64_t>(slot_row) * width + 32U * half;
    const unsigned dq = threadIdx.x % 16U;
    const float* delta_src = deltas + m0 + 4U * dq;
    const bool stage_codes = threadIdx.x < 128U, stage_deltas = Waves == 4U ? threadIdx.x < 16U : threadIdx.x >= 128U && threadIdx.x < 144U;
    dense_a8_v4i sc[2];
    float4 sd;
    const auto load = [&](std::uint32_t p) {
        if (stage_codes) {
            const auto* x = reinterpret_cast<const dense_a8_v4i*>(code_src + 64U * p);
            sc[0] = x[0]; sc[1] = x[1];
        }
        if (stage_deltas) sd = *reinterpret_cast<const float4*>(delta_src + p * pitch);
    };
    const auto store = [&](unsigned buffer) {
        if (stage_codes) { stage.codes[buffer][2U * half][slot] = sc[0]; stage.codes[buffer][2U * half + 1U][slot] = sc[1]; }
        if (stage_deltas) *reinterpret_cast<float4*>(&stage.deltas[buffer][4U * dq]) = sd;
    };
    float acc[TM][TN][8];
#pragma unroll
    for (unsigned t = 0; t < TM; ++t)
#pragma unroll
        for (unsigned u = 0; u < TN; ++u)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) acc[t][u][e] = 0.0F;
    constexpr int kMagic = 0x4B400000;  // |dot| <= 64 * 128 * 127 < 2^22: the magic-number conversion is exact
    constexpr float kMagicF = 12582912.0F;
    const dense_a8_v8i magic{kMagic, kMagic, kMagic, kMagic, kMagic, kMagic, kMagic, kMagic};
    unsigned next[TN][17];
    const auto fetch = [&](std::uint32_t pair) {
#pragma unroll
        for (unsigned u = 0; u < TN; ++u) {
            const auto* q4 = reinterpret_cast<const DenseA8Dword4*>(w[u] + 68U * pair);
            const DenseA8Dword4 c0 = q4[0], c1 = q4[1], c2 = q4[2], c3 = q4[3];
            const unsigned last = reinterpret_cast<const unsigned*>(w[u] + 68U * pair)[16];
            const unsigned q[17]{c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w, c2.x, c2.y, c2.z, c2.w, c3.x, c3.y, c3.z, c3.w, last};
#pragma unroll
            for (unsigned k = 0; k < 17U; ++k) next[u][k] = q[k];
        }
    };
    load(0U);
    store(0U);
    if (active) fetch(0U);
    if (pairs > 1U) load(1U);
    __syncthreads();
    for (std::uint32_t p = 0; p < pairs; ++p) {
        const unsigned buffer = p & 1U;
        if (active) {
            dense_a8_v4i wb[4][TN];
            float dw[TN];
#pragma unroll
            for (unsigned u = 0; u < TN; ++u) {
                const unsigned (&q)[17] = next[u];
                dw[u] = __uint_as_float(q[0]);
                asm volatile("" : "+v"(dw[u]));
#pragma unroll
                for (unsigned c = 0; c < 4U; ++c)
#pragma unroll
                    for (unsigned k = 0; k < 4U; ++k) wb[c][u][k] = static_cast<int>(q[1U + 4U * c + k]);
            }
            fetch(p + 1U < pairs ? p + 1U : p);
#pragma unroll
            for (unsigned t = 0; t < TM; ++t) {
                const float4 lo = *reinterpret_cast<const float4*>(&stage.deltas[buffer][16U * t + 8U * h]);
                const float4 hi = *reinterpret_cast<const float4*>(&stage.deltas[buffer][16U * t + 8U * h + 4U]);
                const float da[8]{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
                dense_a8_v8i c[TN];
#pragma unroll
                for (unsigned u = 0; u < TN; ++u) c[u] = magic;
#pragma unroll
                for (unsigned q = 0; q < 4U; ++q) {
                    const dense_a8_v4i a = stage.codes[buffer][q][16U * t + i];
#pragma unroll
                    for (unsigned u = 0; u < TN; ++u) c[u] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, wb[q][u], c[u], false);
                }
#pragma unroll
                for (unsigned u = 0; u < TN; ++u)
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e)
                        acc[t][u][e] = __builtin_fmaf(dw[u] * da[e], __int_as_float(c[u][e]) - kMagicF, acc[t][u][e]);
                __builtin_amdgcn_sched_barrier(0);
            }
        }
        if (p + 1U < pairs) {
            store(buffer ^ 1U);
            if (p + 2U < pairs) load(p + 2U);
        }
        __syncthreads();
    }
    if (!active) return;
#pragma unroll
    for (unsigned t = 0; t < TM; ++t)
#pragma unroll
        for (unsigned u = 0; u < TN; ++u) {
            const std::uint32_t n = n0 + 16U * u + i;
            if (n < outputs)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) site.store(n, m0 + 16U * t + 8U * h + e, dense_a8_canonical(acc[t][u][e]));
        }
}

// Requantizes the site's weights into the family scratch unless it already holds them (same matrices, same shape).
template <class Site> Segments dense_g64_requantized(const Site& site, hipStream_t stream) {
    auto& cache = dense_g64_weights(a8_family_of<Site>());
    bool same = cache.outputs == site.outputs && cache.width == site.width;
    for (unsigned m = 0; m < kMaxSegments; ++m)
        same = same && cache.source[m] == (m < site.segments.count ? site.segments.weights[m] : nullptr);
    Segments out = site.segments;
    const std::uint64_t row_bytes = static_cast<std::uint64_t>(site.width / 32U) * 34U;
    if (static_cast<std::uint64_t>(site.outputs) * row_bytes > kDenseG64ScratchBytes)
        throw std::invalid_argument("q8_rows: dense A8-G64 weights exceed the requantization scratch");
    std::uint64_t offset = 0;
    for (unsigned m = 0; m < site.segments.count; ++m) {
        const std::uint32_t rows = site.segments.first[m + 1U] - site.segments.first[m];
        out.weights[m] = cache.base + offset;
        if (!same) {
            const std::uint64_t items = static_cast<std::uint64_t>(rows) * (site.width / 64U);
            hipLaunchKernelGGL(HIP_KERNEL_NAME(dense_g64_requant_kernel<>), dim3(static_cast<unsigned>((items + 255U) / 256U)), dim3(256U), 0, stream,
                site.segments.weights[m], cache.base + offset, rows, site.width / 64U);
        }
        offset += rows * row_bytes;
    }
    if (!same) {
        cache.outputs = site.outputs; cache.width = site.width;
        for (unsigned m = 0; m < kMaxSegments; ++m) cache.source[m] = m < site.segments.count ? site.segments.weights[m] : nullptr;
    }
    return out;
}

// Dense A8-G64 launch of 64..256 rows (a multiple of 64): weight requantization (when the matrices changed), the 64-group
// quantizer into the family slot (any-order when the caller allows it) and the barrier-ordered projection.
template <class Site> void launch_dense_g64(const Site& site, hipStream_t stream, bool any_order) {
    static_assert(Site::rows % 64U == 0U && Site::rows <= kDenseA8WideRows);
    if (!dense_a8_width(site.width))
        throw std::invalid_argument("q8_rows: dense A8-G64 requires a width that is a positive multiple of 64, at most 8192");
    for (unsigned m = 0; m < site.segments.count; ++m)
        if (reinterpret_cast<std::uintptr_t>(site.segments.weights[m]) % 4U)
            throw std::invalid_argument("q8_rows: dense A8-G64 requires dword-aligned Q8 weights");
    const Segments weights = dense_g64_requantized(site, stream);
    const auto slot = dense_g64_slot(a8_family_of<Site>());
    const auto items = Site::rows * (site.width / 8U);
    const dim3 blocks((items + kThreads - 1U) / kThreads);
    if (any_order)
        hipExtLaunchKernelGGL(dense_g64_quantize_kernel<Site>, blocks, dim3(kThreads), 0, stream, nullptr, nullptr, hipExtAnyOrderLaunch,
            site, slot.codes, slot.deltas);
    else hipLaunchKernelGGL(dense_g64_quantize_kernel<Site>, blocks, dim3(kThreads), 0, stream, site, slot.codes, slot.deltas);
    constexpr unsigned Waves = 8U;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(project_rows_dense_g64_wmma_kernel<Site, Waves>),
        dim3(Site::rows / 64U * ((site.outputs + 32U * Waves - 1U) / (32U * Waves))), dim3(32U * Waves), 0, stream, site, weights,
        slot.codes, slot.deltas);
}
