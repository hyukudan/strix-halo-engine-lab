#pragma once
// Included by hip_q8_rows_gemv.hpp inside namespace strix::q8_rows.
//
// Dense FP16 projections of fast-prefill-f16wmma-v1 (prefill launches of 32..256 rows, a multiple of 32; not the
// exact or dense A8 contracts): y[r][n] = sum_k fp16(x[r][k]) * fp16(d[n][b] * q[n][k]) with
// v_wmma_f32_16x16x16_f16 (FP32 accumulation over the whole width, hardware order inside a tile).
//   - x: the site's BF16/FP32 activations rounded to FP16 (RNE) by dense_f16_convert_kernel into a
//     per-family scratch slot (rows x width, row-major); finite values beyond the FP16 range count
//     into dense_f16_overflow_count() (they become +-inf).
//   - weights: the effective Q8_0 value d * q (exact in FP32) rounded to FP16 (RNE), dequantized per
//     block into LDS by the projection itself; no FP16 weight copy exists in memory.
// A block owns BM rows x BN outputs (8 waves of WM x WN); per 64-feature stage (one Q8 block pair)
// the rows' FP16 values and the dequantized weights sit in LDS; operands of the next stage are
// prefetched into registers while the current one is multiplied.

typedef _Float16 dense_f16_v16h __attribute__((ext_vector_type(16)));
typedef float dense_f16_v8f __attribute__((ext_vector_type(8)));
constexpr unsigned kDenseF16Pitch = 72U;  // halves per LDS row of one 64-feature stage (144 bytes)

inline unsigned* dense_f16_overflow_counter() {
    static unsigned* const counter = [] {
        void* p = nullptr;
        if (hipMalloc(&p, 256U) != hipSuccess || hipMemset(p, 0, 256U) != hipSuccess)
            throw std::runtime_error("q8_rows: dense FP16 overflow counter allocation failed");
        return static_cast<unsigned*>(p);
    }();
    return counter;
}
// Activations that overflowed FP16 so far (synchronizes the device).
inline unsigned dense_f16_overflow_count() {
    unsigned value = 0;
    if (hipDeviceSynchronize() != hipSuccess ||
        hipMemcpy(&value, dense_f16_overflow_counter(), sizeof(value), hipMemcpyDeviceToHost) != hipSuccess)
        throw std::runtime_error("q8_rows: dense FP16 overflow counter read failed");
    return value;
}
// Two FP16 activation slots per family (rows <= 256, width <= 8192), alternating per launch.
inline _Float16* dense_f16_slot(unsigned family) {
    constexpr std::uint64_t slot_halves = std::uint64_t{kDenseA8WideRows} * kDenseA8MaxWidth;
    static _Float16* const pool = [] {
        void* p = nullptr;
        if (hipMalloc(&p, slot_halves * 2U * 2U * kDenseA8Families) != hipSuccess)
            throw std::runtime_error("q8_rows: dense FP16 scratch allocation failed");
        return static_cast<_Float16*>(p);
    }();
    static std::atomic<unsigned> turn[kDenseA8Families]{};
    family %= kDenseA8Families;
    const unsigned slot = family * 2U + (turn[family].fetch_add(1U, std::memory_order_relaxed) & 1U);
    return pool + slot * slot_halves;
}

template <class Site>
__global__ __launch_bounds__(kThreads) void dense_f16_convert_kernel(Site site, _Float16* __restrict__ a16, unsigned* __restrict__ overflow) {
    const std::uint32_t width = site.width, groups = width / 8U;
    const std::uint32_t item = blockIdx.x * kThreads + threadIdx.x;
    if (item >= Site::rows * groups) return;
    const unsigned r = item / groups;
    const std::uint32_t g = item % groups;
    float x[8];
    load8(row_input(site, r) + 8U * g, x);
    _Float16 h[8];
    bool bad = false;
#pragma unroll
    for (unsigned k = 0; k < 8U; ++k) {
        h[k] = static_cast<_Float16>(x[k]);
        bad |= isfinite(x[k]) && !(fabsf(x[k]) <= 65504.0F);
    }
    uint4 word;
    __builtin_memcpy(&word, h, 16U);
    *reinterpret_cast<uint4*>(a16 + static_cast<std::uint64_t>(r) * width + 8U * g) = word;
    if (bad) atomicAdd(overflow, 1U);
}

__device__ __forceinline__ dense_f16_v16h dense_f16_fragment(const _Float16* p) {
    dense_f16_v16h v;
    const uint4 lo = reinterpret_cast<const uint4*>(p)[0], hi = reinterpret_cast<const uint4*>(p)[1];
    __builtin_memcpy(&v, &lo, 16U);
    __builtin_memcpy(reinterpret_cast<char*>(&v) + 16, &hi, 16U);
    return v;
}

template <class Site, unsigned BM, unsigned BN, unsigned WM, unsigned WN>
__global__ __launch_bounds__(256) void project_rows_dense_f16_wmma_kernel(Site site, const _Float16* __restrict__ a16) {
    constexpr unsigned Rows = Site::rows, WC = BN / WN, TM = WM / 16U, TN = WN / 16U;
    static_assert((BM / WM) * WC == 8U && Rows % BM == 0U && WM % 16U == 0U && WN % 16U == 0U);
    constexpr unsigned ALoads = BM * 8U / 256U;            // 16-byte A loads per thread per stage
    constexpr unsigned BItems = BN * 2U;                   // Q8 blocks per stage
    static_assert(ALoads >= 1U && BItems <= 256U);
    __shared__ __align__(16) _Float16 as[BM * kDenseF16Pitch];
    __shared__ __align__(16) _Float16 bs[BN * kDenseF16Pitch];
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U, wr = wave / WC, wc = wave % WC;
    const std::uint32_t width = site.width, outputs = site.outputs, stages = width / 64U;
    const unsigned m0 = blockIdx.x % (Rows / BM) * BM;
    const std::uint32_t n0 = blockIdx.x / (Rows / BM) * BN;
    // A: thread loads (row i / 8, 8 halves i % 8) for i = threadIdx.x + 256 m.
    const _Float16* asrc[ALoads];
#pragma unroll
    for (unsigned m = 0; m < ALoads; ++m) {
        const unsigned i = threadIdx.x + 256U * m;
        asrc[m] = a16 + static_cast<std::uint64_t>(m0 + i / 8U) * width + (i % 8U) * 8U;
    }
    // B: thread < BItems dequantizes Q8 block (2 stage + odd) of output n0 + threadIdx.x / 2.
    const bool loads_b = threadIdx.x < BItems;
    const unsigned odd = threadIdx.x % 2U, bo = threadIdx.x / 2U;
    const unsigned char* wrow = nullptr;
    {
        unsigned seg; std::uint32_t row;
        wrow = segment_row(site.segments, min(n0 + bo, outputs - 1U), width, seg, row);
    }
    uint4 areg[ALoads];
    unsigned breg[9];
    const auto load = [&](std::uint32_t stage) {
#pragma unroll
        for (unsigned m = 0; m < ALoads; ++m) areg[m] = *reinterpret_cast<const uint4*>(asrc[m] + stage * 64U);
        if (loads_b) {
            const unsigned char* p = wrow + 68U * stage + (odd ? 32U : 0U);
            const auto lo = reinterpret_cast<const uint4*>(p)[0], hi = reinterpret_cast<const uint4*>(p)[1];
            breg[0] = lo.x; breg[1] = lo.y; breg[2] = lo.z; breg[3] = lo.w;
            breg[4] = hi.x; breg[5] = hi.y; breg[6] = hi.z; breg[7] = hi.w;
            breg[8] = reinterpret_cast<const unsigned*>(p)[8];
        }
    };
    const auto store = [&]() {
#pragma unroll
        for (unsigned m = 0; m < ALoads; ++m) {
            const unsigned i = threadIdx.x + 256U * m;
            *reinterpret_cast<uint4*>(&as[(i / 8U) * kDenseF16Pitch + (i % 8U) * 8U]) = areg[m];
        }
        if (loads_b) {
            // Even block: dwords from its scale (codes two bytes in); odd block: from two bytes before its scale.
            const float scale = __half2float(__ushort_as_half(static_cast<unsigned short>(odd ? breg[0] >> 16U : breg[0] & 0xffffU)));
            _Float16* dst = &bs[bo * kDenseF16Pitch + odd * 32U];
#pragma unroll
            for (unsigned q = 0; q < 4U; ++q) {
                _Float16 h[8];
#pragma unroll
                for (unsigned half = 0; half < 2U; ++half) {
                    const unsigned k = 2U * q + half;
                    const unsigned word = odd ? breg[k + 1U] : __builtin_amdgcn_alignbyte(breg[k + 1U], breg[k], 2U);
#pragma unroll
                    for (unsigned j = 0; j < 4U; ++j)
                        h[4U * half + j] = static_cast<_Float16>(scale * static_cast<float>(static_cast<signed char>((word >> (8U * j)) & 255U)));
                }
                uint4 v;
                __builtin_memcpy(&v, h, 16U);
                reinterpret_cast<uint4*>(dst)[q] = v;
            }
        }
    };
    dense_f16_v8f acc[TM][TN];
#pragma unroll
    for (unsigned t = 0; t < TM; ++t)
#pragma unroll
        for (unsigned u = 0; u < TN; ++u) acc[t][u] = dense_f16_v8f{};
    load(0U);
    for (std::uint32_t stage = 0; stage < stages; ++stage) {
        store();
        __syncthreads();
        if (stage + 1U < stages) load(stage + 1U);
#pragma unroll
        for (unsigned k = 0; k < 4U; ++k) {
            dense_f16_v16h b[TN];
#pragma unroll
            for (unsigned u = 0; u < TN; ++u) b[u] = dense_f16_fragment(&bs[(wc * WN + 16U * u + lane % 16U) * kDenseF16Pitch + 16U * k]);
#pragma unroll
            for (unsigned t = 0; t < TM; ++t) {
                const dense_f16_v16h a = dense_f16_fragment(&as[(wr * WM + 16U * t + lane % 16U) * kDenseF16Pitch + 16U * k]);
#pragma unroll
                for (unsigned u = 0; u < TN; ++u) acc[t][u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b[u], acc[t][u]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned t = 0; t < TM; ++t)
#pragma unroll
        for (unsigned u = 0; u < TN; ++u) {
            const std::uint32_t out = n0 + wc * WN + 16U * u + lane % 16U;
            if (out >= outputs) continue;
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) site.store(out, m0 + wr * WM + 16U * t + 2U * e + lane / 16U, acc[t][u][e]);
        }
}

// STRIX_DENSE_F16_TILE = 1 (BM 256, BN 128), 2 (128 x 128), 3 (128 x 64), 4 (64 x 64), 5 (64 x 128): forces
// a block shape where the rows allow it (tuning). Default: 128 x 128 when it gives >= 40 blocks, else the next smaller shape.
template <class Site> void launch_dense_f16(const Site& site, hipStream_t stream, bool any_order) {
    static_assert(Site::rows % 32U == 0U && Site::rows <= kDenseA8WideRows);
    if (!dense_a8_width(site.width))
        throw std::invalid_argument("q8_rows: dense FP16 requires a width that is a positive multiple of 64, at most 8192");
    for (unsigned m = 0; m < site.segments.count; ++m)
        if (reinterpret_cast<std::uintptr_t>(site.segments.weights[m]) % 4U)
            throw std::invalid_argument("q8_rows: dense FP16 requires dword-aligned Q8 weights");
    auto* a16 = dense_f16_slot(a8_family_of<Site>());
    const auto items = Site::rows * (site.width / 8U);
    const dim3 convert_blocks((items + kThreads - 1U) / kThreads);
    if (any_order)
        hipExtLaunchKernelGGL(dense_f16_convert_kernel<Site>, convert_blocks, dim3(kThreads), 0, stream, nullptr, nullptr,
            hipExtAnyOrderLaunch, site, a16, dense_f16_overflow_counter());
    else hipLaunchKernelGGL(dense_f16_convert_kernel<Site>, convert_blocks, dim3(kThreads), 0, stream, site, a16, dense_f16_overflow_counter());
    const auto run = [&]<unsigned BM, unsigned BN, unsigned WM, unsigned WN>() {
        if constexpr (Site::rows % BM == 0U) {
            const unsigned blocks = Site::rows / BM * ((site.outputs + BN - 1U) / BN);
            hipLaunchKernelGGL(HIP_KERNEL_NAME(project_rows_dense_f16_wmma_kernel<Site, BM, BN, WM, WN>), dim3(blocks), dim3(256U), 0, stream,
                site, a16);
            return true;
        } else return false;
    };
    static const unsigned forced = env_unsigned("STRIX_DENSE_F16_TILE", 0U);
    const auto blocks_of = [&](unsigned bm, unsigned bn) { return Site::rows / bm * ((site.outputs + bn - 1U) / bn); };
    if (forced == 1U && run.template operator()<256U, 128U, 64U, 64U>()) return;
    if (forced == 2U && run.template operator()<128U, 128U, 64U, 32U>()) return;
    if (forced == 3U && run.template operator()<128U, 64U, 32U, 32U>()) return;
    if (forced == 4U && run.template operator()<64U, 64U, 32U, 16U>()) return;
    if (forced == 5U && run.template operator()<64U, 128U, 32U, 32U>()) return;
    // 256-row launches: 128 x 128 measured fastest (GDN/QSA in, GDN out, shared gate/up and down); 256 x 128 leaves
    // the last of ~3 block rounds mostly idle on 40 CUs.
    if constexpr (Site::rows % 128U == 0U) {
        if (blocks_of(128U, 128U) >= 40U && run.template operator()<128U, 128U, 64U, 32U>()) return;
        if (run.template operator()<128U, 64U, 32U, 32U>()) return;
    }
    if constexpr (Site::rows % 64U == 0U) {
        if (blocks_of(64U, 128U) >= 40U && run.template operator()<64U, 128U, 32U, 32U>()) return;
        if (run.template operator()<64U, 64U, 32U, 16U>()) return;
    }
    if constexpr (Site::rows == 32U) { run.template operator()<32U, 128U, 16U, 32U>(); return; }  // 2 x 4 waves
}

// Prompt launch of a dense site: with f16 (fast-prefill-f16wmma-v1 prefill launchers) 32..256-row launches
// run launch_dense_f16; otherwise (and for other row counts) the dense A8 form over 64 rows and
// launch(site, a8) up to 64 rows, as the prefill sites did before.
// g64 (component dense8): 64..256-row launches (a multiple of 64) run launch_dense_g64 instead.
template <class Site> void launch_prefill(const Site& site, bool a8, bool f16, hipStream_t stream, bool any_order = false, bool g64 = false) {
    // Only where dense A8 takes its LDS-staged 4 x 2 form (>= 40 blocks of 8 waves): narrower launches (QSA output of
    // 64 rows, shared gate/up) were slower in G64 than in the 4 x 1 A8 form and keep dense A8 (fixture, gfx1151).
    if constexpr (Site::rows >= 64U && Site::rows % 64U == 0U && Site::rows <= kDenseA8WideRows)
        if (g64 && Site::rows / 64U * ((site.outputs + 255U) / 256U) >= 40U) { launch_dense_g64(site, stream, any_order); return; }
    if constexpr (Site::rows >= 32U && Site::rows % 32U == 0U && Site::rows <= kDenseA8WideRows)
        if (f16) { launch_dense_f16(site, stream, any_order); return; }
    if constexpr (Site::rows > 64U) launch_dense_a8(site, stream, any_order);
    else launch(site, a8, stream, any_order);
}
