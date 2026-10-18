// Prefill HC projections of fast-prefill-f16wmma-v1 (prefill-only arithmetic, not the split-K /
// lane-chain contract): v_wmma_f32_16x16x16_f16 with FP32 accumulation over the whole K range.
//
// Operands: the effective Q8_0 weights scale * code (exact in FP32) rounded to FP16 (RNE) by a pack
// kernel per call; the normalized rows rounded to FP16 (RNE) when staged; the low values rounded to
// FP16 for the up projection. The FP32 low values, the injection (an FP32 FMA chain over k ascending
// on the FP32 inputs and weights) and the stream mix (stream order, as the lane-chain kernels) keep
// the FP32 arithmetic of the other forms. Only the two GEMMs change their operand precision and
// reduction order.
//
// WMMA layouts (wave32): A lane l (and l + 16) = row l % 16, 16 halves of K; B lane l = column l % 16;
// D element e of lane l = (row 2 e + l / 16, column l % 16).
typedef _Float16 hcw_v16h __attribute__((ext_vector_type(16)));
typedef float hcw_v8f __attribute__((ext_vector_type(8)));

constexpr unsigned kHcwRank = 320U, kHcwWidth = 10240U, kHcwStages = kHcwWidth / 32U;
// Projection: blocks of 256 rows x 64 low outputs, 8 waves of 32 rows x 64 outputs (2 x 4 tiles).
constexpr unsigned kHcwProjCols = 64U, kHcwProjRows = 256U, kHcwProjColBlocks = kHcwRank / kHcwProjCols;
constexpr unsigned kHcwPitch = 40U;  // halves per staged LDS row of 32 K values (80 bytes)
// Output: blocks of 256 rows x 16 features (x 4 streams), 8 waves of 32 rows.
constexpr unsigned kHcwOutRows = 256U, kHcwOutFeatures = 16U, kHcwK16 = kHcwRank / 16U;
constexpr std::uint64_t kHcwDownPackHalves = std::uint64_t{kHcwRank} * kHcwWidth;
constexpr std::uint64_t kHcwUpPackHalves = std::uint64_t{4U} * 2560U * kHcwRank;
constexpr std::uint32_t kHcwLowRows = 4096U;

__device__ __forceinline__ hcw_v8f hcw_wmma(const hcw_v16h& a, const hcw_v16h& b, const hcw_v8f& c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
__device__ __forceinline__ float hcw_q8_value(const unsigned char* block, unsigned k) {
    const auto bits = static_cast<unsigned short>(block[0] | (block[1] << 8U));
    return __half2float(__ushort_as_half(bits)) * static_cast<float>(static_cast<signed char>(block[2U + k]));
}

// Down weights (Q8_0 [320][10240]) -> FP16 [column block 5][stage 320][output 64][32 K values].
// One thread per (output, Q8 block). Sets *overflow when an effective weight is not finite in FP16.
__global__ __launch_bounds__(256) void hcw_pack_down_kernel(const unsigned char* __restrict__ down, _Float16* __restrict__ out,
    unsigned* __restrict__ overflow) {
    const unsigned index = blockIdx.x * 256U + threadIdx.x;
    if (index >= kHcwRank * kHcwStages) return;
    const unsigned o = index / kHcwStages, j = index % kHcwStages;
    const auto* block = down + (static_cast<std::uint64_t>(o) * kHcwStages + j) * 34U;
    auto* dst = out + ((static_cast<std::uint64_t>(o / kHcwProjCols) * kHcwStages + j) * kHcwProjCols + o % kHcwProjCols) * 32U;
    bool bad = false;
#pragma unroll
    for (unsigned q = 0; q < 4U; ++q) {
        _Float16 h[8];
#pragma unroll
        for (unsigned k = 0; k < 8U; ++k) {
            const float v = hcw_q8_value(block, 8U * q + k);
            h[k] = static_cast<_Float16>(v);
            bad |= !(__builtin_fabsf(v) <= 65504.0F);
        }
        uint4 word;
        __builtin_memcpy(&word, h, 16U);
        reinterpret_cast<uint4*>(dst)[q] = word;
    }
    if (bad) atomicOr(overflow, 1U);
}

// Up weights (Q8_0 [4 * 2560][320], row s * 2560 + f) -> FP16 [feature block 160][k16 20][stream 4][feature 16][16 K values].
__global__ __launch_bounds__(256) void hcw_pack_up_kernel(const unsigned char* __restrict__ up, _Float16* __restrict__ out,
    unsigned hidden, unsigned* __restrict__ overflow) {
    const unsigned index = blockIdx.x * 256U + threadIdx.x;
    if (index >= 4U * hidden * (kHcwRank / 32U)) return;
    const unsigned n = index / (kHcwRank / 32U), j = index % (kHcwRank / 32U), s = n / hidden, f = n % hidden;
    const auto* block = up + static_cast<std::uint64_t>(index) * 34U;
    bool bad = false;
#pragma unroll
    for (unsigned h = 0; h < 2U; ++h) {
        const unsigned k16 = 2U * j + h;
        auto* dst = out + (((static_cast<std::uint64_t>(f / kHcwOutFeatures) * kHcwK16 + k16) * 4U + s) * kHcwOutFeatures +
            f % kHcwOutFeatures) * 16U;
#pragma unroll
        for (unsigned q = 0; q < 2U; ++q) {
            _Float16 v[8];
#pragma unroll
            for (unsigned k = 0; k < 8U; ++k) {
                const float x = hcw_q8_value(block, 16U * h + 8U * q + k);
                v[k] = static_cast<_Float16>(x);
                bad |= !(__builtin_fabsf(x) <= 65504.0F);
            }
            uint4 word;
            __builtin_memcpy(&word, v, 16U);
            reinterpret_cast<uint4*>(dst)[q] = word;
        }
    }
    if (bad) atomicOr(overflow, 1U);
}

__device__ __forceinline__ hcw_v16h hcw_lds_fragment(const _Float16* p) {
    hcw_v16h v;
    const uint4 lo = reinterpret_cast<const uint4*>(p)[0], hi = reinterpret_cast<const uint4*>(p)[1];
    __builtin_memcpy(&v, &lo, 16U);
    __builtin_memcpy(reinterpret_cast<char*>(&v) + 16, &hi, 16U);
    return v;
}

//// Low projection (+ injection): block (column block cb, 256-row block); wave w owns rows 32 w .. + 31.
// Per stage (32 K values) load q = 0..7 of lane l reads row 4 q + l / 8, K values 4 (l % 8) .. + 3 (four
// rows x 128 bytes per instruction), stores them as FP16 into the wave's LDS rows and, in column blocks
// cb < streams, adds them into injection output cb's FP32 partial of that row (4 FMAs, k ascending).
// The 8 partials of a row (lanes of one 8-lane group) are summed by a xor tree at the end.
// The 64 x 32 FP16 weights of a stage are shared by the 8 waves through LDS (double-buffered).
__global__ __launch_bounds__(256) void hcw_projection_kernel(HyperWideRows rows, const _Float16* __restrict__ pack,
    const float* __restrict__ inject, unsigned streams, _Float16* __restrict__ low16) {
    __shared__ __align__(16) _Float16 bs[2][kHcwProjCols * kHcwPitch];
    __shared__ __align__(16) _Float16 as[8][32U * kHcwPitch];
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U, cb = blockIdx.x, sub = lane % 8U, rq = lane / 8U;
    const unsigned row0 = blockIdx.y * kHcwProjRows + wave * 32U;
    const float* x = wide_row(rows.normalized, rows.normalized_stride, min(row0 + rq, rows.count - 1U)) + 4U * sub;
    const std::uint64_t step = 4U * rows.normalized_stride;
    unsigned valid = 0U;
#pragma unroll
    for (unsigned q = 0; q < 8U; ++q) valid |= (row0 + 4U * q + rq < rows.count ? 1U : 0U) << q;
    const bool injecting = inject && cb < streams;
    const float* w = injecting ? inject + static_cast<std::uint64_t>(cb) * kHcwWidth + 4U * sub : nullptr;
    const auto* bsrc = reinterpret_cast<const uint4*>(pack + static_cast<std::uint64_t>(cb) * kHcwStages * kHcwProjCols * 32U) + threadIdx.x;
    const unsigned bdst = (threadIdx.x / 4U) * kHcwPitch + (threadIdx.x % 4U) * 8U;
    float4 xa[8];
    uint4 bw;
    const auto load = [&](unsigned stage) {
        bw = bsrc[stage * (kHcwProjCols * 32U / 8U)];
#pragma unroll
        for (unsigned q = 0; q < 8U; ++q)
            xa[q] = valid >> q & 1U ? *reinterpret_cast<const float4*>(reinterpret_cast<const unsigned char*>(x + stage * 32U) + q * step)
                                    : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    };
    hcw_v8f acc[2][4];
#pragma unroll
    for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
        for (unsigned c = 0; c < 4U; ++c) acc[t][c] = hcw_v8f{};
    float injection[8];
#pragma unroll
    for (unsigned q = 0; q < 8U; ++q) injection[q] = 0.0F;
    load(0U);
    for (unsigned stage = 0; stage < kHcwStages; ++stage) {
        const unsigned buffer = stage & 1U;
        *reinterpret_cast<uint4*>(&bs[buffer][bdst]) = bw;
#pragma unroll
        for (unsigned q = 0; q < 8U; ++q) {
            _Float16 h[4] = {static_cast<_Float16>(xa[q].x), static_cast<_Float16>(xa[q].y), static_cast<_Float16>(xa[q].z),
                static_cast<_Float16>(xa[q].w)};
            uint2 word;
            __builtin_memcpy(&word, h, 8U);
            *reinterpret_cast<uint2*>(as[wave] + (4U * q + rq) * kHcwPitch + 4U * sub) = word;
        }
        if (injecting) {
            const float4 c = *reinterpret_cast<const float4*>(w + stage * 32U);
#pragma unroll
            for (unsigned q = 0; q < 8U; ++q) {
                injection[q] = fmaf(xa[q].x, c.x, injection[q]);
                injection[q] = fmaf(xa[q].y, c.y, injection[q]);
                injection[q] = fmaf(xa[q].z, c.z, injection[q]);
                injection[q] = fmaf(xa[q].w, c.w, injection[q]);
            }
        }
        __syncthreads();
        if (stage + 1U < kHcwStages) load(stage + 1U);
#pragma unroll
        for (unsigned h = 0; h < 2U; ++h) {
            const hcw_v16h a0 = hcw_lds_fragment(as[wave] + (lane % 16U) * kHcwPitch + 16U * h);
            const hcw_v16h a1 = hcw_lds_fragment(as[wave] + (16U + lane % 16U) * kHcwPitch + 16U * h);
#pragma unroll
            for (unsigned c = 0; c < 4U; ++c) {
                const hcw_v16h b = hcw_lds_fragment(&bs[buffer][(16U * c + lane % 16U) * kHcwPitch + 16U * h]);
                acc[0][c] = hcw_wmma(a0, b, acc[0][c]);
                acc[1][c] = hcw_wmma(a1, b, acc[1][c]);
            }
        }
    }
    const float divisor = static_cast<float>(streams);
#pragma unroll
    for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
        for (unsigned c = 0; c < 4U; ++c)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const unsigned r = row0 + 16U * t + 2U * e + lane / 16U, col = cb * kHcwProjCols + 16U * c + lane % 16U;
                if (r >= rows.count) continue;
                const float scaled = acc[t][c][e] / divisor, low = scaled * sigmoid(scaled);
                wide_row(rows.normalized, rows.normalized_stride, r)[kHcwWidth + col] = low;
                low16[static_cast<std::uint64_t>(r) * kHcwRank + col] = static_cast<_Float16>(low);
                // Transposed FP32 copy for the lane-chain output kernel (wide2_hyper_output_kernel).
                if (rows.low_t) rows.low_t[static_cast<std::uint64_t>(col) * rows.low_t_pitch + r] = low;
            }
    if (!injecting) return;
#pragma unroll
    for (unsigned q = 0; q < 8U; ++q) {
        float v = injection[q];
        v = v + wide2_partner<1U>(v);
        v = v + wide2_partner<2U>(v);
        v = v + wide2_partner<4U>(v);
        if (sub == 0U && (valid >> q & 1U))
            wide_row(rows.injection, rows.injection_stride, row0 + 4U * q + rq)[cb] = 2.0F * sigmoid(v / divisor);
    }
}

// Up projection + stream mix: block (256-row block, feature block of 16); the block's FP16 weights
// (16 features x 4 streams x 320) sit in LDS; wave w computes rows 32 w .. + 31 (2 row tiles) x the 4
// streams' tiles, reading its FP16 low fragments from global, then mixes the streams in the lane
// (gate -> sigmoid -> normalized * gate in stream order, / streams) as the lane-chain kernels. The
// epilogue's normalized values are loaded before the GEMM so their DRAM latency overlaps it.
template <typename Storage>
__global__ __launch_bounds__(256) void hcw_output_kernel(HyperWideRows rows, const _Float16* __restrict__ pack,
    const _Float16* __restrict__ low16, std::uint32_t hidden) {
    constexpr unsigned Tile = kHcwK16 * 4U * kHcwOutFeatures * 16U;
    __shared__ __align__(16) _Float16 bt[Tile];
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U, fb = blockIdx.y;
    const auto* src = reinterpret_cast<const uint4*>(pack + static_cast<std::uint64_t>(fb) * Tile);
#pragma unroll
    for (unsigned m = 0; m < Tile / 8U / 256U; ++m)
        reinterpret_cast<uint4*>(bt)[threadIdx.x + 256U * m] = src[threadIdx.x + 256U * m];
    const unsigned row0 = blockIdx.x * kHcwOutRows + wave * 32U;
    const unsigned ar0 = row0 + lane % 16U, ar1 = ar0 + 16U;
    const bool v0 = ar0 < rows.count, v1 = ar1 < rows.count;
    const auto* l0 = reinterpret_cast<const uint4*>(low16 + static_cast<std::uint64_t>(v0 ? ar0 : 0U) * kHcwRank);
    const auto* l1 = reinterpret_cast<const uint4*>(low16 + static_cast<std::uint64_t>(v1 ? ar1 : 0U) * kHcwRank);
    const auto fragment = [](uint4 lo, uint4 hi) {
        hcw_v16h v;
        __builtin_memcpy(&v, &lo, 16U);
        __builtin_memcpy(reinterpret_cast<char*>(&v) + 16, &hi, 16U);
        return v;
    };
    const uint4 zero = make_uint4(0U, 0U, 0U, 0U);
    uint4 p0 = v0 ? l0[0] : zero, p1 = v0 ? l0[1] : zero, q0 = v1 ? l1[0] : zero, q1 = v1 ? l1[1] : zero;
    hcw_v8f acc[2][4];
#pragma unroll
    for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
        for (unsigned s = 0; s < 4U; ++s) acc[t][s] = hcw_v8f{};
    const unsigned feature = fb * kHcwOutFeatures + lane % 16U;
    float nv[2][8][4];
#pragma unroll
    for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) {
            const unsigned r = row0 + 16U * t + 2U * e + lane / 16U;
            const auto* normalized = wide_row(rows.normalized, rows.normalized_stride, r < rows.count ? r : 0U);
#pragma unroll
            for (unsigned s = 0; s < 4U; ++s) nv[t][e][s] = normalized[static_cast<std::uint64_t>(s) * hidden + feature];
        }
    __syncthreads();
    for (unsigned k = 0; k < kHcwK16; ++k) {
        const hcw_v16h a0 = fragment(p0, p1), a1 = fragment(q0, q1);
        if (k + 1U < kHcwK16) {
            p0 = v0 ? l0[2U * k + 2U] : zero; p1 = v0 ? l0[2U * k + 3U] : zero;
            q0 = v1 ? l1[2U * k + 2U] : zero; q1 = v1 ? l1[2U * k + 3U] : zero;
        }
#pragma unroll
        for (unsigned s = 0; s < 4U; ++s) {
            const hcw_v16h b = hcw_lds_fragment(bt + ((k * 4U + s) * kHcwOutFeatures + lane % 16U) * 16U);
            acc[0][s] = hcw_wmma(a0, b, acc[0][s]);
            acc[1][s] = hcw_wmma(a1, b, acc[1][s]);
        }
    }
#pragma unroll
    for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) {
            const unsigned r = row0 + 16U * t + 2U * e + lane / 16U;
            if (r >= rows.count) continue;
            float mixed = 0.0F;
#pragma unroll
            for (unsigned s = 0; s < 4U; ++s) mixed += nv[t][e][s] * sigmoid(acc[t][s][e]);
            store_activation(&wide_row(static_cast<Storage*>(rows.output), rows.output_stride, r)[feature], mixed / 4.0F);
        }
}

struct HcwScratch {
    _Float16* down = nullptr;
    _Float16* up = nullptr;
    _Float16* low16 = nullptr;
    unsigned* overflow = nullptr;
};
HcwScratch hcw_scratch() {
    static const HcwScratch scratch = [] {
        HcwScratch s;
        void* p = nullptr;
        const std::uint64_t bytes = (kHcwDownPackHalves + kHcwUpPackHalves + std::uint64_t{kHcwLowRows} * kHcwRank) * 2U + 256U;
        if (hipMalloc(&p, bytes) != hipSuccess) throw std::runtime_error("HC WMMA rows: scratch allocation failed");
        auto* base = static_cast<_Float16*>(p);
        s.down = base;
        s.up = base + kHcwDownPackHalves;
        s.low16 = s.up + kHcwUpPackHalves;
        s.overflow = reinterpret_cast<unsigned*>(s.low16 + std::uint64_t{kHcwLowRows} * kHcwRank);
        if (hipMemset(s.overflow, 0, 256U) != hipSuccess) throw std::runtime_error("HC WMMA rows: scratch initialization failed");
        return s;
    }();
    return scratch;
}
