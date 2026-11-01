// Prefill option hc_output_wmma (strix/prefill_identity.hpp; on by default, STRIX_PREFILL_HC_OUTPUT_WMMA=0 or
// STRIX_PREFILL_EXACT_FP16=1 restores the ordered FP32 output chains): the HCW_OUTPUT=2 arithmetic inside the fused qpf2k HC route.
// CTA = 32 rows x 128 features, four waves (32 features each), Band16 row-group order.
// One stream at a time: two row tiles x two feature tiles of FP32 WMMA accumulators,
// then sigmoid and the FP32 stream mix. Each accumulator visits K16 = 0..19 exactly
// as hcw_output_kernel; neither K32 staging nor stream scheduling reassociates it.
// Q8 scale * signed code is rounded to FP16 exactly as hcw_pack_up_kernel. The low
// image is the projection's existing RNE FP16 copy, never reconstructed from low_t.
// LDS: two 128 x 40 coefficient buffers + 32 x 40 lows = 23040 bytes. Target 128..160
// VGPRs (compiler resource report / native speed gate still required).
bool qpf2k_hc_out16() {
    return strix::prefill_option_enabled(strix::PrefillOption::hc_output_wmma) && qpf2k_hc_mask() != 0U;
}

template <typename Storage>
__global__ __launch_bounds__(128) void qpf2k_hc_out16_kernel(HyperWideRows rows, const unsigned char* __restrict__ up,
    const _Float16* __restrict__ low16, unsigned hidden) {
    constexpr unsigned Pitch = 40U, Band = 16U;
    __shared__ __align__(16) _Float16 bs[2][128U * Pitch];
    __shared__ __align__(16) _Float16 as[32U * Pitch];
    const unsigned tid = threadIdx.x, lane = tid % 32U, wave = tid / 32U;
    const unsigned groups = gridDim.x, features = gridDim.y, cta = blockIdx.x + groups * blockIdx.y;
    const unsigned base = cta / (Band * features) * Band, count = min(Band, groups - base), local = cta - base * features;
    const unsigned row0 = (base + local % count) * 32U, fb = local / count;
    const unsigned feature0 = fb * 128U + wave * 32U + lane % 16U;
    float mixed[2][2][8]{};
    for (unsigned s = 0; s < 4U; ++s) {
        hcw_v8f acc[2][2]{};
        // Each thread decodes one feature's Q8 block. Prefetch next K32 before
        // consuming this stage's LDS fragments; all global bytes remain read-only.
        _Float16 weights[32];
        uint4 lows;
        const auto load = [&](unsigned j) {
            const auto* b = up + (static_cast<std::uint64_t>(s * hidden + fb * 128U + tid) * 10U + j) * 34U;
#pragma unroll
            for (unsigned k = 0; k < 32U; ++k) weights[k] = static_cast<_Float16>(hcw_q8_value(b, k));
            const unsigned r = row0 + tid / 4U < rows.count ? row0 + tid / 4U : rows.count - 1U;
            lows = *reinterpret_cast<const uint4*>(low16 + static_cast<std::uint64_t>(r) * kHcwRank + j * 32U + (tid % 4U) * 8U);
        };
        load(0U);
        for (unsigned j = 0; j < 10U; ++j) {
            const unsigned buffer = j & 1U;
            // Previous stage's readers must finish before overwriting the lows.
            __syncthreads();
#pragma unroll
            for (unsigned q = 0; q < 4U; ++q) {
                uint4 word;
                __builtin_memcpy(&word, weights + q * 8U, 16U);
                *reinterpret_cast<uint4*>(bs[buffer] + tid * Pitch + q * 8U) = word;
            }
            *reinterpret_cast<uint4*>(as + (tid / 4U) * Pitch + (tid % 4U) * 8U) = lows;
            __syncthreads();
            if (j + 1U < 10U) load(j + 1U);
#pragma unroll
            for (unsigned h = 0; h < 2U; ++h) {
                const hcw_v16h a0 = hcw_lds_fragment(as + (lane % 16U) * Pitch + h * 16U);
                const hcw_v16h a1 = hcw_lds_fragment(as + (16U + lane % 16U) * Pitch + h * 16U);
#pragma unroll
                for (unsigned c = 0; c < 2U; ++c) {
                    const hcw_v16h b = hcw_lds_fragment(bs[buffer] + (wave * 32U + c * 16U + lane % 16U) * Pitch + h * 16U);
                    acc[0][c] = hcw_wmma(a0, b, acc[0][c]);
                    acc[1][c] = hcw_wmma(a1, b, acc[1][c]);
                    // Bound live LDS fragments; this emits no device instruction
                    // and leaves each accumulator's ascending K16 chain intact.
                    asm volatile("" : "+v"(acc[0][c]), "+v"(acc[1][c]) :: "memory");
                }
            }
        }
#pragma unroll
        for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
            for (unsigned c = 0; c < 2U; ++c)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) {
                    const unsigned r = row0 + t * 16U + 2U * e + lane / 16U;
                    const auto* normalized = wide_row(rows.normalized, rows.normalized_stride, r < rows.count ? r : 0U);
                    const float nv = normalized[static_cast<std::uint64_t>(s) * hidden + feature0 + c * 16U];
                    mixed[t][c][e] += nv * sigmoid(acc[t][c][e]);
                }
    }
#pragma unroll
    for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
        for (unsigned c = 0; c < 2U; ++c)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const unsigned r = row0 + t * 16U + 2U * e + lane / 16U;
                if (r < rows.count)
                    store_activation(&wide_row(static_cast<Storage*>(rows.output), rows.output_stride, r)[feature0 + c * 16U], mixed[t][c][e] / 4.0F);
            }
}
