// Draft FAST prefill score identity idxscore16. Included after hip_qsa_prefill_flash.inc.
// Inputs are already BF16-rounded by query norm/RoPE and the pooled-key producer.
// Keep each head's ReLU before the ascending head sum and the existing selection.
namespace qsa_flash {
// Four wave32 tiles, each 16 query rows x 16 keys; one CTA scores 16 x 64.
// K16 WMMA starts at +0, followed by ascending ordinary FP32 panel addition.
// Native C feedback is deliberately not part of this identity.
template <unsigned Rows, bool Batch, bool Packed>
__global__ __launch_bounds__(128) void index_score_wmma_kernel(
    QsaRowsScoreViews<Rows> single, const void* pooled_keys, std::uint32_t scored_blocks,
    float* maxima, std::uint32_t maxima_stride, const ScoreGroup<Rows>* groups = nullptr) {
#if STRIX_HAS_WMMA_BF16_W32
    static_assert(Rows % 16U == 0U);
    using bf16x16 = unsigned short __attribute__((ext_vector_type(16)));
    using f32x8 = float __attribute__((ext_vector_type(8)));
    const auto& views = [&]() -> const QsaRowsScoreViews<Rows>& {
        if constexpr (Batch) return groups[blockIdx.z].views;
        else return single;
    }();
    if constexpr (Batch) {
        const auto& g = groups[blockIdx.z];
        scored_blocks = g.blocks; maxima = g.maxima; maxima_stride = g.maxima_stride;
    }
    const unsigned base = blockIdx.x * 64U, r0 = blockIdx.y * 16U;
    if (base >= scored_blocks) return; // uniform, before any barriers
    const unsigned tid = threadIdx.x, lane = tid % 32U, wave = tid / 32U;
    // 136 BF16 elements = a 16-byte aligned pitch with padding between rows.
    __shared__ __align__(32) std::uint16_t keys[64][136], query[16][136];
    for (unsigned z = tid; z < 64U * kDim; z += 128U) {
        const unsigned b = base + z / kDim, f = z % kDim;
        std::uint16_t value = 0U;
        if (b < scored_blocks) {
            const auto at = static_cast<std::uint64_t>(b) * kDim + f;
            if constexpr (Packed) value = static_cast<const std::uint16_t*>(pooled_keys)[at];
            else value = static_cast<std::uint16_t>(__float_as_uint(static_cast<const float*>(pooled_keys)[at]) >> 16U);
        }
        keys[z / kDim][f] = value;
    }
    const auto fragment = [](const std::uint16_t* p) {
        bf16x16 v;
        const uint4 lo = reinterpret_cast<const uint4*>(p)[0], hi = reinterpret_cast<const uint4*>(p)[1];
        __builtin_memcpy(&v, &lo, 16U);
        __builtin_memcpy(reinterpret_cast<char*>(&v) + 16U, &hi, 16U);
        return v;
    };
    f32x8 total{};
#pragma unroll
    for (unsigned h = 0U; h < kHeads; ++h) {
        for (unsigned z = tid; z < 16U * kDim; z += 128U) {
            const unsigned r = r0 + z / kDim, f = z % kDim;
            // Dense-prefix rows may have no query projection; do not read it.
            query[z / kDim][f] = views.blocks[r] ? static_cast<std::uint16_t>(
                __float_as_uint(views.index_query[r][h * kDim + f]) >> 16U) : 0U;
        }
        __syncthreads();
        f32x8 dot{};
#pragma unroll
        for (unsigned k = 0U; k < kDim; k += 16U) {
            const auto a = fragment(&query[lane % 16U][k]);
            const auto b = fragment(&keys[wave * 16U + lane % 16U][k]);
            const f32x8 panel = STRIX_HIP_WMMA_BF16(a, b, f32x8{});
            dot += panel;
        }
#pragma unroll
        for (unsigned e = 0U; e < 8U; ++e) total[e] += fmaxf(dot[e], 0.0F);
        __syncthreads(); // every wave has finished reading this head's query
    }
    const unsigned b = base + wave * 16U + lane % 16U;
#pragma unroll
    for (unsigned e = 0U; e < 8U; ++e) {
        const unsigned r = r0 + 2U * e + lane / 16U;
        const float score = total[e] / sqrtf(static_cast<float>(kDim));
        if (b < views.blocks[r]) views.block_scores[r][b] = score;
        if (maxima) {
            float m = b < views.blocks[r] ? score : -INFINITY;
            m = fmaxf(m, __shfl_xor(m, 1, 32));
            m = fmaxf(m, __shfl_xor(m, 2, 32));
            if ((lane & 3U) == 0U && b < views.blocks[r])
                maxima[static_cast<std::uint64_t>(r) * maxima_stride + b / kGroup] = m;
        }
    }
#else
    (void)single; (void)pooled_keys; (void)scored_blocks; (void)maxima; (void)maxima_stride; (void)groups;
#if defined(__HIP_DEVICE_COMPILE__)
    __builtin_trap();
#endif
#endif
}
template <unsigned Rows, bool Batch>
void launch_index_score_wmma(QsaRowsScoreViews<Rows> views, const void* keys, bool packed,
    unsigned max_blocks, float* maxima, unsigned maxima_stride, const ScoreGroup<Rows>* groups,
    unsigned count, hipStream_t stream) {
    const auto launch = [&]<bool Packed>() {
        STRIX_HIP_LAUNCH_GGL(Rdna3Layout, HIP_KERNEL_NAME(index_score_wmma_kernel<Rows, Batch, Packed>),
            dim3((max_blocks + 63U) / 64U, Rows / 16U, count), dim3(128), 0, stream,
            views, keys, max_blocks, maxima, maxima_stride, groups);
    };
    if (packed) launch.template operator()<true>();
    else launch.template operator()<false>();
}
} // namespace qsa_flash
