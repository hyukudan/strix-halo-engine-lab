// Frozen from src/cuda/models/qwen/qwen_hc.cu at 093c15dc3, lines 1215-1292 and 1358-1440.
// The infinity spelling is changed for nvcc with the MSVC headers.
namespace frozen {
constexpr unsigned kExperts = 512U;
constexpr float kNegativeInfinity = __builtin_bit_cast(float, std::uint32_t{0xff800000U});

__device__ __forceinline__ bool better(float lv, unsigned li, float rv, unsigned ri) {  // :271-279
    return lv > rv || (lv == rv && li < ri);
}
__global__ __launch_bounds__(256) void route_select_kernel(const float* __restrict__ workspace, std::uint32_t top_k,
    bool normalize, std::uint32_t* __restrict__ ids, float* __restrict__ weights, am::Tables t) {
    constexpr unsigned kThreads = 256U, kWave = 32U, kCandidates = kExperts / kWave;
    const unsigned row = blockIdx.x, lane = threadIdx.x;
    __shared__ float logits[kExperts];
    __shared__ float scratch[kThreads];
    __shared__ float best_values[32];
    __shared__ unsigned best_indices[32];
    for (unsigned e = lane; e < kExperts; e += kThreads) logits[e] = workspace[static_cast<std::size_t>(row) * kExperts + e];
    __syncthreads();
    float local_max = kNegativeInfinity;
    for (unsigned e = lane; e < kExperts; e += kThreads) local_max = fmaxf(local_max, logits[e]);
    scratch[lane] = local_max;
    __syncthreads();
    for (unsigned stride = kThreads / 2U; stride > 0U; stride >>= 1U) {
        if (lane < stride) scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);
        __syncthreads();
    }
    const float maximum = scratch[0];
    __syncthreads();
    float local_sum = 0.0F;
    for (unsigned e = lane; e < kExperts; e += kThreads) {
        const float p = am::amd_expf(t, __fsub_rn(logits[e], maximum));
        logits[e] = p;
        local_sum = __fadd_rn(local_sum, p);
    }
    scratch[lane] = local_sum;
    __syncthreads();
    for (unsigned stride = kThreads / 2U; stride > 0U; stride >>= 1U) {
        if (lane < stride) scratch[lane] = __fadd_rn(scratch[lane], scratch[lane + stride]);
        __syncthreads();
    }
    const float denominator = scratch[0];
    for (unsigned e = lane; e < kExperts; e += kThreads) logits[e] = __fdiv_rn(logits[e], denominator);
    __syncthreads();
    if (lane < kWave) {
        float values[kCandidates];
        unsigned available = 0;
#pragma unroll
        for (unsigned i = 0; i < kCandidates; ++i) {
            values[i] = logits[lane + kWave * i];
            if (values[i] == values[i]) available |= 1U << i;  // NaN probabilities are excluded (:957)
        }
        for (unsigned slot = 0; slot < top_k; ++slot) {
            bool has = false;
            float best = kNegativeInfinity;
            unsigned index = kExperts;
#pragma unroll
            for (unsigned i = 0; i < kCandidates; ++i)
                if ((available & (1U << i)) != 0U) {
                    const unsigned expert = lane + kWave * i;
                    if (!has || better(values[i], expert, best, index)) { has = true; best = values[i]; index = expert; }
                }
#pragma unroll
            for (unsigned mask = kWave / 2U; mask > 0U; mask >>= 1U) {
                const float other = __shfl_xor_sync(0xffffffffU, best, static_cast<int>(mask));
                const unsigned other_index = __shfl_xor_sync(0xffffffffU, index, static_cast<int>(mask));
                const int other_has = __shfl_xor_sync(0xffffffffU, static_cast<int>(has), static_cast<int>(mask));
                if (other_has != 0 && (!has || better(other, other_index, best, index))) { has = true; best = other; index = other_index; }
            }
            if (lane == 0U) { best_values[slot] = has ? best : kNegativeInfinity; best_indices[slot] = has ? index : kExperts; }
            if (has && index % kWave == lane) available &= ~(1U << (index / kWave));
        }
    }
    __syncthreads();
    if (lane == 0U) {
        float selected_sum = 0.0F;
        for (unsigned slot = 0; slot < top_k; ++slot) selected_sum = __fadd_rn(selected_sum, best_values[slot]);
        if (!normalize) selected_sum = 1.0F;
        for (unsigned slot = 0; slot < top_k; ++slot) {
            ids[static_cast<std::size_t>(row) * top_k + slot] = best_indices[slot];
            weights[static_cast<std::size_t>(row) * top_k + slot] = __fdiv_rn(best_values[slot], selected_sum);
        }
    }
}
// Route select with 4 barriers instead of 20: thread t owns experts t and t + 256 (as route_select_kernel); max by shuffles +
// one smem exchange (order independent); the sum tree levels 128/64/32 are read by warp 0 from smem (a_m = a[l + 32 m]:
// (m, m + 4), (m, m + 2), (0, 1)), levels 16..1 are shfl_down. Same adds in the same pairs: bit-identical.
__global__ __launch_bounds__(256) void route_select_warp_kernel(const float* __restrict__ workspace, std::uint32_t top_k,
    bool normalize, std::uint32_t* __restrict__ ids, float* __restrict__ weights, am::Tables t) {
    constexpr unsigned kWave = 32U, kCandidates = kExperts / kWave;
    const unsigned row = blockIdx.x, tid = threadIdx.x, lane = tid % 32U, warp = tid / 32U;
    __shared__ float sa[256];
    __shared__ float sl[kExperts];
    __shared__ float wmax[8];
    __shared__ float sden;
    __shared__ float best_values[32];
    __shared__ unsigned best_indices[32];
    const float x0 = workspace[static_cast<std::size_t>(row) * kExperts + tid], x1 = workspace[static_cast<std::size_t>(row) * kExperts + tid + 256U];
    float local_max = fmaxf(fmaxf(kNegativeInfinity, x0), x1);
#pragma unroll
    for (unsigned mask = 16U; mask > 0U; mask >>= 1U) local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffU, local_max, mask));
    if (lane == 0U) wmax[warp] = local_max;
    __syncthreads();
    float maximum = wmax[0];
#pragma unroll
    for (unsigned w = 1; w < 8U; ++w) maximum = fmaxf(maximum, wmax[w]);
    const float p0 = am::amd_expf(t, __fsub_rn(x0, maximum)), p1 = am::amd_expf(t, __fsub_rn(x1, maximum));
    sa[tid] = __fadd_rn(__fadd_rn(0.0F, p0), p1);
    __syncthreads();
    if (warp == 0U) {
        float a[8];
#pragma unroll
        for (unsigned m = 0; m < 8U; ++m) a[m] = sa[lane + 32U * m];
        float t1[4], t2[2];
#pragma unroll
        for (unsigned m = 0; m < 4U; ++m) t1[m] = __fadd_rn(a[m], a[m + 4U]);
#pragma unroll
        for (unsigned m = 0; m < 2U; ++m) t2[m] = __fadd_rn(t1[m], t1[m + 2U]);
        float v = __fadd_rn(t2[0], t2[1]);
#pragma unroll
        for (unsigned stride = 16U; stride > 0U; stride >>= 1U) v = __fadd_rn(v, __shfl_down_sync(0xffffffffU, v, stride));
        if (lane == 0U) sden = v;
    }
    __syncthreads();
    const float denominator = sden;
    sl[tid] = __fdiv_rn(p0, denominator);
    sl[tid + 256U] = __fdiv_rn(p1, denominator);
    __syncthreads();
    if (warp != 0U) return;
    float values[kCandidates];
#pragma unroll
    for (unsigned i = 0; i < kCandidates; ++i) values[i] = sl[lane + kWave * i];
    unsigned available = 0;
#pragma unroll
    for (unsigned i = 0; i < kCandidates; ++i)
        if (values[i] == values[i]) available |= 1U << i;
    for (unsigned slot = 0; slot < top_k; ++slot) {
        bool has = false;
        float best = kNegativeInfinity;
        unsigned index = kExperts;
#pragma unroll
        for (unsigned i = 0; i < kCandidates; ++i)
            if ((available & (1U << i)) != 0U) {
                const unsigned expert = lane + kWave * i;
                if (!has || better(values[i], expert, best, index)) { has = true; best = values[i]; index = expert; }
            }
#pragma unroll
        for (unsigned mask = kWave / 2U; mask > 0U; mask >>= 1U) {
            const float other = __shfl_xor_sync(0xffffffffU, best, static_cast<int>(mask));
            const unsigned other_index = __shfl_xor_sync(0xffffffffU, index, static_cast<int>(mask));
            const int other_has = __shfl_xor_sync(0xffffffffU, static_cast<int>(has), static_cast<int>(mask));
            if (other_has != 0 && (!has || better(other, other_index, best, index))) { has = true; best = other; index = other_index; }
        }
        if (lane == 0U) { best_values[slot] = has ? best : kNegativeInfinity; best_indices[slot] = has ? index : kExperts; }
        if (has && index % kWave == lane) available &= ~(1U << (index / kWave));
    }
    __syncwarp();
    if (lane == 0U) {
        float selected_sum = 0.0F;
        for (unsigned slot = 0; slot < top_k; ++slot) selected_sum = __fadd_rn(selected_sum, best_values[slot]);
        if (!normalize) selected_sum = 1.0F;
        for (unsigned slot = 0; slot < top_k; ++slot) {
            ids[static_cast<std::size_t>(row) * top_k + slot] = best_indices[slot];
            weights[static_cast<std::size_t>(row) * top_k + slot] = __fdiv_rn(best_values[slot], selected_sum);
        }
    }
}

}  // namespace frozen
