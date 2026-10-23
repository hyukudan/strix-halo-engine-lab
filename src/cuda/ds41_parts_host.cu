// deepseek41 CUDA parts-host kernels: resident-expert routes and the seeded combine (cuda_ds41_parts_host.hpp).

#include "strix/cuda_ds41_parts_host.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace strix::ds41::cuda {

namespace {

constexpr unsigned kFull = 0xffffffffU;
constexpr int kThreads = 256;

// Block-wide exclusive scan of one value per thread (blockDim.x a multiple of 32, <= 1024); returns the total.
__device__ std::uint32_t block_exclusive_scan(std::uint32_t& value, std::uint32_t* warp_sums) {
    const std::uint32_t lane = threadIdx.x & 31U, warp = threadIdx.x / 32U, warps = blockDim.x / 32U;
    std::uint32_t inclusive = value;
    for (int d = 1; d < 32; d *= 2) {
        const std::uint32_t other = __shfl_up_sync(kFull, inclusive, d);
        if (lane >= static_cast<std::uint32_t>(d)) inclusive += other;
    }
    if (lane == 31U) warp_sums[warp] = inclusive;
    __syncthreads();
    if (warp == 0U) {
        std::uint32_t s = lane < warps ? warp_sums[lane] : 0U, inc = s;
        for (int d = 1; d < 32; d *= 2) {
            const std::uint32_t other = __shfl_up_sync(kFull, inc, d);
            if (lane >= static_cast<std::uint32_t>(d)) inc += other;
        }
        if (lane < warps) warp_sums[lane] = inc - s;
        if (lane == 31U) warp_sums[32] = inc;
    }
    __syncthreads();
    value = warp_sums[warp] + inclusive - value;
    const std::uint32_t total = warp_sums[32];
    __syncthreads();
    return total;
}

__global__ __launch_bounds__(1024) void resident_routes_kernel(ResidentRoutesView v) {
    __shared__ std::uint32_t count[1024];
    __shared__ std::uint32_t offset_of[1024];
    __shared__ std::uint32_t warp_sums[33];
    __shared__ std::uint32_t s_max;
    const std::uint32_t slots = v.rows * v.top_k, t = threadIdx.x;
    if (t < 1024U) count[t] = 0U;
    if (t == 0U) s_max = 0U;
    __syncthreads();
    for (std::uint32_t s = t; s < slots; s += blockDim.x) {
        const std::uint32_t e = v.ids[s];
        if (e < v.experts && v.slot_of[e] >= 0) atomicAdd(&count[e], 1U);
    }
    __syncthreads();
    // one thread per expert id (experts <= 1024): union index and entry offset in ascending id
    const std::uint32_t c = t < v.experts ? count[t] : 0U;
    std::uint32_t present = c > 0U ? 1U : 0U, offset = c;
    if (c) atomicMax(&s_max, c);
    const std::uint32_t unions = block_exclusive_scan(present, warp_sums);
    const std::uint32_t total = block_exclusive_scan(offset, warp_sums);
    if (c) {
        v.out.experts[present] = static_cast<std::uint32_t>(v.slot_of[t]);
        v.out.begins[present] = offset;
    }
    offset_of[t] = offset;
    if (t == 0U) {
        v.out.begins[unions] = total;
        v.out.counts[0] = unions;
        v.out.counts[1] = s_max;
    }
    __syncthreads();
    // fill: warp w takes experts w, w + 32, ...; each present one scans the slots in ascending order
    const std::uint32_t warp = t / 32U, lane = t & 31U, warps = blockDim.x / 32U;
    for (std::uint32_t e = warp; e < v.experts; e += warps) {
        if (count[e] == 0U) continue;
        std::uint32_t at = offset_of[e];
        for (std::uint32_t s0 = 0; s0 < slots; s0 += 32U) {
            const std::uint32_t s = s0 + lane;
            const bool match = s < slots && v.ids[s] == e;
            const unsigned ballot = __ballot_sync(kFull, match);
            if (match) {
                const std::uint32_t pos = at + __popc(ballot & ((1U << lane) - 1U));
                v.out.rows[pos] = s / v.top_k;
                v.out.slots[pos] = s;
            }
            at += __popc(ballot);
        }
    }
    // per row: slots in ascending id, count below split (every slot, resident or not)
    for (std::uint32_t r = t; r < v.rows; r += blockDim.x) {
        std::uint32_t id[16], slot[16];
        for (std::uint32_t k = 0; k < v.top_k; ++k) {
            const std::uint32_t s = r * v.top_k + k, e = v.ids[s];
            std::uint32_t a = k;
            while (a > 0U && id[a - 1U] > e) { id[a] = id[a - 1U]; slot[a] = slot[a - 1U]; --a; }
            id[a] = e; slot[a] = s;
        }
        std::uint32_t low = 0;
        for (std::uint32_t k = 0; k < v.top_k; ++k) {
            v.out.order[r * v.top_k + k] = slot[k];
            low += id[k] < v.split ? 1U : 0U;
        }
        v.out.low_counts[r] = low;
    }
}

__device__ __forceinline__ const float* vector_of(const SeededCombineView& c, std::int32_t src) {
    return c.remote[static_cast<std::uint32_t>(src) >> 24U] + std::uint64_t(static_cast<std::uint32_t>(src) & 0xffffffU) * c.hidden;
}

__global__ __launch_bounds__(kThreads) void seeded_combine_kernel(SeededCombineView c) {
    const std::uint64_t i = std::uint64_t{blockIdx.x} * kThreads + threadIdx.x;
    if (i >= std::uint64_t{c.rows} * c.hidden) return;
    const std::uint32_t r = static_cast<std::uint32_t>(i / c.hidden), o = static_cast<std::uint32_t>(i % c.hidden);
    const std::int32_t* seed = c.tables;
    const std::int32_t* start = c.tables + 2U * c.rows;
    const std::int32_t* source = c.tables + 4U * c.rows;
    const std::uint32_t low_count = c.low_counts[r], K = c.top_k;
    const std::uint32_t skip0 = static_cast<std::uint32_t>(start[r * 2U]), skip1 = static_cast<std::uint32_t>(start[r * 2U + 1U]);
    const std::int32_t sd0 = seed[r * 2U], sd1 = seed[r * 2U + 1U];
    // every load first (independent, so answers in mapped host memory are fetched concurrently), then the folds in
    // the declared order: low over positions [skip0, low_count), high over [low_count + skip1, top_k)
    float vals[16], wts[16];
#pragma unroll
    for (std::uint32_t j = 0; j < 16U; ++j) {
        vals[j] = 0.0F;
        wts[j] = 0.0F;
        const bool low = j < low_count;
        if (j < K && (low ? j >= skip0 : j >= low_count + skip1)) {
            const std::uint32_t slot = c.order[r * K + j];
            const std::int32_t src = source[slot];
            vals[j] = src >= 0 ? vector_of(c, src)[o] : c.raw[std::uint64_t{slot} * c.hidden + o];
            wts[j] = c.weights[slot];
        }
    }
    float half[2];
    half[0] = sd0 >= 0 ? vector_of(c, sd0)[o] : 0.0F;
    half[1] = sd1 >= 0 ? vector_of(c, sd1)[o] : 0.0F;
#pragma unroll
    for (std::uint32_t j = 0; j < 16U; ++j) {
        const bool low = j < low_count;
        if (j < K && (low ? j >= skip0 : j >= low_count + skip1)) {
            const float term = wts[j] * vals[j];
            if (low) half[0] = half[0] + term;
            else half[1] = half[1] + term;
        }
    }
    const float a = c.shared[i] + half[0];
    c.out[i] = a + half[1];
}

__global__ __launch_bounds__(kThreads) void export_window_kernel(ExportWindowView v) {
    const std::uint32_t t = threadIdx.x, n = v.rows * v.top_k;
    for (std::uint32_t i = t; i < n; i += kThreads) { v.host_ids[i] = v.ids[i]; v.host_weights[i] = v.weights[i]; }
    const std::uint32_t code_words = v.rows * v.hidden / 16U, groups = v.rows * v.hidden / 32U;
    const auto* src = reinterpret_cast<const int4*>(v.codes);
    auto* dst = reinterpret_cast<int4*>(v.host_codes);
    for (std::uint32_t i = t; i < code_words; i += kThreads) dst[i] = src[i];
    for (std::uint32_t i = t; i < groups; i += kThreads) v.host_deltas[i] = v.deltas[i];
    __threadfence_system();
    __syncthreads();
    if (t == 0U) {
        __threadfence_system();
        *v.flag = v.sequence;
    }
}

}  // namespace

void enqueue_export_window(const ExportWindowView& v, cudaStream_t stream) {
    if (v.rows == 0U || v.top_k == 0U || v.hidden % 32U || !v.flag) throw std::invalid_argument("ds41 cuda export window: inconsistent arguments");
    export_window_kernel<<<1, kThreads, 0, stream>>>(v);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

void enqueue_resident_routes(const ResidentRoutesView& v, cudaStream_t stream) {
    if (v.rows == 0U || v.top_k == 0U || v.top_k > 16U || v.experts == 0U || v.experts > 1024U || !v.ids || !v.slot_of)
        throw std::invalid_argument("ds41 cuda resident routes: inconsistent arguments");
    resident_routes_kernel<<<1, 1024, 0, stream>>>(v);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

void enqueue_seeded_combine(const SeededCombineView& c, cudaStream_t stream) {
    const std::uint64_t n = std::uint64_t{c.rows} * c.hidden;
    if (!n) return;
    if (c.top_k == 0U || c.top_k > 16U || !c.tables) throw std::invalid_argument("ds41 cuda seeded combine: inconsistent arguments");
    seeded_combine_kernel<<<static_cast<unsigned>((n + kThreads - 1U) / kThreads), kThreads, 0, stream>>>(c);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

}  // namespace strix::ds41::cuda
