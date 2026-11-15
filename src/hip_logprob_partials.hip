// HIP backend core for the shared logprob partial contract (strix/logprob_partials.hpp). Compiled with
// -fno-fast-math -ffp-contract=off (CMakeLists): the nonfinite test and the FP32 lane order must hold as written.
#include "strix/hip_logprob_partials.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace strix {
namespace {
constexpr unsigned kThreads = kLogprobPartialLanes;
constexpr unsigned kTop = kLogprobPartialTop;
static_assert(kThreads == 256U, "the kernel's fold assumes 256 lanes");

__device__ __forceinline__ bool ahead(float va, std::uint32_t ia, float vb, std::uint32_t ib) {
    return va > vb || (va == vb && ia < ib);
}
__device__ __forceinline__ bool finite(float x) { return (__float_as_uint(x) & 0x7F800000U) != 0x7F800000U; }

// LDS: per-thread top lists (256 x 20 x 8 B = 40 KiB) + the fold arrays.
__global__ __launch_bounds__(kThreads) void logprob_partials_kernel(const float* __restrict__ logits, std::uint32_t count,
    std::uint32_t stride, std::uint32_t first, std::uint32_t top, const LogprobTarget* __restrict__ targets,
    LogprobPartial* __restrict__ out) {
    __shared__ float list_v[kTop][kThreads];
    __shared__ std::uint32_t list_i[kTop][kThreads];
    __shared__ float fold_f[kThreads];
    __shared__ std::uint32_t fold_u[kThreads], fold_w[kThreads], fold_x[kThreads];
    const unsigned t = threadIdx.x;
    const float* row = logits + static_cast<std::size_t>(blockIdx.x) * stride;
    // 1. maximum and nonfinite count (order-free)
    float m = -INFINITY;
    std::uint32_t bad = 0;
    for (std::uint32_t i = t; i < count; i += kThreads) {
        const float x = row[i];
        if (finite(x)) m = fmaxf(m, x); else ++bad;
    }
    fold_f[t] = m;
    fold_u[t] = bad;
    __syncthreads();
    for (unsigned half = kThreads / 2U; half > 0U; half /= 2U) {
        if (t < half) { fold_f[t] = fmaxf(fold_f[t], fold_f[t + half]); fold_u[t] += fold_u[t + half]; }
        __syncthreads();
    }
    const float maximum = fold_f[0];
    const std::uint32_t nonfinite = fold_u[0];
    __syncthreads();
    // 2. the contract's FP32 sum: lane t sums entries t, t + 256, ... in turn, then the lanes fold pairwise
    float s = 0.0F;
    for (std::uint32_t i = t; i < count; i += kThreads) {
        const float x = row[i];
        if (finite(x)) s += expf(x - maximum);
    }
    fold_f[t] = s;
    // 3. the target: its logit (this slice, or the caller's), then the entries ahead of it (integers, order-free)
    LogprobTarget target{};
    if (targets != nullptr) target = targets[blockIdx.x];
    const bool has_target = target.id != kLogprobNoTarget;
    const bool found = has_target && target.id >= first && target.id - first < count;
    if (found) { target.logit = row[target.id - first]; target.logit_known = true; }
    std::uint32_t greater = 0, ties = 0;
    if (has_target && target.logit_known)
        for (std::uint32_t i = t; i < count; i += kThreads) {
            const float x = row[i];
            if (!finite(x)) continue;
            if (x > target.logit) ++greater;
            else if (x == target.logit && first + i < target.id) ++ties;
        }
    fold_w[t] = greater;
    fold_x[t] = ties;
    __syncthreads();
    for (unsigned half = kThreads / 2U; half > 0U; half /= 2U) {
        if (t < half) { fold_f[t] += fold_f[t + half]; fold_w[t] += fold_w[t + half]; fold_x[t] += fold_x[t + half]; }
        __syncthreads();
    }
    LogprobPartial* p = out + blockIdx.x;
    if (t == 0U) {
        p->max = maximum;
        p->sum = fold_f[0];
        p->target = found ? target.logit : 0.0F;
        p->flags = (found ? LogprobPartial::kFound : 0U) | (has_target && target.logit_known ? LogprobPartial::kCounted : 0U);
        p->greater = fold_w[0];
        p->ties_lower = fold_x[0];
        p->nonfinite = nonfinite;
    }
    // 4. top: each lane keeps its own sorted list (its IDs ascend, so a later equal value never displaces), then `top`
    // rounds pick the best list head (value descending, lowest ID first)
    std::uint32_t n = 0;
    if (top != 0U)
        for (std::uint32_t i = t; i < count; i += kThreads) {
            const float x = row[i];
            if (!finite(x)) continue;
            const std::uint32_t id = first + i;
            if (n == top && !ahead(x, id, list_v[top - 1U][t], list_i[top - 1U][t])) continue;
            std::uint32_t at = n < top ? n : top - 1U;
            while (at > 0U && ahead(x, id, list_v[at - 1U][t], list_i[at - 1U][t])) {
                list_v[at][t] = list_v[at - 1U][t];
                list_i[at][t] = list_i[at - 1U][t];
                --at;
            }
            list_v[at][t] = x;
            list_i[at][t] = id;
            if (n < top) ++n;
        }
    std::uint32_t head = 0, taken = 0;
    for (std::uint32_t k = 0; k < top; ++k) {
        const bool live = head < n;
        fold_f[t] = live ? list_v[head][t] : -INFINITY;
        fold_u[t] = live ? list_i[head][t] : 0xFFFFFFFFU;
        fold_w[t] = live ? t : 0xFFFFFFFFU;
        __syncthreads();
        for (unsigned half = kThreads / 2U; half > 0U; half /= 2U) {
            if (t < half && fold_w[t + half] != 0xFFFFFFFFU &&
                (fold_w[t] == 0xFFFFFFFFU || ahead(fold_f[t + half], fold_u[t + half], fold_f[t], fold_u[t]))) {
                fold_f[t] = fold_f[t + half];
                fold_u[t] = fold_u[t + half];
                fold_w[t] = fold_w[t + half];
            }
            __syncthreads();
        }
        const std::uint32_t winner = fold_w[0];
        if (winner == 0xFFFFFFFFU) break;  // uniform: every lane reads the same fold_w[0]
        if (t == 0U) { p->top_value[k] = fold_f[0]; p->top_id[k] = fold_u[0]; }
        if (t == winner) ++head;
        ++taken;
        __syncthreads();
    }
    if (t == 0U) p->top_count = taken;
}
} // namespace

void enqueue_hip_logprob_partials(const HipLogprobPartialsView& v, hipStream_t stream) {
    if (v.rows == 0U) return;
    if (v.logits == nullptr || v.out == nullptr || v.count == 0U || v.stride < v.count || v.top > kLogprobPartialTop)
        throw std::invalid_argument("hip logprob partials: bad view");
    hipLaunchKernelGGL(logprob_partials_kernel, dim3(v.rows), dim3(kThreads), 0, stream, v.logits, v.count, v.stride, v.first,
                       v.top, v.targets, v.out);
    if (const auto e = hipGetLastError(); e != hipSuccess)
        throw std::runtime_error(std::string("hip logprob partials: launch failed: ") + hipGetErrorString(e));
}

} // namespace strix
