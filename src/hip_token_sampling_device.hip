#include "strix/hip_token_sampling_device.hpp"
#include "strix/hip_token_sampling.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>

namespace strix {
namespace {

// Rows are split into kBlocks workgroups of kThreads threads; thread t of block b owns the contiguous index
// range [(b * kThreads + t) * per, +per), so every scan below runs in token id order.
constexpr unsigned kThreads = 256, kBlocks = 32, kMaxRows = 8, kCand = device_top_p_candidates, kSmallThreads = 1024;
constexpr std::uint32_t kNone = 0xffffffffU;
enum : std::uint32_t {
    flag_nonfinite = 1U, flag_host = 2U, flag_drafted = 4U, flag_accepted = 8U, flag_residual = 16U,
    flag_top_p = 32U, flag_need_residual = 64U, flag_done = 128U, flag_small = 256U, flag_bad_draft = 512U,
};

// Index i of a row is its i-th entry in token id order: entry order[i] of the logits, token ids[i]. Target rows
// (the full vocabulary) have neither map; draft rows (the draft head's entries) have both.
struct RowArgs {
    const float* logits = nullptr;
    const std::uint32_t* order = nullptr;
    const std::uint32_t* ids = nullptr;
    const std::uint32_t* q_tokens = nullptr; // draft distribution: token ids ascending, zero entries allowed
    const double* q_probs = nullptr;
    double* q_out = nullptr;                 // draft rows: the distribution per index, 0 outside it
    std::uint32_t* token_out = nullptr;      // draft rows: device copy of the sampled token
    std::uint32_t draft = kNone, q_count = 0;
    double u_plain = 0.0, u_accept = 0.0, u_residual = 0.0;
};
struct Rows { RowArgs row[kMaxRows]; };
struct Policy {
    std::uint32_t vocab = 0, top_k = 0, top_p_active = 0, rows = 0;
    double temperature = 1.0, cutoff = -40.0, top_p = 1.0;
};
// Radix selection of the m-th largest order key: after four 8-bit rounds `key` is the m-th key and `need` the
// number of elements equal to it that belong to the m largest (lowest token ids first).
struct Select { std::uint32_t active, prefix, mask, remaining; };
struct RowState {
    float max;
    std::uint32_t flags, cut_count, kept_count, last_kept, last_residual, token, pad;
    double total, residual_mass;
    Select topk, cand;
};
struct Candidate { std::uint32_t key, id; double w; };
struct Scratch {
    float* block_max;
    std::uint32_t *block_count, *block_eq, *hist;
    double *weights, *block_sum, *block_residual;
    Candidate* candidates;
    RowState* state;
};
struct Result { std::uint32_t token, flags; };

__device__ __forceinline__ std::uint32_t order_key(float x) {
    const auto u = __float_as_uint(x);
    return (u & 0x80000000U) ? ~u : (u | 0x80000000U);
}
__device__ __forceinline__ float logit_at(const RowArgs& a, std::uint32_t i) { return a.logits[a.order ? a.order[i] : i]; }
__device__ __forceinline__ std::uint32_t token_at(const RowArgs& a, std::uint32_t i) { return a.ids ? a.ids[i] : i; }
__device__ __forceinline__ std::uint32_t per_thread(std::uint32_t vocab) {
    return (vocab + kBlocks * kThreads - 1U) / (kBlocks * kThreads);
}
__device__ __forceinline__ void thread_range(std::uint32_t vocab, unsigned block, unsigned thread, std::uint32_t& begin,
    std::uint32_t& end) {
    const auto per = per_thread(vocab);
    const auto first = (block * kThreads + thread) * per;
    begin = first < vocab ? first : vocab;
    end = begin + per < vocab ? begin + per : vocab;
}
// sampling-v1 step 1a/2: z = (logit - max logit) / temperature in double.
__device__ __forceinline__ double z_of(float logit, float max, double temperature) {
    return (static_cast<double>(logit) - static_cast<double>(max)) / temperature;
}
// Deterministic block reductions: a fixed LDS tree.
template <typename T, typename Op> __device__ T block_reduce(T value, T* lds, Op op) {
    lds[threadIdx.x] = value;
    __syncthreads();
    for (unsigned s = kThreads / 2U; s > 0U; s >>= 1U) {
        if (threadIdx.x < s) lds[threadIdx.x] = op(lds[threadIdx.x], lds[threadIdx.x + s]);
        __syncthreads();
    }
    const T out = lds[0];
    __syncthreads();
    return out;
}
// Sum of per-thread values in thread order (thread 0 adds them sequentially): the order the searches use.
__device__ double block_sequential_sum(double value, double* lds) {
    lds[threadIdx.x] = value;
    __syncthreads();
    if (threadIdx.x == 0U) { double sum = 0.0; for (unsigned t = 0; t < kThreads; ++t) sum += lds[t]; lds[0] = sum; }
    __syncthreads();
    const double out = lds[0];
    __syncthreads();
    return out;
}
// Exclusive prefix of per-thread counts in thread order (thread 0 sums sequentially).
__device__ std::uint32_t block_exclusive(std::uint32_t value, std::uint32_t* lds) {
    lds[threadIdx.x] = value;
    __syncthreads();
    if (threadIdx.x == 0U) {
        std::uint32_t sum = 0;
        for (unsigned t = 0; t < kThreads; ++t) { const auto v = lds[t]; lds[t] = sum; sum += v; }
    }
    __syncthreads();
    const auto out = lds[threadIdx.x];
    __syncthreads();
    return out;
}

__global__ __launch_bounds__(kThreads) void sample_max_kernel(Rows rows, Policy policy, Scratch s) {
    const auto row = blockIdx.y;
    const RowArgs& a = rows.row[row];
    std::uint32_t begin, end;
    thread_range(policy.vocab, blockIdx.x, threadIdx.x, begin, end);
    float m = -INFINITY;
    bool bad = false;
    for (auto i = begin; i < end; ++i) { const float v = logit_at(a, i); bad = bad || !isfinite(v); m = fmaxf(m, v); }
    __shared__ float lds[kThreads];
    const float block = block_reduce(m, lds, [](float x, float y) { return fmaxf(x, y); });
    if (bad) atomicOr(&s.state[row].flags, flag_nonfinite);
    if (threadIdx.x == 0U) s.block_max[row * kBlocks + blockIdx.x] = block;
}
__device__ float row_max(const Scratch& s, unsigned row) {
    float m = -INFINITY;
    for (unsigned b = 0; b < kBlocks; ++b) m = fmaxf(m, s.block_max[row * kBlocks + b]);
    return m;
}
// Elements passing the cutoff, per block.
__global__ __launch_bounds__(kThreads) void sample_count_kernel(Rows rows, Policy policy, Scratch s) {
    const auto row = blockIdx.y;
    const RowArgs& a = rows.row[row];
    const float max = row_max(s, row);
    std::uint32_t begin, end;
    thread_range(policy.vocab, blockIdx.x, threadIdx.x, begin, end);
    std::uint32_t n = 0;
    for (auto i = begin; i < end; ++i) n += z_of(logit_at(a, i), max, policy.temperature) >= policy.cutoff ? 1U : 0U;
    __shared__ std::uint32_t lds[kThreads];
    const auto total = block_reduce(n, lds, [](std::uint32_t x, std::uint32_t y) { return x + y; });
    if (threadIdx.x == 0U) s.block_count[row * kBlocks + blockIdx.x] = total;
    if (blockIdx.x == 0U && threadIdx.x == 0U) s.state[row].max = max;
}
// After the counts: top-k selection state, and whether the kept set is small enough for one workgroup.
__global__ void sample_after_count_kernel(Policy policy, Scratch s) {
    const auto row = blockIdx.x;
    auto& st = s.state[row];
    std::uint32_t cut = 0;
    for (unsigned b = 0; b < kBlocks; ++b) cut += s.block_count[row * kBlocks + b];
    st.cut_count = cut;
    const bool topk = policy.top_k && cut > policy.top_k;
    st.topk = {topk ? 1U : 0U, 0U, 0U, policy.top_k};
    if ((topk ? policy.top_k : cut) <= kCand) st.flags |= flag_small;
}
// One radix round of selection `which` (0 top-k, 1 top-p candidates) over every finite element.
template <unsigned Which> __device__ Select& selection(RowState& st) { if constexpr (Which == 0U) return st.topk; else return st.cand; }
template <unsigned Which>
__global__ __launch_bounds__(kThreads) void sample_hist_kernel(Rows rows, Policy policy, Scratch s, unsigned shift) {
    const auto row = blockIdx.y;
    const auto sel = selection<Which>(s.state[row]);
    if (!sel.active) return;
    __shared__ std::uint32_t hist[256];
    hist[threadIdx.x] = 0U;
    __syncthreads();
    const RowArgs& a = rows.row[row];
    std::uint32_t begin, end;
    thread_range(policy.vocab, blockIdx.x, threadIdx.x, begin, end);
    for (auto i = begin; i < end; ++i) {
        const auto key = order_key(logit_at(a, i));
        if ((key & sel.mask) == sel.prefix) atomicAdd(&hist[(key >> shift) & 255U], 1U);
    }
    __syncthreads();
    s.hist[(static_cast<std::uint64_t>(row) * kBlocks + blockIdx.x) * 256U + threadIdx.x] = hist[threadIdx.x];
}
template <unsigned Which> __global__ __launch_bounds__(kThreads) void sample_pick_kernel(Scratch s, unsigned shift) {
    const auto row = blockIdx.x;
    auto& sel = selection<Which>(s.state[row]);
    if (!sel.active) return;
    __shared__ std::uint32_t counts[256];
    std::uint32_t c = 0;
    for (unsigned b = 0; b < kBlocks; ++b) c += s.hist[(static_cast<std::uint64_t>(row) * kBlocks + b) * 256U + threadIdx.x];
    counts[threadIdx.x] = c;
    __syncthreads();
    if (threadIdx.x == 0U) {
        std::uint32_t above = 0;
        for (int bin = 255; bin >= 0; --bin) {
            if (above + counts[bin] >= sel.remaining) {
                sel.remaining -= above;
                sel.prefix |= static_cast<std::uint32_t>(bin) << shift;
                sel.mask |= 255U << shift;
                break;
            }
            above += counts[bin];
        }
    }
}
// Elements equal to the selected key, per block.
template <unsigned Which> __global__ __launch_bounds__(kThreads) void sample_equal_kernel(Rows rows, Policy policy, Scratch s) {
    const auto row = blockIdx.y;
    const auto sel = selection<Which>(s.state[row]);
    if (!sel.active) return;
    const RowArgs& a = rows.row[row];
    std::uint32_t begin, end;
    thread_range(policy.vocab, blockIdx.x, threadIdx.x, begin, end);
    std::uint32_t n = 0;
    for (auto i = begin; i < end; ++i) n += order_key(logit_at(a, i)) == sel.prefix ? 1U : 0U;
    __shared__ std::uint32_t lds[kThreads];
    const auto total = block_reduce(n, lds, [](std::uint32_t x, std::uint32_t y) { return x + y; });
    if (threadIdx.x == 0U) s.block_eq[(Which * kMaxRows + row) * kBlocks + blockIdx.x] = total;
}
// Rank (among elements equal to the selected key, token id order) of the first element of this thread's range.
template <unsigned Which>
__device__ std::uint32_t equal_rank_base(const Scratch& s, const Select& sel, const RowArgs& a, std::uint32_t begin,
    std::uint32_t end, unsigned row, std::uint32_t* lds) {
    std::uint32_t mine = 0;
    if (sel.active) for (auto i = begin; i < end; ++i) mine += order_key(logit_at(a, i)) == sel.prefix ? 1U : 0U;
    std::uint32_t base = block_exclusive(mine, lds);
    if (sel.active) for (unsigned b = 0; b < blockIdx.x; ++b) base += s.block_eq[(Which * kMaxRows + row) * kBlocks + b];
    return base;
}
__device__ __forceinline__ bool selected(const Select& sel, std::uint32_t key, std::uint32_t& equal_rank) {
    if (!sel.active) return true;
    if (key > sel.prefix) return true;
    if (key < sel.prefix) return false;
    return equal_rank++ < sel.remaining;
}
__device__ double q_probability(const RowArgs& a, std::uint32_t token) {
    std::uint32_t lo = 0, hi = a.q_count;
    while (lo < hi) { const auto mid = (lo + hi) / 2U; if (a.q_tokens[mid] < token) lo = mid + 1U; else hi = mid; }
    return lo < a.q_count && a.q_tokens[lo] == token ? a.q_probs[lo] : 0.0;
}
__device__ __forceinline__ bool candidate_before(const Candidate& a, const Candidate& b) {
    return a.key > b.key || (a.key == b.key && a.id < b.id);
}
// Bitonic sort of c[0, size) (size a power of two) by (key desc, id asc), all threads of the block.
__device__ void sort_candidates(Candidate* c, unsigned size) {
    for (unsigned span = 2U; span <= size; span <<= 1U)
        for (unsigned stride = span >> 1U; stride > 0U; stride >>= 1U) {
            for (unsigned j = threadIdx.x; j < size; j += blockDim.x) {
                const auto partner = j ^ stride;
                if (partner > j) {
                    const bool ascending = (j & span) == 0U;
                    const bool swap = ascending ? candidate_before(c[partner], c[j]) : candidate_before(c[j], c[partner]);
                    if (swap) { const auto t = c[j]; c[j] = c[partner]; c[partner] = t; }
                }
            }
            __syncthreads();
        }
}

// Small rows: at most kCand entries pass the cutoff and top-k. They are compacted in token order and decided by
// one workgroup exactly as the host does (sequential sums in token id order, or in top-p order).
__global__ __launch_bounds__(kThreads) void sample_small_count_kernel(Rows rows, Policy policy, Scratch s) {
    const auto row = blockIdx.y;
    const auto& st = s.state[row];
    if ((st.flags & (flag_small | flag_nonfinite)) != flag_small) return;
    const RowArgs& a = rows.row[row];
    std::uint32_t begin, end;
    thread_range(policy.vocab, blockIdx.x, threadIdx.x, begin, end);
    __shared__ std::uint32_t ranks[kThreads];
    const auto topk = st.topk;
    const float max = st.max;
    auto rank = equal_rank_base<0U>(s, topk, a, begin, end, row, ranks);
    std::uint32_t n = 0;
    for (auto i = begin; i < end; ++i) {
        const float l = logit_at(a, i);
        n += z_of(l, max, policy.temperature) >= policy.cutoff && selected(topk, order_key(l), rank) ? 1U : 0U;
    }
    __shared__ std::uint32_t counts[kThreads];
    const auto total = block_reduce(n, counts, [](std::uint32_t x, std::uint32_t y) { return x + y; });
    if (threadIdx.x == 0U) s.block_count[row * kBlocks + blockIdx.x] = total;
}
__global__ __launch_bounds__(kThreads) void sample_small_collect_kernel(Rows rows, Policy policy, Scratch s) {
    const auto row = blockIdx.y;
    const auto& st = s.state[row];
    if ((st.flags & (flag_small | flag_nonfinite)) != flag_small) return;
    const RowArgs& a = rows.row[row];
    std::uint32_t begin, end;
    thread_range(policy.vocab, blockIdx.x, threadIdx.x, begin, end);
    __shared__ std::uint32_t ranks[kThreads];
    const auto topk = st.topk;
    const float max = st.max;
    auto rank = equal_rank_base<0U>(s, topk, a, begin, end, row, ranks);
    auto rank_count = rank;
    std::uint32_t n = 0;
    for (auto i = begin; i < end; ++i) {
        const float l = logit_at(a, i);
        n += z_of(l, max, policy.temperature) >= policy.cutoff && selected(topk, order_key(l), rank_count) ? 1U : 0U;
    }
    __shared__ std::uint32_t lds[kThreads];
    auto position = block_exclusive(n, lds);
    for (unsigned b = 0; b < blockIdx.x; ++b) position += s.block_count[row * kBlocks + b];
    Candidate* out = s.candidates + static_cast<std::uint64_t>(row) * kCand;
    for (auto i = begin; i < end; ++i) {
        const float l = logit_at(a, i);
        const double z = z_of(l, max, policy.temperature);
        if (z >= policy.cutoff && selected(topk, order_key(l), rank) && position < kCand) out[position++] = {order_key(l), i, exp(z)};
    }
}
__global__ __launch_bounds__(kSmallThreads) void sample_small_decide_kernel(Rows rows, Policy policy, Scratch s, Result* results) {
    const auto row = blockIdx.x;
    const auto& st = s.state[row];
    if ((st.flags & (flag_small | flag_nonfinite)) != flag_small) return;
    const RowArgs& a = rows.row[row];
    __shared__ Candidate c[kCand];
    __shared__ std::uint32_t kept_shared;
    __shared__ double total_shared;
    if (threadIdx.x == 0U) {
        std::uint32_t kept = 0;
        for (unsigned b = 0; b < kBlocks; ++b) kept += s.block_count[row * kBlocks + b];
        kept_shared = kept < kCand ? kept : kCand;
    }
    __syncthreads();
    const auto n = kept_shared;
    unsigned size = 2U;
    while (size < n) size <<= 1U;
    const Candidate* in = s.candidates + static_cast<std::uint64_t>(row) * kCand;
    for (unsigned j = threadIdx.x; j < size; j += blockDim.x) c[j] = j < n ? in[j] : Candidate{0U, kNone, 0.0};
    if (a.q_out) for (std::uint32_t j = threadIdx.x; j < policy.vocab; j += blockDim.x) a.q_out[j] = 0.0;
    __syncthreads();
    // The host's total: the kept weights summed in token id order.
    if (threadIdx.x == 0U) { double total = 0.0; for (std::uint32_t j = 0; j < n; ++j) total += c[j].w; total_shared = total; }
    __syncthreads();
    const double total = total_shared;
    if (policy.top_p_active) sort_candidates(c, size);
    if (threadIdx.x != 0U) return;
    // With top-p, the shortest (w desc, id asc) prefix reaching top_p * total, normalized by its own mass.
    std::uint32_t count = n;
    double norm = total;
    if (policy.top_p_active) {
        const double target = policy.top_p * total;
        double mass = 0.0;
        count = 0;
        while (count < n) { mass += c[count].w; ++count; if (mass >= target) break; }
        norm = mass;
    }
    const auto p_of = [&](std::uint32_t j) { return c[j].w / norm; };
    const auto sample = [&](double u) {
        double cumulative = 0.0;
        for (std::uint32_t j = 0; j < count; ++j) { cumulative += p_of(j); if (u < cumulative) return token_at(a, c[j].id); }
        return token_at(a, c[count - 1U].id);
    };
    std::uint32_t token = kNone, flags = flag_done;
    if (a.draft == kNone) token = sample(a.u_plain);
    else {
        flags |= flag_drafted;
        const double qd = q_probability(a, a.draft);
        double pd = 0.0;
        for (std::uint32_t j = 0; j < count; ++j) if (token_at(a, c[j].id) == a.draft) { pd = p_of(j); break; }
        if (!(qd > 0.0)) { token = a.draft; flags |= flag_bad_draft; }
        else if (a.u_accept * qd < pd) { token = a.draft; flags |= flag_accepted; }
        else {
            flags |= flag_residual;
            double mass = 0.0;
            for (std::uint32_t j = 0; j < count; ++j) mass += fmax(0.0, p_of(j) - q_probability(a, token_at(a, c[j].id)));
            if (!(mass > 0.0)) token = sample(a.u_residual);
            else {
                const double target = a.u_residual * mass;
                double cumulative = 0.0;
                std::uint32_t last = a.draft;
                for (std::uint32_t j = 0; j < count; ++j) {
                    const auto t = token_at(a, c[j].id);
                    const double r = fmax(0.0, p_of(j) - q_probability(a, t));
                    if (!(r > 0.0)) continue;
                    cumulative += r; last = t;
                    if (target < cumulative) { token = t; break; }
                }
                if (token == kNone) token = last;
            }
        }
    }
    if (a.q_out) for (std::uint32_t j = 0; j < count; ++j) a.q_out[c[j].id] = p_of(j);
    if (a.token_out) *a.token_out = token;
    results[row] = {token, flags};
}

// Large rows: weights w = exp(z) of the kept elements (cutoff, then top-k), written per index (0 elsewhere);
// block sums.
__global__ __launch_bounds__(kThreads) void sample_weights_kernel(Rows rows, Policy policy, Scratch s) {
    const auto row = blockIdx.y;
    auto& st = s.state[row];
    if (st.flags & flag_small) return;
    const RowArgs& a = rows.row[row];
    const float max = st.max;
    std::uint32_t begin, end;
    thread_range(policy.vocab, blockIdx.x, threadIdx.x, begin, end);
    __shared__ std::uint32_t ranks[kThreads];
    const auto topk = st.topk;
    auto rank = equal_rank_base<0U>(s, topk, a, begin, end, row, ranks);
    double* weights = s.weights + static_cast<std::uint64_t>(row) * policy.vocab;
    double sum = 0.0;
    std::uint32_t kept = 0, last = 0;
    for (auto i = begin; i < end; ++i) {
        const float l = logit_at(a, i);
        const double z = z_of(l, max, policy.temperature);
        const bool keep = z >= policy.cutoff && selected(topk, order_key(l), rank);
        const double w = keep ? exp(z) : 0.0;
        weights[i] = w;
        if (keep) { sum += w; ++kept; last = i; }
    }
    __shared__ double lds[kThreads];
    __shared__ std::uint32_t counts[kThreads];
    const double block_sum = block_sequential_sum(sum, lds);
    const auto block_kept = block_reduce(kept, counts, [](std::uint32_t x, std::uint32_t y) { return x + y; });
    if (kept) atomicMax(&st.last_kept, last);
    if (threadIdx.x == 0U) { s.block_sum[row * kBlocks + blockIdx.x] = block_sum; s.block_count[row * kBlocks + blockIdx.x] = block_kept; }
}
__global__ void sample_after_weights_kernel(Policy policy, Scratch s) {
    const auto row = blockIdx.x;
    auto& st = s.state[row];
    if (st.flags & flag_small) return;
    double total = 0.0;
    std::uint32_t kept = 0;
    for (unsigned b = 0; b < kBlocks; ++b) { total += s.block_sum[row * kBlocks + b]; kept += s.block_count[row * kBlocks + b]; }
    st.total = total; st.kept_count = kept;
    if (policy.top_p_active) {
        st.flags |= flag_top_p;
        // The kept set is the largest-key set, so its top kCand entries are the overall top kCand.
        st.cand = {kept > kCand ? 1U : 0U, 0U, 0U, kCand};
    }
}
// Large draft rows: the distribution per index.
__global__ __launch_bounds__(kThreads) void sample_q_out_kernel(Rows rows, Policy policy, Scratch s) {
    const auto row = blockIdx.y;
    const auto& st = s.state[row];
    const RowArgs& a = rows.row[row];
    if (!a.q_out || (st.flags & (flag_small | flag_nonfinite | flag_top_p))) return;
    const double* weights = s.weights + static_cast<std::uint64_t>(row) * policy.vocab;
    std::uint32_t begin, end;
    thread_range(policy.vocab, blockIdx.x, threadIdx.x, begin, end);
    for (auto i = begin; i < end; ++i) a.q_out[i] = weights[i] / st.total;
}
// Top-p candidates: kept elements (and, with more than kCand kept, the kCand largest), compacted in token order.
__global__ __launch_bounds__(kThreads) void sample_candidate_count_kernel(Rows rows, Policy policy, Scratch s) {
    const auto row = blockIdx.y;
    auto& st = s.state[row];
    if (st.flags & flag_small) return;
    const RowArgs& a = rows.row[row];
    const double* weights = s.weights + static_cast<std::uint64_t>(row) * policy.vocab;
    std::uint32_t begin, end;
    thread_range(policy.vocab, blockIdx.x, threadIdx.x, begin, end);
    __shared__ std::uint32_t ranks[kThreads];
    const auto cand = st.cand;
    auto rank = equal_rank_base<1U>(s, cand, a, begin, end, row, ranks);
    std::uint32_t n = 0;
    for (auto i = begin; i < end; ++i) n += weights[i] > 0.0 && selected(cand, order_key(logit_at(a, i)), rank) ? 1U : 0U;
    __shared__ std::uint32_t counts[kThreads];
    const auto total = block_reduce(n, counts, [](std::uint32_t x, std::uint32_t y) { return x + y; });
    if (threadIdx.x == 0U) s.block_count[row * kBlocks + blockIdx.x] = total;
}
__global__ __launch_bounds__(kThreads) void sample_collect_kernel(Rows rows, Policy policy, Scratch s) {
    const auto row = blockIdx.y;
    auto& st = s.state[row];
    if (st.flags & flag_small) return;
    const RowArgs& a = rows.row[row];
    const double* weights = s.weights + static_cast<std::uint64_t>(row) * policy.vocab;
    std::uint32_t begin, end;
    thread_range(policy.vocab, blockIdx.x, threadIdx.x, begin, end);
    __shared__ std::uint32_t ranks[kThreads];
    const auto cand = st.cand;
    auto rank = equal_rank_base<1U>(s, cand, a, begin, end, row, ranks);
    auto rank_count = rank;
    std::uint32_t n = 0;
    for (auto i = begin; i < end; ++i) n += weights[i] > 0.0 && selected(cand, order_key(logit_at(a, i)), rank_count) ? 1U : 0U;
    __shared__ std::uint32_t lds[kThreads];
    auto position = block_exclusive(n, lds);
    for (unsigned b = 0; b < blockIdx.x; ++b) position += s.block_count[row * kBlocks + b];
    Candidate* out = s.candidates + static_cast<std::uint64_t>(row) * kCand;
    for (auto i = begin; i < end; ++i) {
        const auto key = order_key(logit_at(a, i));
        if (weights[i] > 0.0 && selected(cand, key, rank) && position < kCand) out[position++] = {key, i, weights[i]};
    }
}
// Large top-p rows: sampling-v1's (w desc, id asc) prefix, its mass and the decision, in the host's arithmetic.
__global__ __launch_bounds__(kSmallThreads) void sample_top_p_decide_kernel(Rows rows, Policy policy, Scratch s, Result* results) {
    const auto row = blockIdx.x;
    auto& st = s.state[row];
    if (st.flags & (flag_nonfinite | flag_small)) return;
    __shared__ Candidate c[kCand];
    const auto n = st.kept_count < kCand ? st.kept_count : kCand;
    const Candidate* in = s.candidates + static_cast<std::uint64_t>(row) * kCand;
    for (unsigned j = threadIdx.x; j < kCand; j += blockDim.x) c[j] = j < n ? in[j] : Candidate{0U, kNone, 0.0};
    __syncthreads();
    sort_candidates(c, kCand);
    if (threadIdx.x != 0U) return;
    const RowArgs& a = rows.row[row];
    const double target = policy.top_p * st.total;
    double mass = 0.0;
    std::uint32_t count = 0;
    while (count < n) { mass += c[count].w; ++count; if (mass >= target) break; }
    if (mass < target && st.kept_count > n) { results[row] = {0U, flag_host}; return; }
    const auto p_of = [&](std::uint32_t j) { return c[j].w / mass; };
    std::uint32_t token = kNone, flags = 0U;
    const auto sample = [&](double u) {
        double cumulative = 0.0;
        for (std::uint32_t j = 0; j < count; ++j) { cumulative += p_of(j); if (u < cumulative) return token_at(a, c[j].id); }
        return token_at(a, c[count - 1U].id);
    };
    if (a.draft == kNone) token = sample(a.u_plain);
    else {
        flags |= flag_drafted;
        const double qd = q_probability(a, a.draft);
        double pd = 0.0;
        for (std::uint32_t j = 0; j < count; ++j) if (token_at(a, c[j].id) == a.draft) { pd = p_of(j); break; }
        if (!(qd > 0.0)) { token = a.draft; flags |= flag_bad_draft; }
        else if (a.u_accept * qd < pd) { token = a.draft; flags |= flag_accepted; }
        else {
            flags |= flag_residual;
            double rmass = 0.0;
            for (std::uint32_t j = 0; j < count; ++j) rmass += fmax(0.0, p_of(j) - q_probability(a, token_at(a, c[j].id)));
            if (!(rmass > 0.0)) token = sample(a.u_residual);
            else {
                const double rtarget = a.u_residual * rmass;
                double cumulative = 0.0;
                std::uint32_t last = a.draft;
                for (std::uint32_t j = 0; j < count; ++j) {
                    const auto t = token_at(a, c[j].id);
                    const double r = fmax(0.0, p_of(j) - q_probability(a, t));
                    if (!(r > 0.0)) continue;
                    cumulative += r; last = t;
                    if (rtarget < cumulative) { token = t; break; }
                }
                if (token == kNone) token = last;
            }
        }
    }
    if (a.token_out) *a.token_out = token;
    results[row] = {token, flags | flag_done};
}
// Large rows without top-p: find the element where the running sum of `value(i)` first exceeds `target`;
// thread 0 walks the block sums, the block's threads sum their ranges, thread 0 their prefix, one thread its range.
template <typename Value>
__device__ std::uint32_t token_order_search(const double* block_sums, double target, std::uint32_t vocab, Value&& value,
    std::uint32_t fallback, double* lds_d, std::uint32_t* lds_u) {
    __shared__ unsigned chosen_block;
    __shared__ double before_block;
    if (threadIdx.x == 0U) {
        chosen_block = kNone; double cumulative = 0.0;
        for (unsigned b = 0; b < kBlocks; ++b) {
            if (target < cumulative + block_sums[b]) { chosen_block = b; break; }
            cumulative += block_sums[b];
        }
        before_block = cumulative;
    }
    __syncthreads();
    const auto block = chosen_block;
    if (block == kNone) return fallback;
    std::uint32_t begin, end;
    thread_range(vocab, block, threadIdx.x, begin, end);
    double mine = 0.0;
    for (auto i = begin; i < end; ++i) mine += value(i);
    lds_d[threadIdx.x] = mine;
    __syncthreads();
    if (threadIdx.x == 0U) {
        double cumulative = before_block, last_start = before_block;
        std::uint32_t chosen = kNone, last = kNone;
        for (unsigned t = 0; t < kThreads; ++t) {
            const double next = cumulative + lds_d[t];
            if (lds_d[t] > 0.0) { last = t; last_start = cumulative; }
            if (target < next) { chosen = t; break; }
            cumulative = next;
        }
        // Rounding: the block's threads summed from its start can stay just below a target its total exceeded.
        if (chosen == kNone) { chosen = last; cumulative = last_start; }
        lds_u[0] = chosen;
        lds_d[0] = cumulative;
    }
    __syncthreads();
    const auto thread = lds_u[0];
    const double start = lds_d[0];
    __syncthreads();
    if (thread == kNone) return fallback;
    __shared__ std::uint32_t found;
    if (threadIdx.x == 0U) found = fallback;
    __syncthreads();
    if (threadIdx.x == thread) {
        thread_range(vocab, block, thread, begin, end);
        double cumulative = start;
        std::uint32_t last = kNone;
        for (auto i = begin; i < end; ++i) {
            const double v = value(i);
            if (!(v > 0.0)) continue;
            cumulative += v; last = i;
            if (target < cumulative) break;
        }
        // The range's own running sum may round just below the target that its total exceeded.
        if (last != kNone) found = last;
    }
    __syncthreads();
    return found;
}
// Drafted rows are target rows (index = token id).
__global__ __launch_bounds__(kThreads) void sample_decide_kernel(Rows rows, Policy policy, Scratch s, Result* results) {
    const auto row = blockIdx.x;
    auto& st = s.state[row];
    if (st.flags & (flag_nonfinite | flag_top_p | flag_small)) return;
    const RowArgs& a = rows.row[row];
    const double* weights = s.weights + static_cast<std::uint64_t>(row) * policy.vocab;
    const double total = st.total;
    __shared__ double lds_d[kThreads];
    __shared__ std::uint32_t lds_u[kThreads];
    const auto plain = [&](double u) {
        return token_order_search(s.block_sum + row * kBlocks, u * total, policy.vocab, [&](std::uint32_t i) { return weights[i]; },
            st.last_kept, lds_d, lds_u);
    };
    if (a.draft == kNone) {
        const auto token = token_at(a, plain(a.u_plain));
        if (threadIdx.x == 0U) { if (a.token_out) *a.token_out = token; results[row] = {token, flag_done}; }
        return;
    }
    const double qd = q_probability(a, a.draft);
    const double pd = weights[a.draft] / total;
    if (!(qd > 0.0)) {
        if (threadIdx.x == 0U) results[row] = {a.draft, flag_done | flag_drafted | flag_bad_draft};
        return;
    }
    if (a.u_accept * qd < pd) {
        if (threadIdx.x == 0U) results[row] = {a.draft, flag_done | flag_drafted | flag_accepted};
        return;
    }
    if (threadIdx.x == 0U) st.flags |= flag_need_residual;
}
// Residual rows: r = max(0, p - q) per kept token, block sums.
__global__ __launch_bounds__(kThreads) void sample_residual_kernel(Rows rows, Policy policy, Scratch s) {
    const auto row = blockIdx.y;
    auto& st = s.state[row];
    if (!(st.flags & flag_need_residual)) return;
    const RowArgs& a = rows.row[row];
    const double* weights = s.weights + static_cast<std::uint64_t>(row) * policy.vocab;
    std::uint32_t begin, end;
    thread_range(policy.vocab, blockIdx.x, threadIdx.x, begin, end);
    double sum = 0.0;
    std::uint32_t last = 0;
    bool any = false;
    for (auto i = begin; i < end; ++i) {
        if (!(weights[i] > 0.0)) continue;
        const double r = fmax(0.0, weights[i] / st.total - q_probability(a, i));
        if (r > 0.0) { sum += r; last = i; any = true; }
    }
    __shared__ double lds[kThreads];
    const double block = block_sequential_sum(sum, lds);
    if (any) atomicMax(&st.last_residual, last);
    if (threadIdx.x == 0U) s.block_residual[row * kBlocks + blockIdx.x] = block;
}
__global__ __launch_bounds__(kThreads) void sample_residual_decide_kernel(Rows rows, Policy policy, Scratch s, Result* results) {
    const auto row = blockIdx.x;
    auto& st = s.state[row];
    if (!(st.flags & flag_need_residual)) return;
    const RowArgs& a = rows.row[row];
    const double* weights = s.weights + static_cast<std::uint64_t>(row) * policy.vocab;
    __shared__ double lds_d[kThreads];
    __shared__ std::uint32_t lds_u[kThreads];
    double mass = 0.0;
    for (unsigned b = 0; b < kBlocks; ++b) mass += s.block_residual[row * kBlocks + b];
    std::uint32_t token;
    if (!(mass > 0.0))
        token = token_order_search(s.block_sum + row * kBlocks, a.u_residual * st.total, policy.vocab,
            [&](std::uint32_t i) { return weights[i]; }, st.last_kept, lds_d, lds_u);
    else
        token = token_order_search(s.block_residual + row * kBlocks, a.u_residual * mass, policy.vocab,
            [&](std::uint32_t i) { return weights[i] > 0.0 ? fmax(0.0, weights[i] / st.total - q_probability(a, i)) : 0.0; },
            st.last_residual, lds_d, lds_u);
    if (threadIdx.x == 0U) results[row] = {token, flag_done | flag_drafted | flag_residual};
}
__global__ void sample_nonfinite_kernel(Scratch s, Result* results, unsigned rows) {
    const auto row = threadIdx.x;
    if (row < rows && (s.state[row].flags & flag_nonfinite)) results[row] = {0U, flag_nonfinite};
}

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
void require(bool ok, const char* message) { if (!ok) throw std::invalid_argument(message); }

// Scratch for `max_rows` rows of at most `capacity` entries, and the kernel sequence.
struct Engine {
    std::uint32_t capacity, max_rows;
    Scratch scratch{};
    void* device_memory = nullptr;
    Engine(std::uint32_t c, std::uint32_t rows) : capacity(c), max_rows(rows) {
        require(c > 0U && rows > 0U && rows <= kMaxRows, "device token sampler geometry is invalid");
        std::uint64_t at = 0;
        const auto take = [&](std::uint64_t bytes) { const auto out = at; at = (at + bytes + 255U) / 256U * 256U; return out; };
        const auto o_max = take(sizeof(float) * kMaxRows * kBlocks), o_count = take(4ULL * kMaxRows * kBlocks),
            o_eq = take(4ULL * 2U * kMaxRows * kBlocks), o_hist = take(4ULL * kMaxRows * kBlocks * 256U),
            o_w = take(8ULL * rows * c), o_sum = take(8ULL * kMaxRows * kBlocks), o_res = take(8ULL * kMaxRows * kBlocks),
            o_cand = take(sizeof(Candidate) * kMaxRows * kCand), o_state = take(sizeof(RowState) * kMaxRows);
        check(hipMalloc(&device_memory, at), "device token sampler scratch");
        auto* base = static_cast<unsigned char*>(device_memory);
        scratch.block_max = reinterpret_cast<float*>(base + o_max);
        scratch.block_count = reinterpret_cast<std::uint32_t*>(base + o_count);
        scratch.block_eq = reinterpret_cast<std::uint32_t*>(base + o_eq);
        scratch.hist = reinterpret_cast<std::uint32_t*>(base + o_hist);
        scratch.weights = reinterpret_cast<double*>(base + o_w);
        scratch.block_sum = reinterpret_cast<double*>(base + o_sum);
        scratch.block_residual = reinterpret_cast<double*>(base + o_res);
        scratch.candidates = reinterpret_cast<Candidate*>(base + o_cand);
        scratch.state = reinterpret_cast<RowState*>(base + o_state);
    }
    ~Engine() { if (device_memory) (void)hipFree(device_memory); }
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    // Enqueues every kernel for `count` rows of `vocab` entries; `results` (pinned host) is filled when the stream
    // reaches the end of the sequence.
    void enqueue(const Rows& r, unsigned count, std::uint32_t vocab, const TokenSamplingPolicy& policy, Result* results,
        hipStream_t stream) {
        require(count > 0U && count <= max_rows && vocab > 0U && vocab <= capacity, "device token sampling geometry is invalid");
        Policy p;
        p.vocab = vocab; p.top_k = policy.top_k; p.top_p_active = policy.top_p < 1.0F ? 1U : 0U; p.rows = count;
        p.temperature = policy.temperature; p.top_p = policy.top_p;
        // sampling-v1 step 2: the cutoff of z.
        p.cutoff = -40.0;
        if (policy.min_p > 0.0F) p.cutoff = std::max(p.cutoff, std::log(static_cast<double>(policy.min_p)));
        check(hipMemsetAsync(scratch.state, 0, sizeof(RowState) * kMaxRows, stream), "device sampler state reset");
        for (unsigned i = 0; i < count; ++i) results[i] = {0U, 0U};
        const dim3 grid(kBlocks, count);
        hipLaunchKernelGGL(sample_max_kernel, grid, dim3(kThreads), 0, stream, r, p, scratch);
        hipLaunchKernelGGL(sample_count_kernel, grid, dim3(kThreads), 0, stream, r, p, scratch);
        hipLaunchKernelGGL(sample_after_count_kernel, dim3(count), dim3(1), 0, stream, p, scratch);
        if (p.top_k) {
            for (unsigned shift = 24U, round = 0U; round < 4U; ++round, shift -= 8U) {
                hipLaunchKernelGGL(sample_hist_kernel<0U>, grid, dim3(kThreads), 0, stream, r, p, scratch, shift);
                hipLaunchKernelGGL(sample_pick_kernel<0U>, dim3(count), dim3(kThreads), 0, stream, scratch, shift);
            }
            hipLaunchKernelGGL(sample_equal_kernel<0U>, grid, dim3(kThreads), 0, stream, r, p, scratch);
        }
        hipLaunchKernelGGL(sample_small_count_kernel, grid, dim3(kThreads), 0, stream, r, p, scratch);
        hipLaunchKernelGGL(sample_small_collect_kernel, grid, dim3(kThreads), 0, stream, r, p, scratch);
        hipLaunchKernelGGL(sample_small_decide_kernel, dim3(count), dim3(kSmallThreads), 0, stream, r, p, scratch, results);
        hipLaunchKernelGGL(sample_weights_kernel, grid, dim3(kThreads), 0, stream, r, p, scratch);
        hipLaunchKernelGGL(sample_after_weights_kernel, dim3(count), dim3(1), 0, stream, p, scratch);
        if (p.top_p_active) {
            for (unsigned shift = 24U, round = 0U; round < 4U; ++round, shift -= 8U) {
                hipLaunchKernelGGL(sample_hist_kernel<1U>, grid, dim3(kThreads), 0, stream, r, p, scratch, shift);
                hipLaunchKernelGGL(sample_pick_kernel<1U>, dim3(count), dim3(kThreads), 0, stream, scratch, shift);
            }
            hipLaunchKernelGGL(sample_equal_kernel<1U>, grid, dim3(kThreads), 0, stream, r, p, scratch);
            hipLaunchKernelGGL(sample_candidate_count_kernel, grid, dim3(kThreads), 0, stream, r, p, scratch);
            hipLaunchKernelGGL(sample_collect_kernel, grid, dim3(kThreads), 0, stream, r, p, scratch);
            hipLaunchKernelGGL(sample_top_p_decide_kernel, dim3(count), dim3(kSmallThreads), 0, stream, r, p, scratch, results);
        } else {
            hipLaunchKernelGGL(sample_q_out_kernel, grid, dim3(kThreads), 0, stream, r, p, scratch);
            hipLaunchKernelGGL(sample_decide_kernel, dim3(count), dim3(kThreads), 0, stream, r, p, scratch, results);
            hipLaunchKernelGGL(sample_residual_kernel, grid, dim3(kThreads), 0, stream, r, p, scratch);
            hipLaunchKernelGGL(sample_residual_decide_kernel, dim3(count), dim3(kThreads), 0, stream, r, p, scratch, results);
        }
        hipLaunchKernelGGL(sample_nonfinite_kernel, dim3(1), dim3(kMaxRows), 0, stream, scratch, results, count);
        check(hipGetLastError(), "device token sampling launch");
    }
};

Result* pinned_results(std::uint32_t count) {
    Result* out = nullptr;
    check(hipHostMalloc(reinterpret_cast<void**>(&out), sizeof(Result) * count, hipHostMallocDefault), "device token sampler results");
    return out;
}
bool env_is(const char* name, const char* value) {
    const char* v = std::getenv(name);
    return v && std::strcmp(v, value) == 0;
}

} // namespace

TokenSamplingPath token_sampling_path(const TokenSamplingPolicy& policy) noexcept {
    // -1 auto, else the forced path.
    static const int forced = [] {
        if (env_is("STRIX_SAMPLE_HOST", "1") || env_is("STRIX_SAMPLE_PATH", "host")) return 1;
        if (env_is("STRIX_SAMPLE_PATH", "device")) return 0;
        return -1;
    }();
    if (forced >= 0) return forced ? TokenSamplingPath::host : TokenSamplingPath::device;
    // auto: the device path for every policy. Single smoke runs measured it faster than the host path for every
    // kind of decision (window rows of each size, serial rows, draft steps) with temperature only, top-k, top-p,
    // min-p and their combination; the host only won with small kept sets before the one-workgroup path existed.
    static_cast<void>(policy);
    return TokenSamplingPath::device;
}
bool device_draft_sampling_enabled(const TokenSamplingPolicy& policy) noexcept {
    static const bool draft_host = env_is("STRIX_SAMPLE_DRAFT_HOST", "1");
    return !draft_host && device_token_sampling_enabled(policy) &&
           (policy.top_p >= 1.0F || (policy.top_k > 0U && policy.top_k <= device_top_p_candidates));
}

struct HipTokenSamplerDevice::Impl {
    std::uint32_t vocab, max_rows;
    Engine engine;
    Result* results = nullptr;                // pinned host
    struct QBuffer { std::uint32_t* tokens = nullptr; double* probs = nullptr; std::uint32_t capacity = 0; };
    QBuffer q[kMaxRows];                      // pinned host, read by the kernels
    std::uint64_t fallbacks = 0;
    std::vector<float> host_logits;
    hipEvent_t start = nullptr, stop = nullptr;
    Impl(std::uint32_t v, std::uint32_t rows) : vocab(v), max_rows(rows), engine(v, rows), results(pinned_results(kMaxRows)) {}
    ~Impl() {
        for (auto& b : q) { if (b.tokens) (void)hipHostFree(b.tokens); if (b.probs) (void)hipHostFree(b.probs); }
        if (results) (void)hipHostFree(results);
        if (start) (void)hipEventDestroy(start);
        if (stop) (void)hipEventDestroy(stop);
    }
    // A host draft distribution, token id ascending for the kernels' binary search.
    void stage_q(unsigned row, const TokenDistribution& d, RowArgs& a) {
        auto& b = q[row];
        const auto n = static_cast<std::uint32_t>(d.tokens().size());
        if (b.capacity < n) {
            if (b.tokens) { check(hipHostFree(b.tokens), "free draft tokens"); check(hipHostFree(b.probs), "free draft probabilities"); }
            const auto cap = std::max<std::uint32_t>(n, 4096U);
            check(hipHostMalloc(reinterpret_cast<void**>(&b.tokens), 4ULL * cap, hipHostMallocDefault), "draft tokens");
            check(hipHostMalloc(reinterpret_cast<void**>(&b.probs), 8ULL * cap, hipHostMallocDefault), "draft probabilities");
            b.capacity = cap;
        }
        std::vector<std::uint32_t> order(n);
        for (std::uint32_t i = 0; i < n; ++i) order[i] = i;
        const auto tokens = d.tokens(); const auto probs = d.probabilities();
        std::sort(order.begin(), order.end(), [&](std::uint32_t x, std::uint32_t y) { return tokens[x] < tokens[y]; });
        for (std::uint32_t i = 0; i < n; ++i) { b.tokens[i] = tokens[order[i]]; b.probs[i] = probs[order[i]]; }
        a.q_tokens = b.tokens; a.q_probs = b.probs; a.q_count = n;
    }
    void set_draft(unsigned row, const TokenDraft& draft, RowArgs& a) {
        require(draft.token < vocab, "draft token lies outside the vocabulary");
        a.draft = draft.token;
        if (draft.device.probabilities) {
            a.q_tokens = draft.device.tokens; a.q_probs = draft.device.probabilities; a.q_count = draft.device.count;
        } else stage_q(row, draft.distribution, a);
    }
    void run(const Rows& r, unsigned count, const TokenSamplingPolicy& policy, hipStream_t stream) {
        const bool timed = sample_timing_enabled();
        if (timed && !start) { check(hipEventCreate(&start), "sampler timing event"); check(hipEventCreate(&stop), "sampler timing event"); }
        if (timed) check(hipEventRecord(start, stream), "sampler timing start");
        engine.enqueue(r, count, vocab, policy, results, stream);
        if (timed) check(hipEventRecord(stop, stream), "sampler timing stop");
        check(hipStreamSynchronize(stream), "device token sampling completion");
        if (timed) {
            float ms = 0.0F;
            check(hipEventElapsedTime(&ms, start, stop), "sampler timing");
            record_sample_timing(SampleTimingKind::kernels, count, 0.0, static_cast<double>(ms) * 1e-3, count);
        }
    }
};

HipTokenSamplerDevice::HipTokenSamplerDevice(std::uint32_t vocabulary, std::uint32_t max_rows)
    : impl_(std::make_unique<Impl>(vocabulary, max_rows)) {}
HipTokenSamplerDevice::~HipTokenSamplerDevice() = default;
std::uint64_t HipTokenSamplerDevice::host_fallbacks() const noexcept { return impl_->fallbacks; }

TokenDraft host_token_draft(const TokenDraft& draft, void* stream) {
    if (!draft.device.probabilities) return draft;
    const auto n = draft.device.count;
    std::vector<std::uint32_t> tokens(n);
    std::vector<double> probs(n);
    const auto s = static_cast<hipStream_t>(stream);
    check(hipMemcpyAsync(tokens.data(), draft.device.tokens, 4ULL * n, hipMemcpyDeviceToHost, s), "draft tokens readback");
    check(hipMemcpyAsync(probs.data(), draft.device.probabilities, 8ULL * n, hipMemcpyDeviceToHost, s), "draft probabilities readback");
    check(hipStreamSynchronize(s), "draft distribution readback");
    TokenDraft out;
    out.position = draft.position; out.token = draft.token;
    out.distribution = make_token_distribution_from_probabilities(tokens, probs);
    return out;
}

std::vector<TokenDecision> HipTokenSamplerDevice::decide(TokenSampler& sampler, std::span<const DeviceSampleRow> rows, void* stream) {
    auto& i = *impl_;
    require(!rows.empty() && rows.size() <= i.max_rows, "device token sampling row count is invalid");
    const auto& policy = sampler.policy();
    Rows r{};
    for (unsigned k = 0; k < rows.size(); ++k) {
        const auto& row = rows[k];
        require(row.logits.address && row.logits.bytes >= 4ULL * i.vocab, "sampled logits row is smaller than the vocabulary");
        auto& a = r.row[k];
        a.logits = reinterpret_cast<const float*>(row.logits.address);
        const auto u = sampler.uniforms(row.position, row.draft != nullptr);
        a.u_plain = u.plain; a.u_accept = u.accept; a.u_residual = u.residual;
        if (row.draft) {
            require(row.draft->position == row.position, "draft position differs from the decided position");
            i.set_draft(k, *row.draft, a);
        }
    }
    const auto s = static_cast<hipStream_t>(stream);
    i.run(r, static_cast<unsigned>(rows.size()), policy, s);
    std::vector<TokenDecision> out(rows.size());
    for (unsigned k = 0; k < rows.size(); ++k) {
        const auto result = i.results[k];
        require(!(result.flags & flag_nonfinite), "token distribution requires finite logits");
        require(!(result.flags & flag_bad_draft), "draft token must have positive draft probability");
        if (result.flags & flag_host || !(result.flags & flag_done)) {
            // Oversized top-p prefix: the host transform over this row's logits.
            ++i.fallbacks;
            read_hip_logits_row(rows[k].logits, i.vocab, stream, i.host_logits);
            const auto draft = rows[k].draft ? std::optional<TokenDraft>(host_token_draft(*rows[k].draft, stream)) : std::nullopt;
            out[k] = decide_token(make_token_distribution(i.host_logits, {}, policy), draft ? &*draft : nullptr,
                r.row[k].u_plain, r.row[k].u_accept, r.row[k].u_residual);
            continue;
        }
        out[k].token = result.token;
        out[k].drafted = (result.flags & flag_drafted) != 0U;
        out[k].accepted = (result.flags & flag_accepted) != 0U;
        out[k].residual = (result.flags & flag_residual) != 0U;
    }
    return out;
}

struct HipTokenDraftSamplerDevice::Impl {
    std::uint32_t capacity, max_steps, rows = 0, uploaded = 0;
    Engine engine;
    Result* results = nullptr;          // pinned host, one per step
    std::uint32_t* order = nullptr;     // device: index (token id order) -> head entry
    std::uint32_t* sorted_ids = nullptr; // device: index -> token id, ascending
    double* q = nullptr;                // device: [max_steps][capacity]
    std::uint32_t* staging = nullptr;   // pinned host: order then ids
    std::vector<std::uint64_t> positions;
    Impl(std::uint32_t c, std::uint32_t steps)
        : capacity(c), max_steps(steps), engine(c, 1U), results(pinned_results(std::max<std::uint32_t>(steps, 1U))), positions(steps) {
        require(steps > 0U, "device draft sampler needs a step");
        check(hipMalloc(reinterpret_cast<void**>(&order), 4ULL * c), "device draft order");
        check(hipMalloc(reinterpret_cast<void**>(&sorted_ids), 4ULL * c), "device draft ids");
        check(hipMalloc(reinterpret_cast<void**>(&q), 8ULL * c * steps), "device draft distributions");
        check(hipHostMalloc(reinterpret_cast<void**>(&staging), 8ULL * c, hipHostMallocDefault), "device draft staging");
    }
    ~Impl() {
        if (results) (void)hipHostFree(results);
        if (staging) (void)hipHostFree(staging);
        for (void* p : {static_cast<void*>(order), static_cast<void*>(sorted_ids), static_cast<void*>(q)}) if (p) (void)hipFree(p);
    }
    void set_vocabulary(std::span<const std::uint32_t> ids, std::uint32_t n, hipStream_t stream) {
        require(n > 0U && n <= capacity && (ids.empty() || ids.size() == n), "device draft vocabulary is invalid");
        rows = n;
        if (n == uploaded) return;
        // The staging copy of the previous upload must be complete before it is rewritten.
        check(hipStreamSynchronize(stream), "device draft vocabulary staging");
        std::uint32_t* perm = staging;
        std::uint32_t* sorted = staging + capacity;
        std::iota(perm, perm + n, 0U);
        if (!ids.empty()) std::sort(perm, perm + n, [&](std::uint32_t x, std::uint32_t y) { return ids[x] < ids[y]; });
        for (std::uint32_t i = 0; i < n; ++i) sorted[i] = ids.empty() ? perm[i] : ids[perm[i]];
        for (std::uint32_t i = 1; i < n; ++i) require(sorted[i - 1U] != sorted[i], "token distribution ids must be distinct");
        check(hipMemcpyAsync(order, perm, 4ULL * n, hipMemcpyHostToDevice, stream), "device draft order upload");
        check(hipMemcpyAsync(sorted_ids, sorted, 4ULL * n, hipMemcpyHostToDevice, stream), "device draft ids upload");
        uploaded = n;
    }
    void step(const TokenSamplingPolicy& policy, std::uint32_t k, const float* logits, double u, std::uint32_t* slot,
        hipStream_t stream) {
        require(rows > 0U && k < max_steps && logits, "device draft step is invalid");
        Rows r{};
        auto& a = r.row[0];
        a.logits = logits; a.order = order; a.ids = sorted_ids;
        a.q_out = q + static_cast<std::uint64_t>(k) * capacity; a.token_out = slot;
        a.u_plain = u;
        engine.enqueue(r, 1U, rows, policy, results + k, stream);
    }
    TokenDraft draft(std::uint32_t k) const {
        const auto result = results[k];
        require(!(result.flags & flag_nonfinite), "token distribution requires finite logits");
        if (result.flags & flag_host || !(result.flags & flag_done))
            throw std::logic_error("device draft step needed the host (policy outside the device draft path)");
        TokenDraft d;
        d.position = positions[k]; d.token = result.token;
        d.device = {sorted_ids, q + static_cast<std::uint64_t>(k) * capacity, rows};
        return d;
    }
};

HipTokenDraftSamplerDevice::HipTokenDraftSamplerDevice(std::uint32_t capacity, std::uint32_t max_steps)
    : impl_(std::make_unique<Impl>(capacity, max_steps)) {}
HipTokenDraftSamplerDevice::~HipTokenDraftSamplerDevice() = default;
void HipTokenDraftSamplerDevice::set_vocabulary(std::span<const std::uint32_t> host_ids, std::uint32_t rows, void* stream) {
    impl_->set_vocabulary(host_ids, rows, static_cast<hipStream_t>(stream));
}
void HipTokenDraftSamplerDevice::step(const TokenSampler& sampler, std::uint32_t step, const float* logits, std::uint64_t position,
    std::uint32_t* slot, void* stream) {
    require(step < impl_->max_steps, "device draft step is invalid");
    impl_->positions[step] = position;
    impl_->step(sampler.policy(), step, logits, sampler.draft_uniform(position), slot, static_cast<hipStream_t>(stream));
}
void HipTokenDraftSamplerDevice::finish(TokenSampler& sampler, std::uint32_t steps, std::vector<TokenDraft>& out) {
    require(steps <= impl_->max_steps, "device draft steps exceed the sampler");
    out.clear();
    for (std::uint32_t k = 0; k < steps; ++k) { out.push_back(impl_->draft(k)); sampler.count_draft(); }
}

TokenDecision hip_decide_token_for_testing(std::span<const float> logits, const TokenSamplingPolicy& policy,
    const TokenDraft* draft, double u_plain, double u_accept, double u_residual, bool* host_fallback, bool device_q) {
    const auto vocab = static_cast<std::uint32_t>(logits.size());
    validate_token_sampling_policy(policy);
    HipTokenSamplerDevice::Impl impl(vocab, 1U);
    std::vector<void*> allocations;
    const auto upload = [&](const void* data, std::uint64_t bytes) {
        void* p = nullptr;
        check(hipMalloc(&p, bytes), "testing allocation");
        allocations.push_back(p);
        check(hipMemcpy(p, data, bytes, hipMemcpyHostToDevice), "testing upload");
        return p;
    };
    struct Free { std::vector<void*>& v; ~Free() { for (void* p : v) (void)hipFree(p); } } free_all{allocations};
    Rows r{};
    auto& a = r.row[0];
    a.logits = static_cast<const float*>(upload(logits.data(), 4ULL * vocab));
    a.u_plain = u_plain; a.u_accept = u_accept; a.u_residual = u_residual;
    TokenDraft view;
    if (draft) {
        if (device_q) {
            // Token ids ascending with every third id outside the distribution present as a zero entry.
            std::vector<std::uint32_t> ids;
            std::vector<double> probs;
            const auto& d = draft->distribution;
            std::vector<std::uint32_t> members(d.tokens().begin(), d.tokens().end());
            std::sort(members.begin(), members.end());
            std::size_t m = 0;
            for (std::uint32_t t = 0; t < vocab; ++t) {
                const bool member = m < members.size() && members[m] == t;
                if (member) { ids.push_back(t); probs.push_back(d.probability(t)); ++m; }
                else if (t % 3U == 0U) { ids.push_back(t); probs.push_back(0.0); }
            }
            view.position = draft->position; view.token = draft->token;
            view.device = {static_cast<const std::uint32_t*>(upload(ids.data(), 4ULL * ids.size())),
                           static_cast<const double*>(upload(probs.data(), 8ULL * probs.size())),
                           static_cast<std::uint32_t>(ids.size())};
            impl.set_draft(0U, view, a);
        } else impl.set_draft(0U, *draft, a);
    }
    impl.run(r, 1U, policy, nullptr);
    const auto result = impl.results[0];
    require(!(result.flags & flag_nonfinite), "token distribution requires finite logits");
    require(!(result.flags & flag_bad_draft), "draft token must have positive draft probability");
    if (host_fallback) *host_fallback = (result.flags & flag_host) != 0U;
    if (result.flags & flag_host) return decide_token(make_token_distribution(logits, {}, policy), draft, u_plain, u_accept, u_residual);
    TokenDecision out;
    out.token = result.token;
    out.drafted = (result.flags & flag_drafted) != 0U;
    out.accepted = (result.flags & flag_accepted) != 0U;
    out.residual = (result.flags & flag_residual) != 0U;
    return out;
}

TokenDraft hip_draft_for_testing(std::span<const float> logits, std::span<const std::uint32_t> ids,
    const TokenSamplingPolicy& policy, double u) {
    validate_token_sampling_policy(policy);
    const auto n = static_cast<std::uint32_t>(logits.size());
    HipTokenDraftSamplerDevice::Impl impl(n, 1U);
    float* device_logits = nullptr;
    std::uint32_t* slot = nullptr;
    check(hipMalloc(reinterpret_cast<void**>(&device_logits), 4ULL * n), "testing draft logits");
    check(hipMalloc(reinterpret_cast<void**>(&slot), 4ULL), "testing draft slot");
    struct Free { float* l; std::uint32_t* s; ~Free() { (void)hipFree(l); (void)hipFree(s); } } free_all{device_logits, slot};
    check(hipMemcpy(device_logits, logits.data(), 4ULL * n, hipMemcpyHostToDevice), "testing draft logits upload");
    impl.set_vocabulary(ids, n, nullptr);
    impl.positions[0] = 0U;
    impl.step(policy, 0U, device_logits, u, slot, nullptr);
    check(hipStreamSynchronize(nullptr), "testing draft step");
    auto d = impl.draft(0U);
    std::uint32_t written = 0;
    check(hipMemcpy(&written, slot, 4ULL, hipMemcpyDeviceToHost), "testing draft slot readback");
    require(written == d.token, "device draft slot differs from the reported token");
    return host_token_draft(d, nullptr);
}

} // namespace strix
