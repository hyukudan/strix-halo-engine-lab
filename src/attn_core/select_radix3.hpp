#pragma once
// Exact ordered block top-k, "Radix3" schedule of the shared select family (select.hpp has Chunked / Segmented /
// FourPerThread, select_threshold.hpp the Threshold schedule of the same rule). Rule, every schedule: the k largest order
// keys, equal keys by the smaller position, emitted in ascending position. This schedule: one workgroup per row, a
// three-pass radix (11, 11, 10 key bits) whose per-wave votes use one atomic per run of lanes sharing a digit, a two-digits
// per lane bin walk over the 2048 bins with a workgroup scan, then one ordered compaction (greater | equal packed scan).
// Elements are score blocks of `ratio` consecutive tokens: a selected block expands to its tokens, the row's partial-block
// tail follows, and rows with fewer than k finite blocks put block 0 first (zero fill). Entries > -inf only count: NaN and
// -inf never win; +0 and -0 are one key. Integer only: the schedule never changes the selected positions.
// Lifted unchanged from the Qwen indexed-attention selector (host statement: select_radix3_rule.hpp).
//
// Template parameters:
//   Threads  block size, 1024 (the bin walk gives each thread two of the 2048 bins; waves = Threads / 32 = 32).
//   Control  per-launch control of a graph-captured launch; members (called by every thread, uniformly):
//              bool graph() const;  bool visible(std::uint32_t& tokens) const  (false = this launch does nothing);
//              std::uint32_t block_topk() const  (k when graph() is true).
//            NoControl is the eager form (graph() == false).
#include <hip/hip_runtime.h>
#include <cstdint>

namespace strix::attn_core::select::radix3 {

inline constexpr unsigned kBins = 2048;

// Monotone float -> uint32 key: larger float, larger key, and +0 == -0 (NaN by its bits; callers exclude it by value > -inf).
__device__ __forceinline__ std::uint32_t order_key(float value) {
    auto bits = __float_as_uint(value);
    if ((bits & 0x7fffffffU) == 0) bits = 0;
    return bits & 0x80000000U ? ~bits : bits | 0x80000000U;
}

struct NoControl {
    __device__ bool graph() const { return false; }
    __device__ bool visible(std::uint32_t&) const { return true; }
    __device__ std::uint32_t block_topk() const { return 0; }
};

// Workgroup LDS of the radix passes and scans: the calling kernel declares the __shared__ objects (so each kernel keeps its
// own LDS layout) and passes pointers to them: histogram[kBins], waves[Threads / 32] and three words of scratch.
template <unsigned Threads>
struct Lds {
    std::uint32_t* histogram;
    std::uint32_t* waves;
    std::uint32_t* found_digit;
    std::uint32_t* found_need;
    std::uint32_t* found_count;
};

// Inclusive workgroup prefix sum; total is the workgroup sum.
template <unsigned Threads>
__device__ __forceinline__ std::uint32_t scan(std::uint32_t value, std::uint32_t* waves, std::uint32_t& total) {
    const auto lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    for (unsigned d = 1; d < 32U; d <<= 1U) { const auto up = __shfl_up(value, d, 32); if (lane >= d) value += up; }
    if (lane == 31U) waves[wave] = value;
    __syncthreads();
    if (wave == 0) {
        auto sum = waves[lane];
        for (unsigned d = 1; d < 32U; d <<= 1U) { const auto up = __shfl_up(sum, d, 32); if (lane >= d) sum += up; }
        waves[lane] = sum;
    }
    __syncthreads();
    total = waves[Threads / 32U - 1U];
    value += wave ? waves[wave - 1U] : 0U;
    __syncthreads();
    return value;
}

// The radix passes over n values value(i): (prefix, mask) of the need-th largest key, need = how many of the keys equal to
// prefix (under mask) remain to take. Returns false when fewer than need values are > -inf (prefix = mask = 0 then, every
// counted value is taken). With zeros != nullptr the early end also sets *zeros = the missing values and need = the counted
// number (the zero fill of select_blocks).
template <unsigned Threads, class Value>
__device__ __forceinline__ bool radix_prefix(const Lds<Threads>& s, Value value, std::uint32_t n, std::uint32_t& prefix,
                                             std::uint32_t& mask, std::uint32_t& need, std::uint32_t* zeros = nullptr) {
    const auto tid = static_cast<std::uint32_t>(threadIdx.x);
    prefix = 0; mask = 0;
    for (unsigned pass = 0; pass < 3U; ++pass) {
        const unsigned shift = pass == 0 ? 21U : pass == 1 ? 10U : 0U;
        const auto digits = pass == 2 ? 1024U : 2048U;
        for (auto i = tid; i < kBins; i += Threads) s.histogram[i] = 0;
        __syncthreads();
        for (std::uint32_t base = 0; base < n; base += Threads) {
            const auto i = base + tid;
            const float v = i < n ? value(i) : -INFINITY;
            const auto key = order_key(v);
            const bool counted_here = v > -INFINITY && (key & mask) == prefix;
            const auto digit = (key >> shift) & (digits - 1U);
            // One atomic for the lanes sharing the first counted lane's digit.
            const auto votes = __ballot(counted_here);
            if (votes) {
                const auto leader = static_cast<unsigned>(__builtin_ctzll(votes));
                const auto lead = __shfl(digit, static_cast<int>(leader), 32);
                const auto same = __ballot(counted_here && digit == lead);
                if (counted_here && digit != lead) atomicAdd(&s.histogram[digit], 1U);
                else if (tid % 32U == leader) atomicAdd(&s.histogram[lead], static_cast<std::uint32_t>(__popcll(same)));
            }
        }
        __syncthreads();
        const auto high = 2U * tid + 1U < digits ? s.histogram[2U * tid + 1U] : 0U;
        const auto low = 2U * tid < digits ? s.histogram[2U * tid] : 0U;
        std::uint32_t counted_total = 0;
        const auto above = [&] { const auto inclusive = scan<Threads>(high + low, s.waves, counted_total); return counted_total - inclusive; }();
        if (pass == 0 && counted_total <= need) {
            const bool full = counted_total == need;
            if (zeros) { *zeros = need - counted_total; need = counted_total; }
            return full;
        }
        if (above < need && need <= above + high) { *s.found_digit = 2U * tid + 1U; *s.found_need = need - above; *s.found_count = high; }
        else if (above + high < need && need <= above + high + low) { *s.found_digit = 2U * tid; *s.found_need = need - above - high; *s.found_count = low; }
        __syncthreads();
        prefix |= *s.found_digit << shift; mask |= (digits - 1U) << shift; need = *s.found_need;
        if (*s.found_count == need) break;
    }
    return true;
}

// Workgroup-parallel selection of the select_blocks first blocks of block_scores[0 .. num_blocks) (every thread calls it).
// selected receives the token ids: k blocks in ascending block order, each expanded to `compress_ratio` tokens, then the
// partial-block tail up to visible_tokens. Selected scores (and block 0 on zero fill) become -inf unless preserve_scores.
template <unsigned Threads, class Control = NoControl>
__device__ void select_blocks(float* block_scores, std::uint32_t num_blocks, std::uint32_t select_blocks_k,
                              std::uint32_t compress_ratio, std::uint32_t visible_tokens, std::uint32_t* selected,
                              Control control, bool preserve_scores = false) {
    const auto tid = static_cast<std::uint32_t>(threadIdx.x);
    if (control.graph()) {
        if (!control.visible(visible_tokens)) return;
        num_blocks = visible_tokens / compress_ratio;
        select_blocks_k = control.block_topk();
    }
    // Full-visible shortcut for both forms (host rule: select_tokens): at most k blocks keep every visible token in order;
    // scores are not touched. The zero-fill path below only serves rows with more than k blocks and fewer than k finite ones.
    if (num_blocks <= select_blocks_k) {
        for (auto token = tid; token < visible_tokens; token += Threads) selected[token] = token;
        return;
    }
    // k == 0: nothing is selected, the output is the partial-block tail alone. The radix passes need need >= 1 (they would
    // leave the found-digit words unwritten); scores are not touched. The branch is uniform over the workgroup.
    if (select_blocks_k == 0) {
        for (auto token = num_blocks * compress_ratio + tid; token < visible_tokens; token += Threads)
            selected[token - num_blocks * compress_ratio] = token;
        return;
    }
    __shared__ std::uint32_t histogram[kBins];
    __shared__ std::uint32_t waves[Threads / 32U];
    __shared__ std::uint32_t found_digit, found_need, found_count;
    const Lds<Threads> s{histogram, waves, &found_digit, &found_need, &found_count};
    std::uint32_t prefix = 0, mask = 0, need = select_blocks_k, zeros = 0;
    radix_prefix<Threads>(s, [&](std::uint32_t i) { return block_scores[i]; }, num_blocks, prefix, mask, need, &zeros);
    for (auto token = tid; token < zeros * compress_ratio; token += Threads) selected[token] = token % compress_ratio;
    std::uint32_t greater_before = 0, equal_before = 0;
    for (std::uint32_t base = 0; base < num_blocks; base += 4U * Threads) {
        const auto first = base + 4U * tid;
        unsigned kind[4], greater = 0, equal = 0;
        for (unsigned j = 0; j < 4U; ++j) {
            const auto block = first + j;
            const float value = block < num_blocks ? block_scores[block] : -INFINITY;
            const auto masked = order_key(value) & mask;
            kind[j] = !(value > -INFINITY) ? 0U : masked > prefix ? 1U : masked == prefix ? 2U : 0U;
            greater += kind[j] == 1U; equal += kind[j] == 2U;
        }
        std::uint32_t tile = 0;
        const auto packed = scan<Threads>((greater << 16U) | equal, s.waves, tile);
        auto g = greater_before + (packed >> 16U) - greater, e = equal_before + (packed & 0xffffU) - equal;
        for (unsigned j = 0; j < 4U; ++j) {
            if (kind[j] == 1U || (kind[j] == 2U && e < need)) {
                const auto block = first + j, slot = zeros + g + min(e, need);
                for (std::uint32_t offset = 0; offset < compress_ratio; ++offset)
                    selected[slot * compress_ratio + offset] = block * compress_ratio + offset;
                if (!preserve_scores) block_scores[block] = -INFINITY;
            }
            g += kind[j] == 1U; e += kind[j] == 2U;
        }
        greater_before += tile >> 16U; equal_before += tile & 0xffffU;
    }
    if (zeros && tid == 0 && !preserve_scores) block_scores[0] = -INFINITY;
    const auto complete_tokens = num_blocks * compress_ratio, selected_complete = select_blocks_k * compress_ratio;
    for (auto token = complete_tokens + tid; token < visible_tokens; token += Threads)
        selected[selected_complete + token - complete_tokens] = token;
}

}  // namespace strix::attn_core::select::radix3
