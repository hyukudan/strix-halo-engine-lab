#pragma once

// Tail policies of the microscaled R1 down-combine kernel (B8 "combine in the epilogue", MoE tail fold).
//
// The R1 down kernel already applies the route weights and adds the experts in ascending order. A tail continues from
// the stored combined value of one hidden channel inside the CTA that owns the channel, so a per-channel consumer of the
// MoE output needs no node of its own. Each tail runs the FP32 operations of the nodes it replaces, in their order,
// on the same operands, through the same shared definitions:
//   NoTail            nothing (the incumbent kernel).
//   ResidualTail      sum = [round] (residual + moe)                           == core::residual_mix add node.
//   ConvResidualTail  conv = core::causal_conv per-channel arithmetic (taps, epilogue, round_output, decode history
//                     commit), optional sum = [round] (residual + conv)        == causal_conv decode node + add node.
// The sum rounding is the residual add node's (residual_add below, the group_indexer BF16 round that keeps the NaN
// payload), the convolution rounding the convolution node's (causal_conv::round_output).
// The kernel stages a tail's operands (thread 0, after its own expert dot, before the CTA barrier) and finishes each
// live channel after the incumbent store, so the stored MoE output itself is unchanged.
//
// Legality is part of the plan (admits / footprint):
//   - a convolution reads the previous row of the same channel, which another CTA owns in a rows launch, so
//     ConvResidualTail admits one routed row only; ResidualTail admits every rows count;
//   - every addressed range fits the address space (no index or pointer wrap);
//   - a CTA stages its inputs before any CTA finishes, so no written range may overlap a range the tail reads (a
//     separate node would read the value the earlier node stored), and written ranges are pairwise disjoint (a
//     partial overlap is a race between CTAs). The one exception is the exact in-place sum (sum == residual, same
//     shape): each element is read and written by the thread that owns its channel.

#include "core/short_convolution.hpp"
#include "strix/attn/group_indexer.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_R1_TAIL_HD __host__ __device__ inline
#else
#define STRIX_R1_TAIL_HD inline
#endif

namespace strix::moe_core::r1_tail {

namespace fp = core::causal_conv::fp;

// The residual add node: FP32 add, optional BF16 round of core::residual_mix (AddRound / pointwise::Add).
STRIX_R1_TAIL_HD float residual_add(float residual, float value, bool round) {
    const float sum = fp::add(residual, value);
    return round ? attn::group_indexer::bf16_round(sum) : sum;
}

// Byte range [begin, end) of a buffer; empty for a null buffer or no elements. valid = false when the range wraps.
struct Range {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
    bool valid = true;
};

// `elements` elements of `element_bytes` from p; elements == SIZE_MAX marks an overflowed count.
inline Range range_of(const void* p, std::size_t elements, std::size_t element_bytes) {
    if (p == nullptr || elements == 0U) return {};
    const auto begin = reinterpret_cast<std::uintptr_t>(p);
    if (elements > std::numeric_limits<std::uintptr_t>::max() / element_bytes) return {0, 0, false};
    const auto bytes = static_cast<std::uintptr_t>(elements * element_bytes);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) return {0, 0, false};
    return {begin, begin + bytes, true};
}

inline bool overlap(const Range& a, const Range& b) {
    return a.begin != a.end && b.begin != b.end && a.begin < b.end && b.begin < a.end;
}

// Element count of `rows` rows of `width` elements at `stride`; SIZE_MAX when it overflows.
inline std::size_t strided_elements(unsigned rows, std::size_t stride, std::size_t width) {
    if (rows == 0U || width == 0U) return 0U;
    constexpr auto max = std::numeric_limits<std::size_t>::max();
    const std::size_t steps = rows - 1U;
    if (steps != 0U && stride > (max - width) / steps) return max;
    return steps * stride + width;
}

// What a tail reads and writes in one launch (at most three of each; unused entries stay empty). in_place marks the
// exact sum == residual alias, the only permitted read/write overlap.
struct Footprint {
    Range reads[3];
    Range writes[3];
    const void* in_place_read = nullptr;
    const void* in_place_write = nullptr;
};

inline bool valid(const Footprint& f) {
    for (const auto& r : f.reads) if (!r.valid) return false;
    for (const auto& w : f.writes) if (!w.valid) return false;
    return true;
}

// Written ranges pairwise disjoint and disjoint from read ranges (except the exact in-place alias).
inline bool disjoint(const Footprint& f) {
    constexpr unsigned n = 3;
    for (unsigned i = 0; i < n; ++i) {
        for (unsigned j = i + 1U; j < n; ++j)
            if (overlap(f.writes[i], f.writes[j])) return false;
        for (unsigned j = 0; j < n; ++j) {
            if (!overlap(f.writes[i], f.reads[j])) continue;
            const bool in_place = f.in_place_write != nullptr && f.writes[i].begin == f.reads[j].begin &&
                                  f.writes[i].end == f.reads[j].end &&
                                  reinterpret_cast<std::uintptr_t>(f.in_place_write) == f.writes[i].begin &&
                                  reinterpret_cast<std::uintptr_t>(f.in_place_read) == f.reads[j].begin;
            if (!in_place) return false;
        }
    }
    return true;
}

struct NoTail {
    template <unsigned Columns>
    struct Staged {};
    static constexpr bool enabled = false;
    template <unsigned Columns>
    STRIX_R1_TAIL_HD Staged<Columns> stage(unsigned, unsigned, unsigned) const { return {}; }
    template <unsigned Columns>
    STRIX_R1_TAIL_HD void finish(const Staged<Columns>&, unsigned, unsigned, unsigned, float) const {}
};

// residual and sum are [row][stride] FP32 rows; sum may alias residual exactly (in-place residual stream).
struct ResidualTail {
    const float* residual;
    float* sum;
    std::size_t stride;
    bool round;

    template <unsigned Columns>
    struct Staged {
        float residual[Columns];
    };
    static constexpr bool enabled = true;
    template <unsigned Columns>
    STRIX_R1_TAIL_HD Staged<Columns> stage(unsigned row, unsigned first, unsigned live) const {
        Staged<Columns> staged{};
        for (unsigned i = 0; i < Columns && i < live; ++i)
            staged.residual[i] = residual[std::size_t{row} * stride + first + i];
        return staged;
    }
    // slot = channel - first of the staged range.
    template <unsigned Columns>
    STRIX_R1_TAIL_HD void finish(const Staged<Columns>& staged, unsigned slot, unsigned row, unsigned channel,
                                 float moe) const {
        sum[std::size_t{row} * stride + channel] = residual_add(staged.residual[slot], moe, round);
    }
};

// The one input row of a decode commit when the convolved value is held, not stored: every subscript reads it.
struct Broadcast {
    float value;
    STRIX_R1_TAIL_HD float operator[](std::size_t) const { return value; }
};

// Decode-only causal short convolution of the MoE output (static weights, no dynamic groups) and an optional residual add.
// weights [lag][channel] (lag 0 = current row), history [taps - 1][channel] oldest first, exactly as
// core::causal_short_convolution. Taps is a compile-time constant equal to plan.taps, so every staged operand stays in
// registers. conv_output may be null (the convolved row is then not stored); residual null means no add (sum unused).
// The arithmetic is core::causal_conv's: weighted_taps, epilogue, round_output and the one-row commit_history (which
// re-reads the old history rows of its own channel: this thread staged them and nothing else writes them).
template <unsigned Taps>
struct ConvResidualTail {
    static_assert(Taps >= 1U);
    static constexpr unsigned kHistory = Taps - 1U;
    core::ShortConvolution plan;
    const float* weights;
    float* history;
    float* conv_output;
    const float* residual;
    float* sum;
    bool round_conv;
    bool round_sum;

    template <unsigned Columns>
    struct Staged {
        float weight[Columns][Taps];
        float history[Columns][kHistory > 0U ? kHistory : 1U];
        float residual[Columns];
    };
    static constexpr bool enabled = true;

    template <unsigned Columns>
    STRIX_R1_TAIL_HD Staged<Columns> stage(unsigned, unsigned first, unsigned live) const {
        Staged<Columns> staged{};
        const auto channels = plan.channels;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
        for (unsigned i = 0; i < Columns; ++i) {
            if (i >= live) break;
            const std::size_t channel = first + i;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
            for (unsigned lag = 0; lag < Taps; ++lag) staged.weight[i][lag] = weights[lag * channels + channel];
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
            for (unsigned h = 0; h < kHistory; ++h) staged.history[i][h] = history[h * channels + channel];
            staged.residual[i] = residual ? residual[channel] : 0.0F;
        }
        return staged;
    }

    template <unsigned Columns>
    STRIX_R1_TAIL_HD void finish(const Staged<Columns>& staged, unsigned slot, unsigned, unsigned channel,
                                 float moe) const {
        namespace conv = core::causal_conv;
        const auto channels = plan.channels;
        const auto& weight = staged.weight[slot];
        const auto& old = staged.history[slot];
        // Decode row 0: lag 0 is the current input, lag l reads history row taps - 1 - l.
        const float taps = conv::weighted_taps<true>(
            Taps, [&](unsigned lag) { return lag == 0U ? moe : old[kHistory - lag]; },
            [&](unsigned lag) { return weight[lag]; });
        const float raw = conv::epilogue(plan, taps, moe);
        const float result = round_conv ? conv::round_output(raw) : raw;
        if (conv_output) conv_output[channel] = result;
        if constexpr (kHistory > 0U)
            conv::commit_history(std::size_t{kHistory}, std::size_t{1}, channels, std::size_t{channel}, history,
                                 Broadcast{moe}, history);
        if (residual) sum[channel] = residual_add(staged.residual[slot], result, round_sum);
    }
};

// Staged operands of an optional tail (empty pack = no tail).
template <unsigned Columns, class... Tail>
struct StagedSelect {
    using type = NoTail::Staged<Columns>;
};
template <unsigned Columns, class Tail>
struct StagedSelect<Columns, Tail> {
    using type = typename Tail::template Staged<Columns>;
};
template <unsigned Columns, class... Tail>
using StagedOf = typename StagedSelect<Columns, Tail...>::type;

// Host footprints. rows = routed rows of the launch, hidden = the R1 hidden width.
inline Footprint footprint(const NoTail&, unsigned, unsigned) { return {}; }
inline Footprint footprint(const ResidualTail& t, unsigned rows, unsigned hidden) {
    const auto elements = strided_elements(rows, t.stride, hidden);
    Footprint f{};
    f.reads[0] = range_of(t.residual, elements, sizeof(float));
    f.writes[0] = range_of(t.sum, elements, sizeof(float));
    f.in_place_read = t.residual;
    f.in_place_write = t.sum;
    return f;
}
template <unsigned Taps>
inline Footprint footprint(const ConvResidualTail<Taps>& t, unsigned, unsigned) {
    const auto channels = t.plan.channels;
    constexpr auto max = std::numeric_limits<std::size_t>::max();
    const auto rows_of = [&](std::size_t count) { return count != 0U && channels > max / count ? max : count * channels; };
    // History is read (staged) and written (committed) only by the thread of its own channel, so it is listed once,
    // as a write: it must stay clear of the weights, the residual and the other stores.
    Footprint f{};
    f.reads[0] = range_of(t.weights, rows_of(Taps), sizeof(float));
    f.writes[0] = range_of(t.history, rows_of(ConvResidualTail<Taps>::kHistory), sizeof(float));
    f.writes[1] = range_of(t.conv_output, channels, sizeof(float));
    if (t.residual) {
        f.reads[1] = range_of(t.residual, channels, sizeof(float));
        f.writes[2] = range_of(t.sum, channels, sizeof(float));
        f.in_place_read = t.residual;
        f.in_place_write = t.sum;
    }
    return f;
}

// Plan legality (host), without the R1 launch (microscaled_r1::admits_tail adds the launch's own buffers).
inline bool admits(const NoTail&, unsigned, unsigned) { return true; }
inline bool admits(const ResidualTail& t, unsigned rows, unsigned hidden) {
    if (!(rows >= 1U && t.residual && t.sum && t.stride >= hidden)) return false;
    const auto f = footprint(t, rows, hidden);
    return valid(f) && disjoint(f);
}
template <unsigned Taps>
inline bool admits(const ConvResidualTail<Taps>& t, unsigned rows, unsigned hidden) {
    const auto& p = t.plan;
    if (!(rows == 1U && core::valid_short_convolution_plan(p) && p.dynamic_groups == 0 && p.channels == hidden &&
          p.taps == Taps && t.weights && (p.taps == 1U || t.history) && (t.residual == nullptr || t.sum != nullptr) &&
          (t.conv_output != nullptr || t.residual != nullptr)))
        return false;
    const auto f = footprint(t, rows, hidden);
    return valid(f) && disjoint(f);
}

}  // namespace strix::moe_core::r1_tail
