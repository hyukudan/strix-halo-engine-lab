#pragma once

// Host oracle of the hyper-connection contract (hc_contract.hpp): the reference every device body is compared
// against. Streams and outputs are FP32 values; with Store::bf16 they carry bfloat16 values (rounded by the oracle).

#include "strix/hc/hc_contract.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strix::hc {

enum class FnLayout { mix_major, feature_major };

struct SiteWeights {
    std::span<const std::uint16_t> fn_f16;  // FnType::f16: [mix_rows][flat]
    std::span<const float> fn_f32;          // FnType::f32: [mix_rows][flat]
    std::span<const float> base;            // [mix_rows]
    std::span<const float> scale;           // [scale_count]
    FnLayout layout = FnLayout::mix_major;  // feature_major F32: [flat][mix_rows]
};

struct Mix {
    std::vector<float> pre, post;  // [streams], post empty on a head site
    std::vector<float> comb;       // [streams][streams] (row = source copy), empty unless Comb::sinkhorn
};

// mix[m] = dot(fn[m], x) * r. The spec places norm_eps inside or outside the RMS square root.
[[nodiscard]] std::vector<float> raw_mixes(const Spec& spec, std::span<const float> streams, const SiteWeights& w);
// pre / post / comb from the raw mixes (the split, then the Sinkhorn stage when Comb::sinkhorn).
[[nodiscard]] Mix split(const Spec& spec, std::span<const float> raw, std::span<const float> base,
                        std::span<const float> scale);
[[nodiscard]] Mix mixes(const Spec& spec, std::span<const float> streams, const SiteWeights& w);
// out[i] = collapse_at(...) for i < hidden.
void collapse(const Spec& spec, std::span<const float> streams, std::span<const float> pre, std::span<float> out);
// Compute RMS variance on the selected side of the collapse store. The
// numerator sees the store in both cases and the normalized output is BF16.
void collapse_rms_bf16(const Spec& spec, std::span<const float> streams, std::span<const float> pre,
                       std::span<const float> gain, float epsilon, CollapseVariance variance,
                       std::span<float> out);
// out[k][i] = expand_at(...) for k < streams, i < hidden (sublayer sites only).
void expand(const Spec& spec, std::span<const float> sub, std::span<const float> residual, const Mix& mix,
            std::span<float> out);

// Stream-major [streams][hidden] to feature-major [hidden][streams]: the order of a mixing matrix whose columns
// are grouped by feature (FnLayout and checkpoint conventions that interleave the streams of one feature).
[[nodiscard]] inline std::vector<float> interleave_features(const Spec& spec, std::span<const float> streams) {
    std::vector<float> ordered(streams.size());
    for (std::size_t d = 0; d < spec.hidden; ++d)
        for (std::size_t c = 0; c < spec.streams; ++c) ordered[d * spec.streams + c] = streams[c * spec.hidden + d];
    return ordered;
}

} // namespace strix::hc
