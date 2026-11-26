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

// mix[m] = dot(fn[m], x) * r, r = 1 / declared_sqrt(dot(x, x) / n + norm_eps). Throws on bad geometry.
[[nodiscard]] std::vector<float> raw_mixes(const Spec& spec, std::span<const float> streams, const SiteWeights& w);
// pre / post / comb from the raw mixes (the split, then the Sinkhorn stage when Comb::sinkhorn).
[[nodiscard]] Mix split(const Spec& spec, std::span<const float> raw, std::span<const float> base,
                        std::span<const float> scale);
[[nodiscard]] Mix mixes(const Spec& spec, std::span<const float> streams, const SiteWeights& w);
// out[i] = collapse_at(...) for i < hidden.
void collapse(const Spec& spec, std::span<const float> streams, std::span<const float> pre, std::span<float> out);
// out[k][i] = expand_at(...) for k < streams, i < hidden (sublayer sites only).
void expand(const Spec& spec, std::span<const float> sub, std::span<const float> residual, const Mix& mix,
            std::span<float> out);

} // namespace strix::hc
