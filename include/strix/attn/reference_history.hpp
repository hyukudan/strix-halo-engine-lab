#pragma once

#include <cstdint>
#include <vector>

namespace strix::attn {

struct CompressedReferenceHistory {
    std::vector<float> window, window_values, compressed, carry_values, carry_logits;
};

struct LatentReferenceHistory {
    std::vector<float> latents, rotary_keys, inverse_key_rms;
    std::vector<std::uint8_t> index_keys;
};

} // namespace strix::attn
