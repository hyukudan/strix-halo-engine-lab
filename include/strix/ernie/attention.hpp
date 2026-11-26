#pragma once

#include "strix/ernie/engine.hpp"

#include <cstdint>
#include <string>

namespace strix::ernie {

using AttentionProject = std::function<void(const std::string&, std::span<const float>, std::span<float>)>;

// One layer of one sequence. Copies own their complete causal history.
class AttentionHistory {
public:
    [[nodiscard]] unsigned position() const noexcept { return position_; }

private:
    friend std::vector<float> attention_reference(unsigned, const ReferenceShape&, const ReferenceRead&,
        std::span<const float>, AttentionHistory&, const AttentionProject&);
    friend std::vector<float> attention_reference_first_hca(const std::string&, const ReferenceShape&,
        const ReferenceRead&, std::span<const float>, const AttentionProject&);

    struct {
        std::vector<float> window, compressed, carry_values, carry_logits;
    } hca_;
    struct {
        std::vector<float> latents, rotary_keys, inverse_key_rms;
        std::vector<std::uint8_t> index_keys;
    } mla_;
    unsigned layer_ = kLayerCount;
    unsigned position_{};
};

// Scalar target attention. Appends one row on success. The attention ordering,
// precision and index gate are candidate checkpoint semantics pending a trace.
[[nodiscard]] std::vector<float> attention_reference(unsigned layer, const ReferenceShape& shape,
    const ReferenceRead& read, std::span<const float> input, AttentionHistory& history,
    const AttentionProject& project = {});

// Position-zero diagnostic binding, with no compressor reads or cache writes.
[[nodiscard]] std::vector<float> attention_reference_first_hca(const std::string& prefix,
    const ReferenceShape& shape, const ReferenceRead& read, std::span<const float> input,
    const AttentionProject& project = {});

} // namespace strix::ernie
