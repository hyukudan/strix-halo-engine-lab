#pragma once

#include "strix/ernie/engine.hpp"
#include "strix/attn/reference_executor.hpp"
#include "strix/attn/reference_history.hpp"

#include <cstdint>
#include <string>

namespace strix::ernie {

using AttentionProject = ReferenceProject;
using AttentionHeadRange = attn::ReferenceHeadRange;
using AttentionHeadExecutor = attn::ReferenceHeadExecutor;

// One layer of one sequence. Copies own their complete causal history.
class AttentionHistory {
public:
    AttentionHistory() = default;
    explicit AttentionHistory(std::size_t max_rows);
    [[nodiscard]] unsigned position() const noexcept { return position_; }

private:
    friend std::vector<float> attention_reference(unsigned, const ReferenceShape&, const ReferenceRead&,
        std::span<const float>, AttentionHistory&, const AttentionProject&, const AttentionHeadExecutor&, bool);
    friend std::vector<float> attention_reference_first_hca(const std::string&, const ReferenceShape&,
        const ReferenceRead&, std::span<const float>, const AttentionProject&);

    attn::CompressedReferenceHistory hca_;
    attn::LatentReferenceHistory mla_;
    std::vector<float> folded_mix_;
    unsigned layer_ = kLayerCount;
    unsigned position_{};
    std::size_t max_rows_ = data::dims::max_context;
};

// Scalar target attention. Appends one row on success. The attention ordering,
// precision and index gate are candidate checkpoint semantics pending a trace.
[[nodiscard]] std::vector<float> attention_reference(unsigned layer, const ReferenceShape& shape,
    const ReferenceRead& read, std::span<const float> input, AttentionHistory& history,
    const AttentionProject& project = {}, const AttentionHeadExecutor& execute_heads = {}, bool index_prefill = true);

// Position-zero diagnostic binding, with no compressor reads or cache writes.
[[nodiscard]] std::vector<float> attention_reference_first_hca(const std::string& prefix,
    const ReferenceShape& shape, const ReferenceRead& read, std::span<const float> input,
    const AttentionProject& project = {});

} // namespace strix::ernie
