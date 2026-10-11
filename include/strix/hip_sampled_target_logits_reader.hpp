#pragma once

#include "strix/compiled_sampled_speculation_backend.hpp"

namespace strix {

class HipTargetLogitsReader final
    : public TargetLogitsReader {
public:
    [[nodiscard]] std::vector<float> read_fp32_logits(
        const ExecutionBufferAddress& buffer,
        std::uint32_t rows,
        std::uint64_t vocab_size) override;
};

}  // namespace strix
