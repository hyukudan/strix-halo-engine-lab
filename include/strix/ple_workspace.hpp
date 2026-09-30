#pragma once

#include "strix/model_contract.hpp"

#include <cstdint>

namespace strix {

struct PleWorkspaceLayout {
    std::uint64_t alignment = 256;
    std::uint64_t gathered_offset = 0;
    std::uint64_t gathered_bytes = 0;
    std::uint64_t key_offset = 0;
    std::uint64_t key_bytes = 0;
    std::uint64_t value_offset = 0;
    std::uint64_t value_bytes = 0;
    std::uint64_t gated_offset = 0;
    std::uint64_t gated_bytes = 0;
    std::uint64_t normalized_offset = 0;
    std::uint64_t normalized_bytes = 0;
    std::uint64_t gate_offset = 0;
    std::uint64_t gate_bytes = 0;
    std::uint64_t total_bytes = 0;
};

[[nodiscard]] PleWorkspaceLayout build_ple_workspace_layout(
    const ModelContract& model,
    std::uint64_t alignment = 256);

}  // namespace strix
