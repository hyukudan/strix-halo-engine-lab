#pragma once

#include <cstdint>

namespace strix::minimax::engine {
// One decode row: a sequence slot, the token at `pos`, and its position.
struct Row {
    std::uint32_t slot = 0, token = 0;
    std::uint64_t pos = 0;
};

struct PfProgress {
    std::uint64_t id = 0, parent = 0;
    std::uint64_t generation = 0, c0 = 0, chunk = 0, rows = 0;
    std::uint32_t slot = 0, bank = 0, layer = 0, next = 0;
    bool open = false;
    bool operator==(const PfProgress&) const = default;
};
}  // namespace strix::minimax::engine
