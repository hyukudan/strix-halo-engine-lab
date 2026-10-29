#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace strix::ds41::dspark {
// Physical FP8 rings, including inactive slots; start is the skip_to lower bound.
struct ControllerSnapshot {
    struct Costs {
        std::array<float, 7> verify{};
        std::array<std::uint32_t, 7> samples{};
        float draft = 0.0F, extra = 0.0F;
    } costs;
    std::array<std::array<std::uint32_t, 8>, 5> seen{}, hits{};
    std::uint64_t decisions = 0;
    std::uint32_t sweep = 0;
    bool explore = false;
    void validate() const {
        if (sweep > 5) throw std::invalid_argument("DSpark controller sweep");
        for (unsigned d = 0; d < 5; ++d) for (unsigned b = 0; b < 8; ++b)
            if (hits[d][b] > seen[d][b]) throw std::invalid_argument("DSpark controller counts");
    }
};
struct ContextSnapshot {
    std::uint64_t start = 0, next = 0;
    std::uint32_t core = 129280;
    std::vector<std::uint32_t> extra;
    std::array<std::vector<std::uint8_t>, 3> rings;
    void validate() const {
        if (start > next || next > UINT32_MAX || core > 129280 || (core == 0 && extra.empty()))
            throw std::invalid_argument("DSpark context cursor/vocabulary");
        std::vector<bool> seen(129280);
        for (auto id : extra) {
            if (id < core || id >= 129280 || seen[id]) throw std::invalid_argument("DSpark extra vocabulary");
            seen[id] = true;
        }
        for (const auto& r : rings)
            if (r.size() != 128U * 528U) throw std::invalid_argument("DSpark ring geometry");
    }
};
} // namespace strix::ds41::dspark
