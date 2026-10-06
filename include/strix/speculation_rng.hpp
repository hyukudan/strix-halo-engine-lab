#pragma once

#include <cstdint>

namespace strix {

enum class SpeculationRandomDrawClass : std::uint8_t {
    serial_target = 0,
    draft_acceptance = 1,
    residual_fallback = 2,
};

struct SpeculationRngIdentity {
    std::uint64_t seed = 0;
    std::uint64_t request_identity = 0;
    std::uint64_t sampling_policy_identity = 0;
};

[[nodiscard]] std::uint64_t speculation_rng_identity(
    const SpeculationRngIdentity& identity);

[[nodiscard]] std::uint64_t speculation_random_u64(
    const SpeculationRngIdentity& identity,
    std::uint64_t absolute_position,
    SpeculationRandomDrawClass draw_class,
    std::uint32_t draw_index = 0);

[[nodiscard]] double speculation_uniform01(
    const SpeculationRngIdentity& identity,
    std::uint64_t absolute_position,
    SpeculationRandomDrawClass draw_class,
    std::uint32_t draw_index = 0);

}  // namespace strix
