#pragma once
// Engine-bound shared runner; the adaptive arithmetic remains HIP-free in glm_spec_host.
#include "glm/spec/glm_spec_host.hpp"
#include "strix/glm/engine.hpp"
#include <array>

namespace strix::glm::spec {
inline constexpr std::uint32_t kEligibilityMerge = 0x53504543U;
inline constexpr std::uint32_t kConfigurationMismatchMerge = 0x53504544U;
enum class BootstrapPolicy { Reference, Chunked };
struct Seed {
    std::uint32_t y0 = 0, draft = 0;
    float confidence = 0;
    const float* hp = nullptr;
    bool flagged = false;
    std::uint64_t p = 0;
};
struct DecodePolicy {
    unsigned kmax = 3;
    bool adaptive = true;
    unsigned fixed_k = 3;
    std::uint64_t capacity = 0;
};
struct CycleRecord {
    std::uint64_t p = 0;
    unsigned k = 0, replay = 0, accepted = 0, emitted = 0, rows = 0;
    double extend = 0, verify = 0, commit = 0, catchup = 0, other = 0;
    std::uint32_t table_version = 0, checksum = 0;
};
struct TimingSample { unsigned kind = 0, r = 0, replay = 0; double elapsed = 0; };
struct ControlState {
    unsigned rank = 0;
    std::uint32_t cycle = 0, step = 0;
    CostTable publisher = seeded_cost_table(), merged;
    bool table_valid = false;
    Calibration calibration;
    std::function<void(const CycleRecord&)> on_cycle;
    std::function<void(const TimingSample&)> on_sample;
    engine::Exchange::MergeCtl next(unsigned kind);
    void consume(const engine::Exchange::MergeCtl&);
    void observe(unsigned index, double ms);
};
struct DepthRecord { unsigned offered = 0, reached = 0, accepted = 0; };
struct RunResult {
    std::vector<std::uint32_t> tokens;
    std::array<DepthRecord, 8> depths{};
    std::array<unsigned, 8> histogram{};
    unsigned cycles = 0, offered = 0, accepted = 0;
    double loop_ms = 0, tail_ms = 0;
};
Seed bootstrap(engine::Engine&, std::uint32_t slot, std::span<const std::uint32_t> prompt,
               BootstrapPolicy, ControlState&);
// Budget includes seed.y0. Always flushes deferred committed rows before returning.
RunResult decode_greedy(engine::Engine&, std::uint32_t slot, Seed, const DecodePolicy&, unsigned budget, ControlState&);
}  // namespace strix::glm::spec
