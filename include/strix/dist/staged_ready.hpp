#pragma once
#include <cstdint>

namespace strix::dist {
// Dedicated coherent cache line, never overlaid with a payload or its trailer.
// CPU prepares ready/done before exposing a view; the last GPU block resets its
// cacheable counter before publishing done (no memset/event per exchange).
struct alignas(64) StagedReady {
    std::uint64_t done = 0;
    std::uint64_t ready = 0;
    unsigned* block_counter = nullptr; // GPU-cacheable scratch; reset by the last block, never by CPU in the hot path
};
static_assert(sizeof(StagedReady) == 64);
struct ProducerStage {
    void* data = nullptr;
    StagedReady* flag = nullptr; // device alias, not a CPU pointer to be dereferenced by callers
    std::uint64_t token = 0;
    unsigned* counter = nullptr; // == flag->block_counter (device memory), for the lean helpers below
};
struct ConsumerStage {
    const void* data = nullptr;
    StagedReady* flag = nullptr;
    std::uint64_t token = 0;
    unsigned* counter = nullptr;
};

}
