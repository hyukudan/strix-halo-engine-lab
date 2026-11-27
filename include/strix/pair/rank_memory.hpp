#pragma once
// One exact byte admission for a rank of a two-rank layer split: a packed weight image, per-layer state, the
// activation seam slots, an optional logits buffer, the load staging chunk, scratch and a deployment reserve, all
// aligned and charged against the rank's usable device memory. Models declare the per-layer bytes; the sums,
// alignment and overflow checks live here.
#include "strix/deploy/binding.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace strix::pair {

struct RankMemory {
    std::uint64_t usable{}, weight_payload{}, weight_resident{}, kv_payload{}, kv_resident{}, carry{};
    std::uint64_t logits{}, exchange{}, load_staging{}, scratch{}, reserve{}, required{};
    [[nodiscard]] bool admitted() const noexcept { return usable > 0 && required <= usable; }
};

struct LayerState { std::uint64_t kv_bytes{}, carry_bytes{}; };

// Everything beyond the packed weights that one rank holds. seam_slots counts the activation buffers of
// rows x row_bytes (two seam slots plus the runner's in/out buffer = 3).
struct RankTerms {
    std::uint64_t alignment{}, load_chunk_bytes{}, largest_tensor{};
    std::uint64_t row_bytes{}, rows{}, seam_slots = 3, logits_bytes{}, scratch{}, reserve{};
    std::span<const LayerState> layers;
};

// Fills the state, exchange, staging, scratch, reserve and required terms of `memory`, whose usable,
// weight_payload and weight_resident are already set. Requires a power-of-two alignment.
inline void account_rank_memory(RankMemory& memory, const RankTerms& terms) {
    if (!std::has_single_bit(terms.alignment)) throw std::invalid_argument("pair rank memory: alignment must be a power of two");
    using deploy::aligned;
    using deploy::checked_add;
    const auto align = [&](std::uint64_t bytes) { return aligned(bytes, terms.alignment); };
    for (const auto& layer : terms.layers) {
        memory.kv_payload = checked_add(memory.kv_payload, layer.kv_bytes);
        memory.kv_resident = checked_add(memory.kv_resident, align(layer.kv_bytes));
        memory.carry = checked_add(memory.carry, align(layer.carry_bytes));
    }
    memory.logits = terms.logits_bytes != 0U ? align(terms.logits_bytes) : 0U;
    const auto slot = align(deploy::checked_mul(terms.row_bytes, terms.rows));
    memory.exchange = deploy::checked_mul(slot, terms.seam_slots);
    memory.load_staging = align(std::min(terms.load_chunk_bytes, terms.largest_tensor));
    memory.scratch = align(terms.scratch);
    memory.reserve = align(terms.reserve);
    for (const auto bytes : {memory.weight_resident, memory.kv_resident, memory.carry, memory.logits,
                             memory.exchange, memory.load_staging, memory.scratch, memory.reserve})
        memory.required = checked_add(memory.required, bytes);
}

} // namespace strix::pair
