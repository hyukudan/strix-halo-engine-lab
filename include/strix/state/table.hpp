#pragma once

#include "strix/state/decl.hpp"

#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace strix::state {

// For paged states base addresses a directory of 64-bit page bases. For a
// fixed state it addresses the currently published image. Otherwise it is
// the base of the slot's row storage. Frontier, pending and capacity are in
// token positions. The table is state-major, then slot-major.
struct SlotEntry {
    std::uint64_t base;
    std::uint32_t frontier, pending, capacity, generation;
};
struct StateTable {
    SlotEntry* entries;
    std::uint32_t states, slots;
};
static_assert(sizeof(SlotEntry) == 24);
static_assert(std::is_trivial_v<SlotEntry> && std::is_standard_layout_v<SlotEntry>);
static_assert(std::is_trivial_v<StateTable> && std::is_standard_layout_v<StateTable>);

inline constexpr std::uint32_t page_rows = 64;
inline constexpr std::uint32_t no_instance = std::numeric_limits<std::uint32_t>::max();

struct InstanceLayout {
    std::uint32_t declaration, layer, visit, pass;
    std::uint32_t storage;            // this instance, or the producer instance of an alias
    std::uint64_t pool_offset, slot_bytes;
    std::uint64_t master_bytes, shadow_bytes, row_stride;
    std::uint64_t carry_offset, input_offset;
    std::uint64_t page_index;         // first directory entry of this instance
    std::uint64_t rows;               // physical rows per slot, excluding carry and replay inputs
    std::uint32_t pages, ring, window, row_divisor, carry_rows;
    std::uint32_t replay_input_bytes, checkpoint_interval;
    Rollback rollback;
    Owner owner;
    bool position_range;
    [[nodiscard]] std::uint64_t payload_bytes() const noexcept { return master_bytes + shadow_bytes; }
    [[nodiscard]] std::uint64_t slot_offset(std::uint32_t slot) const noexcept {
        return pool_offset + std::uint64_t{slot} * slot_bytes;
    }
};

struct Layout {
    std::uint64_t identity{}, prefix_block = 1;
    std::uint64_t pool_bytes{}, page_entries{}, table_entries{};
    std::uint32_t context{}, slots{}, max_pending{}, pool_alignment = 1;
    std::vector<InstanceLayout> instances;
};

// Bind a generated layout to backend addresses (or zero-origin offsets for
// the CPU body). Both spans are host-accessible staging for the eventual table.
[[nodiscard]] StateTable bind_table(const Layout&, std::span<SlotEntry>, std::span<std::uint64_t> pages,
                                    std::uint64_t pool_base, std::uint64_t page_directory_base);

}  // namespace strix::state
