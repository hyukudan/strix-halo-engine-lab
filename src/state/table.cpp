#include "strix/state/txn.hpp"

#include "strix/session/kv_row_append.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace strix::state {
namespace {

std::size_t index(const Layout& plan, std::uint32_t state, std::uint32_t slot) {
    if (state >= plan.instances.size() || slot >= plan.slots) throw std::out_of_range("state table index");
    return static_cast<std::size_t>(state) * plan.slots + slot;
}

std::uint64_t initial_base(const InstanceLayout& piece, std::uint32_t slot,
                           std::uint64_t pool, std::uint64_t directory) {
    if (piece.pages) return directory + (piece.page_index + std::uint64_t{slot} * piece.pages) * sizeof(std::uint64_t);
    return pool + piece.slot_offset(slot);
}

}  // namespace

StateTable bind_table(const Layout& plan, std::span<SlotEntry> entries, std::span<std::uint64_t> pages,
                      std::uint64_t pool_base, std::uint64_t page_directory_base) {
    constexpr auto limit = std::numeric_limits<std::uint64_t>::max();
    if (entries.size() < plan.table_entries || pages.size() < plan.page_entries ||
        pool_base % plan.pool_alignment || page_directory_base % alignof(std::uint64_t))
        throw std::invalid_argument("incomplete or misaligned state table binding");
    if (plan.pool_bytes > limit - pool_base || plan.page_entries > (limit - page_directory_base) / sizeof(std::uint64_t))
        throw std::overflow_error("state table address overflow");

    const auto states = static_cast<std::uint32_t>(plan.instances.size());
    for (std::uint32_t state = 0; state < states; ++state) {
        const auto& piece = plan.instances[state];
        const auto& storage = plan.instances[piece.storage];
        for (std::uint32_t slot = 0; slot < plan.slots; ++slot) {
            entries[index(plan, state, slot)] = {initial_base(storage, slot, pool_base, page_directory_base),
                                                  0, 0, plan.context, 0};
            if (piece.storage != state) continue;
            const auto first = piece.page_index + std::uint64_t{slot} * piece.pages;
            const auto base = pool_base + piece.slot_offset(slot);
            for (std::uint32_t p = 0; p < piece.pages; ++p) {
                pages[static_cast<std::size_t>(first + p)] = base + std::uint64_t{p} * page_rows * piece.row_stride;
            }
        }
    }
    return {entries.data(), states, plan.slots};
}

CpuState::CpuState(Layout plan, std::span<const Update> updates)
    : layout_(std::move(plan)),
      entries_(session::cache_buffer_elements<SlotEntry>(layout_.table_entries)),
      pages_(session::cache_buffer_elements<std::uint64_t>(layout_.page_entries)),
      pool_(session::cache_buffer_elements<std::byte>(layout_.pool_bytes)),
      updates_(updates.begin(), updates.end()) {
    if (layout_.instances.empty()) throw std::invalid_argument("CPU state requires at least one declared instance");
    if (!updates_.empty() && updates_.size() != layout_.instances.size())
        throw std::invalid_argument("state update array does not match its layout");
    for (std::size_t i = 0; i < layout_.instances.size(); ++i) {
        const auto& piece = layout_.instances[i];
        if (piece.storage == i && piece.rollback != Rollback::rewind && (updates_.empty() || !updates_[i]))
            throw std::invalid_argument("fixed state requires a CPU update");
    }
    (void)bind_table(layout_, entries_, pages_, 0, 0);
}

const SlotEntry& CpuState::entry(std::uint32_t state, std::uint32_t slot) const {
    return entries_[index(layout_, state, slot)];
}

SlotEntry& CpuState::mutable_entry(std::uint32_t state, std::uint32_t slot) {
    return entries_[index(layout_, state, slot)];
}

std::span<std::byte> CpuState::bytes(std::uint64_t offset, std::uint64_t count) {
    return std::span(pool_).subspan(static_cast<std::size_t>(offset), static_cast<std::size_t>(count));
}

std::uint64_t CpuState::append_base(const InstanceLayout& piece, const SlotEntry& stored, std::uint32_t position) const {
    if (piece.pages) {
        const auto page = stored.base / sizeof(std::uint64_t) + position / page_rows;
        return pages_[static_cast<std::size_t>(page)] + (position % page_rows) * piece.row_stride;
    }
    if (piece.row_divisor > 1) {
        // Only completed groups are durable rows. The open group has a
        // position-tagged overlay that can retain the old carry on rejection.
        if ((position + 1U) % piece.row_divisor == 0)
            return stored.base + (position / piece.row_divisor) * piece.row_stride;
        const auto row = attn::cache_rows::slot({1, piece.carry_rows}, position);
        return stored.base + piece.carry_offset + row * piece.row_stride;
    }
    return stored.base + attn::cache_rows::row_offset({1, piece.ring}, position) * piece.row_stride;
}

Row CpuState::committed_row(std::uint32_t state, std::uint32_t slot, std::uint32_t position) const {
    const auto& stored = entry(state, slot);
    const auto& piece = layout_.instances[layout_.instances[state].storage];
    if (piece.rollback != Rollback::rewind) throw std::invalid_argument("expected position-addressed state");
    if (position >= stored.frontier || (piece.window && stored.frontier - position > piece.window)) return {};
    if (piece.row_divisor > 1 && (position + 1U) % piece.row_divisor != 0 && position + 1U != stored.frontier)
        return {};
    const auto base = append_base(piece, stored, position);
    const auto row = Row(pool_).subspan(static_cast<std::size_t>(base), static_cast<std::size_t>(piece.payload_bytes()));
    std::uint32_t tag{};
    std::memcpy(&tag, row.data() + row.size(), sizeof(tag));
    if (tag != position + 1U) return {};
    return row;
}

Row CpuState::current(std::uint32_t state, std::uint32_t slot) const {
    const auto& stored = entry(state, slot);
    const auto& piece = layout_.instances[state];
    if (piece.rollback == Rollback::rewind) throw std::invalid_argument("expected fixed state");
    return Row(pool_).subspan(static_cast<std::size_t>(stored.base), static_cast<std::size_t>(piece.payload_bytes()));
}

void CpuState::reset(std::uint32_t slot) {
    const auto& previous = entry(0, slot);
    if (previous.pending) throw std::logic_error("cannot reset a pending state slot");
    if (previous.generation == std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("state slot generation exhausted");
    const auto generation = previous.generation + 1U;
    for (std::uint32_t state = 0; state < layout_.instances.size(); ++state) {
        const auto& piece = layout_.instances[layout_.instances[state].storage];
        auto& stored = mutable_entry(state, slot);
        stored = {initial_base(piece, slot, 0, 0), 0, 0, layout_.context, generation};
        if (piece.storage == state && piece.rollback != Rollback::rewind)
            std::ranges::fill(bytes(stored.base, piece.payload_bytes()), std::byte{});
    }
}

}  // namespace strix::state
