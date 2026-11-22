#pragma once

#include "strix/state/table.hpp"

#include <cstddef>
#include <functional>
#include <span>
#include <vector>

namespace strix::state {

using Row = std::span<const std::byte>;
// Fill the destination image, including any shadows, using the same order on
// the first execution and on replay. Recompute receives only the position.
using Update = std::function<void(std::uint32_t position, Row previous, Row input, std::span<std::byte> next)>;

class Txn;

// CPU reference executor. Views borrow the storage until it is mutated.
// Calls on a given slot, including reads and reset, are externally serialized.
// Updates shared by concurrently executed slots must be safe to call concurrently.
class CpuState {
public:
    explicit CpuState(Layout, std::span<const Update> = {});
    CpuState(const CpuState&) = delete;
    CpuState& operator=(const CpuState&) = delete;

    [[nodiscard]] const Layout& plan() const noexcept { return layout_; }
    [[nodiscard]] const SlotEntry& entry(std::uint32_t state, std::uint32_t slot) const;
    // Empty for a position outside the published rows. Compressed states
    // expose completed groups and the latest unfinished group, if any.
    [[nodiscard]] Row committed_row(std::uint32_t state, std::uint32_t slot, std::uint32_t position) const;
    [[nodiscard]] Row current(std::uint32_t state, std::uint32_t slot) const;  // fixed states only
    void reset(std::uint32_t slot);

private:
    friend class Txn;
    [[nodiscard]] SlotEntry& mutable_entry(std::uint32_t state, std::uint32_t slot);
    [[nodiscard]] std::span<std::byte> bytes(std::uint64_t, std::uint64_t);
    [[nodiscard]] std::uint64_t append_base(const InstanceLayout&, const SlotEntry&, std::uint32_t position) const;

    Layout layout_;
    std::vector<SlotEntry> entries_;
    std::vector<std::uint64_t> pages_;
    std::vector<std::byte> pool_;
    std::vector<Update> updates_;
};

// A window is owned by one Txn until commit, abort or destruction. Failed
// updates discard that window and leave the published contents available.
class Txn {
public:
    Txn(CpuState&, std::uint32_t slot);
    ~Txn();
    Txn(const Txn&) = delete;
    Txn& operator=(const Txn&) = delete;

    void begin(std::uint32_t rows);
    // One token's epilogues, indexed by state instance. Append states take
    // a row image, snapshot states take update input, replay states take
    // their declared input image. Aliases and recompute take an empty span.
    void append(std::span<const Row> inputs);
    void commit(std::uint32_t accepted);
    void abort() noexcept;

private:
    [[nodiscard]] std::uint64_t candidate(const InstanceLayout&, const SlotEntry&, std::uint32_t row) const;
    void prepare_accept(std::uint32_t state, std::uint32_t accepted);
    void update(std::uint32_t state, std::uint32_t position, std::uint64_t previous, Row input,
                std::uint64_t next);

    CpuState& state_;
    std::uint32_t slot_;
    std::uint32_t pending_{};
    std::uint32_t written_{};
};

}  // namespace strix::state
