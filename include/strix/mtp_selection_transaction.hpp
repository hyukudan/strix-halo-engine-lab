#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix {

// Only policy selection refusals cross this boundary; execution errors remain fatal.
class MtpSelectionError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Receipts stay unpublished until model commit and predictor reconciliation succeed.
// Destroying a receipt or discarding an attempt has no committed side effects.
class MtpSelectionReceipt {
public:
    virtual ~MtpSelectionReceipt() = default;
    virtual void publish() = 0;
    [[nodiscard]] virtual std::size_t size() const noexcept = 0;
};
class MtpSelectionTransaction {
public:
    virtual ~MtpSelectionTransaction() = default;
    virtual void reset() = 0;
    virtual void discard() noexcept = 0;
    virtual std::uint32_t select(std::span<const float>) = 0;
    virtual std::unique_ptr<MtpSelectionReceipt> seal() = 0;
};

enum class MtpSelectionStop { eos, output_limit, row_limit, no_draft, rejection, selection_failure };
struct MtpSelectionPrefix {
    std::vector<std::uint32_t> selected_ids;
    std::optional<std::string> selection_failure;
    std::size_t decided_rows = 0;
    std::size_t accepted_draft_inputs = 0;
    MtpSelectionStop stop = MtpSelectionStop::row_limit;
};

// Resolve before drafting. Every ConstraintSpec, including scores/transforms only,
// uses this same admission; false owner admission never starts speculation.
struct MtpSelectionRoute {
    bool mtp=false;
    const char* fallback=nullptr;
};
[[nodiscard]] inline MtpSelectionRoute resolve_mtp_selection_route(bool enabled,bool window,bool serial_rows) noexcept {
    if(!enabled)return {false,"selection-mtp-off"};
    if(!window)return {false,"no-window-owner"};
    if(!serial_rows)return {false,"window-arithmetic-not-serial"};
    return {true,nullptr};
}

// The callback reads and selects exactly one reachable logical row. No row/draft
// spans escape this call. Limits exclude the already emitted pending input.
MtpSelectionPrefix walk_mtp_selection(std::span<const std::uint32_t> proposals,
    std::size_t row_limit, std::size_t output_limit, std::span<const std::uint32_t> eos_ids,
    const std::function<std::uint32_t(std::size_t)>& select_row);

}  // namespace strix
