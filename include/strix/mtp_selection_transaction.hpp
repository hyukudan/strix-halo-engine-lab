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
    // Optional hint after reset(): the drafts the window is about to verify. Row r is reached only when rows
    // 0..r-1 selected drafts 0..r-1, so a policy may prepare those rows' work while the device runs. It never
    // changes what select()/select_known() return.
    virtual void expect(std::span<const std::uint32_t>) {}
    virtual void discard() noexcept = 0;
    virtual std::uint32_t select(std::span<const float>) = 0;
    // The row's unmasked lowest-ID maximum and finite status (device greedy reply). Returns the selected
    // token when it equals select() on the full row; nullopt: read the row and call select().
    virtual std::optional<std::uint32_t> select_known(std::uint32_t, std::uint32_t) { return std::nullopt; }
    virtual std::unique_ptr<MtpSelectionReceipt> seal() = 0;
    // Wall time of one window row (readback and selection); profiling only.
    virtual void note_row_ns(std::uint64_t) noexcept {}
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

// Device-greedy engines (the host receives each verified row's argmax only; a non-finite row fails the window before
// selection, so every row is finite): the window's selected prefix under a selection transaction. Window row r takes
// the input row r (row 0 = the pending token, row r > 0 = drafts[r - 1]) and replies argmax[r]. Rows are decided from
// row 0 by select_known(argmax[r], 0) until a selected token differs from the next input row, or the last row. The
// engine then commits ids.size() rows, emits ids, feeds ids.back() as its next input and publishes the receipt after
// its commit. When the policy forces nothing this is the greedy rule (ids = argmax[0, 1 + accepted drafts)); a forced
// token (a thinking-budget closure) ends the prefix at its row unless it equals that row's draft. Stops and output
// limits stay the engine's (no eos or output limit here). Throws MtpSelectionError (transaction discarded) when a row
// needs host logits: such a policy is not decidable from argmax replies and must be refused at admission.
struct GreedyWindowSelection {
    std::vector<std::uint32_t> ids;
    std::unique_ptr<MtpSelectionReceipt> receipt;
};
[[nodiscard]] GreedyWindowSelection select_greedy_window(MtpSelectionTransaction& transaction,
    std::span<const std::uint32_t> drafts, std::span<const std::uint32_t> argmax);

}  // namespace strix
