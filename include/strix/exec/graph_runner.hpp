#pragma once

#include "strix/exec/graph_submission.hpp"
#include "strix/exec/width_replay.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace strix::exec {

enum class GraphRows { cohort, single, inclusive_window };

// Captured device arguments can only name a different request via the uploaded block.
// Refuse a half-initialized mask before a kernel can read id zero or skip a state write.
inline void validate_graph_rows(const StepBlock& block, GraphRows layout = GraphRows::cohort) {
    if (block.h.rows == 0U || block.h.rows > kStepMaxRows)
        throw std::invalid_argument("invalid graph descriptor width");
    if (block.h.stop_count > kStepMaxStops || (block.h.stop_count && !step_stop_address(block.h)))
        throw std::invalid_argument("invalid graph stop table");
    if (layout == GraphRows::single && block.h.rows != 1U)
        throw std::invalid_argument("single-row graph requires one row");
    std::uint32_t valid = 0;
    for (std::uint32_t i = 0; i < block.h.rows; ++i) {
        const auto bit = std::uint32_t{1} << i;
        const auto& row = block.row[i];
        if (row.flags & step_valid) valid |= bit;
        if (layout == GraphRows::inclusive_window &&
            (row.flags != step_valid || row.slot != block.row[0].slot ||
             row.position != std::uint64_t{block.row[0].position} + i ||
             row.parent != (i == 0U ? kStepNoParent : i - 1U) || row.ancestors != ((bit - 1U) | bit)))
            throw std::invalid_argument("invalid inclusive verify window");
    }
    if (valid != block.h.valid_mask || (block.h.feed_mask & ~valid) != 0U)
        throw std::invalid_argument("inconsistent graph descriptor masks");
}

// A binding contains only the owner's capture-time dependencies. Device-read values belong
// in the StepBlock. A fully indirect owner with stable allocations needs no additional key.
struct IndirectGraphBinding {
    bool operator==(const IndirectGraphBinding&) const = default;
};

// A qualified rectangular program for each active stream count. Entry B - 1 is the number
// of consecutive rows of each stream, including its anchor. A zero entry has no program.
struct StreamGraphPrograms {
    std::array<std::uint32_t, kStepMaxRows> rows_per_stream{};
};

// One serial owner per program family and stream. A runtime records work with capture(),
// instantiates and inspects the candidate, uploads into stable storage, then launches it.
// The serial owner retires staging reads before submitting again. Only a pre-launch decline
// may fall back to eager work; an uncertain submission is never retried here.
template <class Binding = IndirectGraphBinding>
class StepGraphRunner {
public:
    explicit StepGraphRunner(GraphRows layout = GraphRows::cohort) : layout_(layout) {}
    explicit StepGraphRunner(const StreamGraphPrograms& programs) : stream_programs_(programs) {
        for (std::uint32_t streams = 1; streams <= kStepMaxRows; ++streams) {
            if (programs.rows_per_stream[streams - 1U] > kStepMaxRows / streams)
                throw std::invalid_argument("stream graph program exceeds the descriptor width");
        }
    }

    template <class Runtime, class Stream, class Work>
    bool run(Runtime& runtime, Stream stream, bool enabled, bool stable, const StepBlock& block,
             const Binding& binding, Work& work) {
        if (stream_programs_) return false;
        return run_program(runtime, stream, enabled, stable, block, binding, work, block.h.rows);
    }

    template <class Runtime, class Stream, class Work>
    bool run_for_streams(Runtime& runtime, Stream stream, bool enabled, bool stable, const StepBlock& block,
                         const Binding& binding, Work& work, std::uint32_t active_streams) {
        const auto rows = rows_per_stream(active_streams);
        if (rows == 0U || block.h.rows != active_streams * rows) return false;
        if (enabled && stable) {
            validate_graph_rows(block);
            // A shorter window or a different grouping does not fit the recorded launch.
            for (std::uint32_t base = 0; base < block.h.rows; base += rows) {
                const auto& first = block.row[base];
                for (std::uint32_t other = 0; other < base; other += rows) {
                    if (block.row[other].slot == first.slot) return false;
                }
                for (std::uint32_t i = 0; i < rows; ++i) {
                    const auto& row = block.row[base + i];
                    if (!step_row_valid(block, base + i) || row.slot != first.slot ||
                        row.position != std::uint64_t{first.position} + i)
                        return false;
                }
            }
        }
        return run_program(runtime, stream, enabled, stable, block, binding, work, active_streams);
    }

    [[nodiscard]] bool uses_stream_programs() const noexcept { return stream_programs_.has_value(); }

    [[nodiscard]] std::uint32_t rows_per_stream(std::uint32_t active_streams) const noexcept {
        const auto at = WidthReplayPolicy::index(active_streams);
        return stream_programs_ && at < WidthReplayPolicy::kWidths ? stream_programs_->rows_per_stream[at] : 0U;
    }

    template <class Runtime>
    void invalidate(Runtime& runtime) noexcept {
        for (std::uint32_t i = 0; i < WidthReplayPolicy::kWidths; ++i) runtime.discard(i);
        policy_.invalidate();
    }

    template <class Runtime>
    void reset(Runtime& runtime) noexcept {
        invalidate(runtime);
        bindings_ = {};
        policy_.reset();
    }

    // The replay policy is indexed by physical width in width mode and by active streams in stream mode.
    [[nodiscard]] const WidthReplayPolicy& policy() const noexcept { return policy_; }

private:
    template <class Runtime, class Stream, class Work>
    bool run_program(Runtime& runtime, Stream stream, bool enabled, bool stable, const StepBlock& block,
                     const Binding& binding, Work& work, std::uint32_t key) {
        const auto at = WidthReplayPolicy::index(key);
        if (at >= WidthReplayPolicy::kWidths) return false;
        if (enabled && stable && !stream_programs_) validate_graph_rows(block, layout_);
        if (bindings_[at] && *bindings_[at] != binding) {
            runtime.discard(at);
            policy_.invalidate_width(key);
            bindings_[at].reset();
        }
        const auto action = policy_.select(enabled, key, stable);
        if (action == WidthReplayAction::eager) {
            runtime.discard(at);
            return false;
        }
        return submit_graph(action == WidthReplayAction::capture, [&] {
            bindings_[at] = binding;
            if (!runtime.capture(stream, at, work)) return false;
            policy_.captured(key);
            return true;
        }, [&] {
            runtime.upload(stream, block);
            return true;
        }, [&] { runtime.launch(stream, at); }, [&] {
            runtime.discard(at);
            policy_.fail(key);
        });
    }

    GraphRows layout_ = GraphRows::cohort;
    std::optional<StreamGraphPrograms> stream_programs_;
    WidthReplayPolicy policy_;
    std::array<std::optional<Binding>, WidthReplayPolicy::kWidths> bindings_{};
};

}  // namespace strix::exec
