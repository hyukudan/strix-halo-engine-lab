#include "strix/hip_known_prompt_backend.hpp"

#include "strix/hip_known_prefix_window.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/serial_decode_replay.hpp"
#include "strix/serial_session.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace strix {
namespace {
void add_prompt_stat(std::uint64_t& total, std::uint64_t amount, bool& saturated) noexcept {
    if (amount > std::numeric_limits<std::uint64_t>::max() - total) {
        total = std::numeric_limits<std::uint64_t>::max();
        saturated = true;
    } else total += amount;
}
void check_prompt_copy(hipError_t status) {
    if (status != hipSuccess)
        throw std::runtime_error(std::string("known prompt terminal logits copy: ") + hipGetErrorString(status));
}
} // namespace

struct HipKnownPromptBackend::Impl {
    const std::shared_ptr<HipKnownPrefixWindow> window;
    const std::shared_ptr<const HipSetupQuiescenceGuard> guard;
    SerialSessionContext& session;
    const ExecutionMemoryBinding scratch;
    const std::uint64_t vocab_size;
    const std::uint64_t logits_bytes;
    bool failed = false;
    HipKnownPromptBackendStats observed;

    Impl(std::shared_ptr<HipKnownPrefixWindow> native_window, SerialSessionContext& live_session,
        ExecutionMemoryBinding memory, std::uint64_t vocabulary,
        std::shared_ptr<const HipSetupQuiescenceGuard> original_guard)
        : window(std::move(native_window)), guard(std::move(original_guard)), session(live_session),
          scratch(memory), vocab_size(vocabulary), logits_bytes(checked_logits_bytes(vocabulary)) {
        if (!window || !guard || !scratch.execution_identity || !scratch.allocation_generation ||
            !scratch.arena_base || !scratch.arena_bytes || logits_bytes > scratch.arena_bytes ||
            scratch.arena_bytes > std::numeric_limits<std::uintptr_t>::max() - scratch.arena_base)
            throw std::invalid_argument("known prompt adapter requires its actual retained window arena and guard");
        if (current_hip_launch_context())
            throw std::invalid_argument("known prompt adapter construction cannot nest a HIP launch scope");
        guard->require_original();
    }

    static std::uint64_t checked_logits_bytes(std::uint64_t vocabulary) {
        if (!vocabulary || vocabulary > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1ULL ||
            vocabulary > std::numeric_limits<std::size_t>::max() / sizeof(float))
            throw std::invalid_argument("known prompt adapter vocabulary does not fit a complete FP32 row");
        return vocabulary * sizeof(float);
    }

    void validate_terminal(const ExecutionBufferAddress& terminal) const {
        if (terminal.use.buffer.buffer_class != ExecutionBufferClass::logits ||
            !terminal.address || terminal.address % alignof(float) != 0U ||
            terminal.allocation_generation != scratch.allocation_generation || terminal.bytes < logits_bytes ||
            terminal.bytes > std::numeric_limits<std::uintptr_t>::max() - terminal.address ||
            terminal.address < scratch.arena_base || terminal.address - scratch.arena_base > scratch.arena_bytes ||
            terminal.bytes > scratch.arena_bytes - (terminal.address - scratch.arena_base))
            throw std::invalid_argument("known prompt terminal logits are outside the retained window arena");
    }

    std::vector<float> execute(std::span<const std::uint32_t> tokens, std::uint64_t position) {
        if (failed) throw std::logic_error("known prompt adapter cannot be reused after an exception");
        const auto committed_before = session.committed_tokens;
        const auto rows = window->logical_rows();
        bool window_attempted = false;
        try {
            guard->require_original();
            if (current_hip_launch_context() || session.health != SerialSessionHealth::ready ||
                session.arithmetic_lineage.tainted || position != session.next_position ||
                tokens.size() != rows || (rows != 2U && rows != 4U && rows != 8U) ||
                position > std::numeric_limits<std::uint64_t>::max() - rows ||
                committed_before > std::numeric_limits<std::uint64_t>::max() - rows)
                throw std::invalid_argument("known prompt window differs from the live eager session");
            for (const auto token : tokens)
                if (static_cast<std::uint64_t>(token) >= vocab_size)
                    throw std::invalid_argument("known prompt token exceeds vocabulary");
            // Allocate host return storage before any model-state consumption.
            std::vector<float> logits(static_cast<std::size_t>(vocab_size));
            window_attempted = true;
            const auto result = window->execute(session, tokens);
            if (session.health != SerialSessionHealth::ready || session.next_position != position + rows ||
                session.committed_tokens != committed_before + rows || result.logits.size() != rows)
                throw std::logic_error("known prompt native window did not commit its configured causal rows");
            validate_terminal(result.logits[rows - 1U]);
            // execute() has already fenced its owned producer stream and checked
            // D1 before publication. A synchronous full-row download introduces
            // no borrowed stream, no head omission and no additional D1 phase.
            guard->require_original();
            check_prompt_copy(hipMemcpy(logits.data(), reinterpret_cast<const void*>(result.logits[rows - 1U].address),
                static_cast<std::size_t>(logits_bytes), hipMemcpyDeviceToHost));
            guard->require_original();
            validate_serial_decode_engine_logits(logits, vocab_size);
            // Publish only completed successful adapter returns, from the
            // actual native result. Saturation is telemetry, never admission.
            auto next = observed;
            add_prompt_stat(next.windows, 1U, next.saturated);
            add_prompt_stat(next.completed_input_rows, rows, next.saturated);
            add_prompt_stat(next.shared_rows_dispatches, result.shared_rows_dispatches, next.saturated);
            add_prompt_stat(next.gdn_rows_dispatches, result.gdn_rows_dispatches, next.saturated);
            add_prompt_stat(next.r2_dispatches, result.r2_dispatches, next.saturated);
            add_prompt_stat(next.row_loop_dispatches, result.row_loop_dispatches, next.saturated);
            add_prompt_stat(next.logical_projection_weight_bytes_avoided,
                result.logical_projection_weight_bytes_avoided, next.saturated);
            add_prompt_stat(next.projection_launches_saved, result.projection_launches_saved, next.saturated);
            static_assert(std::tuple_size_v<decltype(result.r2_variant_dispatches)> ==
                std::tuple_size_v<decltype(next.r2_variant_dispatches)>);
            for (std::size_t i = 0; i < next.r2_variant_dispatches.size(); ++i)
                add_prompt_stat(next.r2_variant_dispatches[i], result.r2_variant_dispatches[i], next.saturated);
            observed = next;
            return logits;
        } catch (...) {
            failed = true;
            if (session.committed_tokens != committed_before) {
                session.health = SerialSessionHealth::poisoned;
                session.arithmetic_lineage.tainted = true;
            }
            // The native owner also contains its own failure paths. This drain
            // covers output-copy/validation failures after a successful commit.
            // Pure admission has submitted nothing. In particular, rejecting a
            // caller's active scope must not synchronize its live capture.
            if (window_attempted) guard->recover_or_fail_stop();
            throw;
        }
    }
};

HipKnownPromptBackend::HipKnownPromptBackend(std::shared_ptr<HipKnownPrefixWindow> window,
    SerialSessionContext& session, ExecutionMemoryBinding window_scratch, std::uint64_t vocab_size,
    std::shared_ptr<const HipSetupQuiescenceGuard> guard)
    : impl_(std::make_unique<Impl>(std::move(window), session, window_scratch, vocab_size, std::move(guard))) {}
HipKnownPromptBackend::~HipKnownPromptBackend() = default;
std::vector<float> HipKnownPromptBackend::execute_window(std::array<std::uint32_t,2> tokens, std::uint64_t position) {
    return impl_->execute(tokens, position);
}
std::uint32_t HipKnownPromptBackend::logical_rows() const noexcept {
    return impl_->window->logical_rows();
}
std::vector<float> HipKnownPromptBackend::execute_rows(std::span<const std::uint32_t> tokens, std::uint64_t position) {
    return impl_->execute(tokens, position);
}
HipKnownPromptBackendStats HipKnownPromptBackend::stats() const noexcept {
    return impl_->observed;
}
} // namespace strix
