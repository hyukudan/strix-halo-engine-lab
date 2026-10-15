#pragma once

#include "strix/compiled_execution.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strix {
// This schedule admits exactly one active layer. Outside a fork, steps run in
// serial order. At mix completion shared may overlap route -> experts. Combine
// starts only after BOTH branches complete, and completes before the next step.
struct MoeForkJoinArenaLayer {
    std::uint32_t layer = 0;
    std::uint32_t mix_step = 0;
    std::uint32_t route_step = 0;
    std::uint32_t shared_step = 0;
    std::uint32_t experts_step = 0;
    std::uint32_t combine_step = 0;
    ExecutionBufferUse input;
    ExecutionBufferUse shared_output;
    ExecutionBufferUse shared_temporary;
    ExecutionBufferUse routed_output;
    ExecutionBufferUse routed_temporary;
};

// Conservative allocation plan, not a scheduler or completion ticket. It keeps
// the serial base offsets and relocates each shared output/temporary into an
// appended region. Regions are reused across layers only after the required
// join. The source compiled plan and its physical binding remain unchanged.
class MoeForkJoinArenaPlan final {
public:
    [[nodiscard]] std::uint64_t source_execution_identity() const noexcept { return source_identity_; }
    [[nodiscard]] std::uint64_t layout_identity() const noexcept { return layout_identity_; }
    [[nodiscard]] std::uint64_t serial_arena_bytes() const noexcept { return serial_bytes_; }
    [[nodiscard]] std::uint64_t additional_bytes() const noexcept { return arena_.total_bytes - serial_bytes_; }
    [[nodiscard]] std::uint64_t shared_region_offset() const noexcept { return serial_bytes_; }
    [[nodiscard]] std::uint64_t shared_region_bytes() const noexcept { return additional_bytes(); }
    [[nodiscard]] const ExecutionArenaPlan& arena() const noexcept { return arena_; }
    [[nodiscard]] std::span<const MoeForkJoinArenaLayer> layers() const noexcept { return layers_; }

    // Exact comparison with a privately retained source snapshot, including
    // DAG, IO, bindings, flags and arena; a reused source ID alone is not proof.
    // Recompile launchers and create physical bindings AFTER this transformation.
    [[nodiscard]] CompiledSerialExecution rebind_execution(const CompiledSerialExecution& source) const;
    void validate_execution(const CompiledSerialExecution& rebound) const;

    // Requires a separately owned allocation tagged with layout_identity(), not
    // the source execution identity. This does not prove GPU visibility, owner
    // retention, absence of external aliases, event completion or graph safety.
    void validate_binding(const ExecutionMemoryBinding& memory) const;
    [[nodiscard]] ExecutionBufferAddress resolve(const ExecutionMemoryBinding& memory,
                                                ExecutionBufferUse use) const;
private:
    MoeForkJoinArenaPlan() = default;
    friend MoeForkJoinArenaPlan prepare_moe_fork_join_arena(
        const CompiledSerialExecution&, std::uint64_t);
    std::uint64_t source_identity_ = 0;
    std::uint64_t layout_identity_ = 0;
    std::uint64_t serial_bytes_ = 0;
    ExecutionArenaPlan arena_;
    CompiledSerialExecution source_;
    std::vector<MoeForkJoinArenaLayer> layers_;
};

// budget_bytes caps the COMPLETE contiguous arena, including retained serial
// holes and alignment. Preparation allocates host metadata only, never enqueues
// device work, and rejects malformed source lifetimes or an unsupported DAG/IO.
[[nodiscard]] MoeForkJoinArenaPlan prepare_moe_fork_join_arena(
    const CompiledSerialExecution& compiled, std::uint64_t budget_bytes);
} // namespace strix
