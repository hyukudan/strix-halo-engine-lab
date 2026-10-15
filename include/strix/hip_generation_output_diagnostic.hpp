#pragma once

// Test-only host fault boundaries. Definitions exist only with BUILD_TESTING.
// Real device operations preceding a boundary still run and must be drained.
// These simulate a reported failure; they do not inject a device fault or prove
// recovery from a real driver failure. Calls are serialized on the test thread.
namespace strix {
enum class HipGreedyOutputDiagnosticFault {
    none,
    partial_allocation,
    reply_allocation,
    pinned_allocation,
    after_poison,
    after_partials,
    after_finish,
    after_readback,
    after_completion,
    corrupt_epoch,
    corrupt_position,
    corrupt_identity,
    corrupt_status,
    corrupt_token
};
void arm_hip_greedy_output_diagnostic_fault(HipGreedyOutputDiagnosticFault fault);
void clear_hip_greedy_output_diagnostic_fault() noexcept;
} // namespace strix
