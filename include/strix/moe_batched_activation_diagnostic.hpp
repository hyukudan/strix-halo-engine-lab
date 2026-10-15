#pragma once

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strix::moe_batched_activation_diagnostic {
enum class Mode { reference, batched };
struct Shape { std::uint32_t intermediate = 0, slots = 0; };
struct Span { std::uintptr_t address = 0; std::uint64_t bytes = 0; };
struct Bindings { Span workspace, status; };
struct Launch { std::uint32_t grid_x = 0, grid_y = 0, block_x = 256U; };
inline Launch validate(Shape shape, const Bindings& bindings, Mode mode) {
    if (!shape.intermediate || shape.intermediate > 640U || !shape.slots || shape.slots > 10U ||
        (mode != Mode::reference && mode != Mode::batched))
        throw std::invalid_argument("batched routed activation diagnostic shape/mode is invalid");
    const auto check = [](Span view, std::uint64_t expected) {
        if (!view.address || view.address % alignof(float) || view.bytes != expected ||
            view.bytes > std::numeric_limits<std::uintptr_t>::max() - view.address)
            throw std::invalid_argument("batched routed activation diagnostic view extent/alignment differs");
    };
    check(bindings.workspace, static_cast<std::uint64_t>(shape.slots) * 2U * shape.intermediate * sizeof(float));
    check(bindings.status, sizeof(std::uint32_t));
    if (bindings.workspace.address < bindings.status.address + bindings.status.bytes &&
        bindings.status.address < bindings.workspace.address + bindings.workspace.bytes)
        throw std::invalid_argument("batched routed activation diagnostic workspace/status alias");
    return {(shape.intermediate + 255U) / 256U, mode == Mode::batched ? shape.slots : 1U, 256U};
}
// Standalone test seam for the actual production kernels. Reference enqueues the
// original SiLU kernel once per slot; batched enqueues the new two-dimensional
// kernel once. Workspace is FP32 [slot][gate then up], EXACTLY the existing A8
// slot layout. Only gate values change. The up planes and sticky uint32 status
// are read-only. A nonzero status suppresses every workspace write. The raw
// kernels propagate nonfinites; finite pre/post scans remain the full operator's
// responsibility. This seam clears no status and does not validate finiteness.
// No allocation, synchronization, status publication or model qualification.
// Retain the explicit non-default stream and every owner until device completion,
// also after enqueue failure and throughout graph capture/replay. No concurrent
// writer may change status or workspace during this same-stream phase.
void enqueue(Shape, const Bindings&, Mode, hipStream_t);
const void* kernel_symbol(Mode);
} // namespace strix::moe_batched_activation_diagnostic
