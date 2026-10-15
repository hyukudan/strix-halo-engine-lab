#pragma once

#include <hip/hip_runtime_api.h>

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strix::gdn_recurrent_layout_diagnostic {
enum class Layout { reference, stride32, stride33, transposed };
struct Shape {
    std::uint32_t key_heads = 0;
    std::uint32_t value_heads = 0;
    std::uint32_t head_dim = 128;
};
struct Span { std::uintptr_t address = 0; std::uint64_t bytes = 0; };
struct Bindings { Span qkv, beta, gate, state, core, graph_status; };
struct Launch { std::uint32_t grid_x = 0, block_x = 0; };

// All views are FP32 except the optional uint32 sticky graph status. This is a
// bounded diagnostic of the incumbent 128-thread/32-column recurrence. It does
// not validate finiteness: the raw incumbent propagates nonfinites, while an
// already nonzero graph status skips every state/core write. Full-operator
// finite validation remains outside this single-kernel diagnostic boundary.
inline Launch validate(Shape shape, const Bindings& bindings, Layout layout) {
    if (!shape.key_heads || shape.key_heads > 64U || !shape.value_heads ||
        shape.value_heads > 64U || shape.value_heads % shape.key_heads || shape.head_dim != 128U)
        throw std::invalid_argument("GDN recurrent layout diagnostic shape is not admitted");
    if (layout != Layout::reference && layout != Layout::stride32 &&
        layout != Layout::stride33 && layout != Layout::transposed)
        throw std::invalid_argument("GDN recurrent layout diagnostic layout is invalid");
    const auto check = [](Span view, std::uint64_t required) {
        if (!view.address || view.address % alignof(float) || view.bytes != required ||
            view.bytes > std::numeric_limits<std::uintptr_t>::max() - view.address)
            throw std::invalid_argument("GDN recurrent layout diagnostic view extent/alignment differs");
    };
    check(bindings.qkv, (2ULL * shape.key_heads + shape.value_heads) * 128U * sizeof(float));
    check(bindings.beta, static_cast<std::uint64_t>(shape.value_heads) * sizeof(float));
    check(bindings.gate, static_cast<std::uint64_t>(shape.value_heads) * sizeof(float));
    check(bindings.state, static_cast<std::uint64_t>(shape.value_heads) * 128U * 128U * sizeof(float));
    check(bindings.core, static_cast<std::uint64_t>(shape.value_heads) * 128U * sizeof(float));
    if (bindings.graph_status.address || bindings.graph_status.bytes)
        check(bindings.graph_status, sizeof(std::uint32_t));
    const auto overlaps = [](Span a, Span b) {
        return a.address && b.address && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
    };
    const std::array inputs{bindings.qkv, bindings.beta, bindings.gate};
    const std::array outputs{bindings.state, bindings.core, bindings.graph_status};
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        for (const auto input : inputs) if (overlaps(outputs[i], input))
            throw std::invalid_argument("GDN recurrent layout diagnostic writable/read-only views alias");
        for (std::size_t j = i + 1U; j < outputs.size(); ++j) if (overlaps(outputs[i], outputs[j]))
            throw std::invalid_argument("GDN recurrent layout diagnostic writable views alias");
    }
    return {shape.value_heads * 4U, 128U};
}

// Available only in BUILD_TESTING builds with the diagnostic macro enabled.
// No allocation, synchronization, status clear or production dispatch change.
// The caller owns the explicit stream and every view through completion,
// including after an enqueue exception and across graph capture/replay.
void enqueue(Shape, const Bindings&, Layout, hipStream_t);
const void* kernel_symbol(Layout);
} // namespace strix::gdn_recurrent_layout_diagnostic
