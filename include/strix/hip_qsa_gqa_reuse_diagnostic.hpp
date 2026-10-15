#pragma once

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strix::qsa_gqa_reuse_diagnostic {
enum class Mode { reference, heads2, heads4 };
inline constexpr std::uint32_t compression = 4U;
inline constexpr std::uint32_t block_topk = 512U;
inline constexpr std::uint32_t invalid_position = 2U;
inline constexpr std::uint32_t nonfinite_source = 4U;
inline constexpr std::uint32_t invalid_arithmetic = 8U;
inline constexpr std::uint32_t invalid_selection = 16U;
struct Shape {
    std::uint32_t query_heads = 24U, kv_heads = 2U, head_dim = 256U;
    std::uint32_t context_tokens = 0U;
};
struct Span { std::uintptr_t address = 0; std::uint64_t bytes = 0; };
struct Bindings { Span q_gate, kv_state, selected, position, status, scores, output; };
struct Launch {
    std::uint32_t maximum_selected = 0U;
    std::uint32_t score_blocks = 0U, value_blocks = 0U;
    std::uint32_t score_threads = 256U, value_threads = 256U;
    std::uint32_t heads_per_block = 1U, score_lds_bytes = 1024U;
};
// The maximum is over EVERY occupied position up to capacity: a capacity exactly
// divisible by four must still retain the preceding position's three tail slots.
inline std::uint32_t maximum_selected_tokens(Shape shape) {
    if (!shape.context_tokens || shape.context_tokens > 262144U)
        throw std::invalid_argument("QSA GQA diagnostic context capacity is not admitted");
    return std::min(shape.context_tokens, block_topk * compression + compression - 1U);
}
inline Launch validate(Shape shape, const Bindings& bindings, Mode mode) {
    if (!shape.query_heads || shape.query_heads > 64U || !shape.kv_heads ||
        shape.kv_heads > shape.query_heads || shape.query_heads % shape.kv_heads || shape.head_dim != 256U)
        throw std::invalid_argument("QSA GQA diagnostic admits C1 GQA with head dimension 256 only");
    std::uint32_t heads = 0;
    switch (mode) {
    case Mode::reference: heads = 1U; break;
    case Mode::heads2: heads = 2U; break;
    case Mode::heads4: heads = 4U; break;
    default: throw std::invalid_argument("QSA GQA diagnostic mode is invalid");
    }
    const auto maximum = maximum_selected_tokens(shape);
    const auto check = [](Span view, std::uint64_t required, std::uint64_t alignment) {
        if (!view.address || view.address % alignment || view.bytes != required ||
            view.bytes > std::numeric_limits<std::uintptr_t>::max() - view.address)
            throw std::invalid_argument("QSA GQA diagnostic view extent/alignment differs");
    };
    check(bindings.q_gate, static_cast<std::uint64_t>(shape.query_heads) * 256U * 2U * sizeof(float), alignof(float));
    check(bindings.kv_state, static_cast<std::uint64_t>(shape.context_tokens) * shape.kv_heads * 256U * 2U * sizeof(std::uint16_t),
          alignof(std::uint16_t));
    check(bindings.selected, static_cast<std::uint64_t>(maximum) * sizeof(std::uint32_t), alignof(std::uint32_t));
    check(bindings.position, sizeof(std::uint64_t), alignof(std::uint64_t));
    check(bindings.status, sizeof(std::uint32_t), alignof(std::uint32_t));
    check(bindings.scores, static_cast<std::uint64_t>(shape.query_heads) * maximum * sizeof(float), alignof(float));
    check(bindings.output, static_cast<std::uint64_t>(shape.query_heads) * 256U * sizeof(float), alignof(float));
    const std::array views{bindings.q_gate, bindings.kv_state, bindings.selected, bindings.position,
                           bindings.status, bindings.scores, bindings.output};
    for (std::size_t a = 0; a < views.size(); ++a)
        for (std::size_t b = a + 1U; b < views.size(); ++b)
            if (views[a].address < views[b].address + views[b].bytes && views[b].address < views[a].address + views[a].bytes)
                throw std::invalid_argument("QSA GQA diagnostic views alias");
    const auto group = shape.query_heads / shape.kv_heads;
    const auto head_blocks = shape.kv_heads * ((group + heads - 1U) / heads);
    return {maximum, head_blocks * maximum, head_blocks, 256U, 256U, heads, heads * 1024U};
}

// Test-only, no allocation/synchronization/status clear or production routing.
// All calls, eager and captured, read uint64 position from device. Visible tokens
// are position+1; count=min(visible/4,512)*4+visible%4. selected[0:count]
// must be strictly chronological, unique and below visible. Its remaining slots
// are not read. Q/gate and scores are FP32; KV is the incumbent interleaved BF16
// [token][K then V][kv_head][feature]. Output is FP32 storage containing the SAME
// BF16-rounded values as the incumbent attention_output_kernel, not raw FP32 V.
//
// A common device precheck validates finite sources/selected IDs before all
// attention writes. Existing graph score/softmax arithmetic status remains in
// force, and a final check validates output. Nonzero sticky status invalidates
// the result; arithmetic failure may leave partially written scores/output.
// Rejection policy is diagnostic; no claim of legacy-invalid-input equivalence.
// The caller owns explicit non-default stream, all views and producer ordering
// through completion, also on enqueue failure and across capture/replay. Status
// is read only after completion; return means submission, not publication.
void enqueue(Shape, const Bindings&, Mode, hipStream_t);
// Same admission/ownership contract, but stops after source validation and QK
// so tests can compare unnormalized scores before softmax hides differences.
// All views remain required and output is left untouched (two kernel submits).
void enqueue_scores(Shape, const Bindings&, Mode, hipStream_t);
const void* score_kernel_symbol(Mode);
const void* value_kernel_symbol(Mode);
} // namespace strix::qsa_gqa_reuse_diagnostic
