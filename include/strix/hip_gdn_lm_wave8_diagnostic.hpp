#pragma once
#include "strix/gdn_lm_wave8_diagnostic_contract.hpp"
#include <hip/hip_runtime_api.h>

namespace strix::gdn_lm_wave8_diagnostic {
// Test-only entry points, built with BUILD_TESTING. No allocation, clear,
// synchronization, status publication, or production dispatch selection.
// The caller retains all views and the stream until completion, including
// after an enqueue exception. GDN reference/prefetch1 both select its incumbent
// prefetch8 kernel; LM reference selects the incumbent per-feature load body.
namespace detail {
void enqueue_gdn(Shape, Span, Span, Span, Arm, hipStream_t);
void enqueue_lm(Shape, Span, Span, Span, Kind, Arm, hipStream_t);
const void* gdn_symbol(Arm);
const void* lm_symbol(Kind, Arm);
}
inline void enqueue(Shape shape, Span input, Span weights, Span output,
                    Kind kind, Arm arm, hipStream_t stream) {
    if (kind == Kind::gdn_bf16)
        detail::enqueue_gdn(shape, input, weights, output, arm, stream);
    else
        detail::enqueue_lm(shape, input, weights, output, kind, arm, stream);
}
inline const void* kernel_symbol(Kind kind, Arm arm) {
    return kind == Kind::gdn_bf16 ? detail::gdn_symbol(arm) : detail::lm_symbol(kind, arm);
}
} // namespace strix::gdn_lm_wave8_diagnostic
