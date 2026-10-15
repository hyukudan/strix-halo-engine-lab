#pragma once

#include "strix/shared_expert_rows_diagnostic_contract.hpp"
#include <hip/hip_runtime_api.h>

namespace strix::shared_expert_rows_diagnostic {
// Test-only whole shared operator. No allocation, clear, sync or publication.
// Caller owns every view and an explicit nondefault stream through completion,
// including after an enqueue exception. No decoder/catalog integration.
void enqueue(Shape, Span input, Weights, Span workspace, Span output, Arm, hipStream_t);
// Projection identities for future linked-ISA and captured-graph qualification.
// Gate and up use the same symbol; down uses a separate BF16-store symbol.
const void* projection_symbol(Arm, std::uint32_t rows, bool down);
} // namespace strix::shared_expert_rows_diagnostic
