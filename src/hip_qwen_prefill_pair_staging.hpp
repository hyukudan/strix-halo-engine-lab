#pragma once
#include "strix/qwen_prefill_handoff.hpp"
#include <hip/hip_runtime.h>
#include "xport/hip_pipe_staging.hpp"
#include "xport/hip_state_staging.hpp"

namespace strix::qwen_pp {
std::shared_ptr<PipeStaging> make_hip_pipe_staging(std::size_t capacity) {
    return pair::make_hip_pipe_staging(capacity, [] { return hip_execution_stream(); });
}

std::unique_ptr<StateStaging> make_hip_state_staging() {
    return pair::make_hip_state_staging(seam_piece_bytes);
}
} // namespace strix::qwen_pp
