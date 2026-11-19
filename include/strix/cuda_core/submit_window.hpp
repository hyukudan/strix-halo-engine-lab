#pragma once
// Bounded GPU submission + chunk ramp for large-M prefill loops (rtxlong3 11-16, after two RTX PRO losses during 27B long
// prefills that queued the WHOLE prompt, e.g. 128K = ~27K kernels / ~34 s of GPU work, before a single host sync).
//
// * SubmitWindow: before the host queues unit i it waits for the completion event of unit i - depth, so at most `depth`
//   units (prefill chunks) are ever queued ahead of the GPU. Scheduling only: the same kernels are queued in the same
//   order on the same streams with the same event dependencies, so results are bitwise identical. depth 0 = unbounded
//   (the previous form). Same idea as the Flash-Next block loop, which syncs every block (qwen_prefill_driver.cu, "block").
// * plan_chunks (on the shared strix/prefill_piece_plan.hpp): chunk boundaries for n rows with full chunks of C rows;
//   with ramp0 > 0 the first chunks grow ramp0, 2*ramp0, ... (< C) so the card steps up from idle instead of jumping to its full power at the first launch.
//   Rows are whole multiples of `align` (ramp chunks), the tail is whatever remains. Bitwise only where the model's
//   prefill is chunk-size invariant (the caller proves it with a logits hash; the 27B is, measured 3968..8192 rows).
// Host-only header: no CUDA types except cudaEvent_t / cudaEventSynchronize in SubmitWindow::admit.
#include "strix/prefill_piece_plan.hpp"

#include <cstdint>
#include <utility>
#include <vector>

#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>
#define STRIX_SUBMIT_WINDOW_CUDA 1
#endif

namespace strix::cuda_core {

using ChunkSpan = ::strix::prefill::PieceSpan;  // {pos, rows}: first row of the chunk (relative to the prompt start), rows >= 1

// n rows, full chunks of C rows (C >= 1); ramp0 = 0: [C, C, ..., tail]; ramp0 > 0: [r, 2r, 4r, ... (< C), C, ..., tail]
// with r = ramp0 rounded down to `align` (>= align). Never returns an empty chunk; n == 0 gives an empty plan.
// The shared prompt piece planner (strix/prefill_piece_plan.hpp) with its doubling_ramp schedule.
inline std::vector<ChunkSpan> plan_chunks(std::uint32_t n, std::uint32_t C, std::uint32_t ramp0 = 0, std::uint32_t align = 128) {
    if (n == 0 || C == 0) return {};
    return ::strix::prefill::piece_spans(::strix::prefill::greedy_pieces(n, ::strix::prefill::doubling_ramp(C, ramp0, align)));
}

#ifdef STRIX_SUBMIT_WINDOW_CUDA
// done[i] = event recorded after unit i's last GPU work. Call admit(done, i) BEFORE queuing unit i.
struct SubmitWindow {
    std::uint32_t depth = 0;  // 0 = unbounded
    std::uint32_t waits = 0;  // host waits issued (for the effective-config / receipt line)
    cudaError_t admit(const cudaEvent_t* done, std::uint32_t i) {
        if (depth == 0 || i < depth) return cudaSuccess;
        ++waits;
        return cudaEventSynchronize(done[i - depth]);
    }
};
#endif

}  // namespace strix::cuda_core
