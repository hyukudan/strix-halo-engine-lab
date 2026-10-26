#pragma once
// minimax-m3 fused post-exchange kernel (gfx1151): residual + part0 + part1, Gemma norm, optional router top-4.
// Equals strix/minimax_moe.hpp (post_exchange_row, route_row) bit for bit. One 256-thread block per row; rows are
// independent (batch invariance). No inter-block flags: completion is the kernel boundary on the stream.
#include <cstdint>

#include <hip/hip_runtime.h>

#include "strix/minimax_moe.hpp"

namespace strix::minimax::moe::hip {

struct PostExchangeView {
    std::uint32_t rows;
    float* h;                   // R x 6144, residual (updated in place)
    const float* part0;         // R x 6144, rank 0's partial
    const float* part1;         // R x 6144, rank 1's partial
    const float* norm_w;        // 6144 stored Gemma weights
    float* y;                   // R x 6144, normed output
    const float* router_w;      // 128 x 6144 FP32, or null (no routing: attention exchange / dense layers)
    const float* bias;          // 128
    std::uint32_t* ids;         // R x 4
    float* weights;             // R x 4
    // Optional in-kernel wait (exchange module recv_view, STRIX_MM_XB_WAIT): before reading part0 / part1 every block
    // polls until *rx_flag >= rx_seq (system-scope acquire). nullptr: the stream already waited (today's path).
    const std::uint64_t* rx_flag = nullptr;
    std::uint64_t rx_seq = 0;
    // Optional fold of the next linear's input transform (STRIX_MM_XB_PREP): for each of prep_jobs (<= 4) matrices with
    // 6144 inputs, the normed row y is transformed with its sign vector prep_suh[i] into prep_xt[i] + r * 6144 (halves),
    // exactly as the linears' prepare launch does (same device function), so that call skips its prepare launch.
    // STRIX_MM_XB_POSTILP: issue every load of h / part0 / part1 before the first store (h must not overlap a partial;
    // checked at enqueue). Same arithmetic, same bits.
    unsigned ilp = 0;  // 0 plain, 1 loads-first, 2 partials staged through LDS with 16-byte loads (same bits)
    unsigned prep_jobs = 0;
    const std::uint16_t* prep_suh[4] = {nullptr, nullptr, nullptr, nullptr};
    void* prep_xt[4] = {nullptr, nullptr, nullptr, nullptr};  // _Float16 rows
};
void enqueue_post_exchange(const PostExchangeView& v, hipStream_t stream);

// Routes normalized global y only; batch_rows is 4 or 8. Tail rows are masked.
void enqueue_router_batch(std::uint32_t rows, const float* y, const float* router_w, const float* bias,
                          std::uint32_t* ids, float* weights, unsigned batch_rows, hipStream_t stream);

}  // namespace strix::minimax::moe::hip
