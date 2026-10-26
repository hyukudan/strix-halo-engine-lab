#pragma once

// Multi-row form of the Q8_0 x A8 decode sites (STRIX_DS41_RN_WIDE=1, default off): 2-8 rows of a decode
// window (DSpark verify windows, batches) with every output bitwise the product of enqueue_q8_0_a8, i.e. of
// product_q8_0 (lane l takes the 32-weight groups l, l + 32, ... ascending, term float(integer dot) *
// (scale * delta), then the butterfly). Only the launch shape differs from enqueue_q8_0_a8:
//  - a block of `waves` waves owns waves * outputs_per_wave consecutive outputs of one group; each wave owns
//    outputs_per_wave of them and keeps the current and the next 32-weight group of every one in flight;
//  - the first group of every weight row is issued before the activations are staged (early), so the weight
//    stream starts at the top of the block instead of after the staging barrier;
//  - the block stages exactly `rows` activation rows of its group in LDS (rows * width codes, rows * width / 32
//    deltas), or reads them in place from the caches (lds = false);
//  - the row count is a template parameter (no masked rows, fewer registers).
// Same view contract as enqueue_q8_0_a8 (per_group, x_stride, out_stride, first, row_width / k_offset, accumulate,
// packed 0 / 4 / 5 / 6). Not planar.

#include "strix/hip_ds41_layer.hpp"

#include <cstdint>

namespace strix::ds41 {

struct RnWideShape {
    std::uint32_t waves = 16;            // 8 or 16
    std::uint32_t outputs_per_wave = 2;  // 1, 2 or 4
    bool lds = true;                     // stage the block's activation rows in LDS
    bool flat = false;                   // flat form: (output, group) items over all lanes, terms summed from LDS
    // 0: the form given by `flat`; 2: flat with wave-level sync; 3 / 4: segments of 8 / 4 lanes per output, terms in
    // registers (attn_q_b's 40 groups only; outputs_per_wave = 4 / 8 segments x outputs per lane). Q8_0 and U6.
    std::uint32_t form = 0;
};
// STRIX_DS41_RN_WIDE=1 (read once).
[[nodiscard]] bool rn_wide_enabled();
// The shape used by enqueue_q8_0_a8 when the flag is on (STRIX_DS41_RN_SHAPE=W,waves,lds[,form] overrides, e.g. "1,16,1",
// "4,4,0,1" (flat) or "4,8,0,3" (segments of 8 lanes); form 1 = flat).
[[nodiscard]] RnWideShape rn_wide_default_shape(const Q8A8View& view);
// True when the view can take this form with `shape` (rows 1..8, a whole block inside one group, LDS fits).
[[nodiscard]] bool rn_wide_supported(const Q8A8View& view, const RnWideShape& shape);
// Every shape gives the same bits. Shapes outside the registered set throw (P 4 / 5 only take the default ones).
void enqueue_q8_0_a8_rn(const Q8A8View& view, const RnWideShape& shape, void* stream);
// Same selected form and arithmetic, with coherent producer completion. False
// means no launch (the caller must retain its unfused fallback).
[[nodiscard]] bool enqueue_q8_0_a8_rn_ready(const Q8A8View& view, const RnWideShape& shape,
                                          dist::ProducerStage stage, void* stream);

} // namespace strix::ds41
