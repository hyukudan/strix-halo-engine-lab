#pragma once

// Per site x format x rows choice between the engine kernel of enqueue_q8_0_a8 and the multi-row form
// (hip_ds41_rn_wide.hpp) for the Q8_0 x A8 decode sites at 2-8 rows. Every choice gives the same bits; only the time
// differs. STRIX_DS41_RN_WIDE (read once):
//   0: engine kernel (the multi-row form is never used);
//   1: the multi-row form with the round-1 default shape for every site (rn_wide_default_shape);
//   2: this table (cold bench strix-hip-ds41-rn-sel-bench): per (outputs per group N, 32-weight groups Kg, format,
//      rows) either the engine kernel or one shape. Views not in the table take the =1 default shape.
//      The default since P13 (unset = 2).
// RnWideOff (scoped, per thread): mode 0 while it lives, so the prefill keeps its own kernels (P12: the multi-row
// form is -1 % there).
// STRIX_DS41_RN_SHAPE=W,waves,lds[,flat] forces one shape in modes 1 and 2 (A/B).

#include "strix/hip_ds41_rn_wide.hpp"

namespace strix::ds41 {

[[nodiscard]] int rn_wide_mode();
struct RnWideOff {
    RnWideOff();
    ~RnWideOff();
    RnWideOff(const RnWideOff&) = delete;
    RnWideOff& operator=(const RnWideOff&) = delete;
    bool previous;
};
// Mode 2's choice for a view with 2..8 rows: true and `shape` = the multi-row form to launch, false = the engine
// kernel. The caller still checks rn_wide_supported(view, shape).
[[nodiscard]] bool rn_table_select(const Q8A8View& view, RnWideShape& shape);

} // namespace strix::ds41
