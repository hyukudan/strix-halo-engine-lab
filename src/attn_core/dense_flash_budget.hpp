#pragma once
// Occupancy target of the shared dense flash prefill core (dense_flash_prefill.hpp: flash_prefill<P>), host-safe
// arithmetic. Placement only: the target changes register allocation, never the arithmetic (same bits per row).
//
// live = the registers the key loop keeps: the Q fragments (one per term) and the O accumulators (8 VGPRs per 16-wide
// fragment each), the K/V prefetch of the next tile (two 16-B pieces of 4 VGPRs per staged piece and term) and the working
// registers (scores, P, softmax state, addresses: 72 + 16 per term). The target is the highest occupancy whose VGPR
// budget holds live (core/vgpr_budget.hpp), at most 8 waves: a spill in the key loop costs more than one resident
// work-group, because its reload waits vmcnt(0), which (in order) also waits for the next tile's prefetch issued just
// before it, so every key tile pays a full memory round trip. Exception: when the 8-wave cap doubles the resident
// work-groups per WGP (4 SIMDs; 16-wave groups: two instead of one) the target stays 8 (release toolchain, D 512
// group 4: 4.77 ms at 8 waves, 5.01 ms at 6). Measured otherwise (tests/hip_dense_flash_bench.hip, gfx1151, release
// toolchain, same output bits): MHA D 256 -22 %, group 6 -12 %, two-term D 80 -34 %, two-term D 64 within 3 %; D 256
// group 4 -6 % in the bench but 0.6 % slower in the Gemma4 split-K prefill, which keeps 8 (FlashPolicy::kWavesPerEu).
#include "core/vgpr_budget.hpp"

namespace strix::attn_core::dense {

// frags = 16-wide head-dim fragments per wave (Shape::frags), per_thread = staged 16-B K and V pieces per thread
// (Shape::per_thread), terms = FP16 terms per operand (1, or 2 for hi + lo).
constexpr unsigned live_vgprs(unsigned frags, unsigned per_thread, unsigned terms) {
    return 8U * frags * (terms + 1U) + 8U * per_thread * terms + 72U + 16U * terms;
}
// waves = waves of one work-group (Shape::waves).
constexpr unsigned occupancy_target(unsigned frags, unsigned per_thread, unsigned terms, unsigned waves) {
    const unsigned fit = ::strix::core::vgpr_budget::waves_for(live_vgprs(frags, per_thread, terms));
    if (fit >= 8U) return 8U;
    const unsigned per_simd = (waves + 3U) / 4U;  // waves of one group on each SIMD of the WGP
    return 8U / per_simd >= 2U * (fit / per_simd) ? 8U : fit;
}

}  // namespace strix::attn_core::dense
