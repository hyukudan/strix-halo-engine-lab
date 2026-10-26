#pragma once
// Exact faster forms of the GLM post-exchange consumers (design notes
// section 3, measured timings). Each computes every value with the SAME sequence of IEEE operations as the
// Tier A kernel it replaces (src/glm/kernels/glm_tier_a.hip), only the memory access and the work distribution change;
// gated bit for bit against Tier A in glmxch-bench `gates`.
//   hc_pre_v3: dev::hc_pre. The 24 R1 chains (ascending fmaf over 16384 values, one accumulator per mix) are fed by
//     variant 0: each chain lane streams its fn row with float4 loads, one 16-value block prefetched ahead;
//     variant 1: double-buffered LDS tiles (24 rows x 256 values + the F tile) loaded by waves 1..7 while wave 0's lanes
//                0..23 run the chains over the previous tile (one LDS read per step: measured 669 us);
//     variant 2: lds_chains: 4-stage 24 x 128 tiles loaded 3 tiles ahead, chains read register batches of 16 values with
//                the next batch prefetched (measurement: 231 us: each tile still waits one global latency at its barrier);
//     variant 3: 1024-thread block, lds_chains_deep: every loader keeps 8 tiles of its item in flight in registers.
//     R6 on one wave as hc_pre_v2; the collapse as Tier A.
//   router_dense: dev::dense(x, router, F16, 4096, 0, 288, logits) (R1, F16 weights converted exactly), LDS tiles of 32
//     outputs x 256 inputs per block, chains on wave 0.
//   route_v2: dev::route (R5 key = ordered_key(sigmoid(l) + bias), top-8 by (key desc, index asc), ids ascending,
//     R14 den = ((0 + s_0) + ...) + eps in ascending id, w = (s / den) * 2.5): every expert's rank is counted in
//     parallel instead of 8 sequential argmax scans on one lane; the same set, order and arithmetic.
#include <hip/hip_runtime.h>

#include <cstdint>

namespace strix::glm::xfer {

void hc_pre_v3(const float* X, const float* fn, const float* base, const float* scale, float* F, float* mix, float* xc,
               unsigned rows, hipStream_t s, int variant = 7);
// hc_pre_v3 variant 4 = variant 3 with LDS-only tile barriers: measured 102 us at rows 1-8 (variant 3: 117-119, Tier A 371).
// 5/6/7 (design notes) = 4 + consumer LDS prefetch pinned 1 batch ahead / + across the tile barrier / 3 batches ahead
// across the barrier: measured ABA (4,V,4 x3) 81.4-81.7 / 77.8-78.4 / 77.4-77.9 us at rows 1-8. Default 7.
// router_dense variant 0: 2-stage tiles, one LDS read per chain step (measured 260 us); 1: lds_chains, 3 stages, batches of
// 16 (156 us); 2: 1024 threads, lds_chains_deep (measured rows 1/2/4/8: 51/54/54/83 us); 3: 36 blocks x 8 outputs with
// LDS-only tile barriers (38/45/62/109 us). Default -1 = auto: 3 for rows <= 2, else 2 (all variants bitwise == Tier A).
void router_dense(const float* x, const std::uint16_t* router, float* logits, unsigned rows, hipStream_t s, int variant = -1);
// Diagnostic (timing only): hc_pre's 24 consumer chains fed from one preloaded LDS stage, no barriers; pf 0 / 1 / 3.
void hc_chains_lds_only(const float* F, const float* fn, float* z, hipStream_t s, int pf);
void route_v2(const float* logits, const float* bias, std::uint32_t* ids, float* w, unsigned rows, hipStream_t s);

}  // namespace strix::glm::xfer
