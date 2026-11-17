#pragma once
// Qwen dense (qwen35) HIP speculative decode: ONE resolver for every entry point that drives the HipEngine
// (strix-hip-qwen35-decode, strix-hip-qwen35-spec; no server binds the qwen35 HipEngine).
//   MTP source  "auto" (default) = the model's own next-token-prediction layer when the GGUF carries it (the IU4 pack keeps
//               blk.<layers>.* as Q8_0: 0.45 GB per draft position instead of 0.86 GB for the bf16 side file), else none;
//               a path = that side GGUF; "off" / "none" = plain greedy decode.
//   draft rows  draft-only subset head (strix/spec/draft_rows.hpp, group 16 = one IU4 tile column), on whenever MTP
//               drafting is on; off = the full 248K-row head per draft position.
// Speculation is exact: verify windows are bitwise equal to row-by-row decode (hip_engine.hpp), and drafts (MTP layer
// precision, head subset) only steer how many tokens a window accepts, never which tokens are emitted.
// Home: shared host resolver of the qwen35 HIP family (both apps call it); the mechanisms are shared (strix/spec).
#include <cstddef>
#include <cstdint>
#include <string>

#include "strix/qwen35/hip_engine.hpp"
#include "strix/qwen35/iu4_draft_head.hpp"
#include "strix/spec/block_drafter.hpp"
#include "strix/spec/draft_rows.hpp"
#include "strix/spec/mtp_chain.hpp"

namespace strix::qwen35 {

struct HipDenseSpecPlan {
    std::string mtp_source;  // GGUF with the MTP layer; empty = no MTP drafting
    std::string mtp_reason;  // "model" | "file" | "off" | "absent"
    bool draft_rows = false;
};

// `mtp`: "auto" / "" / "off" / "none" / path; `draft_rows`: false = full head (off switch).
inline HipDenseSpecPlan resolve_hip_dense_spec(const std::string& model, const std::string& mtp, bool draft_rows) {
    HipDenseSpecPlan p;
    if (mtp == "off" || mtp == "none") {
        p.mtp_reason = "off";
    } else if (mtp.empty() || mtp == "auto") {
        if (HipEngine::has_mtp_layer(model)) {
            p.mtp_source = model;
            p.mtp_reason = "model";
        } else {
            p.mtp_reason = "absent";
        }
    } else {
        p.mtp_source = mtp;
        p.mtp_reason = "file";
    }
    p.draft_rows = draft_rows && !p.mtp_source.empty();
    return p;
}

// Subset rows of the draft head: the shared DraftRows defaults (core + fallback prefix, prompt and emitted ids) at the IU4
// tile-column granularity.
inline spec::DraftRowsConfig hip_dense_draft_rows_config(std::uint32_t vocab) {
    spec::DraftRowsConfig c;
    c.vocab = vocab;
    c.group = kIu4DraftGroup;
    return c;
}

// Shape of the verify-window cost (ms per pass at rows 1/2/4/8/12/16, Strix Halo gfx1151, 27B IU4 pack, 4.1K context;
// rows <= 8 are one weight stream, rows > 8 two). The arbiter takes it as the curve's shape; measured walls set the scale
// and replace it where measured (strix/spec/drafting.hpp Arbiter::seed_cost), so no timing runs at startup.
inline constexpr std::uint32_t kHipDenseVerifyRows[] = {1U, 2U, 4U, 8U, 12U, 16U};
inline constexpr double kHipDenseVerifyMs[] = {79.0, 81.3, 92.0, 123.7, 200.4, 220.7};

// Drafting block of the qwen35 HIP family: windows up to 16 rows (the engine's max), chain depth 7 (rows 9+ re-stream the
// weights: 200 ms at 12 rows), lookup + MTP chain + hybrid. Chain step seed = byte estimate per draft position (MTP layer +
// head at ~182 GB/s), replaced by the measured wall after the first steps.
inline spec::BlockConfig hip_dense_block_config(const HipDenseSpecPlan& plan) {
    spec::BlockConfig bc;
    bc.max_block = HipEngine::max_rows();
    bc.chain_max_k = 7U;
    bc.chain = bc.hybrid = !plan.mtp_source.empty();
    bc.chain_seed_step_ms = plan.draft_rows ? 4.0 : 9.0;
    return bc;
}

// Device glue of the shared MTP chain (strix/spec/mtp_chain.hpp) over HipEngine.
struct HipChainDev {
    static float* alloc(HipEngine& e, std::size_t n) { return e.alloc_f32(n); }
    static void release(HipEngine& e, float* p) { e.free_f32(p); }
    static void copy(HipEngine& e, float* d, const float* s, std::size_t n) { e.copy_f32_async(d, s, n); }
};
using HipMtpChain = spec::MtpChain<HipEngine, HipChainDev>;

}  // namespace strix::qwen35
