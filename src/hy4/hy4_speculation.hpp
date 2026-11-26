#pragma once
// Hy4-flash decode route settings shared by the native rank (apps/hy4_rank_main.cpp) and the serve family
// (src/hy4/serve/session_service.cpp): the model only declares its speculative capability; the shared default
// resolver (strix/session/speculation_default.hpp) decides. Host-only.
#include "strix/session/speculation_default.hpp"

namespace strix::hy4 {

// Exact speculative decode, the same in every tier: the pack's next-token (MTP) block drafts through the shared MtpChain
// (copy / chain / hybrid arms, the Arbiter picks the rows), a verify pass of R rows (each row bitwise the tier's own step,
// the decode head copy included: hy4_hip_rows.inc linear_vrows) emits only the target's greedy ids, so the ids equal the
// tier's plain greedy decode. Hy4 only declares the capability; the shared default resolver decides
// (strix/session/speculation_default.hpp: --no-mtp / STRIX_MTP=0 opt out). Measured faster on the record route
// (strix-hy4-rows --mtp, 11-21, ids == plain): one node code 32.2 / prose 22.93 tok/s vs plain 17.9; TP pair code 47.24.
// The layer split holds no whole model (no block on either stage): unavailable there.
struct SpeculationRun {
    bool block_in_pack = false;          // GGUF <arch>.nextn_predict_layers > 0 (the block ships in the pack)
    bool whole_model = true;             // one node or a tensor-parallel rank (each holds the whole pack)
    bool no_mtp = false;                 // --no-mtp
    const char* environment = nullptr;   // STRIX_MTP
    bool explicit_mtp = false;           // --mtp
    bool per_step_route = false;         // the caller asked for every step's logits / forced tokens / stage timing
};
[[nodiscard]] inline ::strix::session::SpeculationCapability speculation_capability(bool available) {
    return {"hy4", available, true, true, false};
}
[[nodiscard]] inline ::strix::session::SpeculationDecision speculation(const SpeculationRun& run) {
    ::strix::session::SpeculationRequest request;
    request.cli_opt_out = run.no_mtp;
    request.environment = run.environment;
    request.explicit_speculation = run.explicit_mtp;
    request.explicit_route = run.per_step_route;
    return ::strix::session::resolve_speculation_default(speculation_capability(run.block_in_pack && run.whole_model), request);
}

}  // namespace strix::hy4
