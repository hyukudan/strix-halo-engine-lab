#pragma once
// Hy4-flash decode route settings shared by the native rank (apps/hy4_rank_main.cpp) and the serve family
// (src/hy4/serve/session_service.cpp): the model only declares its speculative capability; the shared default
// resolver (strix/session/speculation_default.hpp) decides. Host-only.
#include "strix/session/speculation_default.hpp"

#include <string>
#include <string_view>
#include <vector>

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
// The decode-route facts the rank's argument list spells. Only the options named here own a route; the tier, the KV
// storage, the prompt, the generation budget and the context are route-neutral (a launcher adds them to every command).
// `--top 0` (any spelling of the number zero) is the plain production form (device greedy row); any other --top asks for
// every step's logits. `--teacher F` asks for the per-step route (the rank refuses a teacher file without ids).
// `environment` is the caller's STRIX_MTP value (read through the shared environment home, never here).
[[nodiscard]] inline bool top_asks_logits(std::string_view value) {
    return value.empty() || value.find_first_not_of('0') != std::string_view::npos;
}
[[nodiscard]] inline SpeculationRun speculation_run_from_arguments(const std::vector<std::string>& args,
                                                                   const char* environment) {
    SpeculationRun run;
    run.environment = environment;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (::strix::session::speculation_opt_out_argument(arg)) run.no_mtp = true;
        else if (arg == "--mtp") run.explicit_mtp = true;
        else if (arg == "--teacher" || arg == "--dump-logits" || arg == "--timing" || arg == "--gate-ref") run.per_step_route = true;
        else if (arg == "--top" && i + 1 < args.size() && top_asks_logits(args[i + 1])) run.per_step_route = true;
    }
    return run;
}
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
