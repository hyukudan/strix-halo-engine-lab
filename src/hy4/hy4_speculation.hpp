#pragma once
// Hy4-flash decode route settings shared by the native rank (apps/hy4_rank_main.cpp) and the serve family
// (src/hy4/serve/session_service.cpp): the model only declares its speculative capability; the shared default
// resolver (strix/session/speculation_default.hpp) decides. Host-only.
#include "strix/session/speculation_default.hpp"
#include "strix/env_flags.hpp"

#include <cstdint>
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
// `--top 0` is the plain production form (device greedy row); any other --top asks for every step's logits.
// Dry-run callers only know the command spelling; the rank supplies the final parsed file state.
struct SpeculationRouteFiles {
    bool teacher_has_ids = true;
    bool dump_has_path = true;
};
[[nodiscard]] inline SpeculationRun speculation_run_from_arguments(const std::vector<std::string>& args,
                                                                   SpeculationRouteFiles files = {}) {
    SpeculationRun run;
    run.environment = ::strix::envcfg::lookup_unrecorded(::strix::session::speculation_environment);
    bool top_route = false;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        const bool more = i + 1 < args.size();
        if (::strix::session::speculation_opt_out_argument(arg)) run.no_mtp = true;
        else if (arg == "--mtp") run.explicit_mtp = true;
        else if (arg == "--timing" || arg == "--gate-ref") run.per_step_route = true;
        else if (arg == "--top" && more) {
            // Match the rank parser: numeric zero is zero, and the last --top wins.
            top_route = static_cast<std::uint32_t>(std::stoul(args[++i])) != 0U;
        } else if ((arg == "--teacher" || arg == "--dump-logits") && more) {
            run.per_step_route = run.per_step_route ||
                (arg == "--teacher" ? files.teacher_has_ids : files.dump_has_path);
            ++i;
        } else if (more && (::strix::session::speculation_neutral_value_option(arg) ||
                   arg == "--ids-file" || arg == "--gen" || arg == "--layers" || arg == "--context" ||
                   arg == "--prompt-vrows" || arg == "--verify-rows" || arg == "--state-in" ||
                   arg == "--state-out" || arg == "--until" || arg == "--budget" || arg == "--score" ||
                   arg == "--save-at" || arg == "--window-costs" || arg == "--cut" || arg == "--stage-map" ||
                   arg == "--repeat" || arg == "--port" || arg == "--pair-model" || arg == "--pair-receipt" ||
                   arg == "--pair-placement" || arg == "--verify-ms" || arg == "--tp-rank" ||
                   arg == "--tp-transport" || arg == "--tp-trip-us" || arg == "--tp-rates" || arg == "--pipe-guess")) {
            // A value (e.g. an ids file named --no-mtp) is not a route switch.
            ++i;
        }
    }
    run.per_step_route = run.per_step_route || top_route;
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
