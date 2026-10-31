#pragma once
#include "hip_decode_options.hpp"
#include "strix/moe_route_bf16.hpp"
#include "strix/serve/json.hpp"

namespace strix::decode_cli {
// Candidate storage arithmetic is resolved after the ordinary prefill owner, before device setup.
template<class Environment>
inline void resolve_moe_route_bf16(Options& o, const Environment& env, bool gsq, bool hip_fp16, std::string& error) {
    const auto value = [&](const char* name) -> std::string_view {
        const auto it = env.find(name); return it == env.end() ? std::string_view{} : it->second;
    };
    const auto requested = value("STRIX_QPF2K_MOE_ROUTE_BF16");
    const bool on = o.prefill_moe_route_bf16_requested.value_or(!requested.empty() && requested.front() != '0');
    o.prefill_moe_route_bf16 = false;
    auto& reason = o.prefill_moe_route_bf16_reason;
    reason = "off";
    if (!on) return;
    if (o.baseline || value("STRIX_BASELINE") == "1") {
        reason = "baseline"; return;
    }
    if (value("STRIX_QPF2K_DEFAULTS").starts_with('0')) { reason = "STRIX_QPF2K_DEFAULTS=0"; return; }
    const auto decline = [&](const char* why) {
        reason = why; error = "MoE route BF16 candidate declined: " + reason;
    };
    if (!hip_fp16) return decline("HIP FP16 prefill required; CUDA is unsupported");
    if (gsq) return decline("GSQ writers are unsupported");
    const auto pair = numerical_profile_pair(o);
    if (!numerical_profile_fast_prefill(pair.prefill) || !(pair.fast_components & kFastPrefillMoe) ||
        !o.known_prefix_window || !o.known_prefix_full_rows || !o.known_prefix_expert_union ||
        o.known_prefix_prefill_rows < 16U || o.known_prefix_prefill_rows % 8U || !o.disabled_prefill_features.empty())
        return decline("full-row known-prefix fast FP16 MoE prefill required; decode and known-rows are unsupported");
    if (o.mtp.enabled || o.mtp.supplied || !o.mtp.cohort.empty() || o.http_serve || o.http_batch ||
        !o.serving_prefix_files.empty() || o.serving_concurrency != 1U || o.serving_tensor_cohort_rows != 1U ||
        o.serving_prefill_rows || o.prompt_lookup || o.prefix_cache_host_bytes || o.prefix_cache_roundtrip ||
        !o.session_load.empty() || !o.session_save.empty() || o.session_tiled_suffix ||
        o.session_live_control || o.session_management_requested || !o.mtp.copy_wide_rows.empty())
        return decline("MTP, nested, serving, cache and session routes are unsupported");
    for (const auto rows : o.known_prefix_ladder_tail_rows)
        if (rows < 16U || rows % 8U) return decline("known-rows tail writers are unsupported");
    for (const auto rows : o.known_prefix_tail_rows)
        if (rows < 16U || rows % 8U) return decline("known-rows tail writers are unsupported");
    if (o.rotary.mode() != RotaryMode::native)
        return decline("native rotary required");
    const auto require = [&](const char* name, std::string_view expected) {
        const auto it = env.find(name); return it == env.end() || std::string_view(it->second) == expected;
    };
    if (!require("STRIX_QPF2K_MOE_DA16", "1") || !require("STRIX_QPF2K_MOE_ROUTE", "14") ||
        !require("STRIX_QPF2K_MOE_MC", "1") || !require("STRIX_QPF2K_MOE", "0") ||
        !require("STRIX_QPF2K_MOE_SMALL", "0") || !require("STRIX_INTPREFILL_MOE_TUNE", "82"))
        return decline("requires DA16=1, ROUTE=14, MC=1, core=0, SMALL=0 and tune=82 (effective 80/82/88); other writers/readers unsupported");
    o.prefill_moe_route_bf16 = true;
    reason = "candidate; needs qualification; DA16 certificate 80/82/88 and ordered scalar/vector merge/commit";
}
inline void moe_route_bf16_capabilities(serve::Json& out) {
    out.set("prefill_moe_route_bf16", serve::Json(serve::Json::Object{{"supported", false}, {"default", false},
        {"qualification", "candidate"}, {"reason", "known-prefix CLI only; serving excluded"}}));
}
} // namespace strix::decode_cli
