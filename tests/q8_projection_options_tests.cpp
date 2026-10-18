#include "../apps/hip_decode_options.hpp"

#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strix;
using namespace strix::decode_cli;
namespace {
int failures = 0;
void expect(bool good, const char* message) {
    if (!good) { ++failures; std::cerr << message << '\n'; }
}
Options parse(std::initializer_list<const char*> args) {
    std::vector<std::string> storage;
    for (const auto* arg : args) storage.emplace_back(arg);
    std::vector<char*> argv;
    for (auto& arg : storage) argv.push_back(arg.data());
    return parse_options(static_cast<int>(argv.size()), argv.data());
}
template<class F> void rejects(F&& action, const char* message) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, message);
}
}
int main() {
    const auto model = ModelContract::qwen38_flash_next();
    gguf::FileInfo info;
    ModelManifest manifest; manifest.architecture = "qwen3next";
    manifest.layout_identity = model_manifest_layout_identity(manifest);
    auto defaults = parse({"decoder", "m", "1"});
    expect(!defaults.prefetch_q8_lm_head && !defaults.wave8_q8_lm_head && !defaults.wave8_q8_gdn_input &&
        !defaults.wave8_q8_gdn_output && !defaults.wave8_q8_qsa_output,
        "default never selects a new organization");
    const auto resolve = [&](Options& options, bool has_a8) {
        return resolve_optimization_profile(options, model, info, manifest,
            validated_model_content_identity, has_a8);
    };
    expect(!defaults.prefetch_hc_projection_next_tile, "HC next-tile overlay defaults off");
    expect(!defaults.q51_slot_parallel_down, "slot-parallel Q5_1 overlay defaults off");
    auto q51 = parse({"decoder", "m", "1", "--q51-slot-parallel-down"});
    expect(!q51.any_low_level_flag && resolve(q51, true).automatic_applied,
        "slot-parallel Q5_1 overlay retains automatic base prerequisites");
    validate_resolved_q51_slot_parallel_down(q51);
    for (auto member : {&Options::grouped_exact_q51_down, &Options::q4k_a8g32_gate_up}) {
        auto missing = q51; missing.*member = false;
        rejects([&] { validate_resolved_q51_slot_parallel_down(missing); },
            "slot-parallel Q5_1 rejects each missing resolved prerequisite");
    }
    auto q51_fallback = parse({"decoder", "m", "1", "--q51-slot-parallel-down"});
    expect(resolve(q51_fallback, false).reason == "a8-runtime-not-built", "Q5_1 overlay preserves automatic fallback");
    rejects([&] { validate_resolved_q51_slot_parallel_down(q51_fallback); }, "Q5_1 fallback cannot invent A8 prerequisites");
    auto q51_manual = parse({"decoder", "m", "1", "--optimization-profile", "manual", "--q51-slot-parallel-down"});
    (void)resolve(q51_manual, false);
    rejects([&] { validate_resolved_q51_slot_parallel_down(q51_manual); }, "manual slot-parallel Q5_1 requires explicit grouped A8");
    q51_manual.grouped_exact_q51_down = true;
    q51_manual.q4k_a8g32_gate_up = true;
    validate_resolved_q51_slot_parallel_down(q51_manual);
    rejects([] { (void)parse({"decoder", "m", "1", "--q51-slot-parallel-down", "--q51-slot-parallel-down"}); }, "duplicate Q5_1 overlay rejects");
    rejects([] { (void)parse({"decoder", "m", "1", "--q51-slot-parallel-down", "--execution-mode", "graph"}); }, "Q5_1 overlay excludes graph before model IO");
    rejects([] { (void)parse({"decoder", "m", "1", "--q51-slot-parallel-down", "--integer-lm-head", "a8-iu8"}); }, "Q5_1 overlay excludes integer head before model IO");
    auto hc_next = parse({"decoder", "m", "1", "--prefetch-hc-projection-next-tile"});
    expect(!hc_next.any_low_level_flag && resolve(hc_next, true).automatic_applied,
        "HC next-tile overlays auto without disabling its prerequisites");
    validate_resolved_hc_projection_next_tile(hc_next);
    for (auto member : {&Options::parallel_hyper_mix, &Options::staged_hyper_projections,
                        &Options::fused_staged_hyper_mix, &Options::typed_hyper_projection,
                        &Options::prefetch_typed_hyper_rows}) {
        auto missing = hc_next; missing.*member = false;
        rejects([&] { validate_resolved_hc_projection_next_tile(missing); },
            "HC next-tile rejects each unresolved prerequisite");
    }
    auto hc_fallback = parse({"decoder", "m", "1", "--prefetch-hc-projection-next-tile"});
    expect(resolve(hc_fallback, false).reason == "a8-runtime-not-built", "HC next-tile preserves automatic fallback reason");
    rejects([&] { validate_resolved_hc_projection_next_tile(hc_fallback); }, "fallback cannot invent HC prerequisites");
    auto hc_manual_missing = parse({"decoder", "m", "1", "--optimization-profile", "manual", "--prefetch-hc-projection-next-tile"});
    (void)resolve(hc_manual_missing, false);
    rejects([&] { validate_resolved_hc_projection_next_tile(hc_manual_missing); }, "manual HC overlay needs explicit prerequisites");
    auto hc_manual = parse({"decoder", "m", "1", "--optimization-profile", "manual",
        "--parallel-hyper-mix", "--staged-hyper-projections", "--fused-staged-hyper-mix",
        "--typed-fused-hyper-projections", "--prefetch-typed-hyper-rows", "--prefetch-hc-projection-next-tile"});
    expect(!resolve(hc_manual, false).automatic_applied, "manual HC overlay preserves manual mode");
    validate_resolved_hc_projection_next_tile(hc_manual);
    rejects([] { (void)parse({"decoder", "m", "1", "--prefetch-hc-projection-next-tile", "--prefetch-hc-projection-next-tile"}); }, "duplicate HC overlay rejects");
    rejects([] { (void)parse({"decoder", "m", "1", "--prefetch-hc-projection-next-tile", "--execution-mode", "graph"}); }, "HC overlay rejects graph before model IO");
    rejects([] { (void)parse({"decoder", "m", "1", "--prefetch-hc-projection-next-tile", "--integer-lm-head", "a8-iu8"}); }, "HC overlay rejects integer head before model IO");
    for (const bool wave8 : {false, true}) {
        auto candidate = wave8 ? parse({"decoder", "m", "1", "--prefetch-q8-lm-head", "--wave8-q8-lm-head", "--wave8-q8-gdn-input"})
            : parse({"decoder", "m", "1", "--prefetch-q8-lm-head"});
        expect(!candidate.any_low_level_flag, "new overlays preserve automatic base selection");
        const auto selected = resolve(candidate, true);
        validate_resolved_q8_projection_organization(candidate);
        expect(selected.automatic_applied && candidate.virtual_q8_lm_head &&
            candidate.prefetch_q8_lm_head && candidate.wave8_q8_lm_head == wave8,
            "explicit overlay composes with automatic base without claiming its qualification");
    }
    auto no_a8 = parse({"decoder", "m", "1", "--prefetch-q8-lm-head"});
    expect(resolve(no_a8, false).reason == "a8-runtime-not-built", "missing capability retains original automatic fallback");
    rejects([&] { validate_resolved_q8_projection_organization(no_a8); }, "fallback cannot silently supply missing virtual LM prerequisite");
    auto manual = parse({"decoder", "m", "1", "--optimization-profile", "manual", "--virtual-q8-lm-head",
        "--prefetch-q8-lm-head", "--wave8-q8-lm-head", "--virtual-q8-gdn-input", "--prefetch-q8-gdn-input", "--wave8-q8-gdn-input"});
    expect(!resolve(manual, false).automatic_applied, "manual Q8 candidates need no A8 build");
    validate_resolved_q8_projection_organization(manual);
    rejects([] { (void)parse({"decoder", "m", "1", "--wave8-q8-lm-head"}); }, "wave8 LM requires explicit prefetch option");
    for (const auto* flag : {"--prefetch-q8-lm-head", "--wave8-q8-lm-head", "--wave8-q8-gdn-input",
                             "--wave8-q8-gdn-output", "--wave8-q8-qsa-output"})
        rejects([&] { (void)parse({"decoder", "m", "1", flag, flag}); }, "duplicate organization option rejects");
    rejects([] { (void)parse({"decoder", "m", "1", "--prefetch-q8-lm-head", "--execution-mode", "graph"}); }, "graph rejects before model IO");
    rejects([] { (void)parse({"decoder", "m", "1", "--wave8-q8-gdn-input", "--integer-lm-head", "a8-iu8"}); }, "integer head excluded before model IO");
    auto missing_gdn = parse({"decoder", "m", "1", "--optimization-profile", "manual", "--wave8-q8-gdn-input"});
    (void)resolve(missing_gdn, false);
    rejects([&] { validate_resolved_q8_projection_organization(missing_gdn); }, "manual GDN requires both existing input prerequisites");
    auto output_auto = parse({"decoder", "m", "1", "--wave8-q8-gdn-output", "--wave8-q8-qsa-output"});
    expect(!output_auto.any_low_level_flag, "output organizations overlay the automatic base");
    expect(resolve(output_auto, true).automatic_applied, "output organizations retain qualified automatic base");
    validate_resolved_q8_projection_organization(output_auto);
    expect(output_auto.wave8_q8_gdn_output && output_auto.wave8_q8_qsa_output &&
        output_auto.virtual_q8_gdn_output, "automatic GDN prerequisite resolves before output validation");
    auto output_manual = parse({"decoder", "m", "1", "--optimization-profile", "manual",
        "--virtual-q8-gdn-output", "--wave8-q8-gdn-output", "--wave8-q8-qsa-output"});
    expect(!resolve(output_manual, false).automatic_applied, "manual output modes preserve manual selection");
    validate_resolved_q8_projection_organization(output_manual);
    auto missing_output = parse({"decoder", "m", "1", "--optimization-profile", "manual",
        "--wave8-q8-gdn-output"});
    (void)resolve(missing_output, false);
    rejects([&] { validate_resolved_q8_projection_organization(missing_output); },
        "manual output organization requires explicit virtual output");
    auto output_no_a8 = parse({"decoder", "m", "1", "--wave8-q8-gdn-output"});
    expect(resolve(output_no_a8, false).reason == "a8-runtime-not-built", "output overlay preserves base fallback");
    rejects([&] { validate_resolved_q8_projection_organization(output_no_a8); },
        "fallback cannot manufacture GDN output prerequisite");
    auto qsa_manual = parse({"decoder", "m", "1", "--optimization-profile", "manual",
        "--wave8-q8-qsa-output"});
    (void)resolve(qsa_manual, false);
    validate_resolved_q8_projection_organization(qsa_manual);
    for (const auto* flag : {"--wave8-q8-gdn-output", "--wave8-q8-qsa-output"}) {
        rejects([&] { (void)parse({"decoder", "m", "1", flag, "--execution-mode", "graph"}); },
            "output organizations exclude graph before model IO");
        rejects([&] { (void)parse({"decoder", "m", "1", flag, "--integer-lm-head", "a8-iu8"}); },
            "output organizations exclude integer head before model IO");
    }
    rejects([&] { (void)parse({"decoder", "m", "1", "--execution-telemetry-trace"}); },
        "bounded trace requires explicit observer");
    auto observed = parse({"decoder", "m", "1", "--host-dispatch-telemetry", "--execution-telemetry-trace"});
    expect(observed.host_dispatch_telemetry && observed.execution_telemetry_trace &&
        observed.optimization_profile == OptimizationProfile::automatic, "observer does not change automatic profile");
    auto tiled = parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-prefill-rows=17",
        "--known-prefix-workspace-budget-bytes", "4096", "--known-prefix-host-metadata-budget-bytes", "2048",
        "--known-prefix-uma-budget-bytes", "8192", "--known-prefix-host-headroom-bytes", "0"});
    expect(tiled.known_prefix_prefill_rows == 17 && tiled.known_prefix_rows == 2 && !tiled.known_prefix_rows_explicit,
        "explicit T supports odd tails without relabelling the small-row selector");
    for (const auto rows : {0U, 15U, 8193U}) {
        auto bad = tiled; bad.known_prefix_prefill_rows = rows;
        rejects([&] { validate_known_prefill_options(bad); }, "prefill closes T bounds");
    }
    for (unsigned missing = 0; missing < 4; ++missing) {
        auto bad = tiled;
        if (missing == 0) bad.known_prefill_workspace_explicit = false;
        if (missing == 1) bad.known_prefill_metadata_explicit = false;
        if (missing == 2) bad.known_prefill_uma_explicit = false;
        if (missing == 3) bad.known_prefill_headroom_explicit = false;
        rejects([&] { validate_known_prefill_options(bad); }, "every prefill reservation input is explicit");
    }
    { auto bad = tiled; bad.known_prefix_rows_explicit = true;
      rejects([&] { validate_known_prefill_options(bad); }, "small and T selectors cannot silently override each other"); }
    { auto bad = tiled; bad.known_prefill_metadata_budget = 1;
      rejects([&] { validate_known_prefill_options(bad); }, "metadata budget admits both retained payload copies"); }
    { auto bad = tiled; bad.prompt_lookup = true;
      rejects([&] { validate_known_prefill_options(bad); }, "lookup verification requires all-row small windows"); }
    { auto only_state = tiled; only_state.known_prefix_output_policy = KnownWindowOutputPolicy::state_only;
      only_state.generate_tokens = 0; validate_known_prefill_options(only_state);
      only_state.generate_tokens = 1;
      rejects([&] { validate_known_prefill_options(only_state); }, "T state-only never fabricates terminal logits"); }
    auto cohort = parse({"decoder", "m", "--serve-prefix-u32le", "p", "--serve-concurrency", "4",
        "--serve-tensor-cohort-rows=2", "--serve-memory-budget-bytes", "1000000",
        "--serve-request-budget-bytes", "1000", "--serve-headroom-bytes", "0", "--generate", "1"});
    expect(cohort.serving_tensor_cohort_rows == 2 && cohort.serving_tensor_cohort_explicit,
        "tensor cohort is a separate explicit overlay, not automatic concurrency");
    cohort.deferred_moe_status = true;
    validate_resolved_serving(cohort);
    rejects([] { (void)parse({"decoder", "m", "1", "--serve-tensor-cohort-rows", "2"}); },
        "tensor cohort cannot be silently ignored by normal decoder");
    rejects([] { (void)parse({"decoder", "m", "--serve-prefix-u32le", "p", "--serve-tensor-cohort-rows", "3"}); },
        "tensor cohort admits exactly two or four");
    rejects([] { (void)parse({"decoder", "m", "--serve-prefix-u32le", "p", "--serve-tensor-cohort-rows", "2",
        "--serve-tensor-cohort-rows=4"}); }, "tensor cohort duplicate rejects instead of overriding");
    { auto bad = cohort; bad.serving_tensor_cohort_rows = 4; bad.serving_concurrency = 2;
      rejects([&] { validate_resolved_serving(bad); }, "tensor cohort width cannot exceed request concurrency"); }
    auto compact_snapshot = parse({"decoder", "m", "1", "--prompt-lookup",
        "--prompt-lookup-snapshot=occupied-kv-v1", "--prompt-lookup-snapshot-bytes", "65536"});
    expect(compact_snapshot.prompt_lookup_snapshot == SessionSnapshotPolicy::occupied_kv_v1 &&
        compact_snapshot.prompt_lookup_snapshot_bytes == 65536U, "compact lookup snapshot is explicitly budgeted");
    expect(defaults.prompt_lookup_snapshot == SessionSnapshotPolicy::full_components &&
        !defaults.prompt_lookup_snapshot_bytes, "snapshot default remains complete legacy state");
    rejects([] { (void)parse({"decoder", "m", "1", "--prompt-lookup-snapshot", "occupied-kv-v1",
        "--prompt-lookup-snapshot-bytes", "4"}); }, "snapshot option cannot be ignored by non-lookup route");
    rejects([] { (void)parse({"decoder", "m", "1", "--prompt-lookup", "--prompt-lookup-snapshot", "occupied-kv-v1"}); },
        "compact snapshot requires positive explicit payload budget");
    rejects([] { (void)parse({"decoder", "m", "1", "--prompt-lookup", "--prompt-lookup-snapshot-bytes", "4"}); },
        "snapshot budget never silently selects compact policy");
    rejects([] { (void)parse({"decoder", "m", "1", "--prompt-lookup", "--prompt-lookup-snapshot", "unknown"}); },
        "snapshot format is a closed enum");
    rejects([] { (void)parse({"decoder", "m", "1", "--prompt-lookup", "--prompt-lookup-snapshot", "occupied-kv-v1",
        "--prompt-lookup-snapshot-bytes", "0"}); }, "compact snapshot rejects zero budget");
    auto token_cost = parse({"decoder", "m", "1", "--generate", "4", "--deferred-moe-status",
        "--moe-shared-overlap", "--moe-overlap-arena-budget-bytes", "1024", "--moe-token-cost-policy"});
    expect(token_cost.moe_token_cost_policy && token_cost.moe_shared_overlap,
        "observed token policy overlays the already requested overlap owner");
    expect(!defaults.moe_token_cost_policy, "online observed policy is default off");
    for (unsigned missing = 0; missing < 3U; ++missing) {
        auto bad = token_cost;
        if (!missing) bad.generate_tokens = 0;
        if (missing == 1U) bad.deferred_moe_status = false;
        if (missing == 2U) bad.moe_shared_overlap = false;
        rejects([&] { validate_moe_token_cost_options(bad); }, "observed policy never turns on its prerequisites");
    }
    { auto bad = token_cost; bad.prompt_lookup = true;
      rejects([&] { validate_moe_token_cost_options(bad); }, "lookup cannot share observed-token owner"); }
    { auto bad = token_cost; bad.known_prefix_window = true;
      rejects([&] { validate_moe_token_cost_options(bad); }, "known windows cannot masquerade as serial token samples"); }
    {
        auto prefill=cohort;prefill.serving_tensor_cohort_rows=1;prefill.serving_tensor_cohort_explicit=false;
        prefill.serving_prefill_explicit=prefill.serving_prefill_workspace_explicit=prefill.serving_prefill_metadata_explicit=true;
        prefill.serving_prefill_workspace_budget=1ULL<<30;prefill.serving_prefill_metadata_budget=1ULL<<24;
        prefill.wave8_q8_gdn_input=prefill.wave8_q8_gdn_output=true;
        prefill.virtual_q8_gdn_input=prefill.prefetch_q8_gdn_input=prefill.virtual_q8_gdn_output=true;
        for(const auto rows:{2U,4U,8U,16U,17U,2048U}) {
            prefill.serving_prefill_rows=rows;validate_resolved_serving(prefill);
            expect(prefill.serving_prefill_rows==rows,"explicit serving prefill retains exact closed native width");
        }
        for(unsigned fault=0;fault<9U;++fault) {
            auto bad=prefill;
            if(!fault) bad.serving_prefill_workspace_budget=0;
            if(fault==1U) bad.serving_prefill_metadata_explicit=false;
            if(fault==2U) bad.serving_tensor_cohort_rows=2;
            if(fault==3U) bad.serving_prefix_files.clear();
            if(fault==4U) bad.serving_prefill_rows=3;
            if(fault==5U) bad.gdn_recurrent_storage=strix::GdnRecurrentStorage::value_key;
            if(fault==6U) bad.wave8_q8_gdn_output=false;
            if(fault==7U) bad.qsa_gqa_heads_per_block=2;
            if(fault==8U) bad.serving_cache_entries_explicit=true;
            rejects([&]{validate_resolved_serving(bad);},"serving prefill never ignores missing budgets/profile or conflicting cohort/layout");
        }
        {
            auto cached=prefill;
            cached.serving_cache_entries_explicit=cached.serving_cache_snapshot_explicit=
                cached.serving_cache_peak_explicit=cached.serving_cache_tokens_explicit=true;
            cached.serving_cache_entries=2;cached.serving_cache_snapshot_bytes=1ULL<<30;
            cached.serving_cache_capture_peak_bytes=1ULL<<30;
            cached.serving_cache_max_prefix_tokens=cached.context_tokens;
            validate_resolved_serving(cached);
            expect(cached.serving_prefill_rows==2048U && cached.serving_cache_entries==2U,
                "bounded prefix cache and native prefill coexist without disabling either request");
            auto incomplete=cached;incomplete.serving_cache_snapshot_explicit=false;
            rejects([&]{validate_resolved_serving(incomplete);},"prefill never bypasses explicit cache budget admission");
            auto cohort_cached=cached;cohort_cached.serving_tensor_cohort_rows=2U;
            rejects([&]{validate_resolved_serving(cohort_cached);},"cache support never silently routes prefill through a cohort");
        }
        rejects([]{(void)parse({"decoder","m","--serve-prefill-rows","3"});},"unsupported native widths are rejected, not relabelled");
        rejects([]{(void)parse({"decoder","m","--serve-prefill-rows","2","--serve-prefill-rows=4"});},"serving prefill duplicate forms reject");
        expect(!defaults.serving_prefill_rows && !defaults.serving_prefill_explicit,"serving prefill is default off");
    }
    return failures ? 1 : 0;
}
