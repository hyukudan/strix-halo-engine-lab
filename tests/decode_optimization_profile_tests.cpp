#include "../apps/hip_decode_options.hpp"
#include <iostream>
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
template<class Function> void rejects(Function operation, std::string_view message) {
    try { operation(); expect(false, "expected CLI rejection"); }
    catch (const std::invalid_argument& error) {
        expect(std::string_view(error.what()) == message, "specific CLI rejection");
    }
}
}
int main() {
    const auto model = ModelContract::qwen38_flash_next();
    gguf::FileInfo info;
    ModelManifest manifest;
    manifest.architecture = "qwen3next";
    manifest.layout_identity = model_manifest_layout_identity(manifest);
    auto defaults = parse({"decoder", "fixture.gguf", "1"});
    expect(defaults.optimization_profile == OptimizationProfile::automatic &&
           defaults.context_tokens == 128 && !defaults.q4k_a8g32_gate_up &&
           !defaults.profile_explicit && !defaults.any_low_level_flag &&
           !defaults.prefetch_q8_qsa_input && !defaults.prefetch_q8_shared_expert &&
           !defaults.deferred_moe_status,
           "parser defers auto until content identification");
    auto on = defaults;
    auto inventory_only = parse({"decoder", "fixture.gguf", "1", "--execution-inventory"});
    expect(inventory_only.execution_inventory && !inventory_only.any_low_level_flag,
           "inventory export is independent of kernel selection");
    auto scalar = parse({"decoder", "fixture.gguf", "1", "--generate", "2", "--gpu-greedy-token"});
    const auto scalar_profile = resolve_optimization_profile(scalar, model, info, manifest,
        validated_model_content_identity, true);
    expect(scalar.gpu_greedy_token && scalar_profile.automatic_applied,
           "explicit scalar output preserves the selected automatic kernel bundle");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--gpu-greedy-token"})); },
        "GPU greedy token output requires eager generation without RAW, reference, roundtrip or integer head");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--generate", "2", "--prefix-cache-roundtrip",
        "--gpu-greedy-token"})); },
        "GPU greedy token output requires eager generation without RAW, reference, roundtrip or integer head");
    auto instrumented = parse({"decoder", "fixture.gguf", "1", "--host-dispatch-telemetry"});
    const auto instrumented_profile = resolve_optimization_profile(instrumented, model, info, manifest,
        validated_model_content_identity, true);
    expect(instrumented.host_dispatch_telemetry && !instrumented.any_low_level_flag &&
           instrumented_profile.automatic_applied,
           "host observation does not silently disable the qualified automatic profile");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--host-dispatch-telemetry",
        "--host-dispatch-telemetry"})); }, "duplicate host dispatch telemetry option");
    const auto selected = resolve_optimization_profile(on, model, info, manifest,
        validated_model_content_identity, true);
    expect(selected.automatic_applied && selected.selected == "canonical-a8-wave-qsa-hc-shared-deferred-v3" &&
           selected.reason == "qualified-checkpoint", "qualified default selects fast profile");
    expect(on.parallel_hyper_mix && on.staged_hyper_projections && on.fused_staged_hyper_mix &&
           on.typed_hyper_projection && on.prefetch_typed_hyper_rows &&
           on.virtual_q8_lm_head && on.parallel_moe_router && on.wave_moe_router_topk &&
           !on.ranked_moe_router_topk && on.virtual_q8_gdn_input && on.prefetch_q8_gdn_input &&
           on.q4k_a8g32_gate_up && on.grouped_exact_q51_down && on.prefetch_q8_hyper_output &&
           on.virtual_q8_gdn_output && on.lane_chain_hyper_output && on.context_tokens == 8192,
           "exact validated flag bundle and qualified implicit context");
    expect(on.prefetch_q8_qsa_input && on.prefetch_q8_shared_expert &&
           on.deferred_moe_status,
           "automatic candidate includes QSA, shared-expert and deferred status");
    const auto fallback = [&](Options input, const char* reason, bool capability = true,
                              std::string_view identity = validated_model_content_identity) {
        const auto before = input;
        const auto result = resolve_optimization_profile(input, model, info, manifest, identity, capability);
        expect(!result.automatic_applied && result.reason == reason,
               "specific automatic fallback reason");
        expect(input.context_tokens == before.context_tokens &&
               input.virtual_q8_lm_head == before.virtual_q8_lm_head &&
               input.q4k_a8g32_gate_up == before.q4k_a8g32_gate_up &&
               input.lane_chain_hyper_output == before.lane_chain_hyper_output &&
               input.wave_moe_router_topk == before.wave_moe_router_topk &&
               input.ranked_moe_router_topk == before.ranked_moe_router_topk &&
               input.prefetch_q8_qsa_input == before.prefetch_q8_qsa_input &&
               input.prefetch_q8_shared_expert == before.prefetch_q8_shared_expert &&
               input.deferred_moe_status == before.deferred_moe_status,
               "fallback retains explicit and previous defaults");
    };
    fallback(defaults, "checkpoint-content-not-qualified", true, "another-checkpoint");
    fallback(defaults, "a8-runtime-not-built", false);
    fallback(parse({"decoder","m","1","--optimization-profile","manual"}), "explicit-manual");
    fallback(parse({"decoder","m","1","--context","4096"}), "context-not-qualified");
    fallback(parse({"decoder","m","1","--residency","copy"}), "residency-not-qualified");
    fallback(parse({"decoder","m","1","--reference","reference.json"}), "explicit-replay-reference");
    fallback(parse({"decoder","m","1","--planned-ple-io"}), "explicit-low-level-options");
    fallback(parse({"decoder","m","1","--optimization-profile","auto","--planned-ple-io"}),
             "planned-ple-io-explicit");
    fallback(parse({"decoder","m","1","--parallel-hyper-mix"}), "explicit-low-level-options");
    fallback(parse({"decoder","m","1","--coalesce-gap","64"}), "explicit-low-level-options");
    auto explicit_increments = parse({"decoder","m","1","--prefetch-q8-qsa-input",
        "--prefetch-q8-shared-expert","--deferred-moe-status"});
    fallback(explicit_increments, "explicit-low-level-options");
    auto explicit_manual = explicit_increments;
    explicit_manual.optimization_profile = OptimizationProfile::manual;
    fallback(explicit_manual, "explicit-manual");
    auto unqualified_explicit = explicit_increments;
    unqualified_explicit.profile_explicit = true;
    fallback(unqualified_explicit, "checkpoint-content-not-qualified", true, "another-checkpoint");
    for (const auto* flag : {"--parallel-hyper-mix","--prefetch-q8-qsa-input",
        "--prefetch-q8-shared-expert","--deferred-moe-status","--lane-chain-hyper-output",
        "--wave-moe-router-topk","--q4k-a8g32-gate-up","--grouped-exact-q51-down"})
        expect(is_low_level_optimization_option(flag), "all experimental optimization flags tracked");
    expect(!is_low_level_optimization_option("--context") &&
           !is_low_level_optimization_option("--optimization-profile"),
           "ordinary context and explicit profile are distinct from low-level choices");
    auto graph = parse({"decoder","m","1","--execution-mode","graph","--residency","copy",
        "--graph-workspace-bytes","1","--graph-executable-bytes","1",
        "--graph-capture-peak-bytes","1","--graph-budget-bytes","3"});
    fallback(graph, "execution-mode-not-qualified");
    auto integer = parse({"decoder","m","1","--residency","copy","--integer-lm-head","a8-iu8",
        "--integer-internal-bytes","1","--integer-retained-bytes","1"});
    fallback(integer, "integer-head-explicit");
    auto ranked = parse({"decoder","m","1","--parallel-moe-router","--ranked-moe-router-topk"});
    const auto ranked_result = resolve_optimization_profile(ranked, model, info, manifest,
        validated_model_content_identity, true);
    expect(ranked.profile_explicit == false && ranked.any_low_level_flag &&
           ranked.ranked_moe_router_topk && !ranked.wave_moe_router_topk &&
           !ranked.lane_chain_hyper_output && !ranked.q4k_a8g32_gate_up &&
           !ranked.prefetch_q8_qsa_input && !ranked.prefetch_q8_shared_expert &&
           !ranked.deferred_moe_status &&
           ranked_result.selected == "manual" && !ranked_result.automatic_applied &&
           ranked_result.reason == "explicit-low-level-options" && ranked.context_tokens == 128,
           "legacy explicit ranked command is never expanded by implicit auto");
    auto ranked_auto = parse({"decoder","m","1","--parallel-moe-router",
        "--ranked-moe-router-topk","--optimization-profile","auto"});
    const auto ranked_auto_result = resolve_optimization_profile(ranked_auto, model, info, manifest,
        validated_model_content_identity, true);
    expect(ranked_auto.profile_explicit && ranked_auto.ranked_moe_router_topk &&
           !ranked_auto.wave_moe_router_topk && ranked_auto.lane_chain_hyper_output &&
           ranked_auto.prefetch_q8_qsa_input && ranked_auto.prefetch_q8_shared_expert &&
           ranked_auto.deferred_moe_status &&
           ranked_auto_result.selected == "canonical-a8-ranked-qsa-hc-shared-deferred-v3" &&
           ranked_auto_result.reason == "explicit-ranked-router",
           "explicit auto retains ranked routing and adds the candidate bundle");
    auto explicit_context = parse({"decoder","m","1","--context","8192","--optimization-profile","auto"});
    expect(resolve_optimization_profile(explicit_context, model, info, manifest,
        validated_model_content_identity, true).automatic_applied, "explicit qualified context eligible");
    auto roundtrip = parse({"decoder","m","1","--generate","511","--prefix-cache-roundtrip"});
    roundtrip.tokens.assign(43,1);
    expect(resolve_optimization_profile(roundtrip, model, info, manifest,
        validated_model_content_identity, true).automatic_applied && roundtrip.context_tokens == 8192,
        "implicit auto resolves roundtrip context before capacity validation");
    validate_resolved_prefix_roundtrip(roundtrip);
    auto manual_roundtrip = parse({"decoder","m","1","--generate","511",
        "--prefix-cache-roundtrip","--optimization-profile","manual"});
    manual_roundtrip.tokens.assign(43,1);
    (void)resolve_optimization_profile(manual_roundtrip, model, info, manifest,
                                     validated_model_content_identity, true);
    validate_resolved_prefix_roundtrip(manual_roundtrip);
    auto explicit_small = parse({"decoder","m","1","--generate","511",
        "--prefix-cache-roundtrip","--context","128"});
    explicit_small.tokens.assign(43,1);
    expect(resolve_optimization_profile(explicit_small, model, info, manifest,
        validated_model_content_identity, true).reason == "context-not-qualified",
        "explicit small context is never expanded");
    validate_resolved_prefix_roundtrip(explicit_small);
    auto unknown_roundtrip = parse({"decoder","m","1","--generate","511","--prefix-cache-roundtrip"});
    unknown_roundtrip.tokens.assign(43,1);
    (void)resolve_optimization_profile(unknown_roundtrip, model, info, manifest,"unknown",true);
    validate_resolved_prefix_roundtrip(unknown_roundtrip);
    auto long_roundtrip = parse({"decoder","m","1","--generate","511","--prefix-cache-roundtrip"});
    long_roundtrip.tokens.assign(200,1);
    expect(resolve_optimization_profile(long_roundtrip, model, info, manifest,
        validated_model_content_identity, true).automatic_applied && long_roundtrip.context_tokens == 8192,
        "prefix beyond implicit legacy context accepted after automatic expansion");
    validate_resolved_prefix_roundtrip(long_roundtrip);
    manual_roundtrip.tokens.assign(200,1);
    rejects([&] { validate_resolved_prefix_roundtrip(manual_roundtrip); },
        "prefix-cache roundtrip requires at least two generated tokens and remaining context");
    manual_roundtrip.tokens.assign(43,1);
    manual_roundtrip.generate_tokens = 86; //43prompt+85generated-inputs exactly128steps.
    validate_resolved_prefix_roundtrip(manual_roundtrip);
    auto altered_model = model;
    altered_model.hidden_size /= 2;
    auto bad_contract = defaults;
    expect(resolve_optimization_profile(bad_contract, altered_model, info, manifest,
        validated_model_content_identity, true).reason == "model-contract-not-qualified",
        "same claimed content cannot bypass incompatible model geometry");
    auto malformed = manifest;
    ++malformed.layout_identity;
    auto bad_manifest = defaults;
    expect(resolve_optimization_profile(bad_manifest, model, info, malformed,
        validated_model_content_identity, true).reason == "manifest-layout-not-qualified",
        "recomputed manifest identity required");
    gguf::MetadataEntry group;
    group.key = "qwen3next.expert_group_count";
    info.metadata.push_back(group);
    auto unsupported = defaults;
    expect(resolve_optimization_profile(unsupported, model, info, manifest,
        validated_model_content_identity, true).reason == "router-contract-not-qualified" &&
        !unsupported.wave_moe_router_topk, "unsupported grouped router falls back without opting in");
    rejects([] { (void)parse({"decoder","m","1","--optimization-profile","fast"}); },
        "--optimization-profile requires auto or manual");
    rejects([] { (void)parse({"decoder","m","1","--optimization-profile"}); },
        "--optimization-profile requires auto or manual");
    rejects([] { (void)parse({"decoder","m","1","--optimization-profile","auto",
        "--optimization-profile","manual"}); }, "duplicate optimization profile option");
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
    auto legacy_full = parse({"decoder","m","1","--parallel-hyper-mix",
        "--staged-hyper-projections","--fused-staged-hyper-mix","--virtual-q8-lm-head",
        "--parallel-moe-router","--ranked-moe-router-topk","--virtual-q8-gdn-input",
        "--prefetch-q8-gdn-input","--typed-fused-hyper-projections","--prefetch-typed-hyper-rows",
        "--q4k-a8g32-gate-up","--grouped-exact-q51-down","--prefetch-q8-hyper-output",
        "--virtual-q8-gdn-output"});
    const auto legacy_result = resolve_optimization_profile(legacy_full, model, info, manifest,
        validated_model_content_identity, true);
    expect(legacy_result.reason == "explicit-low-level-options" &&
           !legacy_result.automatic_applied && legacy_full.q4k_a8g32_gate_up &&
           legacy_full.grouped_exact_q51_down && legacy_full.virtual_q8_gdn_output &&
           legacy_full.prefetch_q8_hyper_output && legacy_full.ranked_moe_router_topk &&
           !legacy_full.wave_moe_router_topk && !legacy_full.lane_chain_hyper_output &&
           !legacy_full.prefetch_q8_qsa_input && !legacy_full.prefetch_q8_shared_expert &&
           !legacy_full.deferred_moe_status &&
           legacy_full.context_tokens == 128,
           "full fourteen-option legacy baseline remains its explicitly requested configuration");
    auto a8_manual = parse({"decoder","m","1","--optimization-profile","manual",
        "--q4k-a8g32-gate-up","--grouped-exact-q51-down"});
    expect(a8_manual.q4k_a8g32_gate_up && a8_manual.grouped_exact_q51_down,
           "actual A8-enabled CLI retains explicitly requested manual kernels");
#else
    rejects([] { (void)parse({"decoder","m","1","--optimization-profile","manual",
        "--q4k-a8g32-gate-up"}); },
        "--q4k-a8g32-gate-up requires the experimental runtime build capability");
#endif
    // Manual is order-independent and never activates a flag the caller omitted.
    auto manual = parse({"decoder","m","1","--parallel-moe-router",
        "--ranked-moe-router-topk","--optimization-profile","manual"});
    const auto manual_result = resolve_optimization_profile(manual, model, info, manifest,
        validated_model_content_identity, true);
    expect(manual_result.requested == "manual" && manual_result.selected == "manual" &&
           manual_result.reason == "explicit-manual" && !manual_result.automatic_applied &&
           manual.ranked_moe_router_topk &&
           !manual.virtual_q8_lm_head && !manual.q4k_a8g32_gate_up &&
           !manual.lane_chain_hyper_output && !manual.prefetch_q8_qsa_input &&
           !manual.prefetch_q8_shared_expert && !manual.deferred_moe_status &&
           manual.context_tokens == 128,
           "actual application CLI preserves manual baseline without auto dependencies");
    auto manual_hc = parse({"decoder","m","1","--optimization-profile","manual",
        "--parallel-hyper-mix","--staged-hyper-projections","--fused-staged-hyper-mix",
        "--typed-fused-hyper-projections","--prefetch-q8-hyper-output","--lane-chain-hyper-output"});
    const auto explicit_hc = resolve_optimization_profile(manual_hc, model, info, manifest,
        validated_model_content_identity, true);
    expect(explicit_hc.requested == "manual" && explicit_hc.selected == "manual" &&
           explicit_hc.reason == "explicit-manual" && !explicit_hc.automatic_applied &&
           manual_hc.lane_chain_hyper_output &&
           !manual_hc.wave_moe_router_topk && !manual_hc.q4k_a8g32_gate_up &&
           !manual_hc.prefetch_q8_qsa_input && !manual_hc.prefetch_q8_shared_expert &&
           !manual_hc.deferred_moe_status,
           "manual HC selection retains its own dependencies without auto wave or A8");
    auto manual_increments = parse({"decoder","m","1","--optimization-profile","manual",
        "--prefetch-q8-qsa-input","--prefetch-q8-shared-expert","--deferred-moe-status"});
    const auto explicit_increments_result = resolve_optimization_profile(manual_increments,
        model, info, manifest, validated_model_content_identity, true);
    expect(explicit_increments_result.requested == "manual" &&
           explicit_increments_result.selected == "manual" &&
           explicit_increments_result.reason == "explicit-manual" &&
           !explicit_increments_result.automatic_applied &&
           manual_increments.prefetch_q8_qsa_input && manual_increments.prefetch_q8_shared_expert &&
           manual_increments.deferred_moe_status && !manual_increments.wave_moe_router_topk &&
           !manual_increments.lane_chain_hyper_output && !manual_increments.q4k_a8g32_gate_up &&
           manual_increments.context_tokens == 128,
           "manual increments preserve their flags without adding the automatic bundle");
    std::cout << "Decode optimization profile failures=" << failures << '\n';
    return failures ? 1 : 0;
}
