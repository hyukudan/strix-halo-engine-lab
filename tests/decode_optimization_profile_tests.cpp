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
           defaults.context_tokens == 128 && !defaults.q4k_a8g32_gate_up,
           "parser defers auto until content identification");
    auto on = defaults;
    const auto selected = resolve_optimization_profile(on, model, info, manifest,
        validated_model_content_identity, true);
    expect(selected.automatic_applied && selected.selected == "canonical-a8-wave-v1" &&
           selected.reason == "qualified-checkpoint", "qualified default selects fast profile");
    expect(on.parallel_hyper_mix && on.staged_hyper_projections && on.fused_staged_hyper_mix &&
           on.typed_hyper_projection && on.prefetch_typed_hyper_rows &&
           on.virtual_q8_lm_head && on.parallel_moe_router && on.wave_moe_router_topk &&
           !on.ranked_moe_router_topk && on.virtual_q8_gdn_input && on.prefetch_q8_gdn_input &&
           on.q4k_a8g32_gate_up && on.grouped_exact_q51_down && on.prefetch_q8_hyper_output &&
           on.virtual_q8_gdn_output && on.context_tokens == 8192,
           "exact validated flag bundle and qualified implicit context");
    expect(!on.prefetch_q8_qsa_input && !on.prefetch_q8_shared_expert &&
           !on.deferred_moe_status && !on.lane_chain_hyper_output,
           "unqualified experiments stay explicit");
    const auto fallback = [&](Options input, const char* reason, bool capability = true,
                              std::string_view identity = validated_model_content_identity) {
        const auto before = input;
        const auto result = resolve_optimization_profile(input, model, info, manifest, identity, capability);
        expect(!result.automatic_applied && result.reason == reason,
               "specific automatic fallback reason");
        expect(input.context_tokens == before.context_tokens &&
               input.virtual_q8_lm_head == before.virtual_q8_lm_head &&
               input.q4k_a8g32_gate_up == before.q4k_a8g32_gate_up &&
               input.wave_moe_router_topk == before.wave_moe_router_topk &&
               input.ranked_moe_router_topk == before.ranked_moe_router_topk,
               "fallback retains explicit and previous defaults");
    };
    fallback(defaults, "checkpoint-content-not-qualified", true, "another-checkpoint");
    fallback(defaults, "a8-runtime-not-built", false);
    fallback(parse({"decoder","m","1","--optimization-profile","manual"}), "explicit-manual");
    fallback(parse({"decoder","m","1","--context","4096"}), "context-not-qualified");
    fallback(parse({"decoder","m","1","--residency","copy"}), "residency-not-qualified");
    fallback(parse({"decoder","m","1","--reference","reference.json"}), "explicit-replay-reference");
    fallback(parse({"decoder","m","1","--planned-ple-io"}), "planned-ple-io-explicit");
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
    expect(ranked.ranked_moe_router_topk && !ranked.wave_moe_router_topk &&
           ranked_result.reason == "explicit-ranked-router", "explicit ranked selection preserved");
    auto explicit_context = parse({"decoder","m","1","--context","8192","--optimization-profile","auto"});
    expect(resolve_optimization_profile(explicit_context, model, info, manifest,
        validated_model_content_identity, true).automatic_applied, "explicit qualified context eligible");
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
    expect(manual_result.requested == "manual" && manual.ranked_moe_router_topk &&
           !manual.virtual_q8_lm_head && !manual.q4k_a8g32_gate_up && manual.context_tokens == 128,
           "actual application CLI preserves manual baseline without auto dependencies");
    std::cout << "Decode optimization profile failures=" << failures << '\n';
    return failures ? 1 : 0;
}
