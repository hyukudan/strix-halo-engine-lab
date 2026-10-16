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
           !defaults.deferred_moe_status && !defaults.gpu_greedy_token && !defaults.prepared_eager && !defaults.moe_shared_overlap && !defaults.batched_moe_activation && !defaults.gdn_islands,
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
    auto prepared = parse({"decoder", "fixture.gguf", "1", "--prepared-eager"});
    const auto prepared_profile = resolve_optimization_profile(prepared, model, info, manifest,
        validated_model_content_identity, true);
    expect(prepared.prepared_eager && !prepared.any_low_level_flag && prepared_profile.automatic_applied,
           "prepared host submission preserves automatic kernel selection");
    auto composed = parse({"decoder", "fixture.gguf", "1", "--generate", "2",
        "--prepared-eager", "--gpu-greedy-token"});
    expect(composed.prepared_eager && composed.gpu_greedy_token,
           "prepared submission and scalar output compose explicitly");
    expect(parse({"decoder", "m", "1", "--reference", "reference.json", "--prepared-eager"}).prepared_eager,
           "prepared diagnostic replay retains full arrays");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--prepared-eager", "--prepared-eager"})); },
        "duplicate prepared eager option");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--prepared-eager", "--prefix-cache-roundtrip", "--generate", "2"})); },
        "prepared eager execution excludes graph, integer head and prefix roundtrip");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--generate", "2", "--gpu-greedy-token", "--reference", "r"})); },
        "--generate and --reference are separate execution modes");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--generate", "2", "--gpu-greedy-token", "--gpu-greedy-token"})); },
        "duplicate GPU greedy token option");
    auto overlap = parse({"decoder", "fixture.gguf", "1", "--moe-shared-overlap"});
    const auto overlap_profile = resolve_optimization_profile(overlap, model, info, manifest,
        validated_model_content_identity, true);
    expect(overlap.moe_shared_overlap && !overlap.any_low_level_flag && overlap_profile.automatic_applied &&
        overlap.deferred_moe_status && overlap.moe_overlap_arena_budget_bytes == 64ULL * 1024ULL * 1024ULL,
        "explicit overlap retains the automatic arithmetic bundle and a bounded arena budget");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--moe-overlap-arena-budget-bytes", "64"})); },
        "MoE overlap arena budget requires enabled overlap and nonzero bytes");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--moe-shared-overlap", "--moe-overlap-arena-budget-bytes", "0"})); },
        "MoE overlap arena budget requires enabled overlap and nonzero bytes");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--moe-shared-overlap", "--moe-shared-overlap"})); },
        "duplicate MoE shared overlap option");
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
    auto batched = parse({"decoder", "fixture.gguf", "1", "--batched-moe-activation"});
    const auto batched_profile = resolve_optimization_profile(batched, model, info, manifest,
        validated_model_content_identity, true);
    expect(batched.batched_moe_activation && !batched.any_low_level_flag && batched_profile.automatic_applied &&
        batched.q4k_a8g32_gate_up && batched.deferred_moe_status,
        "batched activation overlay preserves the existing automatic optimization bundle");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--batched-moe-activation", "--batched-moe-activation"})); },
        "duplicate batched MoE activation option");
#else
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--batched-moe-activation"})); },
        "--batched-moe-activation requires the experimental runtime build capability");
#endif
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
    auto islands = parse({"decoder", "fixture.gguf", "1", "--generate", "2", "--gdn-islands",
        "--prepared-eager", "--gpu-greedy-token", "--moe-shared-overlap"});
    const auto islands_profile = resolve_optimization_profile(islands, model, info, manifest,
        validated_model_content_identity, true);
    validate_resolved_gdn_islands(islands);
    expect(islands.gdn_islands && !islands.any_low_level_flag && islands_profile.automatic_applied &&
        islands.prepared_eager && islands.gpu_greedy_token && islands.moe_shared_overlap &&
        islands.deferred_moe_status && islands.virtual_q8_gdn_input && islands.prefetch_q8_gdn_input &&
        islands.virtual_q8_gdn_output && !islands.wave8_q8_gdn_input && !islands.wave8_q8_gdn_output,
        "GDN islands preserve automatic arithmetic and compose with eager submission overlays");
    auto wave_islands = parse({"decoder", "fixture.gguf", "1", "--gdn-islands",
        "--wave8-q8-gdn-input", "--wave8-q8-gdn-output"});
    const auto wave_islands_profile = resolve_optimization_profile(wave_islands, model, info, manifest,
        validated_model_content_identity, true);
    validate_resolved_gdn_islands(wave_islands);
    expect(wave_islands_profile.automatic_applied && wave_islands.wave8_q8_gdn_input && wave_islands.wave8_q8_gdn_output,
        "paired wave8 island organization preserves explicitly requested geometry without altering defaults");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--gdn-islands", "--gdn-islands"})); },
        "duplicate GDN islands option");
    for (const auto* wave : {"--wave8-q8-gdn-input", "--wave8-q8-gdn-output"})
        rejects([&] { static_cast<void>(parse({"decoder", "m", "1", "--gdn-islands", wave})); },
            "GDN islands require eager floating execution without prefix roundtrip and paired wave8 GDN flags");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--gdn-islands", "--prefix-cache-roundtrip", "--generate", "2"})); },
        "GDN islands require eager floating execution without prefix roundtrip and paired wave8 GDN flags");
    auto unsupported_islands = islands;
    unsupported_islands.execution.mode = DecodeExecutionMode::graph;
    rejects([&] { validate_resolved_gdn_islands(unsupported_islands); },
        "GDN islands require eager floating execution without prefix roundtrip and paired wave8 GDN flags");
    unsupported_islands = islands;
    unsupported_islands.execution.integer_head = DecodeIntegerHead::a8_iu8;
    rejects([&] { validate_resolved_gdn_islands(unsupported_islands); },
        "GDN islands require eager floating execution without prefix roundtrip and paired wave8 GDN flags");
    for (unsigned missing = 0; missing < 4; ++missing) {
        auto missing_dependency = islands;
        if (missing == 0) missing_dependency.deferred_moe_status = false;
        if (missing == 1) missing_dependency.virtual_q8_gdn_input = false;
        if (missing == 2) missing_dependency.prefetch_q8_gdn_input = false;
        if (missing == 3) missing_dependency.virtual_q8_gdn_output = false;
        rejects([&] { validate_resolved_gdn_islands(missing_dependency); },
            "GDN islands require resolved D1 and virtual/prefetch input with virtual output");
    }
    auto manual_islands = parse({"decoder", "m", "1", "--optimization-profile", "manual", "--gdn-islands"});
    rejects([&] { validate_resolved_gdn_islands(manual_islands); },
        "GDN islands require resolved D1 and virtual/prefetch input with virtual output");
#else
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--gdn-islands"})); },
        "--gdn-islands requires the experimental runtime build capability");
#endif
    validate_resolved_gdn_islands(defaults);
    expect(!defaults.qsa_hierarchical_selection, "hierarchical QSA stays opt-in");
    validate_resolved_qsa_hierarchical_selection(defaults);
    expect(!defaults.known_prefix_window, "known-prefix route stays opt-in");
    validate_resolved_known_prefix_window(defaults);
    expect(defaults.offline_tuning_cache_path.empty(), "offline persisted tuning stays opt-in");
    const auto provenance = parse({"decoder", "m", "1", "--offline-tuning-cache", "cache.bin", "--offline-compile-provenance", "context.bin"});
    expect(provenance.offline_compile_provenance_path == "context.bin" && !provenance.any_low_level_flag,
        "explicit evaluated compile context is a prepare-only consumer input");
    rejects([] { (void)parse({"decoder", "m", "1", "--offline-compile-provenance", "context.bin"}); },
        "offline compile provenance requires --offline-tuning-cache");
    rejects([] { (void)parse({"decoder", "m", "1", "--offline-compile-provenance"}); },
        "offline compile provenance requires a file");
    rejects([] { (void)parse({"decoder", "m", "1", "--offline-tuning-cache", "cache.bin", "--offline-compile-provenance", "a", "--offline-compile-provenance", "b"}); },
        "duplicate offline compile provenance option");
    const auto persisted = parse({"decoder", "m", "1", "--offline-tuning-cache", "source-only-cache.bin"});
    expect(persisted.offline_tuning_cache_path == "source-only-cache.bin" && !persisted.any_low_level_flag,
        "persisted tuning does not silently alter base optimization profile");
    rejects([] { (void)parse({"decoder", "m", "1", "--offline-tuning-cache"}); }, "offline tuning cache requires a file");
    rejects([] { (void)parse({"decoder", "m", "1", "--offline-tuning-cache", "a", "--offline-tuning-cache", "b"}); },
        "duplicate offline tuning cache option");
    expect(defaults.logprob_token_ids.empty(), "requested logprobs are opt-in");
    auto logprobs = parse({"decoder", "m", "1", "--generate", "3", "--logprob-token", "0", "--logprob-token", "15", "--logprob-token", "0"});
    expect(logprobs.logprob_token_ids == std::vector<std::uint32_t>({0,15,0}) && !logprobs.any_low_level_flag,
        "requested ID order/duplicates survive without disabling automatic profile");
    validate_resolved_requested_logprobs(logprobs,16);
    for (unsigned mutation = 0; mutation < 9U; ++mutation) {
        auto bad = logprobs;
        if (mutation == 0) bad.generate_tokens = 0;
        if (mutation == 1) bad.gpu_greedy_token = true;
        if (mutation == 2) bad.known_prefix_window = true;
        if (mutation == 3) bad.prompt_lookup = true;
        if (mutation == 4) bad.serving_prefix_files.push_back("source-only-prefix");
        if (mutation == 5) bad.prefix_cache_roundtrip = true;
        if (mutation == 6) bad.raw_recording_path = "source-only-raw";
        if (mutation == 7) bad.reference_path = "source-only-reference";
        if (mutation == 8) bad.generate_tokens = 0;
        rejects([&] { validate_resolved_requested_logprobs(bad,16); },
            "requested logprobs require serial full-output generation without scalar, window, lookup, serving, cache, RAW or reference modes");
    }
    { auto bad = logprobs; bad.logprob_token_ids.push_back(16);
      rejects([&] { validate_resolved_requested_logprobs(bad,16); }, "requested logprob token exceeds model vocabulary"); }
    { auto bad = logprobs; bad.logprob_token_ids.resize(65U);
      rejects([&] { validate_resolved_requested_logprobs(bad,16); }, "requested logprob receipt exceeds bounded row/value capacity"); }
    { auto bad = logprobs; bad.context_tokens = 262145;
      rejects([&] { validate_resolved_requested_logprobs(bad,16); }, "requested logprob receipt exceeds bounded row/value capacity"); }
    { auto accepted = logprobs; accepted.logprob_token_ids.assign(64U,0);accepted.context_tokens=16384;
      accepted.prepared_eager=true;accepted.moe_shared_overlap=true;
      validate_resolved_requested_logprobs(accepted,16);
      ++accepted.context_tokens;
      rejects([&] { validate_resolved_requested_logprobs(accepted,16); }, "requested logprob receipt exceeds bounded row/value capacity"); }
    rejects([] { (void)parse({"decoder", "m", "1", "--logprob-token", "4294967296"}); }, "logprob token exceeds uint32");
    expect(defaults.ple_cache_rows == 0 && defaults.ple_cache_budget_bytes == 0,
        "PLE row cache remains absent by default");
    validate_resolved_ple_cache(defaults);
    auto cache_ple = parse({"decoder", "m", "1", "--ple-cache-rows", "8", "--ple-cache-budget-bytes", "4096"});
    expect(cache_ple.ple_cache_rows == 8 && cache_ple.ple_cache_budget_bytes == 4096 && !cache_ple.any_low_level_flag,
        "PLE cache limits do not silently select a kernel profile");
    rejects([&] { validate_resolved_ple_cache(cache_ple); }, "PLE row cache requires resolved planned PLE I/O");
    cache_ple.planned_ple_io = true;
    validate_resolved_ple_cache(cache_ple);
    rejects([] { (void)parse({"decoder", "m", "1", "--ple-cache-rows", "8"}); },
        "PLE cache requires both positive explicit row and byte limits");
    rejects([] { (void)parse({"decoder", "m", "1", "--ple-cache-rows", "0"}); }, "PLE cache limits must be positive");
    rejects([] { (void)parse({"decoder", "m", "1", "--ple-cache-rows"}); }, "PLE cache option requires a value");
    rejects([] { (void)parse({"decoder", "m", "1", "--ple-cache-rows", "2", "--ple-cache-rows", "4"}); },
        "duplicate PLE cache limit");
    for (unsigned mutation = 0; mutation < 4; ++mutation) {
        auto bad = cache_ple;
        if (mutation == 0) bad.ple_cache_rows_explicit = false;
        if (mutation == 1) bad.ple_cache_budget_explicit = false;
        if (mutation == 2) bad.ple_cache_rows = 0;
        if (mutation == 3) bad.ple_cache_budget_bytes = 0;
        rejects([&] { validate_resolved_ple_cache(bad); }, "PLE cache requires both positive explicit row and byte limits");
    }
    expect(defaults.ple_cpu_helper_threads == 0 && !defaults.ple_cpu_helper_threads_explicit,
        "PLE helper budget default retains legacy backend behavior");
    validate_resolved_ple_cpu_helpers(defaults);
    for (const auto* value : {"1", "2", "8"}) {
        auto explicit_helpers = parse({"decoder", "m", "1", "--ple-cpu-helper-threads", value});
        expect(explicit_helpers.ple_cpu_helper_threads == parse_u64(value, "fixture") &&
            !explicit_helpers.any_low_level_flag && !explicit_helpers.planned_ple_io,
            "helper budget neither changes automatic profile nor implicitly enables planned PLE");
        rejects([&] { validate_resolved_ple_cpu_helpers(explicit_helpers); },
            "PLE CPU helper threads require resolved planned PLE I/O");
        explicit_helpers.planned_ple_io = true;
        validate_resolved_ple_cpu_helpers(explicit_helpers);
    }
    expect(parse({"decoder", "m", "1", "--planned-ple-io", "--ple-cpu-helper-threads=8"}).ple_cpu_helper_threads == 8,
        "explicit planned backend accepts helper cap equals form");
    for (const auto* value : {"0", "9"})
        rejects([&] { (void)parse({"decoder", "m", "1", "--ple-cpu-helper-threads", value}); },
            "PLE CPU helper threads must be 1 through 8 when explicit");
    rejects([] { (void)parse({"decoder", "m", "1", "--ple-cpu-helper-threads", "2", "--ple-cpu-helper-threads=4"}); },
        "duplicate PLE CPU helper threads option");
    { auto invalid = defaults; invalid.ple_cpu_helper_threads = 9;
      rejects([&] { validate_resolved_ple_cpu_helpers(invalid); }, "PLE CPU helper threads must be 1 through 8 when explicit"); }

    auto paired_prompt = defaults;
    paired_prompt.known_prefix_window = true; paired_prompt.generate_tokens = 3;
    paired_prompt.deferred_moe_status = true;
    paired_prompt.virtual_q8_gdn_input = paired_prompt.prefetch_q8_gdn_input = true;
    paired_prompt.virtual_q8_gdn_output = paired_prompt.wave8_q8_gdn_input = paired_prompt.wave8_q8_gdn_output = true;
    validate_resolved_known_prefix_window(paired_prompt);
    expect(parse({"decoder", "m", "1", "--known-prefix-window"}).known_prefix_window,
        "parser records opt-in before resolved profile admission");
    expect(!defaults.known_prefix_expert_union, "known-prefix routed expert union stays opt-in");
    expect(parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-expert-union"}).known_prefix_expert_union &&
        !parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-expert-union"}).any_low_level_flag,
        "union overlay preserves automatic profile selection");
    rejects([] { (void)parse({"decoder", "m", "1", "--known-prefix-expert-union"}); },
        "known-prefix expert union requires --known-prefix-window");
    rejects([] { (void)parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-expert-union", "--known-prefix-expert-union"}); },
        "duplicate known-prefix expert union option");
    { auto selected = paired_prompt; selected.known_prefix_expert_union = true;
      selected.q4k_a8g32_gate_up = selected.grouped_exact_q51_down = selected.q51_slot_parallel_down = true;
      validate_resolved_known_prefix_window(selected);
      for (unsigned missing = 0; missing < 4U; ++missing) {
          auto bad = selected;
          if (missing == 0U) bad.q4k_a8g32_gate_up = false;
          if (missing == 1U) bad.grouped_exact_q51_down = false;
          if (missing == 2U) bad.q51_slot_parallel_down = false;
          if (missing == 3U) bad.deferred_moe_status = false;
          rejects([&] { validate_resolved_known_prefix_window(bad); },
              "known-prefix expert union requires resolved A8/grouped Q5_1/slot-parallel/D1");
      }
    }
    expect(!needs_generation_output_dispatcher(defaults), "ordinary CPU output does not construct an output wrapper");
    { auto state = paired_prompt; state.known_prefix_output_policy = KnownWindowOutputPolicy::state_only;
      expect(needs_generation_output_dispatcher(state) && !state.gpu_greedy_token,
          "state-only known prefix uses head omission without a vocabulary selector");
      state.known_prefix_output_policy = KnownWindowOutputPolicy::terminal_row;
      expect(!needs_generation_output_dispatcher(state), "terminal-row windows retain ordinary serial output");
      state.known_prefix_output_policy = KnownWindowOutputPolicy::all_rows;
      expect(!needs_generation_output_dispatcher(state), "all-row windows preserve serial default output");
      state.known_prefix_window = false; state.gpu_greedy_token = true;
      expect(needs_generation_output_dispatcher(state), "existing GPU scalar output retains its wrapper"); }
    expect(defaults.known_prefix_output_policy == KnownWindowOutputPolicy::all_rows &&
        !defaults.known_prefix_output_policy_explicit, "known-window heads remain all rows by default");
    for (const auto* policy : {"all-rows", "terminal-row", "state-only"}) {
        const auto selected = parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-output-policy", policy});
        expect(known_window_output_policy_name(selected.known_prefix_output_policy) == policy &&
            !selected.any_low_level_flag, "known-window output policy does not change optimization profile");
        auto admitted = paired_prompt; admitted.known_prefix_output_policy = selected.known_prefix_output_policy;
        validate_resolved_known_prefix_window(admitted);
    }
    expect(parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-output-policy=terminal-row"}).known_prefix_output_policy ==
        KnownWindowOutputPolicy::terminal_row, "output policy supports equals form");
    rejects([] { (void)parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-output-policy=other"}); },
        "known-prefix output policy must be all-rows, terminal-row or state-only");
    rejects([] { (void)parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-output-policy"}); },
        "known-prefix output policy requires a value");
    rejects([] { (void)parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-output-policy=all-rows", "--known-prefix-output-policy=state-only"}); },
        "duplicate known-prefix output policy");
    rejects([] { (void)parse({"decoder", "m", "1", "--prompt-lookup", "--known-prefix-output-policy=terminal-row"}); },
        "known-prefix output policy requires --known-prefix-window");
    { auto bad = paired_prompt; bad.known_prefix_output_policy = static_cast<KnownWindowOutputPolicy>(255);
      rejects([&] { validate_resolved_known_prefix_window(bad); }, "unknown known-window output policy"); }
    expect(defaults.known_prefix_rows == 2 && !defaults.known_prefix_rows_explicit,
        "known-prefix logical rows default to two without changing automatic profile");
    for (const auto* text : {"2", "4", "8"}) {
        const auto selected = parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-rows", text});
        expect(selected.known_prefix_rows == parse_u64(text, "fixture") && !selected.any_low_level_flag,
            "known-prefix row selection preserves automatic profile resolution");
        auto admitted = paired_prompt; admitted.known_prefix_rows = selected.known_prefix_rows;
        validate_resolved_known_prefix_window(admitted);
    }
    expect(parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-rows=8"}).known_prefix_rows == 8,
        "known-prefix rows supports explicit equals form");
    for (const auto* text : {"0", "1", "3", "16"})
        rejects([&] { (void)parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-rows", text}); },
            "known-prefix rows must be 2, 4 or 8");
    rejects([] { (void)parse({"decoder", "m", "1", "--known-prefix-rows", "2"}); },
        "known-prefix rows require --known-prefix-window");
    rejects([] { (void)parse({"decoder", "m", "1", "--prompt-lookup", "--known-prefix-rows", "4"}); },
        "known-prefix rows require --known-prefix-window");
    rejects([] { (void)parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-rows", "4", "--known-prefix-rows=8"}); },
        "duplicate known-prefix rows option");
    { auto bad = paired_prompt; bad.known_prefix_rows = 3;
      rejects([&] { validate_resolved_known_prefix_window(bad); }, "known-prefix rows must be 2, 4 or 8"); }
    rejects([] { (void)parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-window"}); },
        "duplicate known-prefix window option");
    for (unsigned mutation = 0; mutation < 10; ++mutation) {
        auto bad = paired_prompt;
        if (mutation == 0) bad.gpu_greedy_token = true;
        if (mutation == 1) bad.prepared_eager = true;
        if (mutation == 2) bad.moe_shared_overlap = true;
        if (mutation == 3) bad.gdn_islands = true;
        if (mutation == 4) bad.qsa_hierarchical_selection = true;
        if (mutation == 5) bad.qsa_gqa_heads_per_block = 2;
        if (mutation == 6) bad.gdn_recurrent_layout = HipGdnRecurrentLayout::stride33;
        if (mutation == 7) bad.execution.mode = DecodeExecutionMode::graph;
        if (mutation == 8) bad.execution.integer_head = DecodeIntegerHead::a8_iu8;
        if (mutation == 9) bad.raw_recording_path = "unused-RAW-path";
        rejects([&] { validate_resolved_known_prefix_window(bad); },
            "known-prefix window requires eager full-head generation without RAW/reference, alternate submission, cache, islands or candidate layouts");
    }
    expect(!defaults.device_queue_trace, "device event profiling is off by default");
    validate_resolved_device_queue_trace(defaults);
    auto trace = defaults; trace.device_queue_trace = true; trace.host_dispatch_telemetry = true; trace.deferred_moe_status = true;
    validate_resolved_device_queue_trace(trace);
    for(unsigned mutation=0;mutation<8;++mutation) {
        auto bad=trace;
        if(mutation==0)bad.host_dispatch_telemetry=false;
        if(mutation==1)bad.deferred_moe_status=false;
        if(mutation==2)bad.execution.mode=DecodeExecutionMode::graph;
        if(mutation==3)bad.serving_prefix_files={"not-opened"};
        if(mutation==4)bad.known_prefix_window=true;
        if(mutation==5)bad.prompt_lookup=true;
        if(mutation==6)bad.prefix_cache_roundtrip=true;
        if(mutation==7)bad.execution.integer_head=DecodeIntegerHead::a8_iu8;
        rejects([&]{validate_resolved_device_queue_trace(bad);},
            "device queue trace requires host telemetry and eager D1 C1 without graph, integer, serving, known-prefix, lookup or cache roundtrip");
    }
    expect(parse({"decoder","m","1","--host-dispatch-telemetry","--device-queue-trace"}).device_queue_trace,
        "explicit device trace does not open a GPU or enable D1 implicitly");
    rejects([]{(void)parse({"decoder","m","1","--device-queue-trace"});},
        "device queue trace requires --host-dispatch-telemetry");
    rejects([]{(void)parse({"decoder","m","1","--host-dispatch-telemetry","--device-queue-trace","--device-queue-trace"});},
        "duplicate device queue trace option");
    expect(!defaults.prompt_lookup_adaptive, "adaptive lookup is default off");
    const auto adaptive_lookup = parse({"decoder", "m", "1", "--prompt-lookup", "--prompt-lookup-adaptive"});
    expect(adaptive_lookup.prompt_lookup_adaptive && !adaptive_lookup.prompt_lookup_depth_explicit &&
        !adaptive_lookup.any_low_level_flag, "adaptive lookup does not replace automatic kernel profile");
    rejects([] { (void)parse({"decoder", "m", "1", "--prompt-lookup-adaptive"}); },
        "adaptive prompt lookup requires --prompt-lookup without an explicit fixed depth");
    rejects([] { (void)parse({"decoder", "m", "1", "--prompt-lookup", "--prompt-lookup-adaptive", "--prompt-lookup-depth", "2"}); },
        "adaptive prompt lookup requires --prompt-lookup without an explicit fixed depth");
    rejects([] { (void)parse({"decoder", "m", "1", "--prompt-lookup", "--prompt-lookup-adaptive", "--prompt-lookup-adaptive"}); },
        "duplicate prompt lookup adaptive option");
    expect(!defaults.prompt_lookup, "prompt lookup remains opt-in");
    validate_resolved_prompt_lookup(defaults);
    auto lookup = paired_prompt;
    lookup.known_prefix_window = false; lookup.prompt_lookup = true;
    for (std::uint32_t depth = 0; depth <= 2; ++depth) {
        lookup.prompt_lookup_depth = depth;
        validate_resolved_prompt_lookup(lookup);
    }
    expect(parse({"decoder", "m", "1", "--prompt-lookup", "--prompt-lookup-depth", "0"}).prompt_lookup_depth == 0,
        "explicit depth zero preserves the device frontier route");
    rejects([] { (void)parse({"decoder", "m", "1", "--prompt-lookup-depth", "1"}); },
        "depth flag cannot silently enable lookup");
    rejects([] { (void)parse({"decoder", "m", "1", "--prompt-lookup", "--prompt-lookup-depth", "3"}); },
        "lookup cannot manufacture an unsupported target depth");
    rejects([] { (void)parse({"decoder", "m", "1", "--prompt-lookup", "--prompt-lookup"}); },
        "duplicate lookup flag");
    for (unsigned mutation = 0; mutation < 14; ++mutation) {
        auto bad = lookup;
        if (mutation == 0) bad.known_prefix_window = true;
        if (mutation == 1) bad.prepared_eager = true;
        if (mutation == 2) bad.moe_shared_overlap = true;
        if (mutation == 3) bad.gdn_islands = true;
        if (mutation == 4) bad.gpu_greedy_token = true;
        if (mutation == 5) bad.execution.mode = DecodeExecutionMode::graph;
        if (mutation == 6) bad.execution.integer_head = DecodeIntegerHead::a8_iu8;
        if (mutation == 7) bad.prefix_cache_roundtrip = true;
        if (mutation == 8) bad.reference_path = "not-opened";
        if (mutation == 9) bad.raw_recording_path = "not-opened";
        if (mutation == 10) bad.qsa_hierarchical_selection = true;
        if (mutation == 11) bad.qsa_gqa_heads_per_block = 4;
        if (mutation == 12) bad.gdn_recurrent_layout = HipGdnRecurrentLayout::stride32;
        if (mutation == 13) bad.generate_tokens = 0;
        rejects([&] { validate_resolved_prompt_lookup(bad); }, "lookup rejects incompatible explicit route without disabling its flags");
    }
    auto missing_lookup_status = lookup; missing_lookup_status.deferred_moe_status = false;
    rejects([&] { validate_resolved_prompt_lookup(missing_lookup_status); }, "lookup requires the actual D1 owner");
    auto missing_lookup_wave = lookup; missing_lookup_wave.wave8_q8_gdn_input = false;
    rejects([&] { validate_resolved_prompt_lookup(missing_lookup_wave); }, "lookup requires the actual paired GDN variant");
    paired_prompt.wave8_q8_gdn_output = false;
    rejects([&] { validate_resolved_known_prefix_window(paired_prompt); },
        "known-prefix window requires resolved D1 and explicit paired wave8 GDN input/output");
    expect(defaults.qsa_gqa_heads_per_block == 1 && defaults.gdn_recurrent_layout == HipGdnRecurrentLayout::baseline,
        "existing QSA and GDN organizations remain default");
    for (const auto* value : {"1", "2", "4"}) {
        auto grouped = parse({"decoder", "fixture.gguf", "1", "--qsa-gqa-heads-per-block", value});
        const auto selected_grouped = resolve_optimization_profile(grouped, model, info, manifest,
            validated_model_content_identity, true);
        validate_resolved_qsa_gqa(grouped);
        expect(selected_grouped.automatic_applied && !grouped.any_low_level_flag,
            "explicit GQA geometry preserves the automatic arithmetic profile");
    }
    rejects([] { (void)parse({"decoder", "m", "1", "--qsa-gqa-heads-per-block", "3"}); },
        "QSA heads per block must be 1, 2 or 4");
    rejects([] { (void)parse({"decoder", "m", "1", "--qsa-gqa-heads-per-block", "2", "--qsa-gqa-heads-per-block", "4"}); },
        "QSA heads per block requires one value");
    for (const auto* layout : {"baseline", "stride32", "stride33", "transposed"}) {
        auto chosen_layout = parse({"decoder", "fixture.gguf", "1", "--gdn-recurrent-layout", layout});
        const auto layout_profile = resolve_optimization_profile(chosen_layout, model, info, manifest,
            validated_model_content_identity, true);
        validate_resolved_gdn_recurrent_layout(chosen_layout);
        expect(chosen_layout.gdn_recurrent_layout_explicit && !chosen_layout.any_low_level_flag && layout_profile.automatic_applied,
            "explicit GDN LDS geometry preserves automatic selection");
    }
    expect(parse({"decoder", "m", "1", "--gdn-recurrent-layout=stride33"}).gdn_recurrent_layout == HipGdnRecurrentLayout::stride33,
        "GDN recurrent layout also accepts the explicit equals form");
    rejects([] { (void)parse({"decoder", "m", "1", "--gdn-recurrent-layout", "other"}); },
        "GDN recurrent layout must be baseline, stride32, stride33 or transposed");
    auto bad_layout = defaults; bad_layout.gdn_recurrent_layout = HipGdnRecurrentLayout::stride32; bad_layout.gdn_islands = true;
    rejects([&] { validate_resolved_gdn_recurrent_layout(bad_layout); },
        "GDN recurrent LDS layout requires eager floating execution without islands");
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
    auto hierarchical = parse({"decoder", "fixture.gguf", "1", "--qsa-hierarchical-selection"});
    const auto hierarchical_profile = resolve_optimization_profile(hierarchical, model, info, manifest,
        validated_model_content_identity, true);
    validate_resolved_qsa_hierarchical_selection(hierarchical);
    expect(hierarchical.qsa_hierarchical_selection && !hierarchical.any_low_level_flag && hierarchical_profile.automatic_applied,
        "hierarchical selection preserves automatic kernel profile selection");
    rejects([] { (void)parse({"decoder", "m", "1", "--qsa-hierarchical-selection", "--qsa-hierarchical-selection"}); },
        "duplicate hierarchical QSA option");
    for (unsigned mutation = 0; mutation < 5; ++mutation) {
        auto bad = hierarchical;
        if (mutation == 0) bad.deferred_moe_status = false;
        if (mutation == 1) bad.execution.mode = DecodeExecutionMode::graph;
        if (mutation == 2) bad.execution.integer_head = DecodeIntegerHead::a8_iu8;
        if (mutation == 3) bad.prefix_cache_roundtrip = true;
        if (mutation == 4) bad.context_tokens = 131073;
        rejects([&] { validate_resolved_qsa_hierarchical_selection(bad); },
            "hierarchical QSA requires eager D1 without integer head or roundtrip and context<=131072");
    }
#else
    rejects([] { (void)parse({"decoder", "m", "1", "--qsa-hierarchical-selection"}); },
        "--qsa-hierarchical-selection requires the experimental runtime build capability");
#endif
    auto serving = defaults;
    serving.tokens.clear(); serving.serving_prefix_files = {"not-opened.u32le"};
    serving.generate_tokens = 4; serving.deferred_moe_status = true;
    serving.serving_memory_budget = 100; serving.serving_request_budget = 10; serving.serving_headroom = 1;
    serving.serving_memory_explicit = serving.serving_request_explicit = serving.serving_headroom_explicit = true;
    for (const auto concurrency : {1U, 2U, 4U}) {
        serving.serving_concurrency = concurrency;
        validate_resolved_serving(serving);
    }
    const auto file_request = parse({"decoder", "m", "--serve-prefix-u32le", "not-opened.u32le", "--generate", "4",
        "--serve-concurrency", "2", "--serve-memory-budget-bytes", "100", "--serve-request-budget-bytes", "10", "--serve-headroom-bytes", "1"});
    expect(file_request.tokens.empty() && file_request.serving_prefix_files.size() == 1 &&
        !file_request.any_low_level_flag, "serving parser records file requests without changing optimization profile");
    rejects([] { (void)parse({"decoder", "m", "1", "--serve-concurrency", "2"}); },
        "serving controls cannot silently change an ordinary request");
    rejects([] { (void)parse({"decoder", "m", "1", "--serve-concurrency", "3"}); },
        "serving geometry is a closed worker family");
    const auto cached_serving = parse({"decoder", "m", "--serve-prefix-u32le", "not-opened.u32le", "--generate", "4",
        "--serve-prefix-cache-entries", "2", "--serve-prefix-cache-snapshot-bytes", "4096",
        "--serve-prefix-cache-capture-peak-bytes", "8192", "--serve-prefix-cache-max-tokens", "16"});
    expect(cached_serving.serving_cache_entries == 2 && cached_serving.serving_cache_snapshot_bytes == 4096 &&
        cached_serving.serving_cache_capture_peak_bytes == 8192 && cached_serving.serving_cache_max_prefix_tokens == 16 &&
        !cached_serving.any_low_level_flag, "serving cache stores explicit independent budgets without opening files");
    expect(!defaults.serving_cache_entries_explicit && defaults.serving_cache_entries == 0,
        "serving cache is absent by default");
    rejects([] { (void)parse({"decoder", "m", "1", "--serve-prefix-cache-entries", "1"}); },
        "serving prefix cache requires serving files and all four positive explicit bounded limits");
    rejects([] { (void)parse({"decoder", "m", "--serve-prefix-u32le", "x", "--generate", "4",
        "--serve-prefix-cache-entries", "1"}); },
        "serving prefix cache requires serving files and all four positive explicit bounded limits");
    rejects([] { (void)parse({"decoder", "m", "1", "--serve-prefix-cache-entries", "0"}); },
        "serving prefix cache limits must be positive");
    rejects([] { (void)parse({"decoder", "m", "1", "--serve-prefix-cache-entries"}); },
        "serving prefix cache option requires a value");
    rejects([] { (void)parse({"decoder", "m", "1", "--serve-prefix-cache-entries", "1", "--serve-prefix-cache-entries", "2"}); },
        "duplicate serving prefix cache option");
    rejects([] { (void)parse({"decoder", "m", "--serve-prefix-u32le", "x", "--generate", "4", "--context", "8",
        "--serve-prefix-cache-entries", "1", "--serve-prefix-cache-snapshot-bytes", "4096",
        "--serve-prefix-cache-capture-peak-bytes", "8192", "--serve-prefix-cache-max-tokens", "16"}); },
        "serving prefix cache requires serving files and all four positive explicit bounded limits");
    for (unsigned mutation = 0; mutation < 6; ++mutation) {
        auto bad = cached_serving;
        if (mutation == 0) bad.serving_cache_entries_explicit = false;
        if (mutation == 1) bad.serving_cache_snapshot_explicit = false;
        if (mutation == 2) bad.serving_cache_peak_explicit = false;
        if (mutation == 3) bad.serving_cache_tokens_explicit = false;
        if (mutation == 4) bad.serving_cache_max_prefix_tokens = bad.context_tokens + 1U;
        if (mutation == 5) bad.serving_prefix_files.clear();
        rejects([&] { validate_serving_prefix_cache_limits(bad); },
            "serving prefix cache requires serving files and all four positive explicit bounded limits");
    }
    for (unsigned mutation = 0; mutation < 13; ++mutation) {
        auto bad = serving;
        if (mutation == 0) bad.serving_headroom_explicit = false;
        if (mutation == 1) bad.serving_request_budget = 0;
        if (mutation == 2) bad.serving_headroom = bad.serving_memory_budget;
        if (mutation == 3) bad.moe_shared_overlap = true;
        if (mutation == 4) bad.gdn_islands = true;
        if (mutation == 5) bad.known_prefix_window = true;
        if (mutation == 6) bad.prompt_lookup = true;
        if (mutation == 7) bad.prefix_cache_roundtrip = true;
        if (mutation == 8) bad.execution.mode = DecodeExecutionMode::graph;
        if (mutation == 9) bad.execution.integer_head = DecodeIntegerHead::a8_iu8;
        if (mutation == 10) bad.tokens = {1};
        if (mutation == 11) bad.execution_inventory = true;
        if (mutation == 12) bad.deferred_moe_status = false;
        rejects([&] { validate_resolved_serving(bad); }, "serving rejects missing reservations and unsupported explicit modes");
    }
    auto stored = defaults;
    stored.gdn_recurrent_storage = GdnRecurrentStorage::value_key;
    validate_resolved_gdn_recurrent_storage(stored);
    expect(parse({"decoder", "m", "1", "--gdn-recurrent-storage=value-key"}).gdn_recurrent_storage == GdnRecurrentStorage::value_key,
        "physical storage has an explicit independent selector");
    for (unsigned mutation = 0; mutation < 5; ++mutation) {
        auto bad = stored;
        if (mutation == 0) bad.gdn_islands = true;
        if (mutation == 1) bad.known_prefix_window = true;
        if (mutation == 2) bad.prompt_lookup = true;
        if (mutation == 3) bad.execution.mode = DecodeExecutionMode::graph;
        if (mutation == 4) bad.execution.integer_head = DecodeIntegerHead::a8_iu8;
        rejects([&] { validate_resolved_gdn_recurrent_storage(bad); }, "physical state storage does not enter unadmitted graph/R2 modes");
    }
    rejects([] { (void)parse({"decoder", "m", "1", "--gdn-recurrent-storage", "unknown"}); }, "storage selector rejects unknown versions");
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
