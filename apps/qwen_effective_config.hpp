#pragma once
#include "hip_decode_options.hpp"
#include "strix/known_prefix_prefill.hpp"
#include "strix/linux_uma_observation.hpp"
#include "strix/uma_owner_inventory.hpp"
#include <map>
#ifdef __linux__
#include <unistd.h>
#endif

namespace strix::decode_cli {
using Environment = std::map<std::string, std::string>;
enum class OptionSource { explicit_value, automatic, baseline };
struct RequestedOptions {
    Options options;
    Environment environment;
    bool permit_generation_defaults = true;
};
struct PrefillCapabilities {
    bool a8_runtime = false;
    bool target_qualified = false;
    bool fp16_operands = false;
};
// These are simultaneous backing reservations, never RSS plus GTT. Registered
// aliases are already part of model_backing and must not be added a second time.
struct PrefillRequirements {
    std::uint64_t model_backing = 0, state_d1 = 0, serial_scratch = 0;
    std::uint64_t workspace = 0, metadata = 0, runtime_payload = 0, tails = 0;
    std::uint64_t device_base = 0;
    std::map<std::string, std::uint64_t> organization{};
};
struct PrefillMemory {
    std::optional<std::uint64_t> available, gtt_limit, gtt_used;
    std::function<PrefillRequirements(const Options&)> estimate;
    // Present only for Linux reclaim support; host tests can supply the observation.
    std::optional<std::uint64_t> ple_reclaim_page{};
};
struct EffectiveConfig {
    Options options;
    OptimizationProfileResolution organization;
    Environment environment;
    std::map<std::string, OptionSource> sources;
    std::string reason = "not-requested";
    std::string arithmetic_fallback;
    std::string tile_fallback;
    std::string error;
    std::uint64_t ple_reclaim_credit = 0;
    std::vector<std::string> suppressed;
    bool admitted = false;
    bool gsq_owner = false;
    std::map<std::string, std::uint64_t> host_compilation;
};
inline bool env_is(const Environment& env, const char* name, std::string_view value) {
    const auto it = env.find(name);
    return it != env.end() && it->second == value;
}
template<class Get>
inline Environment requested_prefill_environment(Get get) {
    Environment env;
    for (const auto& item : qpf2k::defaults)
        if (!(qpf2k::pre_injected & qpf2k::bit(item.item)))
            if (const char* value = get(item.name)) env.emplace(item.name, value);
    for (const char* name : {"STRIX_QPF2K_DEFAULTS", "STRIX_QPF2K_MOE_DEFAULTED", "STRIX_BASELINE", "STRIX_PREFILL_F16",
            "STRIX_KNOWN_PREFIX_WINDOW", "STRIX_KNOWN_PREFIX_FULL_ROWS", "STRIX_KNOWN_PREFIX_EXPERT_UNION",
            "STRIX_KNOWN_WINDOW_ROUTER_ROWS", "STRIX_KNOWN_PREFIX_PREFILL_ROWS",
            "STRIX_RTX_EXPORT_STATE", "STRIX_QRTX_STEP_DUMP", "STRIX_PROMPT_BATCH_ROWS", "STRIX_PREFILL_LOGITS_DIGEST",
            "STRIX_MODEL_COPY_ALL", "STRIX_PLE_RECLAIM_DECODE_BYTES"})
        if (!(qpf2k::pre_injected_moe_marker && std::string_view(name) == "STRIX_QPF2K_MOE_DEFAULTED"))
            if (const char* value = get(name)) env.emplace(name, value);
    return env;
}
inline Environment requested_prefill_environment() {
    return requested_prefill_environment([](const char* name) { return std::getenv(name); });
}
inline void apply_measured_prompt_organization(Options& o) {
    o.q51_slot_parallel_down = true;
    o.batched_moe_activation = true;
    o.wave8_q8_gdn_input = true;
    o.wave8_q8_gdn_output = true;
    o.prefetch_q8_lm_head = true;
    o.wave8_q8_lm_head = true;
    o.wave8_q8_qsa_output = true;
}
inline constexpr DecodeNumericalProfile dense_a8_profile =
    DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_q5k_a8_gateup_dense_a8_v1;
inline std::uint64_t prefill_checked_add(std::uint64_t a, std::uint64_t b) {
    if (b > UINT64_MAX - a) throw std::overflow_error("prefill reservation overflow");
    return a + b;
}
// Predict only the route's reclaim credit. Runtime still verifies the drop and
// admits allocations against the actual credit, including zero on setup failure.
inline std::uint64_t prefill_reclaim_credit(const Options& options, const ModelContract& model,
    const ModelManifest& manifest, const Environment& environment, std::optional<std::uint64_t> page) {
    if (!page || !*page) return 0;
    const auto mode = environment.find("STRIX_PLE_RECLAIMABLE");
    if (mode != environment.end() && mode->second != "auto" && mode->second != "1") return 0;
    const auto copy = environment.find("STRIX_MODEL_COPY_ALL");
    if (copy != environment.end() && !copy->second.empty() && copy->second[0] != '0') return 0;
    HipModelResidencyPolicy policy;
    policy.copy_tier = options.model_copy_tier;
    const bool graph = options.execution.mode != DecodeExecutionMode::eager;
    policy.promote_file_backed_candidates = graph;
    if (!ple_reclaim_route_supported(options, policy, graph)) return 0;
    try {
        const ManifestTensor* table = nullptr;
        std::vector<ple_reclaim::Interval> registered;
        for (const auto& tensor : manifest.tensors) {
            if (tensor.residency == ResidencyPolicy::file_backed_candidate) {
                if (table || tensor.role != TensorRole::ngram_table || tensor.dimensions.size() != 2U ||
                    (tensor.ggml_type != 20U && tensor.ggml_type != 30U) ||
                    (tensor.ggml_type == 20U && tensor.dimensions[0] % 32U)) return 0;
                table = &tensor;
            } else if (tensor.placement.kind == PlacementKind::local) {
                registered.push_back({tensor.absolute_offset / *page * *page,
                    prefill_checked_add(prefill_checked_add(tensor.absolute_offset, tensor.byte_size), *page - 1U) / *page * *page});
            }
        }
        if (!table) return 0;
        std::sort(registered.begin(), registered.end(), [](const auto& a, const auto& b) { return a.begin < b.begin; });
        std::vector<ple_reclaim::Interval> merged;
        for (const auto& range : registered) {
            if (!merged.empty() && range.begin <= prefill_checked_add(merged.back().end, options.coalesce_gap_bytes))
                merged.back().end = std::max(merged.back().end, range.end);
            else merged.push_back(range);
        }
        ple_reclaim::Config config;
        config.page = *page;
        config.table_offset = table->absolute_offset;
        config.table_rows = table->dimensions[1];
        config.stored_row_bytes = table->ggml_type == 20U ? table->dimensions[0] / 32U * 18U : table->dimensions[0] * 2U;
        if (!config.stored_row_bytes || config.table_rows > UINT64_MAX / config.stored_row_bytes ||
            config.stored_row_bytes * config.table_rows != table->byte_size) return 0;
        config.mapping_bytes = prefill_checked_add(config.table_offset, table->byte_size);
        config.reclaim = ple_reclaim::reclaim_intervals(config.table_offset, table->byte_size, *page, std::move(merged));
        const auto contract = build_ple_hash_contract(model);
        const auto allowance = environment.find("STRIX_PLE_RECLAIM_DECODE_BYTES");
        config.decode_allowance_bytes = ple_reclaim::decode_allowance(options.generate_tokens, 1U, contract.head_count, *page,
            allowance == environment.end() ? 0U : std::strtoull(allowance->second.c_str(), nullptr, 10));
        return ple_reclaim::plan_prompt(config, contract, options.tokens).stats.credit_bytes;
    } catch (const std::exception&) { return 0; }
}
inline void require_effective_prefill_execution(const EffectiveConfig& config) {
    if (!config.error.empty()) throw std::invalid_argument(config.error);
}
inline void validate_effective_options(const Options& o, const ModelContract& model) {
    validate_resolved_mtp_cli(o);
    validate_resolved_prefix_roundtrip(o);
    validate_resolved_q8_projection_organization(o);
    validate_resolved_hc_projection_next_tile(o);
    validate_resolved_q51_slot_parallel_down(o);
    validate_resolved_gdn_islands(o);
    validate_resolved_gdn_recurrent_layout(o);
    validate_resolved_gdn_recurrent_storage(o);
    validate_resolved_attention_kv_storage(o);
    validate_resolved_qsa_hierarchical_selection(o);
    validate_resolved_qsa_gqa(o);
    validate_resolved_known_prefix_window(o);
    validate_moe_token_cost_options(o);
    validate_resolved_prompt_lookup(o);
    validate_resolved_serving(o);
    validate_resolved_ple_cpu_helpers(o);
    validate_resolved_ple_cache(o);
    validate_resolved_device_queue_trace(o);
    validate_resolved_requested_logprobs(o, model.vocab_size);
    validate_resolved_teacher_forced_score(o, model.vocab_size);
}
inline EffectiveConfig resolve_effective_config(const RequestedOptions& request,
    const ModelContract& model, const gguf::FileInfo& info, const ModelManifest& manifest,
    std::string_view verified_content, const PrefillCapabilities& capabilities, const PrefillMemory& memory) {
    EffectiveConfig result;
    result.options = request.options;
    result.gsq_owner = qpf2k::has_gsq_gate(manifest);
    result.environment = request.environment;
    for (const auto& name : request.options.supplied_options) result.sources[name] = OptionSource::explicit_value;
    auto& o = result.options;
    const bool baseline = o.baseline || env_is(request.environment, "STRIX_BASELINE", "1");
    const auto inject = [&](unsigned allowed) {
        const unsigned injected = qpf2k::initialize([&](const char* name) -> const char* {
            const auto it = result.environment.find(name);
            return it == result.environment.end() ? nullptr : it->second.c_str();
        }, [&](const char* name, const char* value) { result.environment.emplace(name, value); return true; }, allowed);
        for (const auto& entry : qpf2k::defaults)
            if (result.environment.contains(entry.name)) result.sources[entry.name] =
                injected & qpf2k::bit(entry.item) ? OptionSource::automatic : OptionSource::explicit_value;
        if (result.environment.contains("STRIX_QPF2K_MOE_DEFAULTED"))
            result.sources["STRIX_QPF2K_MOE_DEFAULTED"] = request.environment.contains("STRIX_QPF2K_MOE_DEFAULTED") ?
                OptionSource::explicit_value : OptionSource::automatic;
    };
    // Baseline disables the new policy and all controlled environment items;
    // argv organization/arithmetic still goes through the historical resolver.
    o.baseline = baseline;
    if (baseline) {
        for (const auto& entry : qpf2k::defaults) {
            if (result.environment.contains(entry.name)) result.suppressed.emplace_back(entry.name);
            result.environment[entry.name] = "0";
            result.sources[entry.name] = OptionSource::baseline;
        }
        result.environment["STRIX_QPF2K_DEFAULTS"] = "0";
        result.sources["STRIX_QPF2K_DEFAULTS"] = OptionSource::baseline;
    }
    const auto decline = [&](std::string reason) {
        result.reason = std::move(reason);
        if (!baseline) inject(qpf2k::historical_items);
        if (o.prefill_profile_deferred && !o.numerical_profile_explicit) o.numerical_profile = dense_a8_profile;
        const auto schedule = result.environment.find("STRIX_QPF2K_SCHED");
        const bool injected_gsq = qpf2k::has_gsq_gate(manifest) && schedule != result.environment.end() &&
            !request.environment.contains("STRIX_QPF2K_SCHED");
        apply_default_known_prefill_tails(o, injected_gsq || schedule == result.environment.end() ? "0" : schedule->second.c_str());
        result.organization = resolve_optimization_profile(o, model, info, manifest, verified_content,
            capabilities.a8_runtime, request.permit_generation_defaults);
        auto validation_organization = result.organization;
        apply_numerical_profile(o, model, manifest, verified_content, capabilities.a8_runtime, validation_organization);
        validate_effective_options(o, model);
        if (result.reason == "prompt-exceeds-model-context")
            result.error = "Qwen auto-prefill declined: prompt-exceeds-model-context (prompt + generation exceeds " +
                std::to_string(model.max_position_embeddings) + ")";
        else if (!o.known_prefix_window && o.tokens.size() > o.context_tokens)
            result.error = "Qwen auto-prefill declined: " + result.reason + "; prompt has " + std::to_string(o.tokens.size()) +
                " tokens, exceeding serial prefill context " + std::to_string(o.context_tokens) + "; cannot use serial prefill";
        return result;
    };
    if (baseline) return decline("baseline");
    if (o.route_owned || o.any_low_level_flag || o.known_prefix_window || o.prompt_lookup || o.http_serve ||
        o.prefix_cache_roundtrip || !o.serving_prefix_files.empty() || o.mtp.supplied || o.mtp.enabled || !o.mtp.cohort.empty() || o.teacher_forced_score ||
        !o.raw_recording_path.empty() || !o.validation_final_state_path.empty()) return decline("explicit-route");
    for (const char* name : {"STRIX_RTX_EXPORT_STATE", "STRIX_QRTX_STEP_DUMP", "STRIX_PROMPT_BATCH_ROWS", "STRIX_PREFILL_LOGITS_DIGEST"}) {
        const auto control = request.environment.find(name);
        if (control != request.environment.end() && !control->second.empty() && control->second != "0")
            return decline("explicit-diagnostic-or-execution-control");
    }
    for (const auto& entry : qpf2k::defaults)
        if (!(qpf2k::exact_prefill_items & qpf2k::bit(entry.item)) &&
            env_is(request.environment, entry.name, "0")) return decline("feature-disabled");
    const auto defaults = request.environment.find("STRIX_QPF2K_DEFAULTS");
    if (defaults != request.environment.end() && !defaults->second.empty() && defaults->second[0] == '0')
        return decline("feature-disabled");
    if (env_is(request.environment, "STRIX_PREFILL_F16", "0")) return decline("feature-disabled");
    if (!o.disabled_prefill_features.empty()) return decline("feature-disabled");
    for (const char* name : {"STRIX_KNOWN_PREFIX_WINDOW", "STRIX_KNOWN_PREFIX_FULL_ROWS",
            "STRIX_KNOWN_PREFIX_EXPERT_UNION", "STRIX_KNOWN_WINDOW_ROUTER_ROWS", "STRIX_KNOWN_PREFIX_PREFILL_ROWS"})
        if (env_is(request.environment, name, "0")) return decline("feature-disabled");
    if (o.tokens.size() < 16U || !o.generate_tokens) return decline("prompt-not-eligible");
    if (qpf2k::has_gsq_gate(manifest) || verified_content != validated_model_content_identity)
        return decline("checkpoint-content-not-qualified");
    if (!capabilities.target_qualified) return decline("target-not-qualified");
    if (!capabilities.a8_runtime) return decline("a8-runtime-not-built");
    if (!o.context_explicit && (o.tokens.size() > model.max_position_embeddings ||
        o.generate_tokens > model.max_position_embeddings - static_cast<std::uint64_t>(o.tokens.size())))
        return decline("prompt-exceeds-model-context");
    if (!memory.available || !memory.gtt_limit || !memory.gtt_used || !memory.estimate)
        return decline("memory-observation-unavailable");
    auto candidate = o;
    auto organization = resolve_optimization_profile(candidate, model, info, manifest, verified_content,
        capabilities.a8_runtime, false);
    if (!organization.automatic_applied) return decline(organization.reason);
    try {
        candidate.automatic_prefill = true;
        candidate.known_prefix_window = true;
        candidate.known_prefix_full_rows = true;
        candidate.known_prefix_expert_union = true;
        candidate.known_window_router_rows = true;
        candidate.known_prefix_output_policy = KnownWindowOutputPolicy::terminal_row;
        for (const auto rows : {8192U,4096U,2048U,1024U,256U,64U,16U})
            if (rows <= candidate.tokens.size()) { candidate.known_prefix_prefill_rows = rows; break; }
        if (!o.context_explicit) {
            const auto needed = prefill_checked_add(static_cast<std::uint64_t>(o.tokens.size()), std::max(o.generate_tokens, UINT64_C(512)));
            candidate.context_tokens = std::min(model.max_position_embeddings,
                std::max(UINT64_C(8192), prefill_checked_add(needed, 511U) / 512U * 512U));
            candidate.unqualified_context = candidate.context_tokens != 8192U;
        }
        apply_measured_prompt_organization(candidate);
        if (!candidate.numerical_profile_explicit) candidate.numerical_profile = dense_a8_profile;
        if (!candidate.prefill_numerical_profile && !capabilities.fp16_operands &&
            candidate.numerical_profile == dense_a8_profile && !env_is(request.environment, "STRIX_PREFILL_F16", "0"))
            result.arithmetic_fallback = "fp16-operands-not-qualified";
        if (!candidate.prefill_numerical_profile && capabilities.fp16_operands &&
            candidate.numerical_profile == dense_a8_profile && !env_is(request.environment, "STRIX_PREFILL_F16", "0")) {
            candidate.prefill_numerical_profile = DecodeNumericalProfile::fast_prefill_f16wmma_v1;
            candidate.prefill_profile_automatic = true;
        }
        constexpr std::uint64_t mib = 1024U * 1024U, gib = 1024U * mib;
        const auto uma = std::min(120U * gib, *memory.available) / mib * mib;
        const auto scale = [&](std::uint64_t ceiling) { return (uma / mib * ceiling / 120U) * mib; };
        candidate.known_prefill_workspace_budget = scale(8U);
        candidate.known_prefill_metadata_budget = scale(1U);
        candidate.known_prefill_uma_budget = uma;
        candidate.known_prefill_host_headroom = std::max(gib, scale(4U));
        const auto primary_rows = candidate.known_prefix_prefill_rows;
        const auto primary = candidate;
        const auto arithmetic_fallback = result.arithmetic_fallback;
        const auto credit = prefill_reclaim_credit(candidate, model, manifest, request.environment, memory.ple_reclaim_page);
        std::vector<std::uint32_t> trials{primary_rows};
        for (const auto rows : {4096U,2048U}) if (rows < primary_rows) trials.push_back(rows);
        std::string failed_reason = "insufficient-memory";
        for (const auto rows : trials) {
            candidate = primary;
            result.arithmetic_fallback = arithmetic_fallback;
            candidate.known_prefix_prefill_rows = rows;
            if (rows == 2048U && rows != primary_rows) {
                candidate.known_prefix_tail_rows = {256U,16U};
                candidate.known_prefix_ladder_tail_rows = candidate.known_prefix_tail_rows;
            }
            try {
                apply_default_known_prefill_tails(candidate, result.environment.contains("STRIX_QPF2K_SCHED") ?
                    result.environment.at("STRIX_QPF2K_SCHED").c_str() : "12");
                if (candidate.known_prefix_ladder_tail_rows.empty())
                    candidate.known_prefix_ladder_tail_rows = candidate.known_prefix_tail_rows;
                validate_known_prefill_options(candidate);
                validate_numerical_profile_cli(candidate);
                validate_prefill_numerical_profile_cli(candidate);
                validate_effective_options(candidate, model);
                auto validation_organization = organization;
                apply_numerical_profile(candidate, model, manifest, verified_content, capabilities.a8_runtime, validation_organization);
                PrefillRequirements required;
                try { required = memory.estimate(candidate); }
                catch (const std::exception&) {
                    if (!candidate.prefill_profile_automatic) throw;
                    candidate.prefill_numerical_profile.reset();
                    candidate.prefill_profile_automatic = false;
                    result.arithmetic_fallback = "fp16-plan-not-admitted";
                    validate_prefill_numerical_profile_cli(candidate);
                    required = memory.estimate(candidate);
                }
                const auto add = prefill_checked_add;
                auto live = UmaOwnerInventory{};
                live.reservation_upper_bound_bytes = add(required.model_backing - std::min(required.model_backing, credit),
                    add(required.state_d1, required.serial_scratch));
                const auto growth = add(required.workspace, add(required.metadata, add(required.tails, candidate.known_prefill_host_headroom)));
                if (!required.workspace || !required.metadata || add(required.workspace, required.tails) > candidate.known_prefill_workspace_budget ||
                    required.metadata > candidate.known_prefill_metadata_budget || *memory.gtt_used > *memory.gtt_limit ||
                    add(required.device_base, add(required.workspace, required.tails)) > *memory.gtt_limit - *memory.gtt_used)
                    throw std::runtime_error("insufficient-memory");
                if (add(add(live.reservation_upper_bound_bytes, growth), required.runtime_payload) > uma)
                    throw std::runtime_error("insufficient-memory");
                (void)plan_uma_reservation_growth(live, growth, required.runtime_payload, {uma, uma});
                result.ple_reclaim_credit = credit;
                result.host_compilation = required.organization;
                for (const char* name : {"--known-prefix-window", "--known-prefix-full-rows", "--known-prefix-expert-union",
                        "--known-window-router-rows", "--known-prefix-prefill-rows", "--known-prefix-output-policy",
                        "--known-prefix-workspace-budget-bytes", "--known-prefix-host-metadata-budget-bytes",
                        "--known-prefix-uma-budget-bytes", "--known-prefix-host-headroom-bytes"})
                    result.sources[name] = OptionSource::automatic;
                for (const char* name : {"--parallel-hyper-mix", "--staged-hyper-projections",
                        "--fused-staged-hyper-mix", "--typed-fused-hyper-projections", "--prefetch-typed-hyper-rows",
                        "--prefetch-q8-hyper-output", "--lane-chain-hyper-output", "--prefetch-q8-qsa-input",
                        "--parallel-moe-router", "--wave-moe-router-topk", "--virtual-q8-gdn-input",
                        "--prefetch-q8-gdn-input", "--virtual-q8-gdn-output", "--deferred-moe-status",
                        "--prefetch-q8-shared-expert", "--q4k-a8g32-gate-up", "--grouped-exact-q51-down",
                        "--virtual-q8-lm-head", "--q51-slot-parallel-down", "--batched-moe-activation",
                        "--wave8-q8-gdn-input", "--wave8-q8-gdn-output", "--prefetch-q8-lm-head", "--wave8-q8-lm-head",
                        "--wave8-q8-qsa-output"})
                    result.sources[name] = OptionSource::automatic;
                if (!o.context_explicit) {
                    result.sources["--context"] = OptionSource::automatic;
                    if (candidate.unqualified_context) result.sources["--unqualified-context"] = OptionSource::automatic;
                }
                if (!o.numerical_profile_explicit) result.sources["--numerical-profile"] = OptionSource::automatic;
                if (candidate.prefill_profile_automatic) result.sources["--prefill-numerical-profile"] = OptionSource::automatic;
                if (rows == 2048U && rows != primary_rows) result.sources["--known-prefix-tail-rows"] = OptionSource::automatic;
                result.options = std::move(candidate);
                result.organization = std::move(organization);
                result.reason = "qualified-prompt";
                result.admitted = true;
                inject(~0U);
                return result;
            } catch (const std::overflow_error&) { throw; }
              catch (const KnownPrefillBudgetExceeded&) { failed_reason = "insufficient-memory"; }
              catch (const std::runtime_error&) { failed_reason = "insufficient-memory"; }
              catch (const std::exception&) { failed_reason = "candidate-not-admitted"; }
            if (!result.tile_fallback.empty()) result.tile_fallback += ',';
            result.tile_fallback += "T" + std::to_string(rows) + ':' + failed_reason;
        }
        return decline(failed_reason);
    } catch (const std::overflow_error&) { return decline("memory-overflow"); }
      catch (const std::runtime_error&) { return decline("insufficient-memory"); }
      catch (const std::exception&) { return decline("candidate-not-admitted"); }
    return result;
}
// Read sysfs only when one unambiguously selected KFD GPU is visible. Ambiguous
// visibility is unknown capacity and declines; HIP is never initialized here.
inline PrefillMemory observe_prefill_memory(PrefillCapabilities& capabilities) {
    PrefillMemory result;
    const auto observed = observe_linux_uma_memory();
    if (observed.memory.observation_identity) result.available = observed.memory.mem_available_bytes;
#ifdef __linux__
    const auto page = ::sysconf(_SC_PAGESIZE);
    if (page > 0) result.ple_reclaim_page = static_cast<std::uint64_t>(page);
    for (const char* name : {"HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"})
        if (const char* value = std::getenv(name); value && *value) return result;
    std::error_code error;
    const std::filesystem::path nodes("/sys/class/kfd/kfd/topology/nodes");
    std::optional<std::uint64_t> minor;
    std::uint64_t gfx = 0;
    unsigned node_count = 0;
    for (std::filesystem::directory_iterator node(nodes, error), end; !error && node != end; node.increment(error)) {
        if (++node_count > 256U) return result;
        std::ifstream properties_file(node->path() / "properties", std::ios::binary);
        std::array<char, 65537U> payload{};
        properties_file.read(payload.data(), static_cast<std::streamsize>(payload.size()));
        const auto bytes = properties_file.gcount();
        if (bytes < 0 || bytes > 65536) return result;
        std::istringstream properties(std::string(payload.data(), static_cast<std::size_t>(bytes)));
        std::string key;
        std::uint64_t value = 0, render = 0, target = 0;
        while (properties >> key >> value) {
            if (key == "drm_render_minor") render = value;
            if (key == "gfx_target_version") target = value;
        }
        if (!render) continue;
        if (minor) return result;
        minor = render; gfx = target;
    }
    if (!minor || error) return result;
    const auto device = std::filesystem::path("/sys/class/drm") / ("renderD" + std::to_string(*minor)) / "device";
    const auto read = [&](const char* name) -> std::optional<std::uint64_t> {
        std::ifstream file(device / name);
        std::string text;
        if (!std::getline(file, text) || text.size() > 32U) return {};
        try { return parse_u64(text, "GTT observation"); } catch (const std::exception&) { return {}; }
    };
    result.gtt_limit = read("mem_info_gtt_total");
    result.gtt_used = read("mem_info_gtt_used");
    capabilities.target_qualified = gfx == 110501U;
    capabilities.fp16_operands = capabilities.target_qualified;
#endif
    return result;
}
inline std::string effective_config_notice(const EffectiveConfig& config) {
    if (config.admitted) {
        std::string line = "Qwen auto-prefill: automatic T" + std::to_string(config.options.known_prefix_prefill_rows) +
            " context=" + std::to_string(config.options.context_tokens);
        line += "; reason=" + config.reason;
        if (!config.tile_fallback.empty()) line += "; " + config.tile_fallback;
        if (config.options.context_tokens > 8192U)
            line += "; unqualified-context (beyond qualified 8192; " +
                std::string(config.sources.contains("--unqualified-context") &&
                    config.sources.at("--unqualified-context") == OptionSource::automatic ? "automatic" : "explicit") + ")";
        if (!config.arithmetic_fallback.empty()) line += "; " + config.arithmetic_fallback + "; using dense-A8 tiled prefill";
        return line + "\n";
    }
    if (!config.error.empty()) return config.error + '\n';
    if (config.reason == "baseline" || config.reason == "explicit-route") return {};
    return "Qwen auto-prefill declined: " + config.reason + "; using serial prefill\n";
}
inline std::vector<std::string> effective_prefill_argv(const Options& o) {
    std::vector<std::string> args{"--context", std::to_string(o.context_tokens)};
    if (o.unqualified_context) args.emplace_back("--unqualified-context");
    args.insert(args.end(), {"--execution-mode", o.execution.mode == DecodeExecutionMode::eager ? "eager" : "graph",
        "--residency", o.residency == HipModelResidencyMode::registered_file_mapping ? "register" : "copy",
        "--generate", std::to_string(o.generate_tokens)});
    if (o.parallel_hyper_mix) args.emplace_back("--parallel-hyper-mix");
    if (o.staged_hyper_projections) args.emplace_back("--staged-hyper-projections");
    if (o.fused_staged_hyper_mix) args.emplace_back("--fused-staged-hyper-mix");
    if (o.typed_hyper_projection) args.emplace_back("--typed-fused-hyper-projections");
    if (o.prefetch_typed_hyper_rows) args.emplace_back("--prefetch-typed-hyper-rows");
    if (o.prefetch_q8_hyper_output) args.emplace_back("--prefetch-q8-hyper-output");
    if (o.lane_chain_hyper_output) args.emplace_back("--lane-chain-hyper-output");
    if (o.prefetch_q8_qsa_input) args.emplace_back("--prefetch-q8-qsa-input");
    if (o.parallel_moe_router) args.emplace_back("--parallel-moe-router");
    if (o.wave_moe_router_topk) args.emplace_back("--wave-moe-router-topk");
    if (o.virtual_q8_gdn_input) args.emplace_back("--virtual-q8-gdn-input");
    if (o.prefetch_q8_gdn_input) args.emplace_back("--prefetch-q8-gdn-input");
    if (o.virtual_q8_gdn_output) args.emplace_back("--virtual-q8-gdn-output");
    if (o.deferred_moe_status) args.emplace_back("--deferred-moe-status");
    if (o.prefetch_q8_shared_expert) args.emplace_back("--prefetch-q8-shared-expert");
    if (o.q4k_a8g32_gate_up) args.emplace_back("--q4k-a8g32-gate-up");
    if (o.grouped_exact_q51_down) args.emplace_back("--grouped-exact-q51-down");
    if (o.virtual_q8_lm_head) args.emplace_back("--virtual-q8-lm-head");
    if (o.q51_slot_parallel_down) args.emplace_back("--q51-slot-parallel-down");
    if (o.batched_moe_activation) args.emplace_back("--batched-moe-activation");
    if (o.wave8_q8_gdn_input) args.emplace_back("--wave8-q8-gdn-input");
    if (o.wave8_q8_gdn_output) args.emplace_back("--wave8-q8-gdn-output");
    if (o.prefetch_q8_lm_head) args.emplace_back("--prefetch-q8-lm-head");
    if (o.wave8_q8_lm_head) args.emplace_back("--wave8-q8-lm-head");
    if (o.wave8_q8_qsa_output) args.emplace_back("--wave8-q8-qsa-output");
    if (o.prepared_eager) args.emplace_back("--prepared-eager");
    if (o.moe_shared_overlap) args.emplace_back("--moe-shared-overlap");
    if (o.gdn_islands) args.emplace_back("--gdn-islands");
    if (o.baseline) args.emplace_back("--baseline");
    args.insert(args.end(), {"--optimization-profile", o.optimization_profile == OptimizationProfile::automatic ? "auto" : "manual",
        "--numerical-profile", std::string(numerical_profile_name(o.numerical_profile))});
    if (o.known_prefix_window) args.emplace_back("--known-prefix-window");
    if (o.known_prefix_full_rows) args.emplace_back("--known-prefix-full-rows");
    if (o.known_prefix_rows_explicit) args.insert(args.end(), {"--known-prefix-rows", std::to_string(o.known_prefix_rows)});
    if (o.known_prefix_prefill_rows) args.insert(args.end(), {
        "--known-prefix-workspace-budget-bytes", std::to_string(o.known_prefill_workspace_budget),
        "--known-prefix-host-metadata-budget-bytes", std::to_string(o.known_prefill_metadata_budget),
        "--known-prefix-uma-budget-bytes", std::to_string(o.known_prefill_uma_budget),
        "--known-prefix-host-headroom-bytes", std::to_string(o.known_prefill_host_headroom),
        "--known-prefix-output-policy", o.known_prefix_output_policy == KnownWindowOutputPolicy::terminal_row ? "terminal-row" :
            o.known_prefix_output_policy == KnownWindowOutputPolicy::all_rows ? "all-rows" : "state-only"});
    if (o.known_prefix_window && !o.known_prefix_prefill_rows) args.insert(args.end(), {"--known-prefix-output-policy",
        o.known_prefix_output_policy == KnownWindowOutputPolicy::terminal_row ? "terminal-row" :
        o.known_prefix_output_policy == KnownWindowOutputPolicy::state_only ? "state-only" : "all-rows"});
    if (o.supplied_options.contains("--known-prefix-tail-rows") ||
        (o.automatic_prefill && o.known_prefix_ladder_tail_rows == std::vector<std::uint32_t>{256U,16U})) {
        std::string tails = o.known_prefix_tail_rows_off ? "none" : "";
        for (const auto rows : o.known_prefix_tail_rows) { if (!tails.empty()) tails += ','; tails += std::to_string(rows); }
        args.insert(args.end(), {"--known-prefix-tail-rows", std::move(tails)});
    }
    if (o.known_prefix_expert_union) args.emplace_back("--known-prefix-expert-union");
    if (o.known_window_router_rows) args.emplace_back("--known-window-router-rows");
    if (o.known_prefix_prefill_rows) args.insert(args.end(), {"--known-prefix-prefill-rows", std::to_string(o.known_prefix_prefill_rows)});
    if (o.prefill_numerical_profile) args.insert(args.end(), {"--prefill-numerical-profile", std::string(numerical_profile_name(*o.prefill_numerical_profile))});
    return args;
}
inline std::string effective_config_json(const EffectiveConfig& config) {
    const auto& o = config.options;
    std::ostringstream out;
    out << "{\"http_slots\":{";
    out << "\"mode\":\"" << (o.http_batch_explicit ? (o.http_batch_cap ? "cap" : "single") : "auto")
        << "\",\"cap\":" << o.http_batch_cap << ",\"slots\":";
    if (o.http_serve && o.http_batch) out << "null"; else out << 1;
    out << ",\"route\":\"" << (o.http_batch ? "batch" : "single")
        << "\",\"reason\":\"" << o.http_slot_reason << "\"},";
    out << "\"admitted\":" << (config.admitted ? "true" : "false") << ",\"reason\":\"" << config.reason
        << "\",\"tile_fallback\":\"" << config.tile_fallback
        << "\",\"error\":\"" << config.error
        << "\",\"ple_reclaim_credit_bytes\":" << config.ple_reclaim_credit
        << ",\"arithmetic_fallback\":\"" << config.arithmetic_fallback
        << "\",\"longer_context_qualified\":" << (o.context_tokens > 8192U ? "false" : "null")
        << ",\"context\":" << o.context_tokens << ",\"T\":" << o.known_prefix_prefill_rows
        << ",\"decode_profile\":\"" << numerical_profile_name(o.numerical_profile)
        << "\",\"prefill_profile\":\"" << numerical_profile_name(numerical_profile_pair(o).prefill) << "\",\"numerical_profile_pair_identity\":" << numerical_profile_pair_identity(numerical_profile_pair(o))
        << ",\"tails\":[";
    for (std::size_t i = 0; i < o.known_prefix_tail_rows.size(); ++i) out << (i ? "," : "") << o.known_prefix_tail_rows[i];
    out << "],\"budgets\":[" << o.known_prefill_workspace_budget << ',' << o.known_prefill_metadata_budget << ','
        << o.known_prefill_uma_budget << ',' << o.known_prefill_host_headroom << "],\"environment\":{";
    bool first = true;
    for (const auto& [name,value] : config.environment) {
        // Values are externally supplied strings, so quote every control byte.
        const auto quote = [](std::string_view text) {
            std::ostringstream escaped;
            constexpr char hex[] = "0123456789abcdef";
            for (const char raw : text) {
                const auto ch = static_cast<unsigned char>(raw);
                if (ch == '"' || ch == '\\') escaped << '\\' << static_cast<char>(ch);
                else if (ch < 32U) escaped << "\\u00" << hex[ch >> 4U] << hex[ch & 15U];
                else escaped << static_cast<char>(ch);
            }
            return escaped.str();
        };
        out << (first ? "" : ",") << '"' << quote(name) << "\":{\"value\":\"" << quote(value) << "\",\"source\":\"";
        const auto it = config.sources.find(name);
        out << (it == config.sources.end() || it->second == OptionSource::explicit_value ? "explicit" :
            it->second == OptionSource::baseline ? "baseline" : "automatic") << "\"}";
        first = false;
    }
    out << "},\"gsq_filtered_defaults\":[";
    bool first_filtered = true;
    if (config.gsq_owner) for (const auto& entry : qpf2k::defaults) {
        const auto source = config.sources.find(entry.name);
        if (source == config.sources.end() || source->second != OptionSource::automatic) continue;
        out << (first_filtered ? "" : ",") << '"' << entry.name << '"'; first_filtered = false;
    }
    out << "],\"argv\":[";
    bool first_arg = true;
    for (const auto& arg : effective_prefill_argv(o)) { out << (first_arg ? "" : ",") << '"' << arg << '"'; first_arg = false; }
    out << "],\"organization\":\"" << config.organization.selected << "\",\"configured_tails\":[";
    for (std::size_t i = 0; i < o.known_prefix_ladder_tail_rows.size(); ++i) out << (i ? "," : "") << o.known_prefix_ladder_tail_rows[i];
    out << "],\"suppressed\":[";
    for (std::size_t i = 0; i < config.suppressed.size(); ++i) out << (i ? "," : "") << '"' << config.suppressed[i] << '"';
    out << "],\"option_sources\":{";
    bool first_source = true;
    for (const auto& [name,source] : config.sources) {
        if (!name.starts_with("--")) continue;
        out << (first_source ? "" : ",") << '"' << name << "\":\"" <<
            (source == OptionSource::automatic ? "automatic" : source == OptionSource::baseline ? "baseline" : "explicit") << '"';
        first_source = false;
    }
    out << "},\"host_compilation\":{";
    bool first_field = true;
    for (const auto& [name,value] : config.host_compilation) {
        out << (first_field ? "" : ",") << '"' << name << "\":" << value; first_field = false;
    }
    out << "}}\n";
    return out.str();
}
} // namespace strix::decode_cli
