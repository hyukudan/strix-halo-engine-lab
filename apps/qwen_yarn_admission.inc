// Included inside strix::decode_cli after the native resolver helpers.
// Classify code paths, not default values or environment provenance. New items
// fail closed until their rotary sites have been audited against the B3 inventory.
enum class YarnEnvironmentPath { no_rotary, b3_rotary, uncovered_rotary };
inline YarnEnvironmentPath yarn_environment_path(qpf2k::Item item) {
    using enum qpf2k::Item;
    switch (item) {
    case qsa: case qsa_mask: case norm_group:
        // hip_indexed_attention_launcher.hip:5768-5825,5885-5890,5993-6020;
        // hip_qsa_prefill_batch.inc:193-220 (batched table/norm/pool).
        return YarnEnvironmentPath::b3_rotary;
    case sched: case dense: case gdn: case gdn_gates_batch: case gdn_core16:
    case hc: case hc_mask: case hc_output3: case hc_output3_band:
    case moe_da16: case moe_route: case ple_reclaim: case dense_rows: case pipe:
    case attend_batch: case score_park: case index_rows: case cohort_lean:
    case rows_lean: case dispatch_fuse:
    // Audited at the 86e169e1 merge: score_rows re-tiles the stage-batch scorer over
    // already-rotated pooled keys (hip_qsa_prefill_flash.inc:25-43); ple_batch, shared_pre,
    // moe_mc and moe_hdr are PLE, shared-expert and MoE projections. None adds rotary work.
    case score_rows: case ple_batch: case shared_pre: case moe_mc: case moe_hdr:
        return YarnEnvironmentPath::no_rotary;
    }
    return YarnEnvironmentPath::uncovered_rotary;
}
inline std::string yarn_environment_refusal(const Environment& environment, unsigned factor) {
    if (const auto it = environment.find("STRIX_QPF2K_SCHED");
        it != environment.end() && it->second != "12")
        return "YaRN requires STRIX_QPF2K_SCHED=12; got " + it->second;
    for (const auto& [name, value] : environment) {
        const auto entry = std::find_if(qpf2k::defaults.begin(), qpf2k::defaults.end(),
            [&](const auto& item) { return name == item.name; });
        if (entry != qpf2k::defaults.end() &&
            yarn_environment_path(entry->item) != YarnEnvironmentPath::uncovered_rotary) continue;
        // These controls perform no rotation. MOE_DEFAULTED is a two-bit
        // provenance mask (0..3), not a boolean execution feature.
        if (name == "STRIX_QPF2K_MOE_DEFAULTED" || name == "STRIX_QPF2K_DEFAULTS" ||
            name == "STRIX_PREFILL_F16") continue;
        // Serial decode pool controls: the pool's only rotary site is block_pool_kernel under with_qwen_rotary
        // (qsa_decode_pool_scores, bound to the table by bind_qwen_pool); the self-check re-derives the scores with
        // block_score_kernel from the same table; selection and key storage perform no rotation.
        if (name == "STRIX_QSA_DECODE_POOL" || name == "STRIX_QSA_DECODE_POOL_CHECK" ||
            name == "STRIX_QSA_DECODE_POOL_BF16" || name == "STRIX_QSA_DECODE_SELECT") continue;
        // Compact tiled-prefill memory controls choose where scores, maxima and terminal logits live; the score ring and
        // stage split reuse the same table-bound norm/RoPE, pool and scorer launches, so they add no rotary site.
        if (name == "STRIX_CTX_EXACT_MEMORY" || name == "STRIX_QSA_SCORE_RING" || name == "STRIX_QSA_SCORE_RING_ROWS" ||
            name == "STRIX_TERMINAL_LOGITS_ARENA") continue;
        if (name == "STRIX_QWEN_ROPE_YARN" && value == std::to_string(factor)) continue;
        return "YaRN execution environment not supported: " + name;
    }
    return {};
}
inline std::uint64_t prefill_checked_mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > UINT64_MAX / a) throw std::overflow_error("prefill reservation multiplication overflow");
    return a * b;
}
inline EffectiveConfig resolve_yarn_config(const RequestedOptions& request,
    const ModelContract& model, const gguf::FileInfo& info, const ModelManifest& manifest,
    std::string_view content, const PrefillCapabilities& capabilities, const PrefillMemory& memory) {
    constexpr std::uint64_t gib = UINT64_C(1) << 30U;
    EffectiveConfig result;
    result.options = request.options;
    result.environment = request.environment;
    for (const auto& name : request.options.supplied_options) result.sources[name] = OptionSource::explicit_value;
    result.rotary_identity = model.rotary_identity();
    auto& o = result.options;
    const auto refuse = [&](std::string reason) {
        result.reason = "yarn-refused";
        result.error = std::move(reason);
        return result;
    };
    if (const auto why = yarn_route_refusal(o); !why.empty()) return refuse(why);
    try {
        (void)validate_qwen_rotary_request(o.rotary.factor(), o.context_explicit, o.context_tokens,
            o.baseline || env_is(request.environment, "STRIX_BASELINE", "1"));
        if (o.tokens.empty() || o.tokens.size() > o.context_tokens ||
            o.generate_tokens > o.context_tokens - o.tokens.size())
            return refuse("YaRN prompt + generation exceeds requested context");
        if (model.max_position_embeddings != RotaryDescriptor::original_max ||
            model.rope_theta != RotaryDescriptor::theta || model.rotary_dim() != RotaryDescriptor::rotary_dim ||
            model.rotary_identity().descriptor() != o.rotary || !model.validate().empty())
            return refuse("YaRN resolved rotary/checkpoint contract mismatch");
        if (!memory.available || !memory.total || !memory.gtt_limit || !memory.gtt_used || !memory.estimate)
            return refuse("YaRN memory observation unavailable (MemTotal/MemAvailable/GTT required)");
        if (!capabilities.target_qualified || !capabilities.a8_runtime || content != validated_model_content_identity ||
            qpf2k::has_gsq_gate(manifest)) return refuse("YaRN checkpoint or target route not supported");
        const auto injected = qpf2k::initialize([&](const char* name) -> const char* {
            const auto it = result.environment.find(name);
            return it == result.environment.end() ? nullptr : it->second.c_str();
        }, [&](const char* name, const char* value) {
            result.environment.emplace(name, value); return true;
        });
        for (const auto& entry : qpf2k::defaults)
            if (result.environment.contains(entry.name)) result.sources[entry.name] =
                injected & qpf2k::bit(entry.item) ? OptionSource::automatic : OptionSource::explicit_value;
        if (result.environment.contains("STRIX_QPF2K_MOE_DEFAULTED"))
            result.sources["STRIX_QPF2K_MOE_DEFAULTED"] = request.environment.contains("STRIX_QPF2K_MOE_DEFAULTED") ?
                OptionSource::explicit_value : OptionSource::automatic;
        // Missing with defaults disabled also means execution cannot use schedule 12.
        if (!result.environment.contains("STRIX_QPF2K_SCHED"))
            return refuse("YaRN requires STRIX_QPF2K_SCHED=12; effective value is unset");
        // Classify the complete effective environment before admission callbacks.
        if (const auto why = yarn_environment_refusal(result.environment, o.rotary.factor()); !why.empty())
            return refuse(why);
        if (!o.disabled_prefill_features.empty()) return refuse("YaRN disabled prefill feature not supported");
        o.unqualified_context = true;
        auto organization = resolve_optimization_profile(o, model, info, manifest, content, capabilities.a8_runtime, false);
        if (!organization.automatic_applied) return refuse("YaRN production organization unavailable: " + organization.reason);
        o.automatic_prefill = true;
        o.known_prefix_window = o.known_prefix_full_rows = o.known_prefix_expert_union = o.known_window_router_rows = true;
        o.known_prefix_output_policy = KnownWindowOutputPolicy::terminal_row;
        apply_measured_prompt_organization(o);
        o.numerical_profile = dense_a8_profile;
        // Keep one explicit lineage across all T trials. No FP16-to-A8 fallback.
        if (capabilities.fp16_operands && !env_is(request.environment,"STRIX_PREFILL_F16","0")) {
            o.prefill_numerical_profile = DecodeNumericalProfile::fast_prefill_f16wmma_v1;
            o.prefill_profile_automatic = true;
        }
        o.known_prefill_workspace_budget = 8U * gib;
        o.known_prefill_metadata_budget = gib;
        o.known_prefill_uma_budget = std::min(120U * gib, *memory.total);
        o.known_prefill_host_headroom = 8U * gib;
        o.known_prefix_prefill_rows = 8192U;
        std::uint64_t credit = 0;
        if (memory.verify_ple_credit) credit = memory.verify_ple_credit(o);
        result.ple_reclaim_credit = credit;
        const auto primary = o;
        for (const auto rows : {8192U,4096U,2048U}) {
            o = primary;
            o.known_prefix_prefill_rows = rows;
            apply_default_known_prefill_tails(o, "12");
            auto validation = organization;
            apply_numerical_profile(o, model, manifest, content, capabilities.a8_runtime, validation);
            result.admission = {{"context_capacity",o.context_tokens},{"window_capacity_rows",rows},{"mtp_enabled",0},
                {"parked_session_bytes",0},{"capture_bytes",0},{"restore_peak_bytes",0},{"host_reserve_bytes",8U*gib},{"verified_ple_credit_bytes",credit},
                {"uma_limit_bytes",o.known_prefill_uma_budget},{"available_bytes",*memory.available},
                {"gtt_limit_bytes",*memory.gtt_limit},{"gtt_used_bytes",*memory.gtt_used},{"smallest_T_tried",rows}};
            std::string failure;
            try {
                const auto required = memory.estimate(o);
                auto owners = required.owners;
                if (owners.empty()) owners = {{"model_backing",required.model_backing},{"state_d1",required.state_d1},
                    {"serial_scratch",required.serial_scratch},{"workspace",required.workspace},{"metadata",required.metadata},
                    {"runtime_payload",required.runtime_payload},{"tails",required.tails}};
                owners["rotary_table"] = o.rotary.table_bytes();
                std::uint64_t gross = 8U*gib;
                for (const auto& [name, bytes] : owners) {
                    gross = prefill_checked_add(gross, bytes);
                    result.admission["owner_" + name + "_bytes"] = bytes;
                }
                if (credit > required.model_backing) throw std::invalid_argument("verified credit exceeds model backing");
                const auto net = gross - credit;
                result.admission["gross_required_bytes"] = gross;
                result.admission["required_bytes"] = net;
                // File credit affects the total envelope only. MemAvailable already includes
                // reclaimable cache, so credit is NEVER added to incremental free headroom.
                const auto incremental = gross - required.model_backing;
                result.admission["incremental_bytes"] = incremental;
                const auto device = prefill_checked_add(required.device_base,
                    prefill_checked_add(required.workspace, prefill_checked_add(required.tails, o.rotary.table_bytes())));
                result.admission["device_growth_bytes"] = device;
                auto live = UmaOwnerInventory{};
                live.reservation_upper_bound_bytes = required.model_backing - credit;
                (void)plan_uma_reservation_growth(live, incremental, 0,
                    {o.known_prefill_uma_budget,o.known_prefill_uma_budget});
                if (incremental > *memory.available || *memory.gtt_used > *memory.gtt_limit ||
                    device > *memory.gtt_limit - *memory.gtt_used || !required.workspace ||
                    prefill_checked_add(required.workspace,required.tails) > o.known_prefill_workspace_budget ||
                    required.metadata > o.known_prefill_metadata_budget)
                    throw std::runtime_error("owner payload exceeds available/workspace/metadata/GTT limit");
                result.host_compilation = required.organization;
                result.organization = organization;
                result.reason = "experimental-yarn";
                result.admitted = true;
                return result;
            } catch (const std::overflow_error&) { throw; }
              catch (const std::exception& error) { failure = error.what(); }
            if (!result.tile_fallback.empty()) result.tile_fallback += ',';
            result.tile_fallback += "T" + std::to_string(rows) + ":insufficient-memory";
            if (rows == 2048U) {
                std::string listing = "YaRN admission refused: " + failure + "; context=" + std::to_string(o.context_tokens);
                for (const auto& [name, bytes] : result.admission) listing += "; " + name + '=' + std::to_string(bytes);
                return refuse(listing);
            }
        }
    } catch (const std::exception& error) { return refuse(std::string("YaRN admission refused: ") + error.what()); }
    return refuse("YaRN admission refused");
}
