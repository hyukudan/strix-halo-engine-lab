#include "../apps/qwen_prompt_organization.hpp"
#include "qwen_prefill_contract.hpp"
#include "qwen_serial_contract.hpp"
#include "qwen_historical_resolver.hpp"
#include <iostream>
#include <limits>

using namespace strix;
using namespace strix::decode_cli;
namespace {
int failures = 0;
void expect(bool ok, const char* message) {
    if (!ok) { ++failures; std::cerr << message << '\n'; }
}
Options parse(std::vector<std::string> args) {
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    return parse_options(static_cast<int>(argv.size()), argv.data());
}
std::vector<std::string> mtp_fixture(){
    const std::string sha(64,'1');
    return {"decoder","target.gguf","1","2","3","--generate","512","--mtp",
        "--mtp-sidecar","sidecar.gguf","--mtp-sidecar-logical-name","mtp-sidecar",
        "--mtp-sidecar-sha256",sha,"--mtp-target-sha256",sha,"--mtp-target-content",sha,
        "--mtp-golden","independent.bin","--mtp-golden-sha256",sha,
        "--mtp-source-repository","publisher/model","--mtp-source-revision","pinned-revision",
        "--mtp-source-path","sidecar.gguf","--mtp-converter-revision","converter-revision",
        "--mtp-converter-sha256",sha,"--mtp-norm-converter-sha256",sha,
        "--mtp-conversion","effective-norm-embedding-first-v1","--mtp-normalized-rounding","fp32",
        "--mtp-projection-rounding","bf16-rne","--mtp-sidecar-residency","register",
        "--mtp-sidecar-bytes","1024","--mtp-header-bytes","512",
        "--mtp-sidecar-retained-bytes","2048","--mtp-sidecar-setup-bytes","2048",
        "--mtp-golden-max-bytes","4096","--mtp-replay-host-bytes","8192",
        "--mtp-runtime-bytes","16384","--mtp-target-capture-bytes","8192",
        "--mtp-policy-id","7","--mtp-candidate-depth","2"};
}
constexpr std::uint64_t gib = UINT64_C(1024) * 1024U * 1024U;
// The source command did not enable MTP or cohorts. Their inactive controls
// and disabled experiments are excluded. The logits digest is diagnostic output,
// not part of the prefill arithmetic/organization contract.
constexpr std::array excluded_inactive_env{
    "STRIX_QKERN_QSA_PERSISTENT", "STRIX_QKERN_GDN_SCATTER", "STRIX_QKERN_ROUTER8", "STRIX_QKERN_HEAD_DIRECT",
    "STRIX_QSA_WINDOW_POOLED", "STRIX_QSA_WINDOW_GROUPED", "STRIX_MTP_CONTROLLER", "STRIX_MTP_DRAFT_CACHE",
    "STRIX_MTP_CACHE_AUDIT", "STRIX_QWEN_SHARED_DOWN_NARROW", "STRIX_QPERSIST_DRAFT_SCORE",
    "STRIX_QPERSIST_QSA_TAIL", "STRIX_QPERSIST_MTP_CACHE", "STRIX_QPERSIST_GDN_TAIL", "STRIX_MTP_HC_PERSISTENT",
    "STRIX_MTP_BATCH_JOINT", "STRIX_COHORT_ROWS", "STRIX_COHORT_WIDE",
    "STRIX_PREFILL_LOGITS_DIGEST"};
PrefillMemory ample_memory() {
    return {128U*gib,128U*gib,0U,[](const Options&) {
        return PrefillRequirements{70U*gib,8U*gib,gib,2U*gib,512U*1024U*1024U,128U*1024U*1024U,0U,9U*gib};
    }};
}
void long_context_admission_tests(const RequestedOptions& original, const ModelContract& model,
    const gguf::FileInfo& info, ModelManifest manifest, const PrefillCapabilities& capabilities) {
    const auto contract = build_ple_hash_contract(model);
    ManifestTensor table;
    table.name = "ple.ngram_table.weight";
    table.role = TensorRole::ngram_table;
    table.residency = ResidencyPolicy::file_backed_candidate;
    table.ggml_type = 20U;
    table.dimensions = {160U,contract.minimum_table_rows};
    table.absolute_offset = 4U*gib + 32U;
    table.byte_size = table.dimensions[1] * 90U;
    manifest.tensors.push_back(table);
    manifest.layout_identity = model_manifest_layout_identity(manifest);
    auto observed = ample_memory();
    observed.available = UINT64_C(127800000000);
    observed.gtt_limit = 128U*gib;
    observed.gtt_used = 2U*gib;
    observed.ple_reclaim_page = 4096U;
    observed.estimate = [](const Options& options) {
        auto required = PrefillRequirements{};
        required.model_backing = 96U*gib;
        required.state_d1 = options.context_tokens * 120U*1024U;
        required.serial_scratch = gib;
        required.workspace = options.known_prefix_prefill_rows * (gib / 2048U);
        required.metadata = gib / 2U;
        required.runtime_payload = gib / 8U;
        required.device_base = 40U*gib;
        required.tails = static_cast<std::uint64_t>(options.known_prefix_tail_rows.size()) * (gib / 16U);
        return required;
    };
    const auto resolve = [&](const RequestedOptions& request, const PrefillMemory& memory) {
        return resolve_effective_config(request,model,info,manifest,validated_model_content_identity,capabilities,memory);
    };
    {
        ple_reclaim::Config config;
        config.page = 4096U;
        config.table_offset = table.absolute_offset;
        config.table_rows = table.dimensions[1];
        config.stored_row_bytes = 90U;
        config.mapping_bytes = table.absolute_offset + table.byte_size;
        config.decode_allowance_bytes = ple_reclaim::decode_allowance(1U,1U,contract.head_count,config.page);
        config.reclaim = ple_reclaim::reclaim_intervals(config.table_offset,table.byte_size,config.page,
            {{config.table_offset + 4096U,config.table_offset + 8192U}});
        const auto plan = ple_reclaim::plan_prompt(config,contract,original.options.tokens);
        std::set<std::uint64_t> touched;
        std::array<std::uint32_t,PleHashContract::maximum_ngram - 1U> previous{};
        const auto count = contract.ngram_size - 1U;
        std::fill_n(previous.begin(),count,contract.eos_token_id);
        std::size_t row_index = 0;
        for (const auto token : original.options.tokens) {
            const auto hash = hash_ple_token(contract,token,{previous.data(),count});
            for (std::uint32_t h = 0; h < hash.row_count; ++h) {
                const auto offset = config.table_offset + hash.row_ids[h] * config.stored_row_bytes;
                expect(plan.rows[row_index++] == hash.row_ids[h], "ple_reclaim_prompt_rows: shared planner retains runtime hash rows");
                for (auto page = offset / config.page; page <= (offset + config.stored_row_bytes - 1U) / config.page; ++page)
                    for (const auto& interval : config.reclaim)
                        if (page * config.page >= interval.begin && page * config.page < interval.end) touched.insert(page);
            }
            std::copy_n(hash.next_context.begin(),count,previous.begin());
        }
        std::uint64_t reclaimable = 0;
        for (const auto& interval : config.reclaim) reclaimable += interval.end - interval.begin;
        const auto touched_bytes = touched.size() * config.page;
        const auto admitted_bytes = touched_bytes + (256U << 20U) + (64U << 20U) + touched_bytes / 64U;
        expect(plan.stats.reclaimable_bytes == reclaimable && plan.stats.touched_bytes == touched_bytes &&
            plan.stats.admitted_bytes == admitted_bytes && plan.stats.credit_bytes == reclaimable - admitted_bytes,
            "ple_reclaim_exact_credit: unregistered interior minus unique prompt pages, decode allowance and margin");
        expect(ple_reclaim::decode_allowance(100000U,1U,contract.head_count,config.page) ==
            UINT64_C(100001) * contract.head_count * 2U * config.page,
            "ple_reclaim_decode_bound: generation allowance grows beyond its 256 MiB floor");
        bool overflow = false;
        try { (void)ple_reclaim::decode_allowance(UINT64_MAX,1U,contract.head_count,config.page); }
        catch (const std::overflow_error&) { overflow = true; }
        expect(overflow, "ple_reclaim_decode_overflow: allowance arithmetic is checked");
    }
    auto request = original;
    request.options.generate_tokens = 1U;
    for (const auto n : {131072U,196608U,262143U}) {
        request.options.tokens.resize(n,1U);
        // Repeated, changing IDs exercise the full hash chain without loading a model or prompt file.
        for (std::size_t i = 0; i < request.options.tokens.size(); ++i)
            request.options.tokens[i] = static_cast<std::uint32_t>(i % 1024U);
        const auto config = resolve(request,observed);
        const auto expected_context = std::min(model.max_position_embeddings, static_cast<std::uint64_t>(n) + 512U);
        expect(config.admitted && config.options.known_prefix_prefill_rows == 8192U &&
            config.options.context_tokens == expected_context && config.ple_reclaim_credit > 24U*gib,
            "long_context_l1_T8192: 128K/192K/262143 admit with observed memory and prompt-specific L1 credit");
        const auto text = effective_config_json(config);
        expect(text.find("\"ple_reclaim_credit_bytes\":" + std::to_string(config.ple_reclaim_credit)) != std::string::npos,
            "long_context_l1_receipt: JSON reports the predicted reclaim credit");
        auto no_l1 = observed; no_l1.ple_reclaim_page.reset();
        const auto retained = resolve(request,no_l1);
        expect(!retained.admitted || retained.options.known_prefix_prefill_rows < 8192U,
            "long_context_l1_required_for_T8192: unsupported reclaim earns no credit");
    }
    // Tight GTT exercises the independent device cap: L1 cannot credit registered/device aliases.
    const auto estimate = observed.estimate;
    for (const auto chosen : {4096U,2048U,0U}) {
        auto tight = observed;
        tight.gtt_limit = 2U*gib + 40U*gib + (chosen == 4096U ? 3U*gib : chosen == 2048U ? 2U*gib : gib / 2U);
        std::vector<std::uint32_t> attempts;
        tight.estimate = [&](const Options& candidate) {
            attempts.push_back(candidate.known_prefix_prefill_rows);
            return estimate(candidate);
        };
        const auto config = resolve(request,tight);
        const std::vector<std::uint32_t> expected = chosen == 4096U ? std::vector<std::uint32_t>{8192U,4096U} :
            std::vector<std::uint32_t>{8192U,4096U,2048U};
        expect(attempts == expected && config.admitted == (chosen != 0U) &&
            (!chosen || config.options.known_prefix_prefill_rows == chosen),
            "long_context_tile_fallback_order: T8192 then T4096 then T2048 before decline");
        if (chosen == 2048U) {
            const auto args = effective_prefill_argv(config.options);
            const auto tail = std::find(args.begin(),args.end(),"--known-prefix-tail-rows");
            expect(config.options.known_prefix_tail_rows == std::vector<std::uint32_t>{256U,16U} &&
                config.options.known_prefix_ladder_tail_rows == config.options.known_prefix_tail_rows &&
                config.sources.at("--known-prefix-tail-rows") == OptionSource::automatic &&
                tail != args.end() && std::next(tail) != args.end() && *std::next(tail) == "256,16",
                "long_context_T2048_tails: active/configured tails and replay argv use 256,16");
        }
        const auto notice = effective_config_notice(config);
        expect(chosen ? notice.find("T" + std::to_string(chosen)) != std::string::npos &&
            notice.find("T8192:insufficient-memory") != std::string::npos &&
            effective_config_json(config).find("T8192:insufficient-memory") != std::string::npos :
            notice.find("using serial prefill") == std::string::npos && !config.error.empty(),
            "long_context_fallback_receipt: selection and memory reason reported; oversized serial is refused");
    }
    for (const auto chosen : {4096U,2048U}) {
        auto tight = observed;
        tight.available = 100U*gib;
        const auto total = *tight.available;
        const auto credit = prefill_reclaim_credit(resolve(request,observed).options,model,manifest,{},tight.ple_reclaim_page);
        tight.estimate = [&](const Options& candidate) {
            auto required = estimate(candidate);
            const auto other = required.state_d1 + required.serial_scratch + required.metadata + required.runtime_payload +
                required.tails + candidate.known_prefill_host_headroom;
            const auto allowed_workspace = chosen == 4096U ? 3U*gib : gib + gib / 2U;
            required.model_backing = total + credit - other - allowed_workspace;
            return required;
        };
        const auto config = resolve(request,tight);
        expect(config.admitted && config.options.known_prefix_prefill_rows == chosen,
            "long_context_uma_fallback: retained host backing includes headroom and runtime payload, credit applied once");
    }
    auto compiler_failure = observed;
    for (const auto* message : {"known prefill workspace exceeds its explicit arena budget",
            "known prefill host metadata payload exceeds its budget: T = 8192 needs 1073741825 bytes"}) {
        compiler_failure.estimate = [&](const Options& candidate) {
            if (candidate.known_prefix_prefill_rows == 8192U) throw KnownPrefillBudgetExceeded(message);
            return estimate(candidate);
        };
        const auto compiled = resolve(request,compiler_failure);
        expect(compiled.admitted && compiled.options.known_prefix_prefill_rows == 4096U &&
            compiled.tile_fallback == "T8192:insufficient-memory" && compiled.arithmetic_fallback.empty(),
            "long_context_budget_fallback: workspace and metadata budget rejections report insufficient memory");
    }
    compiler_failure.estimate = [&](const Options& candidate) {
        if (candidate.known_prefix_prefill_rows == 8192U) throw std::invalid_argument("plan budget exceeded");
        return estimate(candidate);
    };
    const auto compiled = resolve(request,compiler_failure);
    expect(compiled.admitted && compiled.options.known_prefix_prefill_rows == 4096U &&
        compiled.tile_fallback == "T8192:candidate-not-admitted" && compiled.arithmetic_fallback.empty(),
        "long_context_compiler_fallback: other validation failures retain candidate-not-admitted");
    for (const auto* value : {"0","00"}) {
        auto disabled = request;
        disabled.environment["STRIX_QPF2K_DEFAULTS"] = value;
        const auto config = resolve(disabled,observed);
        expect(!config.admitted && config.reason == "feature-disabled" && config.ple_reclaim_credit == 0U,
            "long_context_defaults_off: leading zero disables automatic admission and reclaim credit");
    }
    // These pure route checks share the runtime gate, including graph/copy/MTP/IO exclusions.
    const auto admitted = resolve(request,observed);
    expect(admitted.admitted, "long_context_reclaim_route_fixture");
    if (admitted.admitted) {
        const auto credit = [&](const Options& options, const Environment& environment, const ModelManifest& layout) {
            return prefill_reclaim_credit(options,model,layout,environment,observed.ple_reclaim_page);
        };
        for (unsigned excluded = 0; excluded < 10U; ++excluded) {
            auto options = admitted.options;
            if (excluded == 0U) options.residency = HipModelResidencyMode::resident_copy;
            if (excluded == 1U) options.model_copy_tier = HipModelCopyTier::all;
            if (excluded == 2U) options.execution.mode = DecodeExecutionMode::graph;
            if (excluded == 3U) options.known_prefix_prefill_rows = 0U;
            if (excluded == 4U) options.http_serve = true;
            if (excluded == 5U) options.mtp.enabled = true;
            if (excluded == 6U) options.planned_ple_io = true;
            if (excluded == 7U) options.serving_prefix_files.push_back("unused");
            if (excluded == 8U) options.tokens.clear();
            if (excluded == 9U) options.mtp.cohort.push_back({});
            expect(credit(options,{},manifest) == 0U, "long_context_reclaim_exclusions: unsupported runtime route earns no credit");
        }
        for (const auto& environment : {Environment{{"STRIX_MODEL_COPY_ALL","1"}},
                Environment{{"STRIX_MODEL_COPY_ALL","fine"}}, Environment{{"STRIX_MODEL_COPY_ALL","uc"}},
                Environment{{"STRIX_MODEL_COPY_ALL","yes"}},
                Environment{{"STRIX_PLE_RECLAIMABLE","off"}}, Environment{{"STRIX_PLE_RECLAIMABLE","0"}}})
            expect(credit(admitted.options,environment,manifest) == 0U, "long_context_reclaim_environment: copy-all/off earns no credit");
        expect(credit(admitted.options,{{"STRIX_PLE_RECLAIMABLE","1"}},manifest) == admitted.ple_reclaim_credit &&
            credit(admitted.options,{{"STRIX_PLE_RECLAIMABLE","auto"}},manifest) == admitted.ple_reclaim_credit,
            "long_context_reclaim_auto_required: auto and required use identical accounting");
        const auto raised = credit(admitted.options,{{"STRIX_PLE_RECLAIM_DECODE_BYTES",std::to_string(gib)}},manifest);
        expect(admitted.ple_reclaim_credit - raised == gib - (256U << 20U),
            "long_context_reclaim_decode_allowance: explicit allowance only reduces credit");
        auto pinned = manifest;
        auto overlap = table;
        overlap.residency = ResidencyPolicy::resident;
        pinned.tensors.push_back(overlap);
        expect(credit(admitted.options,{},pinned) == 0U, "long_context_reclaim_registered: registered table pages earn no credit");
    }
    for (const auto n : {261632U,261633U,262112U,262143U,262144U,262145U}) {
        request.options.tokens.resize(n,1U);
        for (const auto generate : {1U,32U,512U}) {
            request.options.generate_tokens = generate;
            const auto config = resolve(request,observed);
            if (static_cast<std::uint64_t>(n) + generate <= model.max_position_embeddings) {
                expect(config.admitted && config.options.context_tokens == model.max_position_embeddings,
                    "long_context_clamp: padding is capped at the model contract limit");
            } else {
                bool refused = false;
                try { require_effective_prefill_execution(config); } catch (const std::invalid_argument&) { refused = true; }
                expect(!config.admitted && config.reason == "prompt-exceeds-model-context" && refused &&
                    effective_config_notice(config).find("using serial prefill") == std::string::npos,
                    "long_context_decline: prompt plus generation over model maximum is an explicit execution error");
            }
        }
    }
}

}
int main(int argc, char** arguments) {
    const auto model = ModelContract::qwen38_flash_next();
    gguf::FileInfo info;
    ModelManifest manifest;
    manifest.architecture = "qwen3next";
    const auto tensor = [&](std::uint32_t layer, const char* name, std::uint64_t width, std::uint64_t outputs) {
        ManifestTensor item;
        item.name = "blk." + std::to_string(layer) + "." + name + ".weight";
        item.layer = layer; item.role = TensorRole::dense_projection;
        item.dimensions = {width,outputs}; item.ggml_type = 8U;
        item.byte_size = width / 32U * outputs * 34U;
        manifest.tensors.push_back(std::move(item));
    };
    for (std::uint32_t layer = 0; layer < model.hidden_layers; ++layer) {
        tensor(layer,"hc_attn_up",model.hc_lowrank,model.hidden_size * model.hc_count);
        tensor(layer,"hc_ffn_up",model.hc_lowrank,model.hidden_size * model.hc_count);
        tensor(layer,"ffn_gate_shexp",model.hidden_size,model.shared_expert_intermediate_size);
        tensor(layer,"ffn_up_shexp",model.hidden_size,model.shared_expert_intermediate_size);
        tensor(layer,"ffn_down_shexp",model.shared_expert_intermediate_size,model.hidden_size);
        if ((layer + 1U) % model.full_attention_interval != 0U) continue;
        tensor(layer,"attn_q",model.hidden_size,model.attention_heads * model.head_dim * 2U);
        tensor(layer,"attn_k",model.hidden_size,model.kv_heads * model.head_dim);
        tensor(layer,"attn_v",model.hidden_size,model.kv_heads * model.head_dim);
        tensor(layer,"indexer_q_proj",model.hidden_size,model.indexer_n_heads * model.indexer_head_dim);
        tensor(layer,"indexer_k_proj",model.hidden_size,model.indexer_head_dim);
    }
    tensor(0U,"final_hc_up",model.hc_lowrank,model.hidden_size * model.hc_count);
    manifest.tensors.back().name = "output_hc_up.weight";
    manifest.tensors.back().layer.reset();
    manifest.layout_identity = model_manifest_layout_identity(manifest);
    const PrefillCapabilities capabilities{true,true,true};
    auto memory = ample_memory();
    const auto fixture_requirements = memory.estimate;
    memory.estimate = [&](const Options& options) {
        auto requirements = fixture_requirements(options);
        const auto compiler = prompt_compiler_requests(options,model,manifest);
        const auto buffers = build_serial_execution_buffer_plan(model,compiler.buffers);
        requirements.organization = {{"buffer_peak_internal_bytes",buffers.peak_internal_bytes},
            {"gdn_workspace_bytes",buffers.gdn_workspace_bytes}, {"hyper_workspace_bytes",buffers.hyper_workspace_bytes},
            {"wave8_gdn_input",compiler.binding.wave8_q8_gdn_input},
            {"wave8_gdn_output",compiler.binding.wave8_q8_gdn_output},
            {"wave8_qsa_output",compiler.binding.wave8_q8_qsa_output},
            {"wave8_lm_head",compiler.binding.wave8_q8_lm_head}, {"dense_a8",compiler.binding.dense_a8}};
        return requirements;
    };
    std::vector<std::string> plain_argv{"decoder","MODEL"};
    plain_argv.insert(plain_argv.end(),32768U,"1");
    plain_argv.insert(plain_argv.end(),{"--generate","32"});
    auto request = RequestedOptions{parse(std::move(plain_argv)),{}};
    const auto resolve = [&](const RequestedOptions& value, const PrefillMemory& observation) {
        return resolve_effective_config(value,model,info,manifest,validated_model_content_identity,capabilities,observation);
    };
    // Model the runtime TU's pre-main environment before exercising the same
    // capture/resolver and baseline cleanup used by decode-token.
    {
        Environment startup{{"STRIX_QPF2K_DENSE", "0"}, {"STRIX_QPF2K_GDN", "1"}};
        const auto get = [&](const char* name) -> const char* {
            const auto it = startup.find(name);
            return it == startup.end() ? nullptr : it->second.c_str();
        };
        const auto put = [&](const char* name, const char* value) { return startup.emplace(name,value).second; };
        qpf2k::initialize_before_main(get,put);
        expect((qpf2k::pre_injected & qpf2k::bit(qpf2k::Item::sched)) != 0U &&
            qpf2k::defaulted_items == qpf2k::pre_injected, "runtime startup restores shared default provenance");
        for (const auto& entry : qpf2k::defaults)
            if (static_cast<unsigned>(entry.item) > static_cast<unsigned>(qpf2k::Item::ple_reclaim))
                expect(!startup.contains(entry.name), "non-decoder startup injects none of the new wrapper settings");
        auto captured = request;
        captured.environment = requested_prefill_environment(get);
        expect(!captured.environment.contains("STRIX_QPF2K_SCHED") &&
            !captured.environment.contains("STRIX_QPF2K_MOE_DEFAULTED"), "requested environment excludes startup injections");
        const auto resolved = resolve(captured,memory);
        expect(resolved.sources.at("STRIX_QPF2K_SCHED") == OptionSource::automatic &&
            resolved.sources.at("STRIX_QPF2K_MOE_DEFAULTED") == OptionSource::automatic,
            "pre-injected defaults resolve as automatic");
        expect(resolved.environment.at("STRIX_QPF2K_DENSE") == "0" &&
            resolved.sources.at("STRIX_QPF2K_DENSE") == OptionSource::explicit_value, "explicit startup environment stays explicit");
        expect(resolved.sources.at("STRIX_QPF2K_GDN") == OptionSource::explicit_value,
            "explicit value equal to the default stays explicit");
        captured.options.baseline = true;
        expect(resolve(captured,memory).reason == "baseline", "CLI baseline dominates startup defaults");
        qpf2k::undo_pre_injected([&](const char* name) { startup.erase(name); });
        expect(startup == Environment{{"STRIX_QPF2K_DENSE", "0"}, {"STRIX_QPF2K_GDN", "1"}} && qpf2k::baseline &&
            qpf2k::defaulted_items == 0U && qpf2k::pre_injected == 0U, "baseline unsets exactly its startup injections and clears provenance");
        qpf2k::baseline = false;
        for (const auto* disable : {"STRIX_QPF2K_DEFAULTS", "STRIX_BASELINE"}) {
            startup = {{disable, std::string_view(disable) == "STRIX_BASELINE" ? "1" : "0"}};
            qpf2k::initialize_before_main(get,put);
            expect(startup.size() == 1U && qpf2k::pre_injected == 0U && qpf2k::defaulted_items == 0U,
                "startup honors defaults-off and baseline environment");
            if (std::string_view(disable) == "STRIX_QPF2K_DEFAULTS") {
                captured = request; captured.environment = requested_prefill_environment(get);
                const auto disabled = resolve(captured,memory);
                expect(!disabled.environment.contains("STRIX_QPF2K_SCHED"), "resolver also honors defaults-off after startup");
            }
        }
    }
    long_context_admission_tests(request,model,info,manifest,capabilities);
    const auto automatic = resolve(request,memory);
    expect(automatic.admitted && automatic.reason == "qualified-prompt", "plain default admits the measured prompt recipe");
    const auto argv = effective_prefill_argv(automatic.options);
    expect(std::vector<std::string_view>(argv.begin(),argv.end()) ==
        std::vector<std::string_view>(qwen_contract::argv.begin(),qwen_contract::argv.end()), "plain default matches normalized argv contract");
    for (const auto entry : qwen_contract::env) {
        const auto equal = entry.find('=');
        expect(automatic.environment.at(std::string(entry.substr(0,equal))) == entry.substr(equal+1U), "active environment matches contract");
    }
    for (const auto& entry : qpf2k::defaults) {
        expect(automatic.environment.at(entry.name) == entry.value && automatic.sources.at(entry.name) == OptionSource::automatic,
            "existing and new defaults retain automatic provenance");
    }
    for (const auto* name : excluded_inactive_env)
        expect(!automatic.environment.contains(name), "inactive environment is excluded from actual resolution");
    expect(!automatic.options.mtp.enabled && !automatic.options.prepared_eager && !automatic.options.moe_shared_overlap &&
        !automatic.options.gdn_islands, "prefill organization does not enable speculation, prepared overlap or islands");
    expect(automatic.options.known_prefix_tail_rows.empty() && automatic.options.known_prefix_ladder_tail_rows ==
        std::vector<std::uint32_t>({1024U,256U,64U,16U}), "SCHED=12 retains configured ladder and prunes unused owners");
    const auto json = effective_config_json(automatic);
    if (argc == 2 && std::string_view(arguments[1]) == "--dump-fixture-config") { std::cout << json; return 0; }
    expect(json == effective_config_json(resolve(request,memory)) && json.find("MODEL") == std::string::npos &&
        json.find("\"source\":\"automatic\"") != std::string::npos && !automatic.host_compilation.empty(), "dry-run JSON is deterministic, path-free and reports provenance");
    // Compile organization/buffer requests from each route, not only booleans.
    auto manual_args = std::vector<std::string>{"decoder","MODEL","1"};
    for (const auto arg : qwen_contract::argv) manual_args.emplace_back(arg);
    auto manual = parse(std::move(manual_args));
    manual.tokens = request.options.tokens;
    const auto expected_compiler = prompt_compiler_requests(manual,model,manifest);
    const auto actual_compiler = prompt_compiler_requests(automatic.options,model,manifest);
    expect(actual_compiler.binding.wave8_q8_gdn_input && actual_compiler.binding.wave8_q8_gdn_output &&
        actual_compiler.binding.wave8_q8_qsa_output && actual_compiler.binding.wave8_q8_lm_head &&
        actual_compiler.binding.prefetch_q8_lm_head && actual_compiler.binding.q51_slot_parallel_down &&
        actual_compiler.binding.batched_moe_activation && actual_compiler.binding.a8_down_q51 &&
        actual_compiler.binding.a8_down_q80 && actual_compiler.binding.q5k_a8g32_gate_up &&
        actual_compiler.binding.dense_a8 && actual_compiler.binding.lane_chain_hyper_output_eligible_steps.size() == 97U &&
        actual_compiler.binding.qsa_input_prefetch_layers.size() == 12U &&
        actual_compiler.binding.shared_expert_prefetch_layers.size() == 48U,
        "host compiler receives the complete measured projection, arithmetic and prefetch organization");
    const auto expected_buffers = build_serial_execution_buffer_plan(model,expected_compiler.buffers);
    const auto actual_buffers = build_serial_execution_buffer_plan(model,actual_compiler.buffers);
    expect(expected_buffers.peak_internal_bytes == actual_buffers.peak_internal_bytes &&
        expected_buffers.mlp_intermediate_bytes == actual_buffers.mlp_intermediate_bytes &&
        expected_buffers.gdn_workspace_bytes == actual_buffers.gdn_workspace_bytes &&
        expected_buffers.hyper_workspace_bytes == actual_buffers.hyper_workspace_bytes,
        "normalized recipe and automatic organization compile identical buffer requirements");
    expect(actual_compiler.binding.wave8_q8_gdn_input == expected_compiler.binding.wave8_q8_gdn_input &&
        actual_compiler.binding.wave8_q8_gdn_output == expected_compiler.binding.wave8_q8_gdn_output &&
        actual_compiler.binding.wave8_q8_qsa_output == expected_compiler.binding.wave8_q8_qsa_output &&
        actual_compiler.binding.q51_slot_parallel_down == expected_compiler.binding.q51_slot_parallel_down &&
        actual_compiler.binding.dense_a8 == expected_compiler.binding.dense_a8 &&
        actual_compiler.catalog.numerical_profile == expected_compiler.catalog.numerical_profile,
        "compiler binding and arithmetic organization match the explicit recipe");
    for (const auto rows : {1024U,4096U}) {
        auto short_request = request; short_request.options.tokens.resize(rows);
        const auto short_config = resolve(short_request,memory);
        expect(short_config.admitted && short_config.options.known_prefix_prefill_rows == rows,
            "short prompt chooses its measured T1024/T4096 class");
    }
    auto four_k = request; four_k.options.tokens.resize(4096U);
    auto four_k_receipt = argv;
    four_k_receipt[1] = "8192";
    four_k_receipt.erase(std::find(four_k_receipt.begin(),four_k_receipt.end(),"--unqualified-context"));
    const auto tile_arg = std::find(four_k_receipt.begin(),four_k_receipt.end(),"--known-prefix-prefill-rows");
    *std::next(tile_arg) = "4096";
    expect(effective_prefill_argv(resolve(four_k,memory).options) == four_k_receipt,
        "short_prompt_receipt_argv: 4K preserves the exact historical receipt argv order");
    // Compare all emitted organization/arithmetic/context/budget/tail decisions
    // with the frozen pre-policy resolver, rather than a second call to live logic.
    const auto historical_equal = [&](const RequestedOptions& input, const EffectiveConfig& actual) {
        auto old = input.options;
        const auto expected = resolve_historical_9b1d1c3e(old,model,info,manifest,
            validated_model_content_identity,true,input.permit_generation_defaults);
        auto normalized = actual.options;
        normalized.baseline = old.baseline; // The policy marker itself is new.
        expect(!actual.admitted && effective_prefill_argv(normalized) == effective_prefill_argv(old) &&
            normalized.known_prefix_tail_rows == old.known_prefix_tail_rows &&
            normalized.known_prefix_ladder_tail_rows == old.known_prefix_ladder_tail_rows &&
            actual.organization.selected == expected.selected && actual.organization.reason == expected.reason &&
            actual.organization.automatic_applied == expected.automatic_applied,
            "complete fallback resolution equals the frozen historical resolver");
        Environment expected_env = input.environment;
        const bool baseline = input.options.baseline || env_is(input.environment,"STRIX_BASELINE","1");
        if (baseline) {
            for (const auto& entry : qpf2k::defaults) expected_env[entry.name] = "0";
            expected_env["STRIX_QPF2K_DEFAULTS"] = "0";
        } else {
            qpf2k::initialize([&](const char* name) -> const char* {
                const auto it = expected_env.find(name);
                return it == expected_env.end() ? nullptr : it->second.c_str();
            }, [&](const char* name, const char* value) { return expected_env.emplace(name,value).second; },
                qpf2k::historical_items);
        }
        expect(actual.environment == expected_env, "fallback environment equals historical environment plus explicit controls");
    };
    for (const auto& entry : qpf2k::defaults) {
        auto off = request; off.environment[entry.name] = "0";
        const auto config = resolve(off,memory);
        if (qpf2k::exact_prefill_items & qpf2k::bit(entry.item)) {
            expect(config.admitted && effective_prefill_argv(config.options) == effective_prefill_argv(automatic.options),
                "an exact prefill off switch keeps the admitted route");
            for (const auto& other : qpf2k::defaults)
                if (other.item != entry.item)
                    expect(config.environment.at(other.name) == other.value && config.sources.at(other.name) == OptionSource::automatic,
                        "an exact prefill off switch disables only that item");
            expect(effective_config_json(config).find(std::string("\"") + entry.name +
                "\":{\"value\":\"0\",\"source\":\"explicit\"}") != std::string::npos,
                "effective config reports the explicit per-item off switch");
        } else historical_equal(off,config);
        expect(config.environment.at(entry.name) == "0" && config.sources.at(entry.name) == OptionSource::explicit_value,
            "every feature off switch preserves explicit zero");
    }
    for (const auto* flag : {"--known-prefix-window", "--known-prefix-full-rows", "--known-prefix-expert-union",
            "--known-window-router-rows", "--known-prefix-prefill-rows"}) {
        auto off = request;
        off.options = parse({"decoder","MODEL","1","--generate","32",std::string(flag)+"=0"});
        off.options.tokens = request.options.tokens;
        historical_equal(off,resolve(off,memory));
    }
    for (const auto* name : {"STRIX_KNOWN_PREFIX_WINDOW", "STRIX_KNOWN_PREFIX_FULL_ROWS",
            "STRIX_KNOWN_PREFIX_EXPERT_UNION", "STRIX_KNOWN_WINDOW_ROUTER_ROWS", "STRIX_KNOWN_PREFIX_PREFILL_ROWS"}) {
        auto off = request; off.environment[name] = "0";
        historical_equal(off,resolve(off,memory));
    }
    for (const auto* name : {"STRIX_QPF2K_DEFAULTS", "STRIX_PREFILL_F16"}) {
        auto off = request; off.environment[name] = "0";
        historical_equal(off,resolve(off,memory));
    }
    auto dense_cli = request; dense_cli.options.prefill_numerical_profile = dense_a8_profile;
    expect(!numerical_profile_pair(resolve(dense_cli,memory).options).split(), "explicit dense-A8 prefill wins");
    auto baseline = request; baseline.options.baseline = true;
    baseline.environment["STRIX_PREFILL_PIPE"] = "255";
    const auto base = resolve(baseline,memory);
    historical_equal(baseline,base);
    expect(numerical_profile_name(base.options.numerical_profile) == qwen_contract::baseline_decode &&
        base.options.prepared_eager == qwen_contract::baseline_prepared &&
        base.options.gdn_islands && base.options.moe_shared_overlap && base.options.context_tokens == 8192U &&
        !base.options.unqualified_context && !base.options.prefill_numerical_profile,
        "baseline retains historical exact-v5 prepared overlap islands and implicit 8192 context");
    baseline.options.baseline = false; baseline.environment["STRIX_BASELINE"] = "1";
    historical_equal(baseline,resolve(baseline,memory));
    expect(effective_prefill_argv(resolve(baseline,memory).options) == effective_prefill_argv(base.options), "baseline CLI and env are equivalent");
    for (const auto context : {8192U,65536U}) {
        auto explicit_context = request;
        explicit_context.options.context_explicit = true;
        explicit_context.options.context_tokens = context;
        explicit_context.options.unqualified_context = context > 8192U;
        explicit_context.options.supplied_options.insert("--context");
        if (context > 8192U) explicit_context.options.supplied_options.insert("--unqualified-context");
        const auto config = resolve(explicit_context,memory);
        expect(config.admitted && config.options.context_tokens == context && config.sources.at("--context") == OptionSource::explicit_value &&
            (context == 8192U || config.sources.at("--unqualified-context") == OptionSource::explicit_value), "explicit context is preserved without automatic provenance");
        const auto explicit_notice = effective_config_notice(config);
        expect(context == 8192U ? explicit_notice.find("unqualified-context") == std::string::npos :
            explicit_notice.find("beyond qualified 8192; explicit") != std::string::npos,
            "notice labels explicit long context without inventing automatic acknowledgement");
        explicit_context.options.baseline = true;
        historical_equal(explicit_context,resolve(explicit_context,memory));
    }
    auto unacknowledged = request;
    unacknowledged.options.context_tokens = 65536U; unacknowledged.options.context_explicit = true;
    expect(!resolve(unacknowledged,memory).admitted, "explicit unqualified context still needs acknowledgement");
    const auto auto_notice = effective_config_notice(automatic);
    expect(auto_notice.find("automatic T8192 context=33280") != std::string::npos &&
        auto_notice.find("unqualified-context (beyond qualified 8192; automatic)") != std::string::npos &&
        std::count(auto_notice.begin(),auto_notice.end(),'\n') == 1 &&
        automatic.sources.at("--unqualified-context") == OptionSource::automatic,
        "automatic long context emits one qualified-limit notice and records automatic provenance");
    auto exported = request; exported.options.tokens.resize(1024U); exported.options.route_owned = true; exported.permit_generation_defaults = false;
    const auto export_config = resolve(exported,memory);
    historical_equal(exported,export_config);
    expect(!export_config.options.prepared_eager && !export_config.options.moe_shared_overlap && !export_config.options.gdn_islands,
        "target export excludes ordinary-generation submission defaults independently of route ownership");
    for (unsigned route = 0; route != 4; ++route) {
        auto owned = request;
        if (route == 0 || route == 1) owned.options = parse(mtp_fixture());
        if (route == 1) {
            auto& m = owned.options.mtp;
            m.window_rows = 8U; m.window_rows_set = {2U,4U,8U}; m.candidate_depth = 7U;
            m.window_prefix = m.draft_chain = true;
            m.window_arena_bytes = m.window_capture_bytes = m.window_checkpoint_bytes = m.window_total_bytes = 1024U;
            m.window_history_intervals = 1U;
            m.cohort.push_back({{1U,2U,3U},"cohort.golden",std::string(64,'1')});
            owned.options.wave8_q8_gdn_input = owned.options.wave8_q8_gdn_output = true;
        }
        if (route == 2) owned.options = parse({"decoder","MODEL","--generate","32","--serve-prefix-u32le","prefix.u32",
            "--serve-memory-budget-bytes","8192","--serve-request-budget-bytes","1024","--serve-headroom-bytes","1024"});
        if (route == 3) {
            owned.options = manual;
            owned.options.mtp = parse(mtp_fixture()).mtp;
            auto& m = owned.options.mtp;
            m.window_rows = 8U; m.window_rows_set = {2U,4U,8U}; m.candidate_depth = 7U;
            m.window_prefix = m.draft_chain = true;
            m.window_arena_bytes = m.window_capture_bytes = m.window_checkpoint_bytes = m.window_total_bytes = 1024U;
            m.window_history_intervals = 1U;
            owned.options.http_serve = true;
            validate_http_serve(owned.options);
        }
        owned.environment["STRIX_PREFILL_PIPE"] = "7";
        owned.environment["STRIX_LM_HEAD_COHORT_LEAN"] = "0";
        const auto config = resolve(owned,memory);
        historical_equal(owned,config);
        expect(config.environment.at("STRIX_PREFILL_PIPE") == "7" && config.environment.at("STRIX_LM_HEAD_COHORT_LEAN") == "0",
            "declined serving/MTP/cohort preserves explicit wrapper overrides");
        owned.environment.clear();
        const auto absent = resolve(owned,memory);
        historical_equal(owned,absent);
        expect(!absent.environment.contains("STRIX_PREFILL_PIPE") && !absent.environment.contains("STRIX_LM_HEAD_COHORT_LEAN"),
            "declined serving/MTP/cohort injects no wrapper settings");
    }
    auto dense_baseline = baseline;
    dense_baseline.options.numerical_profile = dense_a8_profile;
    dense_baseline.options.numerical_profile_explicit = true;
    const auto explicit_dense_base = resolve(dense_baseline,memory);
    historical_equal(dense_baseline,explicit_dense_base);
    auto invalid_baseline = explicit_dense_base.options;
    invalid_baseline.prepared_eager = false;
    bool rejected_baseline = false;
    try {
        auto org = explicit_dense_base.organization;
        apply_numerical_profile(invalid_baseline,model,manifest,validated_model_content_identity,true,org);
    } catch (const std::invalid_argument&) { rejected_baseline = true; }
    expect(rejected_baseline, "baseline has no exemption from historical dense-A8 submission prerequisites");
    auto missing = memory; missing.available.reset();
    expect(resolve(request,missing).reason == "memory-observation-unavailable", "missing memory observation cannot assume capacity");
    auto small = memory; small.available = 16U*gib;
    const auto declined = resolve(request,small);
    historical_equal(request,declined);
    const auto notice = effective_config_notice(declined);
    expect(!declined.admitted && !declined.options.known_prefix_window &&
        effective_config_notice(declined).starts_with("Qwen auto-prefill declined: insufficient-memory;") &&
        notice.find("using serial prefill") == std::string::npos && !declined.error.empty() &&
        std::count(notice.begin(),notice.end(),'\n') == 1,
        "shortage names the admission reason and refuses a prompt exceeding serial context");
    auto overflow = request; overflow.options.generate_tokens = UINT64_MAX;
    expect(resolve(overflow,memory).reason == "prompt-exceeds-model-context", "implicit context rejects overflow without adding");
    auto short_serial = request; short_serial.options.tokens.resize(4096U);
    const auto serial = resolve(short_serial,small);
    expect(!serial.admitted && serial.error.empty() &&
        effective_config_notice(serial).find("using serial prefill") != std::string::npos,
        "serial_fallback_fits: short prompts retain the historical serial fallback");
    require_effective_prefill_execution(serial);
    auto diagnostic = request; diagnostic.environment["STRIX_RTX_EXPORT_STATE"] = "1";
    expect(!resolve(diagnostic,memory).admitted, "explicit state-export control owns its existing execution route");
    auto no_target = capabilities; no_target.target_qualified = false;
    expect(!resolve_effective_config(request,model,info,manifest,validated_model_content_identity,no_target,memory).admitted,
        "unqualified target declines before estimating");
    auto f16_failure = memory;
    f16_failure.estimate = [&](const Options& candidate) {
        if (candidate.prefill_numerical_profile) throw std::invalid_argument("unavailable fast operands");
        return memory.estimate(candidate);
    };
    const auto dense_fallback = resolve(request,f16_failure);
    expect(dense_fallback.admitted && !numerical_profile_pair(dense_fallback.options).split() &&
        effective_config_notice(dense_fallback).find("fp16-plan-not-admitted") != std::string::npos,
        "automatic FP16 admission failure retries the complete qualified dense-A8 recipe");
    auto saturated = memory; saturated.gtt_used = *saturated.gtt_limit;
    expect(resolve(request,saturated).reason == "insufficient-memory", "device GTT remaining capacity is capped separately");
    for (const bool gsq : {true,false,true}) {
        qpf2k::defaulted_items = (1U << qpf2k::defaults.size()) - 1U;
        const qpf2k::ModelScope scope(gsq);
        for (const auto& entry : qpf2k::defaults)
            expect(qpf2k::suppressed(entry.item) == gsq, "owner alternation filters every newly injected feature after parsing");
    }
    qpf2k::defaulted_items = 0U;
    auto explicit_route = request; explicit_route.options = manual;
    const auto untouched = resolve(explicit_route,memory);
    expect(!untouched.admitted && effective_prefill_argv(untouched.options) == effective_prefill_argv(manual),
        "explicit route flags are untouched");
    const auto absent_build = resolve_effective_config(request,model,info,manifest,validated_model_content_identity,
        {false,true,true},memory);
    expect(!absent_build.admitted, "missing A8 capability declines");
    auto gsq_manifest = manifest;
    ManifestTensor gsq_gate;
    gsq_gate.name = "blk.0.ffn_gate_exps.weight"; gsq_gate.ggml_type = gsq_a8::type_q2_0;
    gsq_manifest.tensors.push_back(gsq_gate);
    gsq_manifest.layout_identity = model_manifest_layout_identity(gsq_manifest);
    const auto gsq_decline = resolve_effective_config(request,model,info,gsq_manifest,
        validated_model_content_identity,capabilities,memory);
    const auto nonqwen = resolve_effective_config(request,model,info,manifest,"unqualified-content",capabilities,memory);
    auto other_model = model; other_model.hidden_size += 32U;
    const auto noncanonical = resolve_effective_config(request,other_model,info,manifest,
        validated_model_content_identity,capabilities,memory);
    for (const auto& config : {absent_build, gsq_decline, nonqwen, noncanonical, declined, resolve(request,missing)}) {
        expect(!config.admitted && config.environment == export_config.environment,
            "declined build/GSQ/non-Qwen/memory routes have exactly the historical environment");
        expect(effective_config_notice(config).find("unqualified-context") == std::string::npos,
            "automatic long-context qualification notice stays confined to admission");
    }
    const auto baseline_absent = resolve_effective_config(baseline,model,info,manifest,
        "unqualified-content",{false,false,false},missing);
    expect(!baseline_absent.admitted && baseline_absent.options.numerical_profile == DecodeNumericalProfile::exact_v5 &&
        !baseline_absent.options.prefill_numerical_profile && baseline_absent.options.context_tokens == request.options.context_tokens,
        "baseline never forces dense-A8 or grows context when model/build admission fails");
    std::cout << "qwen effective config: " << failures << " failures\n";
    return failures ? 1 : 0;
}
