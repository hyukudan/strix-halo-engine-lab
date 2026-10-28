#include "../apps/qwen_prompt_organization.hpp"
#include "strix/mtp_selection_transaction.hpp"
#include "qwen_prefill_contract.hpp"
#include "rotary_mock.hpp"
#include "strix/memory_plan.hpp"
#include "strix/session_state_cache.hpp"
#include "../apps/qwen_yarn_owner_inventory.hpp"
#include "qwen_serial_contract.hpp"
#include "qwen_historical_resolver.hpp"
#include <iostream>
#include <limits>
#include <set>
#include "strix/q8_projection_organization.hpp"

using namespace strix;
using namespace strix::decode_cli;
namespace {
// Synthetic CPU kernel fixture: compares compiler identities without HIP launchers.
strix::kernels::KernelVariant config_test_variant(
    std::string id,
    strix::kernels::OperatorFamily family,
    const strix::kernels::ShapeClass& shape,
    double measured_seconds) {
    strix::kernels::KernelVariant variant;
    variant.id = std::move(id);
    variant.family = family;
    variant.phase =
        strix::kernels::ExecutionPhase::decode;
    variant.shape = shape;
    variant.wave =
        strix::kernels::WaveSize::wave32;
    variant.resources.workgroup_threads = 256;
    variant.numerics.input_precision = "bf16";
    variant.numerics.accumulator_precision = "fp32";
    variant.numerics.output_precision = "bf16";
    variant.numerics.row_invariant = true;
    variant.numerics.deterministic = true;
    variant.qualification =
        strix::kernels::QualificationStatus::oracle_matched;
    variant.graph_safe = true;
    variant.input_layout = "test";
    variant.output_layout = "test";
    variant.compiler_identity = "test-compiler";
    variant.measured_seconds = measured_seconds;
    return variant;
}

strix::kernels::KernelRegistry config_test_registry(
    const strix::ModelContract& model,
    const strix::ExecutionBindingRequest& request,
    double measured_seconds = 0.0,
    std::string suffix = "") {
    strix::kernels::KernelRegistry registry;
    const auto execution =
        strix::build_serial_decode_execution_plan(model);

    std::set<std::string> added;
    for (const auto& step : execution.steps) {
        const auto family =
            strix::operator_family_for(step.op);
        if (!family.has_value()) {
            continue;
        }

        const auto shape =
            strix::execution_shape_for(
                model, step.op, request);

        const auto key =
            std::to_string(
                static_cast<unsigned>(*family)) +
            "." +
            std::to_string(shape.logical_rows) +
            "." +
            std::to_string(shape.request_concurrency) +
            "." +
            std::to_string(shape.hidden_size) +
            "." +
            std::to_string(shape.auxiliary_size);

        if (!added.insert(key).second) {
            continue;
        }

        auto variant = config_test_variant(
                "test." + key + suffix +
                    (step.op == strix::SerialOp::moe_route && request.staged_moe_router_logits ? ".staged_logits" : "") +
                    (step.op == strix::SerialOp::moe_route && request.wave_moe_router_topk ? ".wave.parallel" :
                     step.op == strix::SerialOp::moe_route && request.ranked_moe_router_topk ? ".ranked.parallel" :
                     step.op == strix::SerialOp::moe_route && request.parallel_moe_router ? ".parallel" : "") +
                    (step.op == strix::SerialOp::moe_experts && request.batched_moe_activation ? ".batched-slot-activation-v1" : "") +
                    (step.op == strix::SerialOp::moe_experts && request.deferred_moe_status ? ".deferred-status-v1" : "") +
                    (step.op == strix::SerialOp::moe_experts && request.q5k_a8g32_gate_up ? ".q5k-a8g32-gateup-v1" : "") +
                    (step.op == strix::SerialOp::moe_experts && request.a8_down_q51 ? ".q51-a8g32-down-v1" : "") +
                    (step.op == strix::SerialOp::moe_experts && request.a8_down_q80 ? ".q80-a8g32-down-v1" : "") +
                    (step.op == strix::SerialOp::moe_experts && request.a8_down_q51 ? ".mixed-a8-down-v1" : "") +
                    (step.op == strix::SerialOp::moe_experts && request.grouped_exact_q51_down && !request.a8_down_q51 ? ".grouped-exact-q51-down-v1" : "") +
                    (step.op == strix::SerialOp::moe_experts && request.q4k_a8g32_gate_up ? ".q4k-a8g32-gateup-v2" : ""),
                *family,
                shape,
                measured_seconds);
        using namespace strix;
        using namespace q8_projection_organization;
        if (step.op==SerialOp::gated_delta_net) {
            if(request.wave8_q8_gdn_input) variant.id+=gdn_wave8_marker;
            if(request.wave8_q8_gdn_output) variant.id+=gdn_output_wave8_marker;
        }
        if(step.op==SerialOp::indexed_attention && request.wave8_q8_qsa_output) variant.id+=qsa_output_wave8_marker;
        if(request.dense_a8 && (step.op==SerialOp::gated_delta_net || step.op==SerialOp::indexed_attention || step.op==SerialOp::shared_expert))
            variant.id+=dense_a8_marker;
        if(step.op==SerialOp::lm_head && request.prefetch_q8_lm_head)
            variant.id+=request.wave8_q8_lm_head ? lm_wave8_marker : lm_prefetch_marker;
        if(*family==kernels::OperatorFamily::hyper_mix && request.prefetch_hc_projection_next_tile) variant.id+=".prefetch-next-tile-v1";
        registry.add(variant);
        // Layer-scoped selection can require either terminal suffix for one family.
        if(step.op==SerialOp::indexed_attention) {variant.id+=qsa_input_prefetch::kernel_suffix; registry.add(variant);}
        if(step.op==SerialOp::shared_expert) {variant.id+=shared_expert_prefetch::kernel_suffix; registry.add(variant);}
        if(*family==kernels::OperatorFamily::hyper_mix) {variant.id+=lane_chain_hyper_output_suffix; registry.add(variant);}
    }

    return registry;
}

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
constexpr std::uint64_t gib_bytes = UINT64_C(1024) * 1024U * 1024U;
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
    return {128U*gib_bytes,128U*gib_bytes,0U,[](const Options&) {
        return PrefillRequirements{70U*gib_bytes,8U*gib_bytes,gib_bytes,2U*gib_bytes,512U*1024U*1024U,128U*1024U*1024U,0U,9U*gib_bytes};
    }};
}
void yarn_admission_tests(const gguf::FileInfo& info, const ModelManifest& manifest) {
    for (const auto factor : {0U,2U,4U}) {
        auto model = ModelContract::qwen38_flash_next();
        if (factor) model.qwen_rotary_table = mock_rotary(factor);
        for (const std::uint64_t capacity : {0U,262143U,262144U,262145U,524288U,524289U,1048576U,1048577U}) {
            const bool fits = capacity && capacity <= model.effective_max_positions();
            bool layout = true, memory = true;
            try { SerialSessionStateLayoutRequest request; request.context_tokens = capacity;
                const auto state = build_serial_session_state_layout(model,request);
                expect(state.context_tokens == capacity,"YaRN layout preserves requested capacity");
            } catch (const std::invalid_argument&) { layout = false; }
            try { MemoryPlanRequest request; request.context_tokens = capacity;
                (void)build_memory_plan(model,request);
            } catch (const std::invalid_argument&) { memory = false; }
            expect(layout == fits && memory == (!factor || fits),"native planning and YaRN execution cap matrix");
        }
    }
    {
        RequestedOptions off;
        off.options = parse({"decoder","MODEL","1","--rope-yarn","off"});
        off.environment["STRIX_QWEN_ROPE_YARN"] = "2";
        const auto native = resolve_effective_config(off,ModelContract::qwen38_flash_next(),info,manifest,
            validated_model_content_identity,{true,true,true},ample_memory());
        expect(native.options.rotary.mode() == RotaryMode::native &&
            native.environment.at("STRIX_QWEN_ROPE_YARN") == "off", "CLI off receipt reports effective native environment");
    }
    auto model = ModelContract::qwen38_flash_next(); model.qwen_rotary_table = mock_rotary(2);
    RequestedOptions request;
    request.options = parse({"decoder","MODEL","1","--rope-yarn","2","--context","300000","--generate","32"});
    request.options.tokens.resize(8192U,1U);
    auto memory = ample_memory(); memory.total = 124U*gib_bytes;
    const auto resolve = [&](const RequestedOptions& r, const PrefillMemory& m) {
        return resolve_effective_config(r,model,info,manifest,validated_model_content_identity,{true,true,true},m);
    };
    {
        SerialSessionStateLayoutRequest state_request; state_request.context_tokens = 300000U;
        const auto state = build_serial_session_state_layout(model,state_request);
        auto sized = request.options; sized.known_prefix_prefill_rows = 8192U;
        PrefillRequirements required;
        required.state_d1 = prefill_checked_add(state.device_total_bytes,
            prefill_checked_add(state.host_total_bytes,prefill_checked_add(state.host_total_bytes,state.host_alignment-1U)));
        required.model_backing = 111334654400ULL;
        append_yarn_owner_requirements(sized,model,state,required.model_backing,required);
        expect(required.owners.at("target_kv") == 7372800000ULL && required.owners.at("target_raw_index") == 921600000ULL,
            "YaRN exact target KV and raw index owners at C300000");
        expect(required.owners.at("qsa_shared_prefill_pool") == 38400000U &&
            required.owners.at("qsa_window_pool_and_slab_bound") == 0U,"YaRN one pool across twelve layers and absent window owners");
        std::uint64_t state_sum = 0;
        for (const auto* name : {"target_kv","target_raw_index","gdn_matrix","gdn_convolution","ple_convolution",
                "ngram_history","position_history","state_alignment_and_d1_shadow"})
            state_sum = prefill_checked_add(state_sum,required.owners.at(name));
        expect(state_sum == required.state_d1,"YaRN state owner inventory reconciles aligned payload and shadow");
        expect(required.owners.at("routed_csr") == 1315584U &&
            required.owners.at("merged_routed_csr") == 0U &&
            !required.owners.contains("tail_64_ple_staging"),
            "YaRN primary CSR present, tail/merged owners absent for exact T prompt");
        const auto primary = required;
        {
            // Compact geometry: the ring and terminal stripes are split out of workspace/tails once, the embedded maxima
            // owners drop to zero and the decode pool is named apart from the state shadow.
            auto compact = PrefillRequirements{};
            compact.model_backing = required.model_backing;
            const auto decode = qsa_decode_pool_bytes(sized,model);
            compact.state_d1 = required.state_d1 + decode;
            compact.workspace = 1000000000U;
            compact.tails = 200000000U;
            const YarnPrefillOwnerGeometry geometry{true,671088640U,993280U,decode,672081920U};
            append_yarn_owner_requirements(sized,model,state,compact.model_backing,compact,geometry);
            expect(compact.owners.at("qsa_selection_maxima")==0 &&
                compact.owners.at("qsa_batch_maxima_growth_peak")==0,
                "compact YaRN removes embedded maxima owners");
            expect(compact.owners.at("qsa_score_ring")+compact.owners.at("terminal_logits_stripes")+
                compact.owners.at("prefill_workspace_and_status")+compact.owners.at("tail_workspaces_and_status") ==
                compact.workspace+compact.tails,"compact ring and logits split workspace exactly once");
            expect(compact.owners.at("qsa_decode_pool")==decode &&
                compact.owners.at("state_alignment_and_d1_shadow")==primary.owners.at("state_alignment_and_d1_shadow"),
                "decode pool is separate from state shadow and charged once");
            auto over = compact; over.workspace = 1U; over.tails = 0U;
            bool refused = false;
            try { append_yarn_owner_requirements(sized,model,state,over.model_backing,over,geometry); }
            catch (const std::invalid_argument&) { refused = true; }
            expect(refused,"compact owners larger than the workspace reservation refuse");
        }
        sized.tokens.resize(299000U,1U);
        sized.known_prefix_tail_rows = {64U,16U};
        sized.known_prefix_ladder_tail_rows = {1024U,256U,64U,16U};
        PrefillRequirements with_tails;
        with_tails.state_d1 = primary.state_d1;
        with_tails.model_backing = primary.model_backing;
        with_tails.workspace = 1U;
        append_yarn_owner_requirements(sized,model,state,with_tails.model_backing,with_tails);
        std::uint64_t tail_ple = 0;
        for (const auto rows : {64U,16U}) {
            const auto prefix = "tail_" + std::to_string(rows) + "_";
            for (const auto* owner : {"ple_staging","ple_projections_and_gates","ple_snapshot_and_hashes"})
                tail_ple = prefill_checked_add(tail_ple,with_tails.owners.at(prefix+owner));
            expect(with_tails.owners.at(prefix+"routed_csr") ==
                q51_known_rows::prefill_scratch_bytes(rows,512U,10U),"YaRN each retained tail has independent CSR");
        }
        expect(tail_ple == 7111040U && with_tails.owners.at("merged_routed_csr") ==
            q51_known_rows::prefill_scratch_bytes(3840U,512U,10U),
            "YaRN tail PLE regression and once-retained merged primary CSR");
        // Mock the same compiled owners present/absent. Every T trial uses the
        // same fixture so fallback cannot conceal an envelope or GTT edge.
        auto absent = with_tails;
        std::uint64_t omitted = 0;
        for (const auto& [name,bytes] : with_tails.owners) {
            if ((name.starts_with("tail_") && name != "tail_workspaces_and_status") ||
                name == "routed_csr" || name == "merged_routed_csr") {
                omitted = prefill_checked_add(omitted,bytes);
                absent.owners.erase(name);
            }
        }
        absent.device_base -= omitted;
        std::uint64_t gross = 8U*gib_bytes + request.options.rotary.table_bytes();
        for (const auto& [name,bytes] : with_tails.owners) { (void)name; gross = prefill_checked_add(gross,bytes); }
        auto edge = memory;
        edge.verify_ple_credit = [](const Options&) { return 16U*gib_bytes; };
        edge.total = gross-16U*gib_bytes-1U; // total-envelope edge after verified credit
        edge.estimate = [=](const Options&) { return absent; };
        expect(resolve(request,edge).admitted,"YaRN missing external owners would admit envelope edge");
        edge.estimate = [=](const Options&) { return with_tails; };
        auto refused = resolve(request,edge);
        expect(!refused.admitted && refused.error.find("owner_tail_64_ple_staging_bytes=") != std::string::npos &&
            refused.error.find("owner_merged_routed_csr_bytes=") != std::string::npos &&
            effective_config_json(refused).find("owner_tail_16_routed_csr_bytes") != std::string::npos,
            "YaRN complete owner refusal and effective-config list each external owner");
        edge.total = 124U*gib_bytes;
        edge.gtt_limit = with_tails.device_base + with_tails.workspace + request.options.rotary.table_bytes()-1U;
        edge.estimate = [=](const Options&) { return absent; };
        expect(resolve(request,edge).admitted,"YaRN missing external owners would admit device edge");
        edge.estimate = [=](const Options&) { return with_tails; };
        expect(!resolve(request,edge).admitted,"YaRN external owners count toward device growth independently");
        auto invalid = model; invalid.num_experts = UINT64_MAX;
        bool csr_guarded = false;
        try { auto value = primary; append_yarn_owner_requirements(sized,invalid,state,value.model_backing,value); }
        catch (const std::invalid_argument&) { csr_guarded = true; }
        expect(csr_guarded,"YaRN CSR dimensions checked before narrowing or layout arithmetic");
        auto bad = model; bad.gdn_key_heads = UINT64_MAX;
        bool guarded = false;
        try { append_yarn_owner_requirements(sized,bad,state,required.model_backing,required); }
        catch (const std::overflow_error&) { guarded = true; }
        expect(guarded,"YaRN owner width products checked before context multiplication");
    }
    {
        // Exercise the process collector, including ordinary shell storage paths.
        const auto saved = [](const char* name) -> std::optional<std::string> {
            const auto* value = std::getenv(name);
            return value ? std::optional<std::string>{value} : std::nullopt;
        };
        const auto home = saved("HOME"), xdg = saved("XDG_CACHE_HOME");
        const auto signature = session_numerical_config("same execution");
        ::setenv("HOME", "storage/home", 1);
        ::setenv("XDG_CACHE_HOME", "storage/cache", 1);
        auto collected = request;
        collected.options.context_tokens = 524288U;
        collected.environment = requested_prefill_environment();
        expect(collected.environment.at("HOME") == "storage/home" &&
            collected.environment.at("XDG_CACHE_HOME") == "storage/cache", "real collector retains storage inputs");
        expect(resolve(collected,memory).admitted, "YaRN admits real collector with shell storage locations");
        expect(session_numerical_config("same execution") == signature, "storage paths do not change execution signature");
        collected.environment["STRIX_MOCK_UNCOVERED_ROTARY"] = "1";
        expect(resolve(collected,memory).error ==
            "YaRN execution environment not supported: STRIX_MOCK_UNCOVERED_ROTARY",
            "storage classification still refuses unknown execution controls");
        if (home) ::setenv("HOME",home->c_str(),1); else ::unsetenv("HOME");
        if (xdg) ::setenv("XDG_CACHE_HOME",xdg->c_str(),1); else ::unsetenv("XDG_CACHE_HOME");
    }
    // Reproduce the complete startup environment from the 8192-ID plain
    // command, including the two-bit MoE marker that previously refused.
    auto startup = request;
    qpf2k::initialize([&](const char* name) -> const char* {
        const auto it = startup.environment.find(name);
        return it == startup.environment.end() ? nullptr : it->second.c_str();
    }, [&](const char* name, const char* value) {
        startup.environment.emplace(name,value); return true;
    }, qpf2k::startup_items);
    const Environment recorded_startup{
        {"STRIX_PLE_RECLAIMABLE","auto"}, {"STRIX_QPF2K_DENSE","1"},
        {"STRIX_QPF2K_GDN","1"}, {"STRIX_QPF2K_GDN_CORE16","1"},
        {"STRIX_QPF2K_GDN_GATES_BATCH","1"}, {"STRIX_QPF2K_HC","1"},
        {"STRIX_QPF2K_HC_MASK","14"}, {"STRIX_QPF2K_HC_OUTPUT3","1"},
        {"STRIX_QPF2K_HC_OUTPUT3_BAND","16"}, {"STRIX_QPF2K_MOE_DA16","1"},
        {"STRIX_QPF2K_MOE_DEFAULTED","3"}, {"STRIX_QPF2K_MOE_ROUTE","14"},
        {"STRIX_QPF2K_QSA","1"}, {"STRIX_QPF2K_QSA_MASK","0x257"},
        {"STRIX_QPF2K_SCHED","12"}};
    expect(startup.environment == recorded_startup, "YaRN exact recorded startup environment");
    expect(resolve(startup,memory).admitted && !resolve(startup,memory).admission.empty(),
        "YaRN exact resolver startup environment reaches full admission");
    for (const auto& item : qpf2k::defaults) {
        expect(yarn_environment_path(item.item) != YarnEnvironmentPath::uncovered_rotary,
            "every resolver item has an audited rotary classification");
        for (const auto* value : {"0", "1", "0xffffffff"}) {
            auto explicit_override = request;
            explicit_override.environment[item.name] = value;
            expect(yarn_environment_refusal(explicit_override.environment,2).empty() ==
                (item.item != qpf2k::Item::sched),
                "explicit overrides use the same rotary classification");
        }
    }
    expect(yarn_environment_path(static_cast<qpf2k::Item>(999U)) == YarnEnvironmentPath::uncovered_rotary,
        "new rotary items fail closed until audited");
    auto uncovered = request;
    uncovered.environment["STRIX_MOCK_UNCOVERED_ROTARY"] = "1";
    unsigned callbacks = 0;
    auto observed = memory;
    observed.verify_ple_credit = [&](const Options&) { ++callbacks; return 0U; };
    observed.estimate = [&](const Options&) { ++callbacks; return PrefillRequirements{}; };
    for (const auto* schedule : {"0","8","13","16","12"}) {
        auto scheduled = request;
        scheduled.options.tokens.resize(299000U,1U);
        scheduled.environment["STRIX_QPF2K_SCHED"] = schedule;
        callbacks = 0;
        const auto planned = resolve(scheduled,observed);
        if (std::string_view(schedule) != "12")
            expect(planned.error.find("STRIX_QPF2K_SCHED") != std::string::npos &&
                callbacks == 0U && planned.admission.empty(), "YaRN schedule refuses before reclaim and estimation");
        else {
            const auto admitted = resolve(scheduled,memory);
            expect(admitted.admitted && admitted.options.known_prefix_tail_rows ==
                std::vector<std::uint32_t>{64U,16U}, "YaRN schedule 12 admits the cursor owner ladder");
        }
        auto native = scheduled;
        native.options.rotary = RotaryDescriptor::native();
        native.options.context_tokens = 8192U;
        native.options.tokens.resize(4096U);
        const auto native_plan = resolve_effective_config(native,ModelContract::qwen38_flash_next(),
            info,manifest,validated_model_content_identity,{true,true,true},memory);
        expect(native_plan.environment.at("STRIX_QPF2K_SCHED") == schedule &&
            native_plan.error.find("YaRN") == std::string::npos, "native schedule remains unchanged");
    }
    callbacks = 0;
    auto defaults_off = request;
    defaults_off.environment["STRIX_QPF2K_DEFAULTS"] = "0";
    expect(resolve(defaults_off,observed).error.find("STRIX_QPF2K_SCHED") != std::string::npos && callbacks == 0U,
        "YaRN unset schedule with injection disabled refuses before callbacks");
    const auto rejected = resolve(uncovered,observed);
    expect(rejected.error == "YaRN execution environment not supported: STRIX_MOCK_UNCOVERED_ROTARY" &&
        callbacks == 0U && rejected.admission.empty(), "uncovered rotary refuses by name before admission");
    auto config = resolve(request,memory);
    expect(config.admitted && config.options.context_tokens == 300000U && config.options.known_prefix_prefill_rows == 8192U,
        "YaRN fits without shrinking context");
    expect(config.admission.at("host_reserve_bytes") == 8U*gib_bytes && config.ple_reclaim_credit == 0U,
        "YaRN reserve and zero unverified credit");
    expect(effective_config_json(config).find("admission") != std::string::npos &&
        effective_config_json(config).find("table_sha256") != std::string::npos,"YaRN inspection resolved identity and owners");
    auto large = memory;
    const auto estimate = memory.estimate;
    large.estimate = [=](const Options& o) { auto value = estimate(o); value.model_backing = 110U*gib_bytes; return value; };
    config = resolve(request,large);
    expect(!config.admitted && config.error.find("owner_model_backing_bytes") != std::string::npos &&
        config.error.find("verified_ple_credit_bytes=0") != std::string::npos &&
        config.error.find("smallest_T_tried=2048") != std::string::npos,"YaRN refusal lists owners credit limits and smallest T");
    expect(config.tile_fallback == "T8192:insufficient-memory,T4096:insufficient-memory,T2048:insufficient-memory",
        "YaRN fallback order");
    bool execution = false;
    try { require_effective_prefill_execution(config); execution = true; } catch (const std::invalid_argument&) {}
    expect(!execution,"YaRN refuses before any execution/allocation callback");
    large.verify_ple_credit = [](const Options&) { return 16U*gib_bytes; };
    expect(resolve(request,large).admitted,"YaRN verified PLE credit admits");
    auto failed_drop = large;
    unsigned estimates_after_drop_failure = 0;
    failed_drop.verify_ple_credit = [](const Options&) -> std::uint64_t { throw std::runtime_error("required drop not verified"); };
    failed_drop.estimate = [&](const Options&) { ++estimates_after_drop_failure; return PrefillRequirements{}; };
    expect(resolve(request,failed_drop).error.find("required drop not verified") != std::string::npos && !estimates_after_drop_failure,
        "YaRN required reclaim failure refuses before candidate growth or device setup");
    large.available = gib_bytes;
    expect(!resolve(request,large).admitted,"YaRN credit never inflates available headroom");
    auto fallback = memory;
    std::vector<std::uint32_t> trials;
    fallback.estimate = [&](const Options& o) { trials.push_back(o.known_prefix_prefill_rows); auto value = estimate(o);
        if (o.known_prefix_prefill_rows > 2048U) value.workspace = 9U*gib_bytes;
        return value; };
    config = resolve(request,fallback);
    expect(config.admitted && trials == std::vector<std::uint32_t>{8192U,4096U,2048U} &&
        config.options.prefill_numerical_profile == DecodeNumericalProfile::fast_prefill_f16wmma_v1,
        "YaRN T fallback keeps numerical lineage");
    auto reserve = memory;
    reserve.estimate = [=](const Options& o) { auto value = estimate(o); value.model_backing = 103U*gib_bytes; return value; };
    expect(!resolve(request,reserve).admitted,"YaRN eight GiB reserve honored");
    for (unsigned route = 0; route < 6; ++route) {
        auto bad = request;
        const char* reason = nullptr;
        if (route == 0) { bad.options.mtp.enabled = true; reason = "YaRN with MTP not yet qualified"; }
        if (route == 1) { bad.options.http_serve = true; reason = "YaRN on HTTP serving not supported"; }
        if (route == 2) { bad.options.teacher_forced_score = true; reason = "YaRN with teacher scoring not yet qualified"; }
        if (route == 3) { bad.options.gdn_islands = true; reason = "YaRN graph capture path not yet qualified"; }
        if (route == 4) { bad.options.session_save = "checkpoint"; reason = "YaRN with session save/restore not yet qualified"; }
        if (route == 5) { bad.options.prefix_cache_host_bytes = 4096; reason = "YaRN with prefix state cache not yet qualified"; }
        expect(resolve(bad,memory).error == reason,"YaRN named route refusal");
    }
    {
        auto unknown = request; unknown.environment["STRIX_QPF2K_SHADOW_MAX_TOKENS"] = "300000";
        expect(resolve(unknown,memory).error.find("YaRN execution environment not supported") != std::string::npos,
            "YaRN unbounded alternate launcher controls refuse");
        for (const char* name : {"STRIX_QSA_DECODE_POOL", "STRIX_QSA_DECODE_POOL_CHECK", "STRIX_QSA_DECODE_POOL_BF16", "STRIX_QSA_DECODE_SELECT"})
            expect(yarn_environment_refusal({{name, "0"}}, 2).empty(), "YaRN accepts the audited serial decode pool controls");
        for (const char* name : {"STRIX_CTX_EXACT_MEMORY", "STRIX_QSA_SCORE_RING", "STRIX_QSA_SCORE_RING_ROWS", "STRIX_TERMINAL_LOGITS_ARENA"})
            expect(yarn_environment_refusal({{name, "0"}}, 2).empty(), "YaRN accepts the compact-memory controls");
        expect(effective_prefill_argv(config.options)[2] == "--rope-yarn","YaRN receipt argv retains explicit opt-in");
    }
    auto overflow = memory;
    overflow.estimate = [=](const Options& o) { auto value = estimate(o); value.model_backing = UINT64_MAX; return value; };
    expect(resolve(request,overflow).error.find("overflow") != std::string::npos,"YaRN checked owner sum");
    bool multiplied = false;
    try { (void)prefill_checked_mul(UINT64_MAX,2U); } catch (const std::overflow_error&) { multiplied = true; }
    expect(multiplied,"YaRN checked width product");
    auto boundary = request; boundary.options.tokens.resize(300000U,1U);
    expect(resolve(boundary,memory).error.find("prompt + generation") != std::string::npos,"YaRN positions are below capacity");
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
    table.absolute_offset = 4U*gib_bytes + 32U;
    table.byte_size = table.dimensions[1] * 90U;
    manifest.tensors.push_back(table);
    manifest.layout_identity = model_manifest_layout_identity(manifest);
    auto observed = ample_memory();
    observed.available = UINT64_C(127800000000);
    observed.gtt_limit = 128U*gib_bytes;
    observed.gtt_used = 2U*gib_bytes;
    observed.ple_reclaim_page = 4096U;
    observed.estimate = [](const Options& options) {
        auto required = PrefillRequirements{};
        required.model_backing = 96U*gib_bytes;
        required.state_d1 = options.context_tokens * 120U*1024U;
        required.serial_scratch = gib_bytes;
        required.workspace = options.known_prefix_prefill_rows * (gib_bytes / 2048U);
        required.metadata = gib_bytes / 2U;
        required.runtime_payload = gib_bytes / 8U;
        required.device_base = 40U*gib_bytes;
        required.tails = static_cast<std::uint64_t>(options.known_prefix_tail_rows.size()) * (gib_bytes / 16U);
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
            config.options.context_tokens == expected_context && config.ple_reclaim_credit > 24U*gib_bytes,
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
        tight.gtt_limit = 2U*gib_bytes + 40U*gib_bytes + (chosen == 4096U ? 3U*gib_bytes : chosen == 2048U ? 2U*gib_bytes : gib_bytes / 2U);
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
        tight.available = 100U*gib_bytes;
        const auto total = *tight.available;
        const auto credit = prefill_reclaim_credit(resolve(request,observed).options,model,manifest,{},tight.ple_reclaim_page);
        tight.estimate = [&](const Options& candidate) {
            auto required = estimate(candidate);
            const auto other = required.state_d1 + required.serial_scratch + required.metadata + required.runtime_payload +
                required.tails + candidate.known_prefill_host_headroom;
            const auto allowed_workspace = chosen == 4096U ? 3U*gib_bytes : gib_bytes + gib_bytes / 2U;
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
        for (unsigned excluded = 0; excluded < 13U; ++excluded) {
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
            if (excluded == 10U) options.session_load = "checkpoint";
            if (excluded == 11U) options.session_live_control = true;
            if (excluded == 12U) options.session_then_prompt = "suffix.ids";
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
        const auto raised = credit(admitted.options,{{"STRIX_PLE_RECLAIM_DECODE_BYTES",std::to_string(gib_bytes)}},manifest);
        expect(admitted.ple_reclaim_credit - raised == gib_bytes - (256U << 20U),
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
    yarn_admission_tests(info,manifest);
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
    {
        auto configured=request;
        configured.environment=requested_prefill_environment([](const char* name) -> const char* {
            return std::string_view(name)=="STRIX_QSA_SCORE_RING_ROWS"?"256":nullptr;
        });
        const auto resolved=resolve(configured,memory);
        expect(resolved.environment.at("STRIX_QSA_SCORE_RING_ROWS")=="256" &&
            effective_config_json(resolved).find("STRIX_QSA_SCORE_RING_ROWS")!=std::string::npos,
            "ring row override is captured in effective configuration");
        configured.environment["STRIX_QSA_SCORE_RING_ROWS"]="192";
        bool rejected=false;
        try { (void)resolve(configured,memory); } catch(const std::invalid_argument&) { rejected=true; }
        expect(rejected,"invalid ring override cannot silently decline the automatic route");
        for(const char* off : {"STRIX_CTX_EXACT_MEMORY","STRIX_QSA_SCORE_RING"}) {
            auto disabled=configured;disabled.environment[off]="0";
            expect(resolve(disabled,memory).admitted,"ring off switches ignore an inactive rows override");
        }
        const auto base=resolve(request,memory);
        const auto plan_ids=[&](const Options& options) {
            std::vector<std::uint64_t> ids;
            for (const auto profile : {options.numerical_profile,numerical_profile_pair(options).prefill}) {
                auto req=prompt_compiler_requests(options,model,manifest);
                if(profile!=options.numerical_profile) req.binding.prefill_arithmetic_identity=numerical_profile_pair_identity(numerical_profile_pair(options));
                const auto registry=config_test_registry(model,req.binding);
                const auto compiled=compile_serial_execution(model,registry,req.binding,req.buffers,256);
                ids.push_back(compiled.execution_identity); ids.push_back(compiled.kernel_plan_identity);
            }
            return ids;
        };
        const auto base_ids=plan_ids(base.options);
        const auto numerical=[](EffectiveConfig c) {
            c.environment.erase("STRIX_PREFIX_CACHE"); c.environment.erase("STRIX_PREFIX_CACHE_RETENTION");
            c.sources.erase("--prefix-cache"); c.sources.erase("--prefix-cache-retention");
            c.options.prefix_cache=true; c.options.prefix_cache_retention="fair";
            c.options.prefix_cache_reason="not-serving";
            return effective_config_json(c);
        };
        for (const auto& suffix : std::vector<std::vector<std::string>>{{},{"--prefix-cache","off"},{"--prefix-cache-retention","lru"}}) {
            std::vector<std::string> args{"decoder","MODEL"}; args.insert(args.end(),32768,"1");
            args.insert(args.end(),{"--generate","32"}); args.insert(args.end(),suffix.begin(),suffix.end());
            auto r=request; r.options=parse(args);
            const auto got=resolve(r,memory);
            expect(got.options.route_owned==base.options.route_owned && numerical(got)==numerical(base) && plan_ids(got.options)==base_ids,
                "operational prefix options preserve full numerical effective config and plan identities");
        }
        auto r=request; r.environment["STRIX_PREFIX_CACHE"]="0";
        const auto got=resolve(r,memory);
        expect(got.options.route_owned==base.options.route_owned && numerical(got)==numerical(base) && plan_ids(got.options)==base_ids,"environment off numerical identity");
    }
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
    expect(json.starts_with("{\"rotary\":{\"rope_mode\":\"native\",\"factor\":0,"),
        "effective-config inspection golden deliberately includes native rotary status; numerical identity unchanged");
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
        if(route==3) {
            expect(!absent.environment.contains("STRIX_DISPATCH_FUSE") &&
                !absent.environment.contains("STRIX_WINDOW_INJECT_NORM") &&
                !absent.environment.contains("STRIX_WINDOW_COMMIT_COMBINE"),
                "default HTTP route does not inject window-only folds");
            const auto policy=resolve_mtp_selection_route(absent.options.http_selection_mtp,true,true);
            expect(policy.mtp && !policy.fallback,"default HTTP policy uses an admitted serial-equivalent owner");
        }
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
    auto small = memory; small.available = 16U*gib_bytes;
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
    {
        Options route;
        route.http_serve=true;
        route.mtp.enabled=true; route.mtp.draft_chain=true;
        route.known_prefix_window=true; route.known_prefix_full_rows=true;
        route.known_prefix_prefill_rows=1024; route.generate_tokens=32;
        Environment env{{"HOME",std::filesystem::temp_directory_path().string()},{"STRIX_SESSION_DISK","0"}};
        for (const bool mtp : {false,true}) for (const unsigned slots : {0U,1U,4U}) {
            auto production=route; production.mtp.enabled=mtp; production.http_batch=slots;
            resolve_session_defaults(production,model,env,128*gib_bytes);
            EffectiveConfig c; c.options=production;
            expect(!production.session_management_requested && !production.session_cache &&
                effective_config_notice(c)=="session cache: off (route not yet GPU-qualified: " +
                    std::string(session_serving_route(production)) + ")\nprefix cache: off (not-serving)\n", "unqualified serving routes fail closed");
        }
        expect(!session_route_gpu_qualified("cli-target") && !session_route_gpu_qualified("unknown"), "serving-only qualification table");
        for (const auto& entry : session_route_qualifications)
            expect(!entry.gpu_qualified, "no serving route qualified today");
        auto explicit_cli=parse({"decoder","MODEL","1","--session-cache","on","--session-load","checkpoint"});
        resolve_session_defaults(explicit_cli,model,env,128*gib_bytes);
        EffectiveConfig cli_config; cli_config.options=explicit_cli;
        const auto cli_json=serve::parse_json(effective_config_json(cli_config));
        expect(!explicit_cli.session_cache && explicit_cli.session_load=="checkpoint" &&
            explicit_cli.session_off_reason=="cli-save-load-explicit" &&
            effective_config_notice(cli_config)=="session cache: off (cli-save-load-explicit)\n" &&
            cli_json.find("session_cache")->find("policy")->as_string()=="qualified-serving-routes-only" &&
            cli_json.find("session_cache")->find("cli_operations")->as_string()=="explicit-save-load",
            "CLI cache switch does not enable automatic save/load");
        route.supplied_options.insert("--session-mtp-fixed");
        route.supplied_options.insert("--session-tiled-suffix");
        auto live=route; live.session_live_control=true;
        resolve_session_defaults(live,model,env,128*gib_bytes);
        expect(live.session_management_requested,"explicit live control qualification remains accessible");
        for (const auto* flag : {"--session-cache", "--session-disk", "--session-dir", "--session-ram-bytes",
                "--session-disk-bytes", "--session-max-parked", "--session-capture-host-bytes"}) {
            auto parsed=parse({"decoder","MODEL","1",flag,
                std::string_view(flag)=="--session-cache" || std::string_view(flag)=="--session-disk" ? "off" : "42"});
            expect(!parsed.route_owned && !parsed.session_options_requested,"storage policy cannot change numerical route");
            auto incumbent=parse({"decoder","MODEL","1"});
            for (auto* candidate : {&parsed,&incumbent}) {
                candidate->tokens.resize(32768,1); candidate->known_prefix_window=true;
                candidate->known_prefix_full_rows=true; candidate->known_prefix_prefill_explicit=true;
                candidate->known_prefix_prefill_rows=8192;
                apply_default_known_prefill_tails(*candidate,"8");
            }
            expect(parsed.known_prefix_tail_rows==incumbent.known_prefix_tail_rows &&
                parsed.known_prefix_ladder_tail_rows==incumbent.known_prefix_ladder_tail_rows,
                "storage policy cannot affect tail pruning");
        }
        for (const unsigned slots : {1U,4U}) for (const bool enabled : {false,true}) {
            auto ordinary=route; ordinary.http_batch=slots; ordinary.session_management_requested=enabled;
            ordinary.http_sampling="sample";
            expect(ordinary_http_batch_slots(ordinary)==slots && ordinary.http_sampling=="sample",
                "unnamed scheduling and sampling survive session admission including batch 1");
        }
        auto on=route; resolve_session_defaults(on,model,env,128*gib_bytes);
        expect(on.session_cache && on.session_management_requested && !on.session_disk &&
            on.session_ram_bytes==8*gib_bytes && on.session_max_parked==8 &&
            on.session_dir==(std::filesystem::temp_directory_path()/".cache/strix-engine/sessions").string(),
            "serving session defaults and RAM-only switch");
        auto cli=parse({"decoder","MODEL","1","--session-cache","off","--session-disk","off","--session-disk-bytes","42"});
        expect(!cli.session_cache && !cli.session_disk && cli.session_disk_bytes==42,"session CLI switches parse");
        auto disabled=route; auto env_off=env; env_off["STRIX_SESSION_CACHE"]="0";
        resolve_session_defaults(disabled,model,env_off,128*gib_bytes);
        expect(!disabled.session_cache && !disabled.session_management_requested && disabled.session_off_reason=="disabled","environment cache off");
        for (const bool baseline_cli : {false,true}) {
            auto base_route=route; auto base_env=env;
            if (baseline_cli) base_route.baseline=true; else base_env["STRIX_BASELINE"]="1";
            resolve_session_defaults(base_route,model,base_env,128*gib_bytes);
            expect(!base_route.session_cache && base_route.session_off_reason=="baseline","baseline disables sessions");
        }
        auto disk_cli=parse({"decoder","MODEL","1","--session-disk","off"});
        auto disk_cli_route=route; disk_cli_route.session_disk=disk_cli.session_disk;
        disk_cli_route.supplied_options.insert(disk_cli.supplied_options.begin(),disk_cli.supplied_options.end());
        auto disk_cli_env=env; disk_cli_env["STRIX_SESSION_DISK"]="1";
        resolve_session_defaults(disk_cli_route,model,disk_cli_env,128*gib_bytes);
        expect(disk_cli_route.session_cache && !disk_cli_route.session_disk,"CLI disk off overrides env on");
        auto cli_off_route=route; cli_off_route.session_cache=cli.session_cache;
        cli_off_route.supplied_options=cli.supplied_options;
        resolve_session_defaults(cli_off_route,model,env,128*gib_bytes);
        expect(!cli_off_route.session_management_requested,"CLI cache off disables serving");
        auto resolver_off=request; resolver_off.options.http_serve=true;
        resolver_off.environment["STRIX_SESSION_CACHE"]="0";
        const auto resolved_off=resolve(resolver_off,memory);
        const auto off_json=serve::parse_json(effective_config_json(resolved_off));
        expect(!off_json.find("session_cache")->find("enabled")->as_bool() &&
            off_json.find("session_cache")->find("reason")->as_string()=="disabled", "full resolver JSON environment off");
        auto legacy_prefix=route; legacy_prefix.prefix_cache_host_bytes=1U<<20;
        resolve_session_defaults(legacy_prefix,model,env,128*gib_bytes);
        expect(!legacy_prefix.session_cache,"existing prefix cache retains unnamed request behavior");
        auto small_route=route; resolve_session_defaults(small_route,model,env,1);
        expect(!small_route.session_cache && small_route.session_off_reason=="insufficient-memory","session memory guard");
        auto unknown=route; resolve_session_defaults(unknown,model,env,{});
        expect(!unknown.session_cache && unknown.session_off_reason=="memory-observation-unavailable","unknown session memory declines");
        auto unqualified=route; unqualified.mtp.draft_chain=false;
        resolve_session_defaults(unqualified,model,env,128*gib_bytes);
        expect(!unqualified.session_cache && unqualified.session_off_reason=="route-not-qualified","unsupported HTTP route declines");
        auto batch=route; batch.http_batch=4; resolve_session_defaults(batch,model,env,128*gib_bytes);
        expect(!batch.session_management_requested && batch.http_batch==4 && batch.session_off_reason=="http-batch-requires-ordinary-scheduler","batch concurrency retained");
        auto disk=route; auto disk_env=env; disk_env.erase("STRIX_SESSION_DISK");
        disk.session_dir=(std::filesystem::temp_directory_path()/"qwen-session-defaults-test").string();
        resolve_session_defaults(disk,model,disk_env,128*gib_bytes);
        expect(disk.session_cache && disk.session_disk && disk.session_disk_bytes>0,"disk default cap and writable directory");
        auto bad_disk=route; bad_disk.session_dir=__FILE__;
        resolve_session_defaults(bad_disk,model,disk_env,128*gib_bytes);
        expect(bad_disk.session_cache && !bad_disk.session_disk && !bad_disk.session_disk_off_reason.empty(),"disk failure preserves RAM");
        EffectiveConfig config; config.options=on;
        const auto session_json=serve::parse_json(effective_config_json(config));
        const auto* fields=session_json.find("session_cache");
        expect(fields && fields->find("enabled")->as_bool() && fields->find("ram_bytes") &&
            fields->find("disk_bytes") && fields->find("dir") && fields->find("reason"),"effective JSON session fields");
        config.options=batch;
        expect(effective_config_notice(config)=="session cache: off (http-batch-requires-ordinary-scheduler)\nprefix cache: off (not-serving)\n","one decline notice per store");
    }
    {
        Options route; route.http_serve=true; route.known_prefix_window=true; route.known_prefix_full_rows=true;
        route.known_prefix_prefill_rows=1024; route.generate_tokens=10; route.context_tokens=8192;
        const auto resolved=[&](Options o, Environment e={}, std::optional<std::uint64_t> a=128*gib_bytes) {
            constexpr SessionRouteQualification test_only[]={{"http-target",true}};
            resolve_prefix_defaults(o,model,e,a,test_only); return o;
        };
        for (const auto& entry : prefix_auto_route_qualifications) {
            expect(!entry.gpu_qualified,"automatic prefix routes remain unqualified");
            auto production=route; production.mtp.enabled=entry.route!="http-target";
            production.http_batch=entry.route=="http-mtp-batch" ? 2 : 0;
            resolve_prefix_defaults(production,model,{},128*gib_bytes);
            expect(production.prefix_cache_host_bytes==0 && production.prefix_cache_source=="off" &&
                production.prefix_cache_reason=="automatic budget not yet GPU-qualified: " + std::string(entry.route),"production auto budget gated");
            EffectiveConfig prefix_notice; prefix_notice.options=production;
            expect(effective_config_notice(prefix_notice).find("prefix cache: off ("+production.prefix_cache_reason+")\n")!=std::string::npos,"qualified route notice");
        }
        const auto auto_prefix=resolved(route);
        expect(auto_prefix.prefix_cache_host_bytes==4*gib_bytes && auto_prefix.prefix_cache_source=="auto","prefix fitted default on");
        expect(resolved(route,{},16*gib_bytes).prefix_cache_host_bytes==gib_bytes,"prefix MemAvailable fraction");
        expect(resolved(route,{},{}).prefix_cache_reason=="memory-observation-unavailable","prefix unknown observation");
        expect(resolved(route,{},1).prefix_cache_reason=="checkpoint-does-not-fit","prefix insufficient checkpoint fit");
        auto wrong=route; wrong.known_prefix_full_rows=false;
        expect(resolved(wrong).prefix_cache_reason=="route-not-accepted","prefix existing acceptance predicate");
        auto nonhttp=route; nonhttp.http_serve=false;
        expect(resolved(nonhttp).prefix_cache_host_bytes==0,"prefix non-HTTP unchanged");
        auto sessions=route; sessions.session_management_requested=true; sessions.session_cache=true;
        auto kept=resolved(sessions);
        expect(kept.session_management_requested && kept.session_cache && kept.prefix_cache_host_bytes==0 &&
            kept.prefix_cache_reason=="sessions-admitted","automatic prefix never disables sessions");
        auto explicit_budget=route; explicit_budget.prefix_cache_host_bytes=12345;
        explicit_budget.supplied_options.insert("--prefix-cache-host-bytes");
        expect(resolved(explicit_budget,{},1).prefix_cache_host_bytes==12345,"explicit budget wins fitting");
        expect(resolved(explicit_budget,{{"STRIX_PREFIX_CACHE","0"}}).prefix_cache_host_bytes==0,"prefix env off");
        auto cli=route; cli.supplied_options.insert("--prefix-cache"); cli.prefix_cache=true;
        cli.supplied_options.insert("--prefix-cache-retention"); cli.prefix_cache_retention="lru";
        auto precedence=resolved(cli,{{"STRIX_PREFIX_CACHE","0"},{"STRIX_PREFIX_CACHE_RETENTION","fair"}});
        expect(precedence.prefix_cache_host_bytes==4*gib_bytes && precedence.prefix_cache_retention=="lru","prefix CLI wins env");
        auto prefix_baseline=resolved(cli,{{"STRIX_BASELINE","1"}});
        EffectiveConfig baseline_notice; baseline_notice.options=prefix_baseline;
        expect(effective_config_json(baseline_notice).find("\"prefix_cache\"")==std::string::npos,"environment baseline receipt omits prefix object");
        expect(effective_config_notice(baseline_notice).find("prefix cache:")==std::string::npos,"baseline has no new prefix stderr line");
        expect(!prefix_baseline.prefix_cache_host_bytes && prefix_baseline.prefix_cache_retention=="lru" && prefix_baseline.prefix_cache_reason=="baseline","baseline wins prefix policy");
        auto staging_cli=parse({"decoder","MODEL","1","--prefix-cache-staging-bytes","123456"});
        expect(staging_cli.prefix_cache_staging_bytes==123456 && operational_session_option("--prefix-cache-staging-bytes"),
            "staging cap is an operational CLI option");
        auto staging_env=resolved(route,{{"STRIX_PREFIX_CACHE_STAGING_BYTES","234567"}});
        expect(staging_env.prefix_cache_staging_bytes==234567,"staging environment cap");
        auto staging_input=route;staging_input.prefix_cache_staging_bytes=345678;
        staging_input.supplied_options.insert("--prefix-cache-staging-bytes");
        expect(resolved(staging_input,{{"STRIX_PREFIX_CACHE_STAGING_BYTES","234567"}}).prefix_cache_staging_bytes==345678,
            "staging CLI wins environment");
        auto parsed=parse({"decoder","MODEL","1","--prefix-cache","off","--prefix-cache-retention","lru"});
        expect(!parsed.prefix_cache && parsed.prefix_cache_retention=="lru","prefix switches parse");
        auto baseline_explicit=explicit_budget; baseline_explicit.baseline=true;
        baseline_explicit.prefix_cache_dir="explicit-dir"; baseline_explicit.prefix_cache_every=4096;
        baseline_explicit.prefix_cache_every_explicit=true;
        baseline_explicit.prefix_cache_then_prompts={"prompt-file"};
        for (const auto& e : {Environment{}, Environment{{"STRIX_BASELINE","1"}}}) {
            auto input=baseline_explicit; input.baseline=e.empty();
            const auto got=resolved(input,e);
            expect(got.prefix_cache_host_bytes==12345 && got.prefix_cache_dir=="explicit-dir" &&
                got.prefix_cache_every==4096 && got.prefix_cache_every_explicit &&
                got.prefix_cache_then_prompts==input.prefix_cache_then_prompts && got.prefix_cache_retention=="lru",
                "baseline preserves every explicit prefix option");
        }
        auto decline=route; decline.http_serve=false; decline.prefix_cache_dir="kept";
        decline.prefix_cache_every_explicit=true; decline.prefix_cache_every=4096;
        decline.prefix_cache_then_prompts={"kept-prompt"};
        auto declined_prefix=resolved(decline);
        expect(declined_prefix.prefix_cache_dir=="kept" && declined_prefix.prefix_cache_every_explicit &&
            declined_prefix.prefix_cache_every==4096 && declined_prefix.prefix_cache_then_prompts==decline.prefix_cache_then_prompts,
            "automatic decline preserves explicit options");
        for (const auto& e : {Environment{{"STRIX_PREFIX_CACHE","invalid"}},
                Environment{{"STRIX_PREFIX_CACHE_RETENTION","invalid"}}}) {
            bool threw=false;
            try { (void)resolved(route,e); } catch (const std::invalid_argument&) { threw=true; }
            expect(threw,"strict shared environment parser rejects invalid policy");
        }
        Options standalone;
        apply_prefix_cache_environment(standalone, [](const char* name) -> std::optional<std::string_view> {
            if (std::string_view(name)=="STRIX_BASELINE") return "1";
            if (std::string_view(name)=="STRIX_PREFIX_CACHE") return "on";
            return std::nullopt;
        });
        expect(standalone.baseline && standalone.prefix_cache_retention=="lru", "non-resolver policy honours baseline env");
        const auto baseline_cli=parse({"decoder","MODEL","1","--baseline","--generate","4",
            "--known-prefix-window","--known-prefix-full-rows","--known-prefix-prefill-rows","1024",
            "--known-prefix-workspace-budget-bytes","1048576","--known-prefix-host-metadata-budget-bytes","1048576",
            "--known-prefix-uma-budget-bytes","1048576","--known-prefix-host-headroom-bytes","0",
            "--deferred-moe-status","--virtual-q8-gdn-input","--prefetch-q8-gdn-input",
            "--virtual-q8-gdn-output","--wave8-q8-gdn-input","--wave8-q8-gdn-output",
            "--prefix-cache-host-bytes","12345","--prefix-cache-dir","explicit-dir",
            "--prefix-cache-every","4096","--prefix-cache-then-prompt","next-prompt"});
        expect(baseline_cli.prefix_cache_host_bytes==12345 && baseline_cli.prefix_cache_dir=="explicit-dir" &&
            baseline_cli.prefix_cache_every_explicit && baseline_cli.prefix_cache_every==4096 &&
            baseline_cli.prefix_cache_then_prompts==std::vector<std::string>{"next-prompt"} &&
            baseline_cli.prefix_cache_retention=="lru", "CLI baseline preserves explicit prefix options before resolver");
        auto off_cli=parse({"decoder","MODEL","1","--prefix-cache","off",
            "--prefix-cache-host-bytes","12345","--prefix-cache-dir","explicit-dir",
            "--prefix-cache-every","4096","--prefix-cache-then-prompt","next-prompt"});
        expect(!prefix_state_cache_requested(off_cli),"explicit off clears explicit prefix settings before validation");
        EffectiveConfig disabled; disabled.options=off_cli;
        expect(effective_config_notice(disabled).find("prefix cache: off (disabled)\n")!=std::string::npos,
            "explicit off prints the decline line outside HTTP too");
        EffectiveConfig c; c.options=auto_prefix;
        const auto j=serve::parse_json(effective_config_json(c));
        expect(j.find("prefix_cache")->find("enabled")->as_bool() &&
            j.find("prefix_cache")->find("source")->as_string()=="auto","effective prefix config JSON");
    }
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
    {
        // Serial decode pool route: plain append-only CLI only; every multi-request or rewriting route is excluded.
        const auto route = [&](std::vector<std::string> extra) {
            std::vector<std::string> args{"decoder","target.gguf","1","2","3","--generate","4"};
            args.insert(args.end(), extra.begin(), extra.end());
            return qsa_decode_pool_route(parse(args), false);
        };
        expect(route({}), "decode pool on for the plain CLI route");
        for(const auto& flags : std::vector<std::vector<std::string>>{
                {"--session-save","checkpoint"},{"--session-load","checkpoint"},
                {"--session-live-control"},{"--session-save","checkpoint","--session-then-prompt","4,5"}})
            expect(route(flags),"decode pool stays on for save/restore/park/continuation");
        expect(!qsa_decode_pool_route(parse({"decoder","target.gguf","1","2","3","--generate","4"}), true), "decode pool off for GSQ models");
        auto cached = parse({"decoder","target.gguf","1","2","3","--generate","4"});
        cached.prefix_cache_host_bytes = UINT64_C(1) << 30U;
        expect(!qsa_decode_pool_route(cached, false), "decode pool off with the prefix state cache");
        cached.prefix_cache_host_bytes = 0U;
        cached.prefix_cache_then_prompts = {"4,5,6"};
        expect(!qsa_decode_pool_route(cached, false), "decode pool off for follow-up prefix-cache prompts");
        expect(!route({"--baseline"}), "decode pool off with --baseline");
        ::setenv("STRIX_QSA_DECODE_POOL", "0", 1);
        expect(!route({}), "STRIX_QSA_DECODE_POOL=0 disables the decode pool");
        ::unsetenv("STRIX_QSA_DECODE_POOL");
        ::setenv("STRIX_QPF2K_DEFAULTS", "0", 1);
        expect(!route({}), "STRIX_QPF2K_DEFAULTS=0 disables the decode pool");
        ::unsetenv("STRIX_QPF2K_DEFAULTS");
        auto sized = parse({"decoder","target.gguf","1","2","3","--generate","4"});
        const auto check_bytes = [&](const Options& options, bool gsq, std::uint64_t key_bytes) {
            for (const auto context : {262144U, 524288U, 1048576U}) {
                auto o = options; o.context_tokens = context;
                const auto blocks = context / 4U;
                const auto expected = UINT64_C(12) * blocks * 128U * key_bytes + (blocks / 4U) * 4U;
                expect(qsa_decode_pool_key_bytes(o, gsq) == key_bytes, "decode pool key width follows the shared policy");
                expect(qsa_decode_pool_bytes(o, model, gsq) == expected, "decode pool exact bytes at 262144/524288/1048576");
            }
        };
        expect(qsa_decode_pool_bf16(sized, false), "BF16 decode pool default on");
        check_bytes(sized, false, 2U);
        // Literal totals also guard the model geometry and maxima row independently.
        for (const auto& [context, bytes] : std::array<std::pair<std::uint64_t,std::uint64_t>,3>{{
                {262144U,201392128U},{524288U,402784256U},{1048576U,805568512U}}}) {
            sized.context_tokens = context;
            expect(qsa_decode_pool_bytes(sized, model, false) == bytes, "BF16 decode pool literal total bytes");
        }
        auto baseline_pool = sized; baseline_pool.baseline = true;
        check_bytes(baseline_pool, false, 4U);
        check_bytes(sized, true, 4U);
        qpf2k::baseline = true;
        check_bytes(sized, false, 4U);
        qpf2k::baseline = false;
        for (const char* name : {"STRIX_QPF2K_DEFAULTS", "STRIX_QSA_DECODE_POOL_BF16", "STRIX_QSA_DECODE_POOL"}) {
            ::setenv(name, "0", 1);
            check_bytes(sized, false, 4U);
            ::unsetenv(name);
        }
        ::setenv("STRIX_QSA_DECODE_POOL_BF16", "1", 1);
        expect(requested_prefill_environment().at("STRIX_QSA_DECODE_POOL_BF16") == "1",
            "requested environment includes the decode pool format switch");
        check_bytes(sized, false, 2U);
        check_bytes(baseline_pool, false, 4U);
        check_bytes(sized, true, 4U);
        ::unsetenv("STRIX_QSA_DECODE_POOL_BF16");
        // --print-effective-config resolves each long-context item and names why it is off.
        const auto items = [&](const Options& options, bool gsq) {
            EffectiveConfig printed; printed.options = options; printed.gsq_owner = gsq;
            return effective_config_json(printed);
        };
        const auto has = [](const std::string& text, const char* field) { return text.find(field) != std::string::npos; };
        auto plain = items(sized, false);
        expect(has(plain, "\"STRIX_QSA_UNION_MERGE\":{\"enabled\":true,\"suppression\":null}") &&
            has(plain, "\"STRIX_QSA_SCORE_ROWS64\":{\"enabled\":true,\"suppression\":null}") &&
            has(plain, "\"STRIX_QSA_DECODE_POOL\":{\"enabled\":true,\"suppression\":null}") &&
            has(plain, "\"STRIX_QSA_DECODE_SELECT\":{\"enabled\":true,\"suppression\":null}") &&
            has(plain, "\"STRIX_QSA_DECODE_POOL_BF16\":{\"enabled\":true,\"suppression\":null}") &&
            has(plain, "\"qsa_decode_pool_key_bytes\":2}"), "effective config resolves the long-context items as on");
        ::setenv("STRIX_QSA_DECODE_SELECT", "0", 1);
        ::setenv("STRIX_QSA_DECODE_POOL_BF16", "0", 1);
        plain = items(sized, false);
        expect(has(plain, "\"STRIX_QSA_DECODE_SELECT\":{\"enabled\":false,\"suppression\":\"explicit-off\"}") &&
            has(plain, "\"STRIX_QSA_DECODE_POOL_BF16\":{\"enabled\":false,\"suppression\":\"explicit-off\"}") &&
            has(plain, "\"STRIX_QSA_DECODE_POOL\":{\"enabled\":true,\"suppression\":null}") &&
            has(plain, "\"qsa_decode_pool_key_bytes\":4}") &&
            has(requested_prefill_environment().at("STRIX_QSA_DECODE_SELECT"), "0"),
            "explicit item switches resolve off while the request stays in the environment");
        ::unsetenv("STRIX_QSA_DECODE_SELECT");
        ::unsetenv("STRIX_QSA_DECODE_POOL_BF16");
        ::setenv("STRIX_QPF2K_DEFAULTS", "0", 1);
        expect(has(items(sized, false), "\"STRIX_QSA_UNION_MERGE\":{\"enabled\":false,\"suppression\":\"STRIX_QPF2K_DEFAULTS=0\"}"),
            "STRIX_QPF2K_DEFAULTS=0 is the reported suppression");
        ::unsetenv("STRIX_QPF2K_DEFAULTS");
        const auto gsq_text = items(sized, true);
        expect(has(gsq_text, "\"STRIX_QSA_SCORE_ROWS64\":{\"enabled\":false,\"suppression\":\"gsq\"}") &&
            has(gsq_text, "\"STRIX_QSA_DECODE_SELECT\":{\"enabled\":false,\"suppression\":\"gsq\"}") &&
            has(gsq_text, "\"qsa_decode_pool_key_bytes\":0}"), "GSQ owners report the gsq suppression");
        qpf2k::baseline = true;
        expect(has(items(sized, false), "\"STRIX_QSA_DECODE_POOL\":{\"enabled\":false,\"suppression\":\"baseline\"}"),
            "--baseline is the reported suppression");
        qpf2k::baseline = false;
        auto mtp = sized; mtp.mtp.enabled = true;
        const auto mtp_text = items(mtp, false);
        expect(has(mtp_text, "\"STRIX_QSA_DECODE_POOL\":{\"enabled\":false,\"suppression\":\"mtp\"}") &&
            has(mtp_text, "\"STRIX_QSA_DECODE_SELECT\":{\"enabled\":false,\"suppression\":\"mtp\"}") &&
            has(mtp_text, "\"STRIX_QSA_UNION_MERGE\":{\"enabled\":true,\"suppression\":null}"),
            "an excluded decode route names the route; prefill items stay on");
        expect(std::string(qsa_decode_pool_route_suppression(cached, false)) == "prefix-state-cache",
            "the prefix state cache is the reported route exclusion");
    }
    for(int master=0;master<4;++master) {
        Options modes; modes.session_tiled_suffix=true; modes.session_mtp_fixed=true;
        modes.supplied_options.insert("--session-tiled-suffix"); modes.supplied_options.insert("--session-mtp-fixed");
        Environment env;
        if(master==1) modes.baseline=true;
        if(master==2) env["STRIX_BASELINE"]="1";
        if(master==3) env["STRIX_QPF2K_DEFAULTS"]="0";
        resolve_session_defaults(modes,model,env,UINT64_C(8)<<30);
        expect(modes.session_tiled_suffix_requested && modes.session_mtp_fixed_requested,"session mode requests retained");
        expect(modes.session_tiled_suffix==(master==0) && modes.session_mtp_fixed==(master==0),"masters suppress both numerical modes");
        auto modes_json=session_modes_json(modes);
        for(const char* name:{"tiled_suffix","mtp_fixed"}) {
            expect(modes_json.find(name)->find("requested")->as_bool(),"mode requested disclosed");
            expect(modes_json.find(name)->find("effective")->as_bool()==(master==0),"mode effective disclosed");
            expect(modes_json.find(name)->find("bit_changing")->as_bool(),"mode bit-changing disclosed");
            expect(modes_json.find(name)->find("suppression")->as_string().empty()==(master==0),"suppression disclosed");
        }
    }
    EffectiveConfig receipt; receipt.options.baseline=true;
    expect(effective_config_json(receipt).find("\"prefix_cache\"")==std::string::npos,"baseline receipt omits additive prefix object");
    receipt.options.baseline=false;
    expect(effective_config_json(receipt).find("\"prefix_cache\"")!=std::string::npos,"nonbaseline receipt includes additive prefix object");
    std::cout << "qwen effective config: " << failures << " failures\n";
    return failures ? 1 : 0;
}
