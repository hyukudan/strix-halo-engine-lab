#include "strix/hip_known_prefix_window.hpp"

#include "strix/deferred_moe_status.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/prepared_tensor_views.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strix {
namespace {
void check(hipError_t error, const char* operation) {
    if (error != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(error));
}
bool empty(const ExecutionMemoryBinding& binding) {
    return !binding.execution_identity && !binding.arena_base && !binding.arena_bytes &&
        !binding.allocation_generation;
}
bool same_layout(const SerialSessionStateLayout& a, const SerialSessionStateLayout& b) {
    if (a.layout_identity != b.layout_identity || a.context_tokens != b.context_tokens ||
        a.device_alignment != b.device_alignment || a.host_alignment != b.host_alignment ||
        a.device_total_bytes != b.device_total_bytes || a.host_total_bytes != b.host_total_bytes ||
        a.recurrent_storage != b.recurrent_storage ||
        a.allocations.size() != b.allocations.size()) return false;
    for (std::size_t i = 0; i < a.allocations.size(); ++i) {
        const auto& x = a.allocations[i]; const auto& y = b.allocations[i];
        if (x.component != y.component || x.state_flag != y.state_flag || x.storage != y.storage ||
            x.offset != y.offset || x.bytes != y.bytes || x.provisional != y.provisional) return false;
    }
    return true;
}
std::uint32_t u32(std::uint64_t value) {
    if (!value || value > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("known-window launcher shape exceeds uint32");
    return static_cast<std::uint32_t>(value);
}
ExecutionBindingRequest binding_request(const HipLauncherCatalogConfig& config) {
    ExecutionBindingRequest request;
    request.minimum_qualification = kernels::QualificationStatus::experimental;
    request.device_graph_inputs = config.enable_device_graph_inputs;
    request.prefetch_q8_lm_head = config.prefetch_q8_lm_head;
    request.wave8_q8_lm_head = config.wave8_q8_lm_head;
    request.wave8_q8_gdn_input = config.wave8_q8_gdn_input;
    request.wave8_q8_gdn_output = config.wave8_q8_gdn_output;
    request.wave8_q8_qsa_output = config.wave8_q8_qsa_output;
    request.parallel_moe_router = config.parallel_moe_router;
    request.ranked_moe_router_topk = config.ranked_moe_router_topk;
    request.wave_moe_router_topk = config.wave_moe_router_topk;
    request.prefetch_q8_qsa_input = config.prefetch_q8_qsa_input;
    request.qsa_input_prefetch_layers = config.qsa_input_prefetch_layers;
    request.q4k_a8g32_gate_up = config.q4k_a8g32_gate_up;
    request.batched_moe_activation = config.batched_moe_activation;
    request.grouped_exact_q51_down = config.grouped_exact_q51_down;
    request.q51_slot_parallel_down = config.q51_slot_parallel_down;
    request.lane_chain_hyper_output = config.lane_chain_hyper_output;
    request.prefetch_hc_projection_next_tile = config.prefetch_hc_projection_next_tile;
    request.deferred_moe_status = config.deferred_moe_status;
    request.prefetch_q8_shared_expert = config.prefetch_q8_shared_expert;
    request.shared_expert_prefetch_layers = config.shared_expert_prefetch_layers;
    return request;
}
} // namespace

struct HipKnownPrefixWindow::Impl {
    // Member order keeps all retained owners alive through destructor recovery.
    std::shared_ptr<const void> actual_owner;
    HipSetupQuiescenceGuard guard;
    CompiledKnownPrefixWindow window;
    std::optional<CompiledKnownPrefixPrefill> prefill;
    std::optional<CompiledSerialExecution> prefill_launcher_source;
    CompiledSerialExecution serial;
    ModelContract model;
    HipLauncherCatalogConfig config;
    ExecutionRuntimeResources resources;
    ExecutionMemoryBinding scratch;
    HipLaunchContext scope;
    RuntimeExecutionIdentities original_identities;
    SerialSessionStateLayout original_state_layout;
    std::unique_ptr<const PreparedTensorViews> metadata;
    std::vector<ExecutionRuntimeResources> row_resources;
    std::vector<std::unique_ptr<HipLauncherCatalog>> catalogs;
    std::vector<ExecutionLauncherRegistry> registries;
    std::vector<CompiledLaunchPlan> launch_plans;
    std::vector<const CompiledLaunchPlan*> launch_pointers;
    std::vector<std::uint64_t> position_readers;
    HipSharedExpertLauncherConfig shared_config, shared_fallback;
    HipGdnLauncherConfig gdn_config;
    HipMoeExpertsLauncherConfig routed_config;
    q51_known_rows::Workspace routed_workspace;
    q51_known_rows::Summary pending_union_summary;
    std::uint32_t last_union_layer = 0;
    std::optional<HipKnownPrefixUnionObservation> last_union_observation;
    std::array<KnownWindowR2Entry, 3> r2_entries;
    std::vector<KnownWindowRowsEntry> rows_entries;
    DeferredMoeFirstError completion_status{};
    bool failed = false;

    Impl(CompiledKnownPrefixWindow source, CompiledSerialExecution serial_source,
        ModelContract model_source, HipLauncherCatalogConfig config_source,
        const ExecutionRuntimeResources& serial_resources, ExecutionMemoryBinding memory,
        HipLaunchContext native_scope, std::shared_ptr<const void> owner,
        q51_known_rows::Workspace union_workspace,std::optional<CompiledKnownPrefixPrefill> tiled = std::nullopt)
        : actual_owner(std::move(owner)), window(std::move(source)), serial(std::move(serial_source)),
          model(std::move(model_source)), config(std::move(config_source)), resources(serial_resources),
          scratch(memory), scope(native_scope), routed_workspace(std::move(union_workspace)) {
        prefill=std::move(tiled);
        if (!actual_owner) throw std::invalid_argument("known-window native owner requires retained actual owners");
        if (const auto error = model.validate(); !error.empty()) throw std::invalid_argument(error);
        if (config.gdn_recurrent_storage != GdnRecurrentStorage::key_value ||
            (resources.session_state_layout && resources.session_state_layout->recurrent_storage != GdnRecurrentStorage::key_value))
            throw std::invalid_argument("known-window native owner requires key-value recurrent storage");
        if (serial.device_graph_inputs || config.enable_device_graph_inputs || serial.logical_rows != 1U ||
            serial.request_concurrency != 1U || config.activation != HipActivationStorage::bf16)
            throw std::invalid_argument("known-window native owner supports eager BF16 C1 plans only");
        validate_compiled_known_prefix_window(window, serial);
        if(prefill) {
            validate_compiled_known_prefix_prefill(*prefill,serial);
            if(prefill->tile8.window_execution_identity!=window.window_execution_identity||
                prefill->tile8.window_kernel_plan_identity!=window.window_kernel_plan_identity)
                throw std::invalid_argument("known prefill owner selection tile differs");
            prefill_launcher_source=serial;
            prefill_launcher_source->execution_identity=prefill->prefill_execution_identity;
            prefill_launcher_source->arena=prefill->arena.row_view[0];
            prefill_launcher_source->whole_plan_graph_safe=false;
        }
        if (!resources.model_manifest || !resources.tensor_access)
            throw std::invalid_argument("known-window native owner requires manifest and tensor access resources");
        validate_known_prefix_window_formats(window, serial, *resources.model_manifest, *resources.tensor_access);
        kernels::KernelRegistry admitted;
        register_complete_hip_kernel_catalog(model, binding_request(config), config, admitted);
        for (const auto& binding : serial.bindings.bindings)
            if (!binding.host_or_runtime_step && (!binding.kernel_id || !admitted.find(*binding.kernel_id)))
                throw std::invalid_argument("known-window serial kernel is not in the configured native catalog");
        original_identities = build_runtime_execution_identities(serial.execution_identity, serial.arena, resources);
        original_state_layout = *resources.session_state_layout;
        metadata = std::make_unique<const PreparedTensorViews>(serial.execution_identity, serial.execution,
            *resources.model_manifest, *resources.model_weights, *resources.tensor_access, actual_owner);
        validate_execution_memory_binding(execution_identity(), physical_arena(), scratch);
        validate_scope();
        const auto launcher_rows=prefill?1U:window.logical_rows;
        row_resources.resize(launcher_rows);
        catalogs.resize(launcher_rows);
        registries.resize(launcher_rows);
        launch_plans.resize(launcher_rows);
        launch_pointers.reserve(launcher_rows);
        for (std::size_t row = 0; row < launcher_rows; ++row) {
            const auto& row_source=prefill?*prefill_launcher_source:window.row_plan[row];
            row_resources[row] = resources;
            row_resources[row].scratch = &scratch;
            catalogs[row] = std::make_unique<HipLauncherCatalog>(model, config);
            catalogs[row]->register_for_compiled_execution(row_source, registries[row], &row_resources[row]);
            if (catalogs[row]->audit(row_source, registries[row]).missing_steps)
                throw std::invalid_argument("known-window native row catalog coverage is incomplete");
            launch_plans[row] = compile_launch_plan(row_source, registries[row]);
            for (const auto& launcher : launch_plans[row].steps)
                if (!launcher.preflight)
                    throw std::invalid_argument("known-window native row contains a launcher without deep preflight");
            launch_pointers.push_back(&launch_plans[row]);
            const auto readers = catalogs[row]->known_window_position_readers(row_source, registries[row]);
            position_readers.insert(position_readers.end(), readers.begin(), readers.end());
        }
        std::sort(position_readers.begin(), position_readers.end());
        position_readers.erase(std::unique(position_readers.begin(), position_readers.end()), position_readers.end());
        setup_r2();
        setup_rows();
        setup_routed_union();
        // Setup only: no execution callback clears or reads this record before
        // all deep admission. Exclusive status ownership preserves zero across
        // successful calls; any exception disables the owner permanently.
        if (config.deferred_moe_status) guard.run([&] {
            check(hipMemcpy(&completion_status, reinterpret_cast<const void*>(scope.deferred_moe_status.arena_base),
                sizeof(completion_status), hipMemcpyDeviceToHost), "read initial known-window deferred status");
            check_deferred_moe_status(completion_status);
        });
    }
    ~Impl() { guard.recover_or_fail_stop(); }
    std::uint64_t execution_identity() const noexcept {return prefill?prefill->prefill_execution_identity:window.window_execution_identity;}
    const ExecutionArenaPlan& physical_arena() const noexcept {return prefill?prefill->arena.physical:window.arena.physical;}
    std::uint32_t logical_rows() const noexcept {return prefill?prefill->logical_rows:window.logical_rows;}

    void status_disjoint(std::uintptr_t address, std::uint64_t bytes) const {
        if (!bytes) return;
        if (!address || bytes > std::numeric_limits<std::uintptr_t>::max() - address)
            throw std::invalid_argument("known-window deferred alias owner range is invalid");
        const auto& status = scope.deferred_moe_status;
        if (status.arena_base < address + bytes && address < status.arena_base + status.arena_bytes)
            throw std::invalid_argument("known-window deferred status aliases a retained resource");
    }
    void validate_scope() const {
        guard.require_original();
        if (scope.graph_capture || !scope.stream || scope.stream == hipStreamLegacy ||
            scope.stream == hipStreamPerThread || scope.arguments != GraphDynamicArguments{} ||
            !empty(scope.status) || !empty(scope.ngram_input) || !empty(scope.ngram_output) ||
            scope.context_tokens || scope.vocabulary_size)
            throw std::invalid_argument("known-window native owner requires empty graph bindings and one eager owned stream");
        hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
        check(hipStreamIsCapturing(scope.stream, &capture), "query known-window native capture");
        if (capture != hipStreamCaptureStatusNone)
            throw std::invalid_argument("known-window native stream is being captured");
        const auto& status = scope.deferred_moe_status;
        if (!config.deferred_moe_status) {
            if (!empty(status)) throw std::invalid_argument("known-window deferred configuration/scope parity differs");
            return;
        }
        if (status.execution_identity != execution_identity() || !status.arena_base ||
            status.arena_base % alignof(std::uint64_t) || status.arena_bytes != sizeof(DeferredMoeFirstError) ||
            !status.allocation_generation || status.arena_bytes > std::numeric_limits<std::uintptr_t>::max() - status.arena_base)
            throw std::invalid_argument("known-window deferred status has stale identity, generation or extent");
        status_disjoint(scratch.arena_base, scratch.arena_bytes);
        status_disjoint(resources.scratch->arena_base, resources.scratch->arena_bytes);
        status_disjoint(resources.session_state_memory->device_base, resources.session_state_memory->device_bytes);
        status_disjoint(resources.session_state_memory->host_base, resources.session_state_memory->host_bytes);
        for (const auto& tensor : resources.model_weights->tensors)
            if (tensor.storage != RuntimeTensorStorage::remote) status_disjoint(tensor.address, tensor.bytes);
    }
    void validate_metadata() const {
        guard.require_original();
        metadata->validate_source_metadata(*resources.model_manifest, *resources.model_weights, *resources.tensor_access);
        const auto current = build_runtime_execution_identities(serial.execution_identity, serial.arena, resources);
        if (current.compatibility_identity != original_identities.compatibility_identity ||
            current.binding_identity != original_identities.binding_identity ||
            !same_layout(original_state_layout, *resources.session_state_layout))
            throw std::invalid_argument("known-window retained runtime bindings or state layout changed");
        validate_scope();
    }
    const HipSharedExpertLauncherConfig& shared_for(std::span<const ExecutionDispatchContext> rows) const {
        if (!rows[0].kernel_id) throw std::invalid_argument("known-window shared row has no serial kernel id");
        return rows[0].kernel_id->ends_with(shared_expert_prefetch::kernel_suffix) ? shared_config : shared_fallback;
    }
    void setup_r2() {
        shared_config.hidden_size = u32(model.hidden_size);
        shared_config.intermediate_size = u32(model.shared_expert_intermediate_size);
        shared_config.activation = config.activation;
        shared_config.prefetch_q8 = config.prefetch_q8_shared_expert;
        shared_config.prefetch_eligible_layers = config.shared_expert_prefetch_layers;
        shared_config.prefetch_stats = config.shared_expert_prefetch_stats;
        shared_fallback = shared_config;
        shared_fallback.prefetch_q8 = false; shared_fallback.prefetch_eligible_layers.clear();
        gdn_config.model = model; gdn_config.activation = config.activation;
        gdn_config.virtual_q8_input_projections = config.virtual_q8_gdn_input;
        gdn_config.prefetch_q8_input_projections = config.prefetch_q8_gdn_input;
        gdn_config.wave8_q8_input_projections = config.wave8_q8_gdn_input;
        gdn_config.virtual_q8_output_projection = config.virtual_q8_gdn_output;
        gdn_config.wave8_q8_output_projection = config.wave8_q8_gdn_output;
        r2_entries[0] = {KnownWindowR2Variant::shared_q8_virtual256_r2_v1, this,
            [](std::span<const ExecutionDispatchContext, 2> rows, void* object) {
                const auto& self = *static_cast<Impl*>(object);
                shared_expert_known_window_r2_preflight(rows, self.shared_for(rows));
            },
            [](std::span<const ExecutionDispatchContext, 2> rows, void* object) {
                const auto& self = *static_cast<Impl*>(object);
                shared_expert_known_window_r2_launch(rows, self.shared_for(rows));
            }};
        r2_entries[1] = {KnownWindowR2Variant::gdn_q8_virtual256_r2_v1, this,
            [](std::span<const ExecutionDispatchContext, 2> rows, void* object) {
                gdn_known_window_r2_preflight(rows, static_cast<Impl*>(object)->gdn_config);
            },
            [](std::span<const ExecutionDispatchContext, 2> rows, void* object) {
                gdn_known_window_r2_launch(rows, static_cast<Impl*>(object)->gdn_config);
            }};
        r2_entries[2] = {KnownWindowR2Variant::gdn_q8_rows2_f32_gates_r1_v1, this,
            [](std::span<const ExecutionDispatchContext, 2> rows, void* object) {
                gdn_known_window_r2_f32_gates_preflight(rows, static_cast<Impl*>(object)->gdn_config);
            },
            [](std::span<const ExecutionDispatchContext, 2> rows, void* object) {
                gdn_known_window_r2_f32_gates_launch(rows, static_cast<Impl*>(object)->gdn_config);
            }};
    }
    void setup_rows() {
        rows_entries.resize(4);
        for(std::size_t i=0;i<2U;++i) {
            rows_entries[i]={i==0U ? KnownWindowR2Variant::shared_q8_virtual256_r4_v1 :
                KnownWindowR2Variant::shared_q8_virtual256_r8_v1,this,
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    const auto& self=*static_cast<Impl*>(object);
                    shared_expert_known_window_rows_preflight(rows,self.shared_for(rows));
                },
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    const auto& self=*static_cast<Impl*>(object);
                    shared_expert_known_window_rows_launch(rows,self.shared_for(rows));
                }};
            rows_entries[i+2U]={i==0U ? KnownWindowR2Variant::gdn_q8_rows4_f32_gates_r1_v1 :
                KnownWindowR2Variant::gdn_q8_rows8_f32_gates_r1_v1,this,
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    gdn_known_window_rows_preflight(rows,static_cast<Impl*>(object)->gdn_config);
                },
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    gdn_known_window_rows_launch(rows,static_cast<Impl*>(object)->gdn_config);
                }};
        }
    }
    void setup_routed_union() {
        const auto& workspace=routed_workspace.memory;
        if(!window.routed_expert_union_steps) {
            if(routed_workspace.owner || !empty(workspace))
                throw std::invalid_argument("unused known-window routed union workspace");
            return;
        }
        if(!config.q4k_a8g32_gate_up || !config.grouped_exact_q51_down ||
            !config.q51_slot_parallel_down || !config.deferred_moe_status || !routed_workspace.owner ||
            workspace.execution_identity!=execution_identity() ||
            !workspace.arena_base || workspace.arena_base%256U || !workspace.allocation_generation ||
            workspace.arena_bytes!=q51_known_rows::required_scratch_bytes(window.logical_rows,
                u32(model.num_experts),u32(model.num_experts_per_token)) ||
            workspace.arena_bytes>std::numeric_limits<std::uintptr_t>::max()-workspace.arena_base)
            throw std::invalid_argument("known-window routed union requires its exact retained extra workspace and configured serial policy");
        status_disjoint(workspace.arena_base,workspace.arena_bytes);
        routed_config.hidden_size=u32(model.hidden_size);
        routed_config.intermediate_size=u32(model.moe_intermediate_size);
        routed_config.expert_count=u32(model.num_experts);
        routed_config.top_k=u32(model.num_experts_per_token);
        routed_config.activation=config.activation;
        routed_config.q4k_a8g32_gate_up=config.q4k_a8g32_gate_up;
        routed_config.batched_slot_activation=config.batched_moe_activation;
        routed_config.grouped_exact_q51_down=config.grouped_exact_q51_down;
        routed_config.q51_slot_parallel_down=config.q51_slot_parallel_down;
        routed_config.deferred_moe_status=config.deferred_moe_status;
        routed_config.q4k_a8_stats=config.q4k_a8_stats;
        routed_config.q4k_a8_quiescence=std::make_shared<HipSetupQuiescenceGuard>();
        for(std::size_t i=0;i<window.mode.size();++i)
            if(window.mode[i]==KnownWindowStepMode::routed_expert_union)
                last_union_layer=serial.execution.steps[i].layer.value();
        const auto variant=window.logical_rows==2U?KnownWindowR2Variant::routed_q51_union_rows2_v1:
            window.logical_rows==4U?KnownWindowR2Variant::routed_q51_union_rows4_v1:KnownWindowR2Variant::routed_q51_union_rows8_v1;
        rows_entries.push_back({variant,this,
            [](std::span<const ExecutionDispatchContext> rows,void* object) {
                const auto& self=*static_cast<Impl*>(object);
                moe_experts_known_window_rows_preflight(rows,self.routed_config,self.routed_workspace);
            },
            [](std::span<const ExecutionDispatchContext> rows,void* object) {
                const auto& self=*static_cast<Impl*>(object);
                moe_experts_known_window_rows_launch(rows,self.routed_config,self.routed_workspace);
            }});
    }
    void complete() {
        guard.require_original();
        if(window.routed_expert_union_steps)
            check(hipMemcpyAsync(&pending_union_summary,
                reinterpret_cast<const void*>(routed_workspace.memory.arena_base),sizeof(pending_union_summary),
                hipMemcpyDeviceToHost,scope.stream),"read latest routed expert union metadata");
        if (config.deferred_moe_status)
            check(hipMemcpyAsync(&completion_status,
                reinterpret_cast<const void*>(scope.deferred_moe_status.arena_base), sizeof(completion_status),
                hipMemcpyDeviceToHost, scope.stream), "read completed known-window deferred status");
        check(hipStreamSynchronize(scope.stream), "complete known-window native stream");
        guard.require_original();
        if (config.deferred_moe_status) check_deferred_moe_status(completion_status);
        if(window.routed_expert_union_steps) {
            const auto& s=pending_union_summary;
            const auto assignments=window.logical_rows*u32(model.num_experts_per_token);
            std::uint64_t experts=0,batches=0,batch_assignments=0,unique_row_assignments=0;
            for(std::uint32_t i=0;i<8U;++i) {
                experts+=s.rows_per_expert_histogram[i];
                unique_row_assignments+=static_cast<std::uint64_t>(i+1U)*s.rows_per_expert_histogram[i];
                batches+=s.batch_size_histogram[i];
                batch_assignments+=static_cast<std::uint64_t>(i+1U)*s.batch_size_histogram[i];
                if(i>=window.logical_rows && (s.rows_per_expert_histogram[i] || s.batch_size_histogram[i]))
                    throw std::runtime_error("routed expert union histogram exceeds its admitted row count");
            }
            if(s.status || s.version!=1U || s.logical_rows!=window.logical_rows || s.assignments!=assignments ||
                !s.touched || s.touched>assignments || s.touched>model.num_experts ||
                s.batches<s.touched || s.batches>assignments || !s.maximum_assignments ||
                s.maximum_assignments>assignments || s.duplicate_assignments>assignments ||
                experts!=s.touched || batches!=s.batches || batch_assignments!=assignments ||
                unique_row_assignments+s.duplicate_assignments!=assignments ||
                std::any_of(s.reserved,s.reserved+8U,[](auto v){return v!=0U;}))
                throw std::runtime_error("routed expert union metadata disagrees with its completed window");
        }
    }
    KnownWindowResult execute(SerialSessionContext& session, std::span<const std::uint32_t> tokens,
        const std::function<void()>& before_publish = {}) {
        if (failed) throw std::logic_error("known-window native owner cannot be reused after an exception");
        bool owned_eager_scope_admitted=false;
        try {
            if(tokens.size()!=logical_rows())
                throw std::invalid_argument("known-window native input row count differs from its compiled window");
            if (current_hip_launch_context())
                throw std::invalid_argument("known-window native owner cannot nest a HIP launch scope");
            validate_metadata();
            owned_eager_scope_admitted=true;
            const HipLaunchScope active(scope);
            KnownWindowExecutionHooks hooks;
            hooks.preflight_scope = [&] { validate_scope(); };
            hooks.preflight_plan = [&] { validate_metadata(); };
            hooks.complete = [&] { complete(); if(before_publish) before_publish(); };
            hooks.quiesce_or_fail_stop = [&] { failed = true; guard.recover_or_fail_stop(); };
            hooks.position_reader_implementations = position_readers;
            const auto first_position=session.next_position;
            auto result=prefill ? execute_known_prefix_prefill_eager(*prefill,serial,resources,scratch,session,tokens,
                launch_plans.front(),r2_entries,rows_entries,hooks) :
                execute_known_prefix_window_eager(window, serial, resources, scratch, session, tokens,
                    launch_pointers, r2_entries, rows_entries, hooks);
            if(window.routed_expert_union_steps)
                last_union_observation=HipKnownPrefixUnionObservation{first_position+(prefill?(prefill->logical_rows/8U-1U)*8U:0U),last_union_layer,pending_union_summary};
            return result;
        } catch (...) {
            failed = true;
            // A rejected foreign capture/scope must never be drained here.
            if(owned_eager_scope_admitted) guard.recover_or_fail_stop();
            throw;
        }
    }
};

HipKnownPrefixWindow::HipKnownPrefixWindow(CompiledKnownPrefixWindow window,
    CompiledSerialExecution serial_snapshot, ModelContract model, HipLauncherCatalogConfig config,
    const ExecutionRuntimeResources& serial_resources, ExecutionMemoryBinding window_scratch,
    HipLaunchContext scope, std::shared_ptr<const void> actual_owner,
    q51_known_rows::Workspace routed_union_workspace)
    : impl_(std::make_unique<Impl>(std::move(window), std::move(serial_snapshot), std::move(model),
          std::move(config), serial_resources, window_scratch, scope, std::move(actual_owner),
          std::move(routed_union_workspace))) {}
HipKnownPrefixWindow::~HipKnownPrefixWindow() = default;
HipKnownPrefixWindow::HipKnownPrefixWindow(CompiledKnownPrefixPrefill prefill,
    CompiledSerialExecution serial,ModelContract model,HipLauncherCatalogConfig config,
    const ExecutionRuntimeResources& resources,ExecutionMemoryBinding memory,HipLaunchContext scope,
    std::shared_ptr<const void> owner,q51_known_rows::Workspace workspace) {
    auto tile=prefill.tile8;
    impl_=std::make_unique<Impl>(std::move(tile),std::move(serial),std::move(model),std::move(config),resources,memory,
        scope,std::move(owner),std::move(workspace),std::move(prefill));
}
KnownWindowResult HipKnownPrefixWindow::execute(SerialSessionContext& session, std::array<std::uint32_t, 2> tokens) {
    return impl_->execute(session, std::span<const std::uint32_t>(tokens));
}
KnownWindowResult HipKnownPrefixWindow::execute(SerialSessionContext& session, std::span<const std::uint32_t> tokens) {
    return impl_->execute(session,tokens);
}
KnownWindowResult HipKnownPrefixWindow::execute_validated(SerialSessionContext& session,
    std::span<const std::uint32_t> tokens,const std::function<void()>& before_publish) {
    return impl_->execute(session,tokens,before_publish);
}
std::uint32_t HipKnownPrefixWindow::logical_rows() const noexcept { return impl_->logical_rows(); }
bool HipKnownPrefixWindow::supports_tiled_prefill() const noexcept {return impl_->prefill.has_value();}
KnownWindowOutputPolicy HipKnownPrefixWindow::output_policy() const noexcept { return impl_->prefill?impl_->prefill->output_policy:impl_->window.output_policy; }
std::optional<HipKnownPrefixUnionObservation> HipKnownPrefixWindow::latest_routed_union_observation() const noexcept {
    return impl_->failed?std::nullopt:impl_->last_union_observation;
}
} // namespace strix
