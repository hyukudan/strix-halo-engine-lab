#pragma once

#include "strix/hip_embedding_launcher.hpp"
#include "strix/hip_gdn_launcher.hpp"
#include "strix/hip_hyper_mix_launcher.hpp"
#include "strix/hip_indexed_attention_launcher.hpp"
#include "strix/hip_lm_head_launcher.hpp"
#include "strix/hip_moe_route_launcher.hpp"
#include "strix/hip_ple_launcher.hpp"
#include "strix/hip_moe_experts_launcher.hpp"
#include "strix/hip_shared_expert_launcher.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/model_contract.hpp"
#include <memory>
#include <vector>

namespace strix {
namespace integer_projection { struct RuntimeConfig; class HipIntegerProjectionRuntimeBundle; }

struct HipLauncherCatalogConfig {
    HipActivationStorage activation =
        HipActivationStorage::bf16;
    bool enable_device_graph_inputs = false;
    bool parallel_hyper_mix = false;
    bool staged_hyper_projections = false;
    bool fused_staged_hyper_mix = false;
    bool typed_hyper_projection = false;
    bool prefetch_typed_hyper_rows = false;
    bool prefetch_q8_hyper_output = false;
    bool lane_chain_hyper_output = false;
    bool prefetch_hc_projection_next_tile = false;
    bool splitk_hyper_projection = false;
    bool splitk_hyper_projection_fp64 = false;
    bool virtual_q8_lm_head = false;
    bool prefetch_q8_lm_head = false;
    bool wave8_q8_lm_head = false;
    bool wave8_q8_gdn_input = false;
    bool wave8_q8_gdn_output = false;
    bool wave8_q8_qsa_output = false;
    bool parallel_moe_router = false;
    bool virtual_q8_gdn_input = false;
    bool ranked_moe_router_topk = false;
    bool wave_moe_router_topk = false;
    std::shared_ptr<const WaveRouterManifestAdmission> wave_router_manifest_admission;
    bool prefetch_q8_gdn_input = false;
    bool q4k_a8g32_gate_up = false;
    bool batched_moe_activation = false;
    bool grouped_exact_q51_down = false;
    bool q51_slot_parallel_down = false;
    bool deferred_moe_status = false;
    std::shared_ptr<moe_q4k_a8::RuntimeDispatchStats> q4k_a8_stats;
    bool virtual_q8_gdn_output = false;
    bool prefetch_q8_qsa_input = false;
    std::vector<std::uint32_t> qsa_input_prefetch_layers;
    std::shared_ptr<qsa_input_prefetch::RuntimeDispatchStats> qsa_input_prefetch_stats;
    bool prefetch_q8_shared_expert = false;
    std::vector<std::uint32_t> shared_expert_prefetch_layers;
    std::shared_ptr<shared_expert_prefetch::RuntimeDispatchStats> shared_expert_prefetch_stats;
};

void register_complete_hip_kernel_catalog(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    HipLauncherCatalogConfig config,
    kernels::KernelRegistry& registry,
    const integer_projection::RuntimeConfig* alternate_lm_head = nullptr);

class HipLauncherCatalog {
public:
    HipLauncherCatalog(
        const ModelContract& model,
        HipLauncherCatalogConfig config = {},
        std::shared_ptr<integer_projection::HipIntegerProjectionRuntimeBundle> alternate_lm_head = {});

    HipLauncherCatalog(
        const HipLauncherCatalog&) = delete;
    HipLauncherCatalog& operator=(
        const HipLauncherCatalog&) = delete;
    HipLauncherCatalog(
        HipLauncherCatalog&&) = delete;
    HipLauncherCatalog& operator=(
        HipLauncherCatalog&&) = delete;

    void register_for_compiled_execution(
        const CompiledSerialExecution& compiled,
        ExecutionLauncherRegistry& registry,
        const ExecutionRuntimeResources* resources = nullptr);

    // Derive the closed QSA position-reader set from this catalog's owned
    // configurations and actual factories, rather than trusting arbitrary
    // identities supplied by a launch plan.
    [[nodiscard]] std::vector<std::uint64_t> known_window_position_readers(
        const CompiledSerialExecution& compiled, const ExecutionLauncherRegistry& registry) const;

    [[nodiscard]] LauncherCoverage audit(
        const CompiledSerialExecution& compiled,
        const ExecutionLauncherRegistry& registry) const;

private:
    bool device_graph_inputs_ = false;
    std::shared_ptr<integer_projection::HipIntegerProjectionRuntimeBundle> alternate_lm_head_;
    HipElementwiseLauncherBundle elementwise_;
    HipPleLauncherBundle ple_;
    HipGdnLauncherBundle gdn_;
    HipIndexedAttentionLauncherBundle indexed_attention_;
    HipEmbeddingLauncherBundle embedding_;
    HipHyperMixLauncherBundle hyper_mix_;
    HipMoeRouteLauncherBundle moe_route_;
    HipMoeExpertsLauncherBundle moe_experts_;
    HipLmHeadLauncherBundle lm_head_;
    HipSharedExpertLauncherBundle shared_expert_;
};

}  // namespace strix
