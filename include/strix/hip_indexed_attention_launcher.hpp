#pragma once

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/model_contract.hpp"
#include "strix/graph_attention_state.hpp"
#include "strix/qsa_input_prefetch.hpp"
#include "strix/qsa_selection_plan.hpp"
#include "strix/session_state_layout.hpp"
#include "strix/session_state_memory.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace strix {

struct HipIndexedAttentionLauncherConfig {
    ModelContract model =
        ModelContract::qwen38_flash_next();
    HipActivationStorage activation =
        HipActivationStorage::bf16;
    bool enable_device_graph_inputs = false;
    bool prefetch_q8_input = false;
    bool wave8_q8_output = false;
    std::vector<std::uint32_t> prefetch_eligible_layers;
    std::shared_ptr<qsa_input_prefetch::RuntimeDispatchStats> prefetch_stats;
    bool qsa_hierarchical_selection = false;
    std::uint32_t qsa_gqa_heads_per_block = 1;
    std::shared_ptr<QsaHierarchicalSelectionStats> selection_stats;
    std::shared_ptr<QsaHierarchicalSelectionStats> gqa_stats;
};

enum class HipQsaInputProjectionMode {legacy256,virtual256,prefetch8};
struct HipQsaInputProjectionView {std::uintptr_t address=0;std::uint64_t bytes=0;};
// Standalone diagnostic, no allocation/sync. Caller retains all owners through
// stream drain (also on enqueue failure). Exact disjoint output views, BF16 input,
// original Q8_0 weights and BF16-rounded values represented in FP32 output.
void launch_hip_qsa_input_projection_probe(HipQsaInputProjectionMode,
    HipQsaInputProjectionView input,HipQsaInputProjectionView weights,
    HipQsaInputProjectionView output,std::uint32_t width,std::uint32_t outputs,void* stream);

#ifdef STRIX_BUILD_QSA_OUTPUT_PROJECTION_DIAGNOSTIC
enum class HipQsaOutputProjectionMode { legacy256, wave8 };
// Test-only explicit-stream entry. FP32 input and original Q8_0 weights produce
// BF16 output. No allocation/synchronization; retain owners through completion.
// Diagnostic capture is supported to inspect geometry, independently of the
// eager-only production launcher's pre-enqueue capture rejection.
void launch_hip_qsa_output_projection_probe(HipQsaOutputProjectionMode,
    HipQsaInputProjectionView input, HipQsaInputProjectionView weights,
    HipQsaInputProjectionView output, std::uint32_t width, std::uint32_t outputs, void* stream);
#endif

// Experimental post-projection C1 norm/RoPE/KV stage. Async on caller's
// explicitly owned stream; retain all resource owners through completion.
// Read status only after completion: 0=stage valid, 2=position out of bounds,
// 4=nonfinite source/norm, 8=nonfinite arithmetic. A nonzero status invalidates
// the stage. This does not publish a serial token/state commit.
void launch_hip_graph_attention_state(
    const GraphAttentionStatePlan& plan, std::uint64_t scalar_control_position,
    void* stream);

void register_hip_indexed_attention_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipIndexedAttentionLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification =
        kernels::QualificationStatus::experimental);

[[nodiscard]] ExecutionLauncher
make_hip_indexed_attention_launcher(
    std::string kernel_id,
    HipIndexedAttentionLauncherConfig* config);

// Optional device journal of raw indexer keys: per absolute layer,
// (compress_ratio-1) partial-block tail cells, then up to eight window row keys.
struct HipQsaIndexJournal { std::uintptr_t address = 0; std::uint64_t bytes = 0; };
[[nodiscard]] std::uint64_t qsa_index_journal_bytes(const ModelContract&);

// Known-prefix window: T = 2/4/8 rows of one QSA step at consecutive positions
// p..p+T-1, eager only. Q/K/V/indexer and output weights are read once for all
// rows; every row's raw-index key and K/V are appended before any row attends,
// and row t sees exactly cells <= p+t. Each row's output, workspace and state
// bytes equal the serial launcher's for this config (the row's serial kernel id
// is checked). A bound journal receives the T raw keys for the tail commit.
void indexed_attention_known_window_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipIndexedAttentionLauncherConfig&, HipQsaIndexJournal journal = {});
void indexed_attention_known_window_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipIndexedAttentionLauncherConfig&, HipQsaIndexJournal journal = {});

// Indexer partial-block tail around a window. Snapshot (before the window)
// copies each QSA layer's raw cells [p - p%ratio, p) into the journal; commit
// restores them and re-appends the journal keys of the first n_keep rows at
// p..p+n_keep-1. p and n_keep are device words (graph-capturable, no sync).
// Cells past the accepted prefix stay stale: they are rewritten before any read.
struct HipQsaIndexTailPlan {
    HipQsaIndexJournal journal;
    std::vector<std::uintptr_t> raw_index;
    std::vector<std::uint32_t> layers;
    std::uint64_t context_tokens = 0;
    std::uint32_t index_dim = 0, ratio = 0;
};
[[nodiscard]] HipQsaIndexTailPlan prepare_qsa_index_tail(const ModelContract&, const SerialSessionStateLayout&,
    const SessionStateMemoryBinding&, HipQsaIndexJournal);
void enqueue_qsa_index_tail_snapshot(const HipQsaIndexTailPlan&, const std::uint64_t* device_position, void* stream);
void enqueue_qsa_index_tail_commit(const HipQsaIndexTailPlan&, const std::uint64_t* device_position,
    const std::uint32_t* device_n_keep, void* stream);

class HipIndexedAttentionLauncherBundle {
public:
    explicit HipIndexedAttentionLauncherBundle(
        HipIndexedAttentionLauncherConfig config)
        : config_(config), fallback_config_(config) {fallback_config_.prefetch_q8_input=false;fallback_config_.prefetch_eligible_layers.clear();}

    HipIndexedAttentionLauncherBundle(
        const HipIndexedAttentionLauncherBundle&) = delete;
    HipIndexedAttentionLauncherBundle& operator=(
        const HipIndexedAttentionLauncherBundle&) = delete;
    HipIndexedAttentionLauncherBundle(
        HipIndexedAttentionLauncherBundle&&) = delete;
    HipIndexedAttentionLauncherBundle& operator=(
        HipIndexedAttentionLauncherBundle&&) = delete;

    void register_for_compiled_execution(
        const CompiledSerialExecution& compiled,
        ExecutionLauncherRegistry& registry);

    // Admit only this bundle's exact current eager QSA launchers. The bundle
    // and its configurations must outlive use of the returned identities.
    [[nodiscard]] std::vector<std::uint64_t> known_window_position_readers(
        const CompiledSerialExecution& compiled,
        const ExecutionLauncherRegistry& registry) const;

private:
    HipIndexedAttentionLauncherConfig config_;
    HipIndexedAttentionLauncherConfig fallback_config_;
};

}  // namespace strix
