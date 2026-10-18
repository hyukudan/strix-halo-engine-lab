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
    // Dense A8 numerical profile (dense-a8-q80-v1): every Q8_0 projection of the
    // step (R1, windows and prompt rows) runs Q8_0 x per-32 int8 activations
    // through q8_rows::launch_dense_a8. Different numerics from the exact path.
    bool dense_a8_projections = false;
};

enum class HipQsaInputProjectionMode {legacy256,virtual256,prefetch8};
struct HipQsaInputProjectionView {std::uintptr_t address=0;std::uint64_t bytes=0;};
// Standalone diagnostic, no allocation/sync. Caller retains all owners through
// stream drain (also on enqueue failure). Exact disjoint output views, BF16 input,
// original Q8_0 weights and BF16-rounded values represented in FP32 output.
void launch_hip_qsa_input_projection_probe(HipQsaInputProjectionMode,
    HipQsaInputProjectionView input,HipQsaInputProjectionView weights,
    HipQsaInputProjectionView output,std::uint32_t width,std::uint32_t outputs,void* stream);

#ifdef STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC
// Test-only QSA top-k block selection over caller-owned device views; no
// allocation/sync. reference = the original single-thread loop, one launch per
// row; otherwise the production kernels: one row = the C1 kernel (with a
// device position: its graph-input form, sizes read on the device), 2/4/8 rows
// = the known-window rows kernel. Without a device position a row with
// visible/ratio <= block_topk takes the full-visible fill. Scores are consumed.
struct HipQsaBlockSelectRow { float* scores = nullptr; std::uint32_t* selected = nullptr; std::uint32_t visible = 0; };
void launch_hip_qsa_block_select_probe(bool reference, std::span<const HipQsaBlockSelectRow> rows, std::uint32_t ratio,
    std::uint32_t block_topk, const std::uint64_t* device_position, std::uint64_t context_tokens, std::uint32_t* device_status,
    void* stream);
#endif

#ifdef STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC
// Test-only kv-int8-tok-v1 writers and readers over caller-owned device memory; no
// allocation/sync. Each of the 8 rows appends k[r] / v[r] (kv_heads x head_dim FP32) at
// position[r]: the decode writer into decode_state, the 8-row window writer into rows_state
// (int8 rows of 2 kv_width + 8 kv_heads bytes). The FP32 query q_gate (attention_heads x 2
// head_dim) then attends the `count` tokens of `selected` in decode_state through the decode
// score/softmax/value kernels (scores_decode, out_decode), every row of the 8-row window
// kernels (scores_rows, out_rows: 8 x) and of the 16-row prefill group kernels (scores_group,
// out_group: 16 x). Scores are heads x maximum_selected, outputs attention_heads x head_dim.
struct HipKvInt8Probe {
    std::uint32_t attention_heads = 24, kv_heads = 2, head_dim = 256, count = 0, maximum_selected = 0;
    const float* k[8]{};
    const float* v[8]{};
    std::uint64_t position[8]{};
    void* decode_state = nullptr;
    void* rows_state = nullptr;
    const float* q_gate = nullptr;
    const std::uint32_t* selected = nullptr;
    float* scores_decode = nullptr;
    float* out_decode = nullptr;
    float* scores_rows = nullptr;
    float* out_rows = nullptr;
    float* scores_group = nullptr;
    float* out_group = nullptr;
};
void launch_hip_attention_kv_int8_probe(const HipKvInt8Probe& probe, void* stream);
#endif

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
#ifdef STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC
// Test-only T = 1/2/4/8/16/32/64 rows of the prefetch8 input (BF16 -> BF16-rounded FP32)
// or wave8 output (FP32 -> BF16) Q8_0 projection: the q8_rows engine (exact or
// A8) or the C1 kernel once per row. Same ownership rules as the probes above.
enum class HipQsaRowsProjection { input, output };
enum class HipQsaRowsProbeMode { rows, rows_a8, row_loop };
void launch_hip_qsa_rows_projection_probe(HipQsaRowsProjection, HipQsaRowsProbeMode,
    std::span<const HipQsaInputProjectionView> input, HipQsaInputProjectionView weights,
    std::span<const HipQsaInputProjectionView> output, std::uint32_t width, std::uint32_t outputs, void* stream);
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
// Prefill: all T rows of a QSA step (8-row tiles; projections 64 rows per launch).
void indexed_attention_known_prefill_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipIndexedAttentionLauncherConfig&, HipQsaIndexJournal journal = {});
void indexed_attention_known_prefill_rows_launch(std::span<const ExecutionDispatchContext> rows,
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
