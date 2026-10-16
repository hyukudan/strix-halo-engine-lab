#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix::kernels {

enum class OperatorFamily {
    rms_norm,
    residual_add,
    partial_rope,
    argmax,
    top_k,
    packed_projection,
    recurrent,
    moe_route,
    moe_expert,
    sparse_lookup,
    embedding_lookup,
    indexed_attention,
    hyper_mix,
    hyper_expand,
    elementwise_inject,
    shared_expert,
    moe_combine,
    lm_head,
};

enum class ExecutionPhase {
    prefill,
    decode,
    speculative_draft,
    speculative_verify,
    commit,
    rollback,
};

enum class WaveSize : std::uint32_t {
    wave32 = 32,
    wave64 = 64,
};

enum class QualificationStatus {
    experimental,
    oracle_matched,
    target_qualified,
    production,
    rejected,
};

struct ShapeClass {
    std::uint32_t logical_rows = 1;
    std::uint32_t request_concurrency = 1;
    std::uint32_t hidden_size = 0;
    std::uint32_t auxiliary_size = 0;
    friend bool operator==(const ShapeClass&, const ShapeClass&) = default;
};

struct ResourceUsage {
    std::uint32_t workgroup_threads = 0;
    std::uint32_t lds_bytes = 0;
    std::uint32_t vgpr_per_thread = 0;
    std::uint32_t sgpr_per_wave = 0;
    std::uint64_t scratch_bytes = 0;
};

struct NumericalContract {
    std::string input_precision;
    std::string accumulator_precision;
    std::string output_precision;
    bool row_invariant = false;
    bool deterministic = false;
};

struct KernelVariant {
    std::string id;
    OperatorFamily family = OperatorFamily::rms_norm;
    ExecutionPhase phase = ExecutionPhase::decode;
    ShapeClass shape;
    WaveSize wave = WaveSize::wave32;
    ResourceUsage resources;
    NumericalContract numerics;
    QualificationStatus qualification = QualificationStatus::experimental;
    bool graph_safe = false;
    std::string input_layout;
    std::string output_layout;
    std::string compiler_identity;
    double measured_seconds = 0.0;
    // Explicit equivalence class, supplied by qualification. Zero is unknown
    // and cannot participate in offline tuning across implementations. This
    // evidence does not change the arithmetic identity of the same kernel ID.
    std::uint64_t numerical_contract_identity = 0;
};

struct KernelQuery {
    OperatorFamily family = OperatorFamily::rms_norm;
    ExecutionPhase phase = ExecutionPhase::decode;
    ShapeClass shape;
    bool require_graph_safe = false;
    bool require_row_invariant = false;
    bool require_deterministic = false;
    QualificationStatus minimum_qualification = QualificationStatus::oracle_matched;
    std::optional<WaveSize> required_wave;
};

// Offline measurement provenance, supplied once during preparation. Timings
// never participate in plan_identity and cannot promote registry qualification.
struct OfflineTuningKey {
    std::string gfx_target, device_sku, compiler_identity, hip_identity;
    std::string binary_sha256, model_content_sha256, compiler_flags_sha256;
    std::string input_precision, accumulator_precision, output_precision;
    std::string weight_format, input_layout, output_layout, state_layout;
    std::string context_bucket, memory_regime;
    std::uint32_t compute_units = 0;
    WaveSize wave = WaveSize::wave32;
    OperatorFamily family = OperatorFamily::rms_norm;
    ExecutionPhase phase = ExecutionPhase::decode;
    ShapeClass shape;
    std::uint64_t m = 0, n = 0, k = 0;
    std::uint64_t numerical_contract_identity = 0;
    friend bool operator==(const OfflineTuningKey&, const OfflineTuningKey&) = default;
};
struct OfflineTuningRecord {
    OfflineTuningKey key;
    std::string kernel_id;
    std::string qualification_receipt_sha256;
    double median_complete_seconds = 0.0;
    std::uint32_t samples = 0;
    bool complete_workload = false;
    bool owners_and_resources_qualified = false;
};
struct OfflineTuningPlan {
    std::span<const OfflineTuningKey> step_keys;
    std::span<const OfflineTuningRecord> records;
};
enum class TuningSelectionReason { qualified_record, invalid_key, ambiguous_records, no_qualified_record, selection_policy_mismatch };
struct TunedKernelSelection {
    const KernelVariant* variant = nullptr;
    TuningSelectionReason reason = TuningSelectionReason::no_qualified_record;
};

[[nodiscard]] bool valid_variant(const KernelVariant& variant) noexcept;
[[nodiscard]] bool matches(const KernelVariant& variant,
                           const KernelQuery& query) noexcept;

class KernelRegistry {
public:
    void add(KernelVariant variant);

    [[nodiscard]] const KernelVariant* find(std::string_view id) const noexcept;

    [[nodiscard]] const KernelVariant* select(const KernelQuery& query) const noexcept;
    // Preparation only: exact provenance match, then qualified complete cost.
    // Invalid/missing/ambiguous records retain the existing selector unchanged.
    [[nodiscard]] TunedKernelSelection select_tuned(const KernelQuery& query,
        const OfflineTuningKey& live_key, std::span<const OfflineTuningRecord> records) const noexcept;

    [[nodiscard]] std::span<const KernelVariant> variants() const noexcept {
        return variants_;
    }

private:
    std::vector<KernelVariant> variants_;
};

[[nodiscard]] std::uint64_t plan_identity(
    std::span<const KernelVariant* const> selected) noexcept;

}  // namespace strix::kernels
