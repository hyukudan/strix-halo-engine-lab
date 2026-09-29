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

[[nodiscard]] bool valid_variant(const KernelVariant& variant) noexcept;
[[nodiscard]] bool matches(const KernelVariant& variant,
                           const KernelQuery& query) noexcept;

class KernelRegistry {
public:
    void add(KernelVariant variant);

    [[nodiscard]] const KernelVariant* find(std::string_view id) const noexcept;

    [[nodiscard]] const KernelVariant* select(const KernelQuery& query) const noexcept;

    [[nodiscard]] std::span<const KernelVariant> variants() const noexcept {
        return variants_;
    }

private:
    std::vector<KernelVariant> variants_;
};

[[nodiscard]] std::uint64_t plan_identity(
    std::span<const KernelVariant* const> selected) noexcept;

}  // namespace strix::kernels
