#pragma once

#include "attn_core/prepared_gqa_attention.hpp"
#include "core/bf16_projection_batch.hpp"
#include "core/residual_mix.hpp"
#include "core/screened_head_plan.hpp"
#include "core/short_convolution.hpp"
#include "cuda/core/dense/rows_bf16_narrow_plan.hpp"
#include "cuda/core/mix_cuda_latency_plan.hpp"
#include "moe_core/expert_activation.hpp"
#include "moe_core/grouped_microscaled_rows.hpp"
#include "moe_core/grouped_rows.hpp"
#include "moe_core/microscaled_r1.hpp"
#include "strix/exec/catalog.hpp"
#include "strix/exec/compiler.hpp"
#include "strix/head_argmax.hpp"

#include <array>

namespace strix::exec::cuda_decode {

struct Embedding {
    const std::uint16_t* table;
    const std::uint32_t* token;
    float* output;
    unsigned vocabulary, width;
};

struct RmsBf16 {
    const float* input;
    const std::uint16_t* gain;
    float* output;
    unsigned groups, width;
    float epsilon;
};

struct AppendKvBf16 {
    const float* key;
    const float* value;
    std::uint16_t* key_cache;
    std::uint16_t* value_cache;
    const std::uint32_t* position;
    unsigned heads, kv_heads, width, capacity, first_position;
    float score_scale;
    bool wrapped;
};

struct PreparedFlashBf16 {
    const float* query;
    const float* relative_bias;
    const float* sinks;
    const std::uint16_t* key_cache;
    const std::uint16_t* value_cache;
    float* output;
    const std::uint32_t* position;
    unsigned heads, kv_heads, capacity, first_position, relative_extent, window;
    float score_scale;
    bool wrapped;
};

struct AppendKvHostBf16 { AppendKvBf16 args; };
struct PreparedFlashHostBf16 { PreparedFlashBf16 args; };

struct PreparedFinite {
    attn_core::PreparedGqaAttentionRows rows;
    unsigned heads, kv_heads, width, relative_extent;
    unsigned* failure;
};

struct CausalConvolution {
    core::ShortConvolution plan;
    const float* input;
    const float* static_weights;
    const float* dynamic_weights;
    float* history;
    float* output;
    bool round_bf16;
};

struct HeadClipBf16 {
    const float* input;
    const std::uint16_t* gain;
    float* output;
    const std::uint32_t* position;
    unsigned heads, width, first_position, scale_floor;
    float epsilon, threshold, scale_alpha;
};

struct Bf16Round {
    float* values;
    std::size_t elements;
};

struct Bf16SigmoidProduct {
    const float* input;
    const float* logits;
    float* output;
    std::size_t elements;
};

struct Bf16Softcap {
    float* values;
    std::size_t elements;
    float cap;
};

struct Bf16RelativeProjection16 {
    const float* input;
    const std::uint16_t* weights;
    float* output;
    unsigned heads, relative_extent;
};

enum class Bf16DenseMode { fp32, mma3, mma3_round };

// A homogeneous group that emits one native launch. The caller preserves the
// incumbent group order when a projection site contains multiple groups.
struct Bf16DenseGroup {
    const float* input;
    unsigned input_width;
    std::array<core::Bf16RowProjection, 4> projections;
    unsigned count;
    Bf16DenseMode mode;
};

struct Bf16Fp32Epilogue {
    Bf16DenseGroup dense;
    graph::Form form;
    float cap = 0.0F;
};

struct RoundedCausalConvolution {
    CausalConvolution convolution;
};

inline constexpr unsigned kPackedExpertsPerToken = 8;

struct PackedGateUp {
    moe_core::microscaled_r1::Args args;
    weight::microscaled::MatrixFormat format;
};

struct PackedDownCombine {
    moe_core::microscaled_r1::Args args;
    weight::microscaled::MatrixFormat format;
};

// Exact one-row latency forms (bitwise against their incumbents, RTX fixtures p39 / p40). A program
// selects them through R1Forms; the default keeps every incumbent leaf and identity.
struct R1Forms {
    bool bank_tiled = false;    // residual.bank.tiled: one-row bank readouts on 512-feature tiles
    bool output_split = false;  // r1.bf16_image.output_split_round: one narrow rounded projection
    bool moe_pipe = false;      // microscaled .k8.pipe: packed gate-up and down-combine

    auto operator<=>(const R1Forms&) const = default;
};

struct BankReadoutTiled {
    core::residual_mix::MixArgs args;
};

struct Bf16OutputSplitRound {
    const std::uint16_t* weight;
    const float* input;
    float* output;
    unsigned input_width, outputs;
};

struct PackedGateUpPipe {
    moe_core::microscaled_r1::Args args;
    weight::microscaled::MatrixFormat format;
};

struct PackedDownCombinePipe {
    moe_core::microscaled_r1::Args args;
    weight::microscaled::MatrixFormat format;
};

// Shape gates of the latency forms; false keeps the incumbent leaf.
[[nodiscard]] inline bool fits(const BankReadoutTiled& a) {
    return ::strix::cuda::mix::latency_grid(a.args).tiles != 0U;
}
[[nodiscard]] inline bool fits(const Bf16OutputSplitRound& a) noexcept {
    return a.weight && a.input && a.output && static_cast<const void*>(a.input) != static_cast<const void*>(a.output) &&
           cuda_core::bf16_narrow_blocks(a.input_width, a.outputs) != 0U;
}

struct GroupedProjection {
    moe_core::grouped_microscaled::Args args;
    weight::microscaled::MatrixFormat format;
};

struct GroupedActivation {
    moe_core::expert_activation::Args args;
    float gate_cap, up_cap;
};

struct GreedyPartial {
    const float* logits;
    head_argmax::Pick* partial;
    std::uint32_t count, chunks;
};

struct GreedyFinal {
    GreedyPartial reduction;
    std::uint32_t* output;
};

// EXACT screened greedy head (src/cuda/core/head_screened.cuh): the five one-launch stages of one screened_head_plan::Launch,
// bound in this order in place of the head projection, its softcap pass and the two greedy leaves. The pick writes the
// GreedyFinal reply contract: output[lane] = token, or head_argmax::kNone when a logit of the lane is non-finite.
struct ScreenedHeadPrep {
    core::screened_head_plan::Launch launch;
};
struct ScreenedHeadPass1 {
    core::screened_head_plan::Launch launch;
};
struct ScreenedHeadCompact {
    core::screened_head_plan::Launch launch;
};
struct ScreenedHeadPass2 {
    core::screened_head_plan::Launch launch;
};
struct ScreenedHeadPick {
    core::screened_head_plan::Launch launch;
    std::uint32_t* output;
};

// R1 entries for native leaves, with no per-engine shapes or storage policy.
// The resident adapter provides durable device pointers, including the token
// and position words within the selected executor descriptor.
[[nodiscard]] std::span<const KernelEntry> entries() noexcept;
[[nodiscard]] KernelBinding bind(const Embedding&);
[[nodiscard]] KernelBinding bind(const RmsBf16&);
[[nodiscard]] KernelBinding bind(const AppendKvBf16&);
[[nodiscard]] KernelBinding bind(const PreparedFlashBf16&);
[[nodiscard]] KernelBinding bind(const PreparedFinite&);
[[nodiscard]] KernelBinding bind(const CausalConvolution&);
[[nodiscard]] KernelBinding bind(const HeadClipBf16&);
[[nodiscard]] KernelBinding bind(const core::residual_mix::NormalizeArgs&);
[[nodiscard]] KernelBinding bind(const core::residual_mix::MixArgs&);
[[nodiscard]] KernelBinding bind(const core::residual_mix::AddArgs&);
[[nodiscard]] KernelBinding bind(const core::residual_mix::ControllerArgs&);
[[nodiscard]] KernelBinding bind(const Bf16Round&);
[[nodiscard]] KernelBinding bind(const Bf16SigmoidProduct&);
[[nodiscard]] KernelBinding bind(const Bf16Softcap&);
[[nodiscard]] KernelBinding bind(const Bf16RelativeProjection16&);
[[nodiscard]] KernelBinding bind(const Bf16DenseGroup&);
// Validate all independent operands and build all owned leaves before the caller emits any of them.
[[nodiscard]] std::vector<KernelBinding> bind_projection_batch(const core::Bf16ProjectionBatch&, const device::Profile&);
[[nodiscard]] KernelBinding bind(const Bf16Fp32Epilogue&);
[[nodiscard]] KernelBinding bind(const RoundedCausalConvolution&);
[[nodiscard]] KernelBinding bind(const core::residual_mix::NormalizePairArgs&);
[[nodiscard]] KernelBinding bind(const PackedGateUp&);
[[nodiscard]] KernelBinding bind(const PackedDownCombine&);
[[nodiscard]] KernelBinding bind(const BankReadoutTiled&);
[[nodiscard]] KernelBinding bind(const Bf16OutputSplitRound&);
[[nodiscard]] KernelBinding bind(const PackedGateUpPipe&);
[[nodiscard]] KernelBinding bind(const PackedDownCombinePipe&);
[[nodiscard]] KernelBinding bind(const GreedyPartial&);
[[nodiscard]] KernelBinding bind(const GreedyFinal&);
[[nodiscard]] KernelBinding bind(const ScreenedHeadPrep&);
[[nodiscard]] KernelBinding bind(const ScreenedHeadPass1&);  // pass1_lanes comes from the owner's plan
[[nodiscard]] KernelBinding bind(const ScreenedHeadCompact&);
[[nodiscard]] KernelBinding bind(const ScreenedHeadPass2&);
[[nodiscard]] KernelBinding bind(const ScreenedHeadPick&);
[[nodiscard]] KernelBinding bind(const AppendKvHostBf16&);
[[nodiscard]] KernelBinding bind(const PreparedFlashHostBf16&);
[[nodiscard]] KernelBinding bind(const moe_core::grouped_rows::GatherArgs&);
[[nodiscard]] KernelBinding bind(const GroupedProjection&);
[[nodiscard]] KernelBinding bind(const GroupedActivation&);
[[nodiscard]] KernelBinding bind(const moe_core::grouped_rows::CombineArgs&);

}  // namespace strix::exec::cuda_decode
