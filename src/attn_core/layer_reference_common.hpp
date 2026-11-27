#pragma once

// Pieces shared by the layer-level attention reference forms (compressed MQA, selected latent):
// the value-head post-mix and the sigmoid output gate. A model binds them with a `Weights` type
// (tensor(role, count) -> std::vector<float>; linear(role, input, output, groups)) and the role
// names of its checkpoint; nothing here knows a model.

#include "attn_core/value_head_mix_reference.hpp"
#include "attn_core/declared_order_reference.hpp"
#include "core/bf16_reference.hpp"
#include "strix/attn/head_rms_reference.hpp"
#include "strix/attn/reference_order.hpp"
#include "strix/attn/reference_executor.hpp"

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace strix::attn_core {

// The layer norms of the reference under the chosen order: sequential = core::rms_norm_bf16_reference /
// attn::head_rms_reference (the shared defaults, unchanged); declared = the device's lane-strided halving sums.
inline void order_rms_norm_bf16_reference(attn::ReferenceOrder order, std::span<float> row, std::span<const float> gain,
                                          float epsilon) {
    core::rms_norm_bf16_reference(row, gain, epsilon, false, order);
}

inline void order_head_rms_reference(attn::ReferenceOrder order, std::span<float> heads, unsigned head_width, float epsilon) {
    if (order == attn::ReferenceOrder::sequential) return attn::head_rms_reference(heads, head_width, epsilon);
    if (head_width == 0 || heads.empty() || heads.size() % head_width != 0 || !(epsilon > 0.0F))
        throw std::invalid_argument("head RMSNorm: invalid shape or epsilon");
    for (std::size_t i = 0; i < heads.size(); i += head_width) {
        const auto head = heads.subspan(i, head_width);
        const float inv = declared_order::rms_inv(head, epsilon);
        for (float& value : head) value *= inv;
    }
}

struct ValueHeadMixRoles {
    std::string_view output_factor, input_factor;
};

// How the value-head mix is applied: factor by factor on every row, or as one folded matrix.
enum class FoldForm : std::uint8_t { residual, folded };

// Precision of the stored intermediates: full single precision, or rounded through BF16.
enum class StorePrecision : std::uint8_t { fp32, bf16 };

// Where the rotary is undone on the attention output heads: before the value-head mix, or on its result.
enum class OutputRope : std::uint8_t { before_mix, after_mix };

// Per-head RMS normalisation of a projected stream: applied, or left as projected.
enum class HeadNorm : std::uint8_t { head_rms, none };

// Value-head mix. The folded matrix is built once per history and reused. A model whose checkpoint
// stores the two factors exchanged names its roles accordingly.
template <class Weights>
void value_head_postmix_reference(const Weights& w, const ValueHeadMixRoles& roles, unsigned heads, unsigned rank,
                                  FoldForm form, std::span<float> values,
                                  std::vector<float>& folded, const attn::ReferenceHeadExecutor& execute) {
    if (!folded.empty()) {
        apply_folded_value_head_mix_reference(values, heads, folded, execute);
        return;
    }
    const auto count = std::size_t{heads} * rank;
    auto u = w.tensor(roles.output_factor, count), v = w.tensor(roles.input_factor, count);
    if (form == FoldForm::folded) {
        folded = fold_value_head_mix_reference(heads, rank, u, v, true);
        apply_folded_value_head_mix_reference(values, heads, folded, execute);
    } else {
        residual_value_head_mix(values, heads, rank, u, v, true);
    }
}

// values *= sigmoid(linear(role, query_lora)) at the declared sigmoid precision. The declared order binds the
// declared exp (the device's DeclaredSigmoidExp).
template <class Weights>
void sigmoid_gate_values_reference(const Weights& w, std::string_view role, std::span<const float> query_lora,
                                   std::span<float> values,
                                   core::SigmoidGatePrecision precision,
                                   attn::ReferenceOrder order = attn::ReferenceOrder::sequential) {
    std::vector<float> gate(values.size());
    w.linear(role, query_lora, gate, 1U);
    core::sigmoid_product_reference(values, gate, values, precision, order == attn::ReferenceOrder::declared);
}

} // namespace strix::attn_core
