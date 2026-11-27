#pragma once

// Pieces shared by the layer-level attention reference forms (compressed MQA, selected latent):
// the value-head post-mix and the sigmoid output gate. A model binds them with a `Weights` type
// (tensor(role, count) -> std::vector<float>; linear(role, input, output, groups)) and the role
// names of its checkpoint; nothing here knows a model.

#include "attn_core/value_head_mix_reference.hpp"
#include "core/bf16_reference.hpp"
#include "strix/attn/reference_executor.hpp"

#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace strix::attn_core {

struct ValueHeadMixRoles {
    std::string_view output_factor, input_factor;
};

// Residual value-head mix. The folded matrix is built once per history and reused.
// `swapped` exchanges the two stored factors.
template <class Weights>
void value_head_postmix_reference(const Weights& w, const ValueHeadMixRoles& roles, unsigned heads, unsigned rank,
                                  bool swapped, bool checkpoint, std::span<float> values,
                                  std::vector<float>& folded, const attn::ReferenceHeadExecutor& execute) {
    if (!folded.empty()) {
        apply_folded_value_head_mix_reference(values, heads, folded, execute);
        return;
    }
    const auto count = std::size_t{heads} * rank;
    auto u = w.tensor(roles.output_factor, count), v = w.tensor(roles.input_factor, count);
    if (swapped) std::swap(u, v);
    if (checkpoint) {
        folded = fold_value_head_mix_reference(heads, rank, u, v, true);
        apply_folded_value_head_mix_reference(values, heads, folded, execute);
    } else {
        residual_value_head_mix(values, heads, rank, u, v, true);
    }
}

// values *= sigmoid(linear(role, query_lora)); BF16 sigmoid on the checkpoint path.
template <class Weights>
void sigmoid_gate_values_reference(const Weights& w, std::string_view role, std::span<const float> query_lora,
                                   std::span<float> values, bool checkpoint) {
    std::vector<float> gate(values.size());
    w.linear(role, query_lora, gate, 1U);
    core::sigmoid_product_reference(values, gate, values, checkpoint
        ? core::SigmoidGatePrecision::bf16_sigmoid : core::SigmoidGatePrecision::f32);
}

} // namespace strix::attn_core
