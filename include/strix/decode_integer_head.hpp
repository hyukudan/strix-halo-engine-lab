#pragma once

#include "strix/decode_execution_options.hpp"
#include "strix/integer_projection_runtime.hpp"
#include <limits>

namespace strix {
// Derive storage from the actual authoritative output tensor. Selecting a mode
// never changes checkpoint codes, relabels a nonlinear codebook or rounds K.
inline integer_projection::RuntimeConfig make_decode_integer_head_config(
    const ModelContract& model, const ModelManifest& manifest,
    const DecodeExecutionOptions& options, integer_projection::DeviceIdentity device) {
    if (options.mode != DecodeExecutionMode::eager || options.integer_head == DecodeIntegerHead::none)
        throw std::invalid_argument("integer decode config requires an explicit eager alternate head");
    options.validate(true, false);
    if (!model.hidden_size || !model.vocab_size || model.hidden_size > UINT32_MAX || model.vocab_size > UINT32_MAX)
        throw std::invalid_argument("integer decode shape exceeds native bounded dimensions");
    const ManifestTensor* head = nullptr;
    for (const auto& tensor : manifest.tensors) if (tensor.role == TensorRole::output_head) {
        if (head) throw std::invalid_argument("integer decode requires one authoritative output tensor");
        head = &tensor;
    }
    if (!head || head->dimensions != std::vector<std::uint64_t>{model.hidden_size, model.vocab_size} ||
        head->placement.kind != PlacementKind::local || head->placement.node != 0 ||
        (head->ggml_type != 2 && head->ggml_type != 23))
        throw std::invalid_argument("integer decode requires local rank-two original Q4_0 or IQ4_XS output weights");
    integer_projection::RuntimeConfig config;
    config.experimental_opt_in = true;
    config.hidden_size = static_cast<std::uint32_t>(model.hidden_size);
    config.vocabulary_size = static_cast<std::uint32_t>(model.vocab_size);
    config.input = integer_projection::RuntimeInput::bf16;
    config.weights = head->ggml_type == 2 ? integer_projection::WeightEncoding::q4_0 : integer_projection::WeightEncoding::iq4_xs;
    const auto block = head->ggml_type == 2 ? 32ULL : 256ULL;
    const auto block_bytes = head->ggml_type == 2 ? 18ULL : 136ULL;
    if (model.hidden_size % block || model.vocab_size > UINT64_MAX / (model.hidden_size / block) / block_bytes ||
        head->byte_size != model.vocab_size * (model.hidden_size / block) * block_bytes)
        throw std::invalid_argument("integer decode original weight blocks or byte extent mismatch");
    config.activations = options.integer_head == DecodeIntegerHead::a8_iu8
        ? low_precision::ActivationEncoding::symmetric_a8 : low_precision::ActivationEncoding::symmetric_a4;
    config.instruction = options.integer_head == DecodeIntegerHead::a4_iu4
        ? integer_projection::Instruction::iu4 : integer_projection::Instruction::iu8;
    if (config.weights == integer_projection::WeightEncoding::iq4_xs && config.instruction == integer_projection::Instruction::iu4)
        throw std::invalid_argument("IQ4_XS nonlinear codes cannot become raw IU4 operands");
    config.device = std::move(device);
    config.maximum_internal_bytes = options.integer_internal_bytes;
    config.maximum_retained_bytes = options.integer_retained_bytes;
    config.approved_numerical_identity = integer_projection::runtime_numerical_identity(config);
    (void)integer_projection::runtime_kernel_id(config);
    return config;
}
} // namespace strix
