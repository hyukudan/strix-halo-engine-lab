#pragma once

#include "backend.hpp"
#include "strix/core/dense_reference.hpp"

#include <memory>
#include <string_view>

namespace strix::intel {

// Admission is pure host code. The only accepted physical device is a
// discrete Xe2 Arc device with device USM. XMX is not needed.
[[nodiscard]] std::string_view reference_refusal(Status, const Capabilities&) noexcept;
[[nodiscard]] std::size_t q8_matrix_bytes(core::Q8Rows);
[[nodiscard]] std::size_t linear_batch(core::F32Rows, std::size_t input, std::size_t output);

// The matrix draft narrows FP32 operands to FP16 and accumulates in FP32.
// This versioned FAST candidate is explicitly requested by a qualification caller.
enum class LinearPolicy { reference, xmx_fp16_f32_v1 };
enum class LinearPath { refused, reference, xmx };
struct LinearPlan {
    LinearPath path;
    std::size_t padded_batch = 0;
    std::string_view reason;
};
[[nodiscard]] LinearPlan plan_linear(Status, const Capabilities&, LinearPolicy,
                                     std::size_t batch, std::size_t rows, std::size_t width) noexcept;
[[nodiscard]] std::size_t attention_score_elements(const core::DenseAttentionRow&, std::size_t output_elements);

// Both factories use the shared executor and may be called before reading a model.
// Q8 rows are borrowed until destruction for the device cache.
[[nodiscard]] std::unique_ptr<core::DenseReference> open_reference();
// Decoded FP32 rows and transformed routing scores are borrowed per call.
[[nodiscard]] std::unique_ptr<core::RoutedReference> open_routed_reference(LinearPolicy = LinearPolicy::reference);

} // namespace strix::intel
