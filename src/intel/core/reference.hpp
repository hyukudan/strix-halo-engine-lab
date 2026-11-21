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
[[nodiscard]] std::size_t attention_score_elements(const core::DenseAttentionRow&, std::size_t output_elements);

// May be called before reading the model. The returned executor borrows any
// matrix it sees until destruction and caches its immutable device copy.
[[nodiscard]] std::unique_ptr<core::DenseReference> open_reference();

} // namespace strix::intel
