#pragma once

#include <cstdint>

namespace strix {

struct PackedProjectionRequest {
    std::uint32_t ggml_type = 0;
    std::uint64_t input_features = 0;
    std::uint64_t output_features = 0;
    std::uint32_t logical_rows = 1;
    std::uint32_t request_concurrency = 1;
    std::uint32_t activation_element_bytes = 4;
    std::uint32_t output_element_bytes = 4;
};

struct PackedProjectionPlan {
    std::uint32_t ggml_type = 0;
    std::uint64_t block_elements = 0;
    std::uint64_t block_bytes = 0;
    std::uint64_t blocks_per_output_row = 0;
    std::uint64_t weight_bytes_per_output_row = 0;
    std::uint64_t total_weight_bytes = 0;
    std::uint64_t input_activation_bytes = 0;
    std::uint64_t output_activation_bytes = 0;
    std::uint64_t serial_weight_traffic_bytes = 0;
    std::uint64_t ideal_reused_weight_traffic_bytes = 0;
    std::uint32_t logical_rows = 0;
    std::uint32_t request_concurrency = 0;
};

[[nodiscard]] bool supported_low_row_class(std::uint32_t logical_rows) noexcept;

[[nodiscard]] PackedProjectionPlan build_packed_projection_plan(
    const PackedProjectionRequest& request);

}  // namespace strix
