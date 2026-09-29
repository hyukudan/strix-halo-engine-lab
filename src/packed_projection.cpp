#include "strix/packed_projection.hpp"

#include "strix/gguf_reader.hpp"

#include <limits>
#include <stdexcept>

namespace strix {
namespace {

std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right) {
    if (left != 0 &&
        right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error("packed projection size overflow");
    }
    return left * right;
}

}  // namespace

bool supported_low_row_class(std::uint32_t logical_rows) noexcept {
    return logical_rows == 1 || logical_rows == 2 ||
           logical_rows == 4 || logical_rows == 8;
}

PackedProjectionPlan build_packed_projection_plan(
    const PackedProjectionRequest& request) {
    if (request.input_features == 0 ||
        request.output_features == 0 ||
        request.request_concurrency == 0 ||
        request.activation_element_bytes == 0 ||
        request.output_element_bytes == 0) {
        throw std::invalid_argument(
            "packed projection dimensions and element sizes must be non-zero");
    }
    if (!supported_low_row_class(request.logical_rows)) {
        throw std::invalid_argument(
            "packed projection logical rows must be one of 1, 2, 4 or 8");
    }

    const auto geometry =
        gguf::tensor_type_geometry(request.ggml_type);
    if (!geometry.has_value()) {
        throw std::invalid_argument(
            "GGML type has no pinned storage geometry");
    }
    if (request.input_features % geometry->block_elements != 0) {
        throw std::invalid_argument(
            "input feature count is not divisible by quantization block size");
    }

    PackedProjectionPlan plan;
    plan.ggml_type = request.ggml_type;
    plan.block_elements = geometry->block_elements;
    plan.block_bytes = geometry->block_bytes;
    plan.blocks_per_output_row =
        request.input_features / geometry->block_elements;
    plan.weight_bytes_per_output_row =
        checked_mul(plan.blocks_per_output_row, plan.block_bytes);
    plan.total_weight_bytes =
        checked_mul(plan.weight_bytes_per_output_row,
                    request.output_features);

    const auto input_elements =
        checked_mul(request.input_features, request.logical_rows);
    const auto output_elements =
        checked_mul(request.output_features, request.logical_rows);
    plan.input_activation_bytes =
        checked_mul(input_elements, request.activation_element_bytes);
    plan.output_activation_bytes =
        checked_mul(output_elements, request.output_element_bytes);

    plan.serial_weight_traffic_bytes =
        checked_mul(plan.total_weight_bytes, request.logical_rows);
    plan.ideal_reused_weight_traffic_bytes =
        plan.total_weight_bytes;
    plan.logical_rows = request.logical_rows;
    plan.request_concurrency = request.request_concurrency;
    return plan;
}

}  // namespace strix
