#include "strix/hip_sampled_target_logits_reader.hpp"

#include <hip/hip_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace strix {
namespace {

void check_hip(
    hipError_t error,
    const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) +
            ": " +
            hipGetErrorString(error));
    }
}

}  // namespace

std::vector<float>
HipTargetLogitsReader::read_fp32_logits(
    const ExecutionBufferAddress& buffer,
    std::uint32_t rows,
    std::uint64_t vocab_size) {
    if (buffer.address == 0 ||
        buffer.allocation_generation == 0 ||
        rows == 0 ||
        vocab_size == 0 ||
        vocab_size >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument(
            "HIP target logits reader received invalid buffer geometry");
    }
    const auto vocab =
        static_cast<std::size_t>(
            vocab_size);
    if (vocab >
        std::numeric_limits<std::size_t>::max() /
            static_cast<std::size_t>(rows)) {
        throw std::overflow_error(
            "HIP target logits element count overflows size_t");
    }
    const auto elements =
        vocab *
        static_cast<std::size_t>(
            rows);
    if (elements >
        std::numeric_limits<std::size_t>::max() /
            sizeof(float)) {
        throw std::overflow_error(
            "HIP target logits byte count overflows size_t");
    }
    const auto bytes =
        elements *
        sizeof(float);
    if (buffer.bytes < bytes) {
        throw std::invalid_argument(
            "HIP target logits buffer is smaller than requested rows");
    }

    std::vector<float> result(
        elements);
    check_hip(
        hipDeviceSynchronize(),
        "hipDeviceSynchronize(target logits)");
    check_hip(
        hipMemcpy(
            result.data(),
            reinterpret_cast<const void*>(
                buffer.address),
            bytes,
            hipMemcpyDeviceToHost),
        "hipMemcpy(target logits D2H)");
    return result;
}

}  // namespace strix
