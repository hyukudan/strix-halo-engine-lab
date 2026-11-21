#pragma once

#include "strix/ggml_dequant.hpp"
#include "strix/hip_embedding_launcher.hpp"

#include <cstdint>
#include <span>
#include <stdexcept>

namespace strix::core {

// The shared gather validates host token IDs before enqueueing any row.
inline void enqueue_hip_bf16_embedding_rows(std::span<const std::uint16_t> table, unsigned vocabulary,
                                            unsigned width, std::span<const std::uint32_t> ids,
                                            std::span<float> output, void* stream) {
    if (width == 0 || table.size() / width != vocabulary || table.size() % width != 0 ||
        ids.size() > output.size() / width) {
        throw std::invalid_argument("HIP BF16 embedding rows: invalid buffer sizes");
    }
    constexpr auto ggml_bf16 = static_cast<std::uint32_t>(ggml_dequant::Type::bf16);
    launch_hip_embedding_rows({table.data(), ggml_bf16, width, vocabulary},
                              HipActivationStorage::fp32, ids, output.data(), width, stream);
}

inline void enqueue_hip_bf16_embedding_device_row(std::span<const std::uint16_t> table, unsigned vocabulary,
                                                  unsigned width, const std::uint32_t* id,
                                                  std::span<float> output, void* stream) {
    if (width == 0 || table.size() / width != vocabulary || table.size() % width != 0 || output.size() < width)
        throw std::invalid_argument("HIP BF16 embedding device row: invalid buffer sizes");
    constexpr auto ggml_bf16 = static_cast<std::uint32_t>(ggml_dequant::Type::bf16);
    launch_hip_embedding_device_row({table.data(), ggml_bf16, width, vocabulary},
                                    HipActivationStorage::fp32, id, output.data(), stream);
}

}  // namespace strix::core
