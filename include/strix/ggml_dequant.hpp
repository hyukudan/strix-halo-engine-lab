#pragma once

// Host dequantization of GGUF tensor rows, written from the block layouts and
// dequantize_row_* of ggml-org/llama.cpp@9d1878a4354e743529fbe09e7350a957f5e39e9a
// (ggml/src/ggml-common.h, ggml/src/ggml-quants.c; MIT, see SOURCES_LEDGER.md).
// Each value is produced with the same float operations in the same order as
// llama.cpp's reference C (no contraction: compile with -ffp-contract=off), so
// the output is bitwise llama.cpp's dequantization. This is the CPU oracle for
// the GSQ-RCO formats (Q2_0, K quants, IQ2_S/IQ3_XXS/IQ3_S, legacy 32-blocks).

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace strix::ggml_dequant {

// ggml type ids used by the Qwen GGUF builds.
enum class Type : std::uint32_t {
    f32 = 0, f16 = 1, q4_0 = 2, q4_1 = 3, q5_0 = 6, q5_1 = 7, q8_0 = 8,
    q2_k = 10, q3_k = 11, q4_k = 12, q5_k = 13, q6_k = 14,
    iq2_xxs = 16, iq2_xs = 17, iq3_xxs = 18, iq4_nl = 20, iq3_s = 21, iq2_s = 22, iq4_xs = 23,
    bf16 = 30, q2_0 = 42,
};

struct Geometry {
    std::size_t block_elements = 0;
    std::size_t block_bytes = 0;
};

// Block geometry of a supported type, nullopt otherwise.
[[nodiscard]] std::optional<Geometry> geometry(std::uint32_t ggml_type) noexcept;
[[nodiscard]] std::string_view name(std::uint32_t ggml_type) noexcept;
[[nodiscard]] bool supported(std::uint32_t ggml_type) noexcept;

// Bytes of one row of `elements` values (elements must be a multiple of the block).
[[nodiscard]] std::size_t row_bytes(std::uint32_t ggml_type, std::size_t elements);

// Dequantize one row: `row` holds row_bytes(type, out.size()) bytes.
void dequantize_row(std::uint32_t ggml_type, std::span<const std::byte> row, std::span<float> out);

// Canonical FP16 -> FP32 (exact; what GGML_FP16_TO_FP32 computes).
[[nodiscard]] float fp16(std::uint16_t bits) noexcept;
[[nodiscard]] float bf16(std::uint16_t bits) noexcept;

}  // namespace strix::ggml_dequant
