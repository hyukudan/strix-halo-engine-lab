#pragma once

#include "strix/exec/descriptor.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

namespace strix::cuda::core::lookup {

enum class Form : std::uint8_t {
    bf16_to_f32, f32, f16_to_f32, f16_payload_to_f32, q8_0_to_f32,
    q8_0_to_bf16, q8_0_to_bf16_row_zero, nvfp4_tile_to_f32,
};

struct Table {
    const void* data;
    std::uint32_t vocabulary, width;
    float tensor_scale = 1.0F;  // NVFP4 only, after the E2M1 x E4M3 product.
};

struct TokenIds { const std::uint32_t* data; };
struct StepTokens { const exec::StepBlock* data; };

struct View {
    Table table;
    Form form;
    void* output;  // [rows][copies][width], float or uint16_t as selected by form.
    std::uint32_t rows;
    std::uint32_t copies = 1;
    float* pre_carry = nullptr;       // [rows][copies], optional F16 payload-route initializer.
    std::uint32_t* invalid_id = nullptr;  // Optional row-zero route status, ORs bit 0.
    bool programmatic = false;       // Select only on a PDL-capable device path.
};

inline constexpr std::uint32_t kMaxIdRows = 0x7fffffffU;

[[nodiscard]] constexpr bool bf16_output(Form form) noexcept {
    return form == Form::q8_0_to_bf16 || form == Form::q8_0_to_bf16_row_zero;
}

// Physical table row equivalent in bytes. NVFP4 bytes are interleaved within
// a 16-row by 128-column tile, so the table must also satisfy its tile shape.
[[nodiscard]] constexpr std::uint64_t row_bytes(Form form, std::uint32_t width) noexcept {
    switch (form) {
    case Form::bf16_to_f32:
    case Form::f16_to_f32:
    case Form::f16_payload_to_f32: return std::uint64_t{width} * 2U;
    case Form::f32: return std::uint64_t{width} * 4U;
    case Form::q8_0_to_f32:
    case Form::q8_0_to_bf16:
    case Form::q8_0_to_bf16_row_zero: return width % 32U == 0U ? std::uint64_t{width / 32U} * 34U : 0U;
    case Form::nvfp4_tile_to_f32: return width % 128U == 0U ? std::uint64_t{width / 128U} * 72U : 0U;
    }
    return 0U;
}

[[nodiscard]] inline bool valid(const View& v, std::uint32_t max_rows) noexcept {
    const auto& [data, vocabulary, width, tensor_scale] = v.table;
    const auto bytes = row_bytes(v.form, width);
    constexpr auto max_bytes = std::numeric_limits<std::size_t>::max();
    const auto element_bytes = bf16_output(v.form) ? sizeof(std::uint16_t) : sizeof(float);
    if (!data || !v.output || vocabulary == 0U || bytes == 0U || v.rows == 0U || v.rows > max_rows ||
        v.copies == 0U || bytes > max_bytes / vocabulary ||
        width > max_bytes / element_bytes / v.copies / v.rows) return false;

    const std::size_t table_alignment = v.form == Form::f32 ? alignof(float) :
        v.form == Form::bf16_to_f32 || v.form == Form::f16_to_f32 || v.form == Form::f16_payload_to_f32
        ? alignof(std::uint16_t) : 1U;
    if (reinterpret_cast<std::uintptr_t>(data) % table_alignment != 0U ||
        reinterpret_cast<std::uintptr_t>(v.output) % element_bytes != 0U) return false;
    if (v.form == Form::nvfp4_tile_to_f32 && (vocabulary % 16U != 0U || !(tensor_scale > 0.0F) ||
                                              tensor_scale > std::numeric_limits<float>::max())) return false;
    if (v.pre_carry && (v.form != Form::f16_payload_to_f32 || v.copies > width ||
                        reinterpret_cast<std::uintptr_t>(v.pre_carry) % alignof(float) != 0U)) return false;
    return !v.invalid_id || (v.form == Form::q8_0_to_bf16_row_zero &&
                            reinterpret_cast<std::uintptr_t>(v.invalid_id) % alignof(std::uint32_t) == 0U);
}

#if defined(__CUDACC__)
// Buffers are non-overlapping device addresses, ordered by the caller's stream.
// X3 uses resolved StepRow::token and validity in program order. Masked rows
// leave outputs and status untouched. Out-of-range active IDs leave the row
// untouched except in the row-zero form, which preserves MTP's fallback.
cudaError_t enqueue(View view, TokenIds ids, cudaStream_t stream);
cudaError_t enqueue(View view, StepTokens ids, cudaStream_t stream);
#endif

}  // namespace strix::cuda::core::lookup
