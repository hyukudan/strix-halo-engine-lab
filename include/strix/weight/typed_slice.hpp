#pragma once

#include "strix/ggml_dequant.hpp"
#include "strix/weight/admission.hpp"
#include "strix/weight/registry.hpp"

#include <bit>
#include <limits>

namespace strix::weight {

enum class SliceDtype { bytes, bf16, f32 };

constexpr SliceDtype slice_dtype(std::string_view wire_dtype) {
    return wire_dtype == "BF16" ? SliceDtype::bf16 :
           wire_dtype == "F32" ? SliceDtype::f32 : SliceDtype::bytes;
}

// Safetensors F32 and BF16 are little endian. Read and widen only the chosen
// contiguous range, retaining the indexed source's stale-source checks.
inline Status read_float_slice(const RawTensor& tensor, std::span<float> out,
                               std::size_t first_element = 0) {
    const auto dtype = slice_dtype(tensor.wire_dtype);
    if (dtype == SliceDtype::bytes)
        return {Error::unknown_format, "plain weight read requires BF16 or F32"};
    const bool f32 = dtype == SliceDtype::f32;
    const std::size_t width = f32 ? 4U : 2U;
    const auto count = tensor.payload.range.bytes / width;
    if (first_element > count || out.size() > count - first_element)
        return {Error::unsupported_shape, "plain weight read outside tensor " + tensor.name};
    if (out.empty()) return {Error::ok, {}};

    const auto bytes = std::uint64_t{out.size()} * width;
    if (bytes > std::numeric_limits<std::size_t>::max())
        return {Error::memory_budget, "plain weight read exceeds addressable storage"};
    Slice slice;
    if (const auto status = subslice(tensor.payload, {std::uint64_t{first_element} * width, bytes}, slice);
        !status) return status;
    if constexpr (std::endian::native == std::endian::little) {
        if (f32) return read_slice(slice, std::as_writable_bytes(out));
    }

    std::vector<Byte> raw(static_cast<std::size_t>(bytes));
    if (const auto status = read_slice(slice, raw); !status) return status;
    if (f32) {
        for (std::size_t i = 0; i < out.size(); ++i) {
            std::uint32_t bits = 0;
            for (unsigned j = 0; j < 4; ++j)
                bits |= std::to_integer<std::uint32_t>(raw[4 * i + j]) << (8U * j);
            out[i] = std::bit_cast<float>(bits);
        }
    } else {
        ggml_dequant::dequantize_row(gguf_type(safetensors_format(tensor.wire_dtype)), raw, out);
    }
    return {Error::ok, {}};
}

}  // namespace strix::weight
