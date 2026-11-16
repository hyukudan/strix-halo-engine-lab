#pragma once

#include "source.cuh"
#include "strix/weight/registry.hpp"
#include "strix/cuda/qwen_quant.cuh"

namespace strix::cuda::weight {

// A row address and its geometry are fixed at launch; the storage type is fixed
// by the source instantiation. The core's accumulation policy stays separate.
struct GgmlView {
    const std::uint8_t* blocks{};
    std::uint64_t row_stride_bytes{}, expert_stride_bytes{};
    std::uint32_t k{}, n{};
};
static_assert(std::is_trivially_copyable_v<GgmlView>);

inline ::strix::weight::Status make_view(const ::strix::weight::WeightView& weight,
                                         const UploadedSlices& uploaded, GgmlView& out) {
    namespace sw = ::strix::weight;
    const auto valid = sw::validate_weight(weight);
    if (!valid) return valid;
    if (weight.shape.rank < 2 || weight.shape.rank > 3)
        return {sw::Error::unsupported_shape, "CUDA source requires matrix or expert matrix"};
    const auto wire = quant::geometry(sw::gguf_type(weight.storage));
    if (!wire.block_elements) return {sw::Error::unsupported_arithmetic, "CUDA scalar source unavailable"};
    const auto k = weight.shape.dims[weight.shape.rank - 1], n = weight.shape.dims[weight.shape.rank - 2];
    if (!k || !n || k > UINT32_MAX || n > UINT32_MAX || k % wire.block_elements)
        return {sw::Error::unsupported_shape, "CUDA source dimensions exceed matrix view"};
    const auto row_bytes = k / wire.block_elements * wire.block_bytes;
    if (n > UINT64_MAX / row_bytes)
        return {sw::Error::unsupported_shape, "CUDA source expert stride overflow"};
    if (!uploaded.payload.data || uploaded.payload.bytes < weight.payload.range.bytes ||
        (reinterpret_cast<std::uintptr_t>(uploaded.payload.data) % (wire.block_bytes == 128 ? 4 : 2)))
        return {sw::Error::malformed, "CUDA source upload missing, truncated or misaligned"};
    out = {static_cast<const std::uint8_t*>(uploaded.payload.data), row_bytes, n * row_bytes,
           static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(n)};
    return {sw::Error::ok, {}};
}


template <std::uint32_t Type, class Arch, class Arithmetic>
struct OperandSource<GgmlTag<Type>, NativeScalarTag, Arch, Arithmetic> {
    static_assert(Type == quant::t_f32 || Type == quant::t_f16 || Type == quant::t_bf16 ||
                  Type == quant::t_q4_0 || Type == quant::t_q5_0 || Type == quant::t_q5_1 || Type == quant::t_q8_0 ||
                  Type == quant::t_q4_k || Type == quant::t_q5_k || Type == quant::t_q6_k || Type == quant::t_iq4_nl ||
                  Type == quant::t_q2_0 || Type == quant::t_iq4_xs || Type == quant::t_iq2_s || Type == quant::t_iq3_xxs ||
                  Type == quant::t_iq3_s, "No CUDA semantic scalar source for this storage");
    using View = GgmlView;
    static constexpr unsigned group_k = 32, alignment = Type == 0 ? 4 : 2, shared_bytes = 0;
    static constexpr bool has_input_transform = false, has_output_transform = false;
    static ::strix::weight::Status validate(const ::strix::weight::WeightView& w, const LaunchShape& shape) {
        namespace sw = ::strix::weight;
        if (sw::gguf_type(w.storage) != Type) return {sw::Error::unknown_format, "CUDA source storage differs"};
        if (!quant::geometry(Type).block_elements) return {sw::Error::unsupported_arithmetic, "CUDA scalar decode unavailable"};
        if (w.shape.rank < 2 || w.shape.rank > 3 ||
            shape.k != w.shape.dims[w.shape.rank - 1] || shape.n != w.shape.dims[w.shape.rank - 2] ||
            shape.k == 0 || shape.k % quant::geometry(Type).block_elements ||
            shape.n == 0 || shape.k > UINT32_MAX || shape.n > UINT32_MAX)
            return {sw::Error::unsupported_shape, "CUDA source requires whole storage blocks"};
        return sw::validate_weight(w);
    }
    STRIX_CUDA_WEIGHT_HD static void group(const View& view, std::uint64_t row, std::uint64_t group_index, float* out) {
        quant::decode_row_group(Type, view.blocks + row * view.row_stride_bytes,
                                static_cast<std::uint32_t>(group_index), out);
    }
    STRIX_CUDA_WEIGHT_HD static float scalar(const View& view, std::uint64_t row, std::uint64_t k) {
        float decoded[32];
        group(view, row, k / 32, decoded);
        return decoded[k % 32];
    }
};

} // namespace strix::cuda::weight
