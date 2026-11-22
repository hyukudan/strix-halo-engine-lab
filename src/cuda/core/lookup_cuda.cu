#include "lookup_cuda.cuh"

#include "cuda/weight/arch/sm120_sources.cuh"
#include "cuda_core/common.cuh"
#include "strix/cuda_core/nvfp4.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <type_traits>

namespace strix::cuda::core::lookup {
namespace {

namespace scalar = ::strix::weight::scalar_float;
namespace cc = ::strix::cuda_core;

template <Form F>
__device__ float decode(Table table, std::size_t id, std::size_t i) {
    const std::size_t element = id * table.width + i;
    if constexpr (F == Form::bf16_to_f32) {
        return __uint_as_float(std::uint32_t{static_cast<const std::uint16_t*>(table.data)[element]} << 16U);
    } else if constexpr (F == Form::f32) {
        return static_cast<const float*>(table.data)[element];
    } else if constexpr (F == Form::f16_to_f32 || F == Form::f16_payload_to_f32) {
        const auto bits = static_cast<const std::uint16_t*>(table.data)[element];
        if constexpr (F == Form::f16_payload_to_f32) {
            // Widen the original sign and payload for the incumbent F16 target route.
            if ((bits & 0x7fffU) > 0x7c00U)
                return __uint_as_float((std::uint32_t{bits & 0x8000U} << 16U) | 0x7f800000U |
                                       (std::uint32_t{bits & 0x03ffU} << 13U));
        }
        return __half2float(__ushort_as_half(bits));
    } else if constexpr (F == Form::nvfp4_tile_to_f32) {
        // The decode tile stores two 8-row halves, four lanes per row,
        // and 16 consecutive input features per E4M3 scale.
        const auto* record = static_cast<const std::uint8_t*>(table.data) +
            ((id / 16U) * (table.width / 128U) + i / 128U) * cc::kNvfp4RecordBytes;
        const auto half = (id % 16U) / 8U, row = id % 8U;
        const auto group = (i % 128U) / 16U, feature = i % 16U;
        const auto lane = row * 4U + (feature % 8U) / 2U;
        const auto packed = record[half * 512U + lane * 16U + group * 2U + feature / 8U];
        const auto code = static_cast<std::uint8_t>((packed >> (4U * (feature % 2U))) & 15U);
        const auto scale = record[1024U + row * 16U + half * 8U + group];
        return __fmul_rn(__fmul_rn(scalar::e2m1(code), scalar::e4m3(scale)), table.tensor_scale);
    } else {
        using Q8 = ::strix::cuda::weight::Q8Block;
        const auto* block = static_cast<const std::uint8_t*>(table.data) + element / Q8::kBlockElements * Q8::kBlockBytes;
        const auto scale = Q8::scale_bits(static_cast<std::uint16_t>(block[0] | (block[1] << 8U)));
        return __fmul_rn(scale, static_cast<float>(Q8::code(block, static_cast<unsigned>(i % Q8::kBlockElements))));
    }
}

template <Form F>
__device__ auto encode(float value) {
    if constexpr (F == Form::q8_0_to_bf16) {
        if (value != value) return std::uint16_t{0x7fc0U};
        return scalar::bf16_encode(value);
    } else if constexpr (F == Form::q8_0_to_bf16_row_zero) {
        // The MTP conversion uses the raw integer RNE formula, including for NaN bits.
        const auto bits = __float_as_uint(value);
        return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
    } else {
        return value;
    }
}

template <Form F, class Ids>
__global__ void gather(View v, Ids ids) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (v.programmatic) {
        cc::pdl_trigger();
        cc::pdl_wait();
    }
#endif

    const auto row = blockIdx.x;
    std::uint32_t id;
    if constexpr (std::is_same_v<Ids, StepTokens>) {
        if (!exec::step_row_valid(*ids.data, row)) return;
        id = exec::step_row(*ids.data, row).token;
    } else {
        id = ids.data[row];
    }
    if (id >= v.table.vocabulary) {
        if constexpr (F != Form::q8_0_to_bf16_row_zero) return;
        if (threadIdx.x == 0U && v.invalid_id) atomicOr(v.invalid_id, 1U);
        id = 0U;
    }

    const std::size_t first_copy = std::size_t{row} * v.copies;
    for (std::size_t i = threadIdx.x; i < v.table.width; i += blockDim.x) {
        const auto value = encode<F>(decode<F>(v.table, id, i));
        auto* out = static_cast<decltype(encode<F>(0.0F))*>(v.output);
        for (std::uint32_t copy = 0; copy < v.copies; ++copy)
            out[(first_copy + copy) * v.table.width + i] = value;
        if constexpr (F == Form::f16_payload_to_f32) {
            if (v.pre_carry && i < v.copies) v.pre_carry[first_copy + i] = i == 0U ? 1.0F : 0.0F;
        }
    }
}

template <Form F, class Ids>
cudaError_t launch(View v, Ids ids, cudaStream_t stream) {
    constexpr unsigned threads = 256;
    if (v.programmatic) return cc::launch_pdl(gather<F, Ids>, dim3(v.rows), dim3(threads), stream, v, ids);
    gather<F, Ids><<<v.rows, threads, 0, stream>>>(v, ids);
    return cudaGetLastError();
}

template <class Ids>
cudaError_t dispatch(View v, Ids ids, cudaStream_t stream) {
    constexpr auto max_rows = std::is_same_v<Ids, StepTokens> ? exec::kStepMaxRows : kMaxIdRows;
    if (!ids.data || reinterpret_cast<std::uintptr_t>(ids.data) % alignof(std::uint32_t) != 0U ||
        !valid(v, max_rows)) return cudaErrorInvalidValue;

    switch (v.form) {
    case Form::bf16_to_f32: return launch<Form::bf16_to_f32>(v, ids, stream);
    case Form::f32: return launch<Form::f32>(v, ids, stream);
    case Form::f16_to_f32: return launch<Form::f16_to_f32>(v, ids, stream);
    case Form::f16_payload_to_f32: return launch<Form::f16_payload_to_f32>(v, ids, stream);
    case Form::q8_0_to_f32: return launch<Form::q8_0_to_f32>(v, ids, stream);
    case Form::q8_0_to_bf16: return launch<Form::q8_0_to_bf16>(v, ids, stream);
    case Form::q8_0_to_bf16_row_zero: return launch<Form::q8_0_to_bf16_row_zero>(v, ids, stream);
    case Form::nvfp4_tile_to_f32: return launch<Form::nvfp4_tile_to_f32>(v, ids, stream);
    }
    return cudaErrorInvalidValue;
}

}  // namespace

cudaError_t enqueue(View view, TokenIds ids, cudaStream_t stream) { return dispatch(view, ids, stream); }
cudaError_t enqueue(View view, StepTokens ids, cudaStream_t stream) { return dispatch(view, ids, stream); }

}  // namespace strix::cuda::core::lookup
