#pragma once

#include "collective.cuh"

namespace strix::cuda::core::chain {

struct Rows {
    const float* input;
    float* output;  // optional when the chain writes its own output
    unsigned count;  // flattened [position][group] rows
    unsigned width;
    std::size_t input_stride = 0;  // floats; zero means width
    std::size_t output_stride = 0;
    unsigned groups = 1;
    unsigned first_position = 0;
    const unsigned* positions = nullptr;  // optional device position per unflattened row
    unsigned first_index = 0;
    const PendingRow* pending = nullptr;  // device [row][state_count]
    unsigned state_count = 0;
};

// Recompute prologue: the consumer CTA loads and transforms the complete row.
// It may keep the lane-owned values or stage them in its own operand layout.
template <unsigned R, class... Steps>
__device__ __forceinline__ RegisterRow<R> load_prologue(List<Steps...> list, const float* input,
                                                        const ChainArgs& args, const TileCtx& ctx) {
    RegisterRow<R> row{};
#pragma unroll
    for (unsigned r = 0; r < R; ++r)
        if (detail::column(r) < ctx.width) row.values[r] = input[detail::column(r)];
    apply(list, row, args, ctx);
    return row;
}

namespace detail {

template <class... Steps>
inline constexpr bool side_output_only = sizeof...(Steps) == 1 &&
    ((Steps::fold == Fold::descending_upper_first_side_output) && ...);

template <unsigned R>
__device__ __forceinline__ void store_row(const RegisterRow<R>& row, float* output, unsigned width) {
#pragma unroll
    for (unsigned r = 0; r < R; ++r)
        if (column(r) < width) output[column(r)] = row.values[r];
}

template <unsigned R, class... Steps>
__global__ __launch_bounds__(kThreads) void standalone(List<Steps...> list, Rows rows, ChainArgs args) {
    extern __shared__ float scratch[];
    const unsigned row = blockIdx.x;
    const auto position_row = row / rows.groups;
    const auto position = rows.positions ? rows.positions[position_row] : rows.first_position + position_row;
    const auto* pending = rows.pending ? rows.pending + std::size_t{row} * rows.state_count : nullptr;
    const TileCtx ctx{rows.width, row, row % rows.groups, position, rows.first_index, scratch, pending};
    if constexpr (side_output_only<Steps...>) {
        RegisterRow<R> tile{};
        apply(list, tile, args, ctx);
    } else {
        const auto tile = load_prologue<R>(list, rows.input + std::size_t{row} * rows.input_stride, args, ctx);
        if (rows.output) store_row(tile, rows.output + std::size_t{row} * rows.output_stride, rows.width);
    }
}

template <class StepType>
[[nodiscard]] bool valid_binding(const Rows& rows, const ChainArgs& args) {
    constexpr auto link = StepType::link;
    constexpr auto arg = form_info(link.form).arg;
    if constexpr (arg == Arg::state)
        if (!rows.pending || args.state[link.arg] >= rows.state_count) return false;
    if constexpr (arg == Arg::aux && link.form != Form::store_f32 && link.form != Form::store_bf16 &&
                  link.form != Form::store_f16 && link.form != Form::store_e4m3 && link.form != Form::store_iu4prep)
        if (!args.aux[link.arg]) return false;
    if constexpr (link.form == Form::clip_l2) return rows.width <= kThreads;
    if constexpr (link.form == Form::quant_a4had || link.form == Form::quant_i8rot || link.form == Form::rotate_hadamard)
        return rows.width % 128U == 0;
    if constexpr (link.form == Form::store_a8g32 || link.form == Form::quant_a8g32) return rows.width % 32U == 0;
    if constexpr (link.form == Form::blocksum) return rows.width % 16U == 0;
    return true;
}

}  // namespace detail

// Compiles one selected instance, with the same steps that a producer uses.
// Descriptors and their pointees must remain live through the stream operation.
// The caller provides valid device addresses and uniform descriptor geometry;
// e.g. Rope::half is in [1, width / 2]. Input/output may alias at equal strides.
template <unsigned R, class... Steps>
cudaError_t launch(List<Steps...> list, Rows rows, ChainArgs args, cudaStream_t stream) {
    static_assert(static_key<List<Steps...>>.has_value());
    static_assert(R >= 1 && R <= kMaxRegisters);
    if (rows.count == 0) return cudaSuccess;
    // A lone folded side-output step needs only its BF16 dot descriptor and row count.
    if constexpr (detail::side_output_only<Steps...>) {
        if (rows.output || rows.groups == 0 || rows.count > 0x7fffffffU ||
            !(detail::valid_binding<Steps>(rows, args) && ...)) return cudaErrorInvalidValue;
    } else {
        if (!rows.input || rows.width == 0 || rows.width > R * kThreads || rows.groups == 0 ||
            rows.count > 0x7fffffffU || rows.width > UINT32_MAX - rows.first_index ||
            (!rows.positions && (rows.count - 1U) / rows.groups > UINT32_MAX - rows.first_position) ||
            !(detail::valid_binding<Steps>(rows, args) && ...)) return cudaErrorInvalidValue;
        if (!rows.input_stride) rows.input_stride = rows.width;
        if (!rows.output_stride) rows.output_stride = rows.width;
        constexpr auto max_stride = SIZE_MAX / sizeof(float);
        const auto stride_limit = rows.count > 1 ? (max_stride - rows.width) / (rows.count - 1U) : max_stride;
        if (rows.input_stride < rows.width || rows.input_stride > stride_limit ||
            (rows.output && (rows.output_stride < rows.width || rows.output_stride > stride_limit)) ||
            (rows.input == rows.output && rows.input_stride != rows.output_stride)) return cudaErrorInvalidValue;
    }
    constexpr bool rope = ((Steps::link.form == Form::rope || Steps::link.form == Form::rope_inverse) || ...);
    constexpr bool row = static_key<List<Steps...>>->scope == Scope::row;
    const unsigned scratch = rope && rows.width > kThreads ? rows.width : row ? kThreads : 0U;
    detail::standalone<R><<<rows.count, kThreads, scratch * sizeof(float), stream>>>(list, rows, args);
    return cudaGetLastError();
}

using Enqueue = cudaError_t (*)(Rows, ChainArgs, cudaStream_t);
struct Instance {
    KernelKey key;
    Enqueue enqueue;
};

// Kernel generation emits only the lists and sizes required by the plans.
template <unsigned R, class ChainList>
inline constexpr Instance instance{
    *static_key<ChainList>,
    [](Rows rows, ChainArgs args, cudaStream_t stream) { return launch<R>(ChainList{}, rows, args, stream); }
};

}  // namespace strix::cuda::core::chain
