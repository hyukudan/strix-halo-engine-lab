#include "catalog_cuda_wr1.hpp"

#include "core/row_transforms.hpp"
#include "cuda/core/head_argmax.cuh"
#include "cuda/core/mix_cuda.cuh"
#include "cuda/core/step_rows.cuh"

#include <cstring>
#include <type_traits>

namespace strix::exec::cuda_decode::wr1 {
namespace {

namespace mix = core::residual_mix;
namespace pointwise = core::pointwise;
namespace transforms = core::row_transforms;
namespace argmax = ::strix::cuda::core::head_argmax;
namespace bank = ::strix::cuda::mix::detail;
using ::strix::cuda::core::StepRows;

template <unsigned Width>
__global__ void bank_kernel(mix::NormalizePairArgs pair, StepRows<Width> rows) {
    if (!rows.valid(blockIdx.x)) return;
    __shared__ float scratch[mix::detail::kThreads];
    mix::detail::normalize_rows(blockIdx.y == 0U ? pair.first : pair.second, scratch);
}

template <unsigned Width, unsigned Threads>
__global__ void clip_kernel(ops::norm::Bf16HeadClipArgs args, StepRows<Width> rows) {
    __shared__ float scratch[Threads];
    ops::norm::clip_body(args, scratch, rows);
}

template <class Transform>
struct Offset {
    Transform transform;
    std::size_t first;
    __device__ void operator()(std::size_t i) const { transform(first + i); }
};

template <unsigned Width, class Transform>
__global__ void pointwise_kernel(StepRows<Width> rows, std::size_t elements, Transform transform) {
    const unsigned row = blockIdx.y;
    if (!rows.valid(row)) return;
    pointwise::visit_elements<true>(elements, Offset<Transform>{transform, row * elements});
}

template <unsigned Width>
__global__ void relative_kernel(Bf16RelativeProjection16 a, StepRows<Width> rows) {
    const unsigned row = blockIdx.y;
    if (!rows.valid(row)) return;
    const auto elements = std::size_t{a.heads} * a.relative_extent;
    pointwise::visit_elements<true>(elements, transforms::ShortBf16Projection<16>{
        a.input + std::size_t{row} * a.heads * 16U, a.weights, a.output + row * elements, a.relative_extent});
}

struct EmitGreedy {
    RowOut* output;
    __device__ void operator()(unsigned row, const head_argmax::Pick2& pick) const {
        output[row] = {pick.first.nonfinite ? head_argmax::kNone : pick.first.index,
                       pick.first.value, pick.value2, RowStatus::picked};
    }
    __device__ void masked(unsigned row) const { output[row].status = RowStatus::masked; }
};

dim3 dimensions(const std::array<unsigned, 3>& axes) { return {axes[0], axes[1], axes[2]}; }

template <Leaf Which, unsigned Width, class Args>
void thunk(std::span<const std::byte> bytes, const Grid& shape, void* stream_handle) {
    static_assert(std::is_trivially_copyable_v<LaunchArgs<Args>>);
    if (bytes.size() != sizeof(LaunchArgs<Args>))
        throw std::invalid_argument("invalid WR1 CUDA launch arguments");
    LaunchArgs<Args> launch;
    std::memcpy(&launch, bytes.data(), sizeof(launch));
    const auto& a = launch.scalar;
    const StepRows<Width> rows{launch.descriptor};
    const auto grid = dimensions(shape.blocks), block = dimensions(shape.threads);
    const auto stream = static_cast<cudaStream_t>(stream_handle);
    cudaError_t status = cudaSuccess;
    if constexpr (Which == Leaf::dense_fp32) {
        status = cuda_core::dense_bf16_fp32_step(a.input, a.input_width, std::span(a.projections).first(a.count),
                                                rows.descriptor, Width, grid, stream);
    } else if constexpr (Which == Leaf::dense_mma3 || Which == Leaf::dense_mma3_round) {
        status = cuda_core::dense_bf16_mma3_step(a.input, a.input_width, std::span(a.projections).first(a.count),
                                                Which == Leaf::dense_mma3_round, rows.descriptor, grid, stream);
    } else if constexpr (Which == Leaf::rms) {
        status = cuda_core::rmsnorm_bf16_step(a.input, a.gain, a.output, a.width, a.epsilon, rows.descriptor, grid, stream);
    } else if constexpr (Which == Leaf::mix) {
        const auto enqueue = [&]<bool Parallel> {
            if (a.normalized) bank::bank_softmax_step<Width, true, Parallel><<<grid, block, 0, stream>>>(a, rows);
            else bank::bank_softmax_step<Width, false, Parallel><<<grid, block, 0, stream>>>(a, rows);
        };
        if (a.bank_count > 1U) enqueue.template operator()<true>();
        else enqueue.template operator()<false>();
    } else if constexpr (Which == Leaf::bank) {
        bank_kernel<<<grid, block, 0, stream>>>(mix::NormalizePairArgs{a, {}}, rows);
    } else if constexpr (Which == Leaf::pair) {
        bank_kernel<<<grid, block, 0, stream>>>(a, rows);
    } else if constexpr (Which == Leaf::clip) {
        const ops::norm::Bf16HeadClipArgs args{a.input, a.gain, a.output, Width, a.heads, a.width,
            a.epsilon, a.threshold, {0, a.scale_floor, a.scale_alpha}};
        if (block.x == 32U) clip_kernel<Width, 32><<<grid, block, 0, stream>>>(args, rows);
        else if (block.x == 128U) clip_kernel<Width, 128><<<grid, block, 0, stream>>>(args, rows);
        else clip_kernel<Width, 256><<<grid, block, 0, stream>>>(args, rows);
    } else if constexpr (Which == Leaf::relative) {
        relative_kernel<<<grid, block, 0, stream>>>(a, rows);
    } else if constexpr (Which == Leaf::greedy_partial || Which == Leaf::greedy_final) {
        const argmax::View view{a.logits, a.count, a.count, Width, 0U, a.chunks,
                                reinterpret_cast<head_argmax::Pick*>(a.partial)};
        if constexpr (Which == Leaf::greedy_partial)
            argmax::detail::partial_step_kernel<<<grid, block, 0, stream>>>(view, rows);
        else argmax::detail::final_step_kernel<<<grid, block, 0, stream>>>(view, EmitGreedy{a.output}, rows);
    } else {
        const auto enqueue = [&](auto transform) {
            pointwise_kernel<<<grid, block, 0, stream>>>(rows, a.elements, transform);
        };
        if constexpr (Which == Leaf::controller) enqueue(mix::detail::Controller{a});
        if constexpr (Which == Leaf::softcap) enqueue(transforms::Bf16Softcap{a.values, a.cap});
        if constexpr (Which == Leaf::round) enqueue(transforms::Bf16Round{a.values});
        if constexpr (Which == Leaf::sigmoid) enqueue(transforms::Bf16SigmoidProduct{a.input, a.logits, a.output});
        if constexpr (Which == Leaf::add) {
            if (a.round_bf16) enqueue(mix::detail::AddRound{a.lhs, a.rhs, a.output});
            else enqueue(pointwise::Add{a.lhs, a.rhs, a.output});
        }
    }
    if (status == cudaSuccess) status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <unsigned Width>
constexpr auto thunks() {
    using enum Leaf;
    return Thunks::value_type{
        thunk<dense_fp32, Width, Bf16DenseGroup>, thunk<dense_mma3, Width, Bf16DenseGroup>,
        thunk<dense_mma3_round, Width, Bf16DenseGroup>, thunk<mix, Width, core::residual_mix::MixArgs>,
        thunk<bank, Width, core::residual_mix::NormalizeArgs>, thunk<pair, Width, core::residual_mix::NormalizePairArgs>,
        thunk<controller, Width, core::residual_mix::ControllerArgs>, thunk<rms, Width, RmsBf16>,
        thunk<relative, Width, Bf16RelativeProjection16>, thunk<clip, Width, HeadClipBf16>,
        thunk<greedy_partial, Width, Greedy>, thunk<greedy_final, Width, Greedy>, thunk<softcap, Width, Bf16Softcap>,
        thunk<round, Width, Bf16Round>, thunk<sigmoid, Width, Bf16SigmoidProduct>, thunk<add, Width, core::residual_mix::AddArgs>
    };
}

}  // namespace

const Thunks& native_thunks() {
    static constexpr Thunks table{thunks<1>(), thunks<2>(), thunks<4>(), thunks<8>(), thunks<16>()};
    return table;
}

}  // namespace strix::exec::cuda_decode::wr1
