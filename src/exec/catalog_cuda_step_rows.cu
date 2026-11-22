#include "catalog_cuda_step_rows.hpp"

#include "cuda/core/head_argmax.cuh"
#include "strix/cuda_core/attention.hpp"
#include "strix/cuda_core/dense.hpp"
#include "strix/cuda_core/gdn.hpp"
#include "strix/cuda_core/norms.hpp"
#include "strix/cuda_core/rope.hpp"
#include "strix/device/runtime.hpp"

#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace strix::exec::cuda_step_rows {
namespace {

namespace cc = cuda_core;
namespace hp = strix::cuda::core::head_argmax;

cc::DenseWeight weight(const Matrix& w) {
    switch (w.format) {
    case Format::bf16: return {w.data, cc::WeightFormat::bf16, w.out, w.in};
    case Format::q8: return {w.data, cc::WeightFormat::q8_0, w.out, w.in};
    case Format::f32: return {w.data, cc::WeightFormat::f32, w.out, w.in};
    default: throw std::invalid_argument("unsupported CUDA row matrix");
    }
}

cc::GroupedView view(const View& v) { return {1, v.groups, v.width, v.row_stride, v.group_stride}; }

cudaError_t enqueue(const Embedding& a, cudaStream_t stream) {
    return cc::embedding_rows(weight(a.weight), a.ids, a.output, 1, stream);
}

cudaError_t enqueue(const Rms& a, cudaStream_t stream) {
    return cc::rmsnorm(a.input, view(a.input_view), a.gain, a.output, view(a.output_view), a.epsilon, stream);
}

cudaError_t enqueue(const Dense& a, cudaStream_t stream) {
    std::array<cc::DenseWeight, 4> weights;
    for (std::size_t i = 0; i < a.count; ++i) weights[i] = weight(a.weights[i]);
    const auto arith = a.mma ? cc::DenseArith::mma3 : cc::DenseArith::fp32;
    if (a.up) {
        auto up = weights.front();
        up.data = a.up;
        return cc::dense_gate_up_silu_rows(weights.front(), up, a.input, a.outputs.front(), 1, stream, arith);
    }
    return cc::dense_rows_multi(weights.data(), a.outputs.data(), static_cast<int>(a.count), a.input, 1,
                               a.residual, stream, arith);
}

cudaError_t enqueue(const Rope& a, cudaStream_t stream) {
    return cc::rope_apply(a.image, view(a.view), a.rotary_width, a.cosines, a.sines, a.position, stream);
}

cudaError_t enqueue(const KvStore& a, cudaStream_t stream) {
    const auto& s = a.shape;
    const cc::AttentionDims dims{s.heads, s.kv_heads, s.width, s.scale, s.capacity};
    return cc::kv_store(dims, a.key, a.value, static_cast<__nv_bfloat16*>(a.key_cache),
                        static_cast<__nv_bfloat16*>(a.value_cache), a.position, 1, stream);
}

cudaError_t enqueue(const Gdn& a, GdnPhase phase, cudaStream_t stream) {
    const cc::GdnDims dims{a.key_heads, a.key_width, a.value_heads, a.value_width, a.conv_taps};
    const cc::GdnLayer layer{a.convolution_weights, a.ssm_a, a.time_bias, a.gain};
    const cc::GdnState state{a.convolution_state, a.recurrent_state};
    const cc::GdnWindow window{a.raw, a.prepared, a.alpha, a.beta};
    if (phase == GdnPhase::convolution)
        return cc::gdn_window_convolution(dims, layer, state, window, 1, stream);
    return cc::gdn_window_recurrence(dims, layer, state, window, a.z, a.output, 1, a.epsilon, a.parity, stream);
}

cudaError_t enqueue(const Attention& a, cc::AttentionDecodePhase phase, cudaStream_t stream) {
    const auto& s = a.shape;
    const cc::AttentionDims dims{s.heads, s.kv_heads, s.width, s.scale, s.capacity};
    return cc::attention_decode_row_phase(dims, a.query, view(a.query_view), a.gate, view(a.gate_view),
        static_cast<const __nv_bfloat16*>(a.key_cache), static_cast<const __nv_bfloat16*>(a.value_cache),
        a.position, a.output, a.scratch, phase, stream);
}

hp::View pick_view(const Pick& a) {
    hp::View result;
    result.logits = a.logits;
    result.count = a.vocabulary;
    result.rows = 1;
    result.partial = static_cast<head_argmax::Pick*>(a.workspace);
    return result;
}

template <class Args, Leaf leaf>
void thunk(std::span<const std::byte> bytes, const Grid&, void* stream) {
    static_assert(std::is_trivially_copyable_v<Args>);
    if (bytes.size() != sizeof(Args)) throw std::invalid_argument("invalid CUDA row argument block");
    Args args;
    std::memcpy(&args, bytes.data(), sizeof(args));
    const auto queue = static_cast<cudaStream_t>(stream);
    cudaError_t status;
    if constexpr (leaf == Leaf::pick_partial) {
        status = hp::pick_partial<hp::Launch::pdl>(pick_view(args), queue);
    } else if constexpr (leaf == Leaf::pick_final) {
        status = hp::pick_final<hp::Launch::pdl>(pick_view(args), hp::EmitIndex{args.ids}, queue);
    } else if constexpr (leaf == Leaf::gdn_convolution) {
        status = enqueue(args, GdnPhase::convolution, queue);
    } else if constexpr (leaf == Leaf::gdn_recurrence) {
        status = enqueue(args, GdnPhase::recurrence, queue);
    } else if constexpr (leaf == Leaf::attention_scores) {
        status = enqueue(args, cc::AttentionDecodePhase::scores, queue);
    } else if constexpr (leaf == Leaf::attention_values) {
        status = enqueue(args, cc::AttentionDecodePhase::values, queue);
    } else if constexpr (leaf == Leaf::attention_combine) {
        status = enqueue(args, cc::AttentionDecodePhase::combine, queue);
    } else {
        status = enqueue(args, queue);
    }
    device::native::check(status, "enqueue bound CUDA row leaf");
}

}  // namespace

const NativeThunks& native_thunks() {
    constexpr PdlLeaf waited{PdlWait::before_step_io};
    constexpr PdlLeaf no_wait{PdlWait::absent, PdlOptOut::kernel_without_wait};
    static constexpr NativeThunks thunks{{
        {thunk<Embedding, Leaf::embedding>, waited},
        {thunk<Rms, Leaf::rms>, waited},
        {thunk<Dense, Leaf::dense>, waited},
        {thunk<Rope, Leaf::rope>, waited},
        {thunk<KvStore, Leaf::kv_store>, waited},
        {thunk<Pick, Leaf::pick_partial>, waited},
        {thunk<Pick, Leaf::pick_final>, waited},
        {thunk<Gdn, Leaf::gdn_convolution>, no_wait},
        {thunk<Gdn, Leaf::gdn_recurrence>, no_wait},
        {thunk<Attention, Leaf::attention_scores>, waited},
        {thunk<Attention, Leaf::attention_values>, waited},
        {thunk<Attention, Leaf::attention_combine>, waited},
    }};
    return thunks;
}

}  // namespace strix::exec::cuda_step_rows
