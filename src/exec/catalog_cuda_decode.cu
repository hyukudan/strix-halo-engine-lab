#include "catalog_cuda_decode.hpp"

#include "core/row_transforms.hpp"
#include "cuda/core/head_argmax.cuh"
#include "cuda_core/dense/bf16_rows.hpp"
#include "strix/cuda_core/attention.hpp"
#include "strix/cuda_core/dense.hpp"
#include "strix/cuda_core/norms.hpp"
#include "strix/device/runtime.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace strix::exec::cuda_decode {
namespace {

namespace residual = core::residual_mix;
namespace transforms = core::row_transforms;
namespace cuda = cuda_core;
namespace packed = moe_core::microscaled_r1;
namespace argmax = ::strix::cuda::core::head_argmax;
using weight::microscaled::MatrixFormat;

cudaError_t enqueue(const Embedding& a, cudaStream_t stream) {
    return cuda::embedding_rows({a.table, cuda::WeightFormat::bf16, a.vocabulary, a.width}, a.token, a.output, 1, stream);
}

cudaError_t enqueue(const RmsBf16& a, cudaStream_t stream) {
    const cuda::GroupedView view{1, a.groups, a.width};
    return cuda::rmsnorm_bf16(a.input, view, a.gain, a.output, view, a.epsilon, stream);
}

cudaError_t enqueue(const AppendKvBf16& a, cudaStream_t stream) {
    const cuda::AttentionDims dims{a.heads, a.kv_heads, a.width, a.score_scale, a.capacity};
    return cuda::kv_store_prepared_bf16(dims, a.key, a.value, reinterpret_cast<__nv_bfloat16*>(a.key_cache),
        reinterpret_cast<__nv_bfloat16*>(a.value_cache), a.first_position, 1, a.wrapped, stream, a.position);
}

constexpr unsigned prepared_head_width = 128;

cudaError_t enqueue(const PreparedFlashBf16& a, cudaStream_t stream) {
    const cuda::AttentionDims dims{a.heads, a.kv_heads, prepared_head_width, a.score_scale, a.capacity};
    const cuda::PreparedBf16Flash prepared{a.relative_bias, a.sinks, a.relative_extent, a.window, a.wrapped};
    return cuda::attention_prefill_prepared_bf16(dims, prepared, a.query,
        reinterpret_cast<const __nv_bfloat16*>(a.key_cache), reinterpret_cast<const __nv_bfloat16*>(a.value_cache),
        a.first_position, 1, a.output, stream, cuda::FlashPipe::pad2, a.position);
}

cudaError_t enqueue(const PreparedFinite& a, cudaStream_t stream) {
    return attn_core::enqueue_validate_prepared_finite(
        a.rows, a.heads, a.kv_heads, a.width, a.relative_extent, a.failure, stream);
}

core::causal_conv::Args convolution_args(const CausalConvolution& a) {
    return {1, a.input, a.static_weights, a.dynamic_weights, a.history, a.output};
}

cudaError_t enqueue(const CausalConvolution& a, cudaStream_t stream) {
    const auto args = convolution_args(a);
    if (a.round_bf16) return core::causal_conv::launch_with_store<true>(a.plan, args, stream);
    return core::causal_conv::launch_with_store<false>(a.plan, args, stream);
}

cudaError_t enqueue(const HeadClipBf16& a, cudaStream_t stream) {
    const transforms::Bf16HeadClipArgs args{a.input, a.gain, a.output, 1, a.heads, a.width,
        a.epsilon, a.threshold, {a.first_position, a.scale_floor, a.scale_alpha}};
    return transforms::l2_clip_bf16(args, stream, a.position);
}

cudaError_t enqueue(const residual::NormalizeArgs& a, cudaStream_t stream) {
    return residual::launch_normalize(a, stream);
}

cudaError_t enqueue(const residual::MixArgs& a, cudaStream_t stream) {
    return residual::launch_mix(a, stream);
}

cudaError_t enqueue(const residual::AddArgs& a, cudaStream_t stream) {
    return residual::launch_add(a, stream);
}

cudaError_t enqueue(const residual::ControllerArgs& a, cudaStream_t stream) {
    return residual::launch_controller(a, stream);
}

cudaError_t enqueue(const residual::NormalizePairArgs& a, cudaStream_t stream) {
    return residual::launch_normalize_pair(a, stream);
}

cudaError_t enqueue(const PackedGateUp& a, cudaStream_t stream) {
    if (a.format == MatrixFormat::nvfp4)
        return packed::launch_gate_up<MatrixFormat::nvfp4, kPackedExpertsPerToken>(a.args, stream);
    return packed::launch_gate_up<MatrixFormat::mxfp8, kPackedExpertsPerToken>(a.args, stream);
}

cudaError_t enqueue(const PackedDownCombine& a, cudaStream_t stream) {
    if (a.format == MatrixFormat::nvfp4)
        return packed::launch_down_combine<MatrixFormat::nvfp4, kPackedExpertsPerToken>(a.args, stream);
    return packed::launch_down_combine<MatrixFormat::mxfp8, kPackedExpertsPerToken>(a.args, stream);
}

argmax::View greedy_view(const GreedyPartial& a) {
    return {a.logits, 0U, a.count, 1U, 0U, a.chunks, a.partial};
}

cudaError_t enqueue(const GreedyPartial& a, cudaStream_t stream) {
    return argmax::pick_partial(greedy_view(a), stream);
}

cudaError_t enqueue(const GreedyFinal& a, cudaStream_t stream) {
    if (!a.output) return cudaErrorInvalidValue;
    return argmax::pick_final(greedy_view(a.reduction), argmax::EmitIndexOrNone{a.output}, stream);
}

cudaError_t enqueue(const Bf16Round& a, cudaStream_t stream) {
    return transforms::elements(a.elements, transforms::Bf16Round{a.values}, stream);
}

cudaError_t enqueue(const Bf16SigmoidProduct& a, cudaStream_t stream) {
    return transforms::elements(a.elements, transforms::Bf16SigmoidProduct{a.input, a.logits, a.output}, stream);
}

cudaError_t enqueue(const Bf16Softcap& a, cudaStream_t stream) {
    return transforms::elements(a.elements, transforms::Bf16Softcap{a.values, a.cap}, stream);
}

cudaError_t enqueue(const Bf16RelativeProjection16& a, cudaStream_t stream) {
    return transforms::elements(std::size_t{a.heads} * a.relative_extent,
        transforms::ShortBf16Projection<16>{a.input, a.weights, a.output, a.relative_extent}, stream);
}

cudaError_t enqueue(const Bf16DenseGroup& a, cudaStream_t stream) {
    static_assert(std::tuple_size_v<decltype(a.projections)> == cuda::kMaxBf16RowProjections);
    if (a.count == 0 || a.count > a.projections.size()) return cudaErrorInvalidValue;
    cuda::DenseWeight weights[cuda::kMaxBf16RowProjections];
    float* output[cuda::kMaxBf16RowProjections];
    for (unsigned i = 0; i < a.count; ++i) {
        const auto& projection = a.projections[i];
        weights[i] = {projection.weights, cuda::WeightFormat::bf16, projection.outputs, a.input_width};
        output[i] = projection.output;
    }
    const auto n = static_cast<int>(a.count);
    if (a.mode == Bf16DenseMode::mma3_round)
        return cuda::dense_bf16_mma3_rows_round(weights, output, n, a.input, 1, stream);
    const auto arithmetic = a.mode == Bf16DenseMode::mma3 ? cuda::DenseArith::mma3 : cuda::DenseArith::fp32;
    return cuda::dense_rows_multi(weights, output, n, a.input, 1, nullptr, stream, arithmetic);
}

template <class Args>
void thunk(std::span<const std::byte> bytes, const Grid&, void* stream) {
    static_assert(std::is_trivially_copyable_v<Args>);
    if (bytes.size() != sizeof(Args)) throw std::invalid_argument("invalid CUDA decode leaf arguments");
    Args args{};
    std::memcpy(&args, bytes.data(), sizeof(args));
    device::native::check(enqueue(args, static_cast<cudaStream_t>(stream)), "enqueue CUDA decode leaf");
}

template <class Args>
int catalogue_enqueue(void* binding, void* stream) {
    return enqueue(*static_cast<const Args*>(binding), static_cast<cudaStream_t>(stream));
}

template <class Args, Bf16DenseMode... Modes>
EntryLaunch bind_site(const LaunchSite& site, EntryBinder&) {
    if (site.rows != 1 || !site.backend_args) return {};
    if constexpr (sizeof...(Modes) != 0) {
        const auto mode = static_cast<const Args*>(site.backend_args)->mode;
        if (!((mode == Modes) || ...)) return {};
    }
    return {catalogue_enqueue<Args>, const_cast<void*>(site.backend_args)};
}

EntryCost unknown_cost(const LaunchSite& site, const device::Profile&) {
    return {std::numeric_limits<double>::infinity(), site.weight_bytes, site.state_bytes, site.macs, 1};
}

void workspace(const LaunchSite& site, WorkspaceNeed& need) { need = site.scratch; }

constexpr LauncherRef embedding{"strix::cuda_core::embedding_rows", "src/cuda_core/dense/dense.cu", "bf16", 1};
constexpr LauncherRef rms{"strix::cuda_core::rmsnorm_bf16", "src/cuda_core/norms/norms.cu", "r1", 1};
constexpr LauncherRef append_kv{"strix::cuda_core::kv_store_prepared_bf16", "src/cuda_core/attention/attention.cu", "step", 1};
constexpr LauncherRef prepared_flash{"strix::cuda_core::attention_prefill_prepared_bf16",
    "src/cuda_core/attention/attention_flash.cu", "step.pad2", 1};
constexpr LauncherRef finite{"strix::attn_core::enqueue_validate_prepared_finite",
    "src/attn_core/prepared_gqa_attention.hpp", "r1", 1};
constexpr LauncherRef convolution{"strix::core::causal_conv::launch_with_store", "src/core/short_convolution.hpp", "r1", 1};
constexpr LauncherRef clip{"strix::ops::norm::l2_clip_bf16", "src/ops/norm/norm_device.hpp", "r1", 1};
constexpr LauncherRef normalize{"strix::core::residual_mix::launch_normalize", "src/core/residual_mix.hpp", "r1", 1};
constexpr LauncherRef mix{"strix::core::residual_mix::launch_mix", "src/core/residual_mix.hpp", "r1", 1};
constexpr LauncherRef add{"strix::core::residual_mix::launch_add", "src/core/residual_mix.hpp", "r1", 1};
constexpr LauncherRef controller{"strix::core::residual_mix::launch_controller", "src/core/residual_mix.hpp", "r1", 1};
constexpr LauncherRef round_bf16{"strix::core::pointwise::elements", "src/core/pointwise.hpp", "bf16_round", 1};
constexpr LauncherRef sigmoid{"strix::core::pointwise::elements", "src/core/pointwise.hpp", "bf16_sigmoid_product", 1};
constexpr LauncherRef softcap{"strix::core::pointwise::elements", "src/core/pointwise.hpp", "bf16_softcap", 1};
constexpr LauncherRef relative{"strix::core::pointwise::elements", "src/core/row_transforms.hpp", "relative_bf16_d16", 1};
constexpr LauncherRef dense_fp32{"strix::cuda_core::dense_rows_multi", "src/cuda_core/dense/dense.cu", "fp32", 1};
constexpr LauncherRef dense_mma3{"strix::cuda_core::dense_rows_multi", "src/cuda_core/dense/dense.cu", "mma3", 1};
constexpr LauncherRef dense_round{"strix::cuda_core::dense_bf16_mma3_rows_round", "src/cuda_core/dense/dense_mma.cu", "mma3_round", 1};
constexpr LauncherRef pair_norm{"strix::core::residual_mix::launch_normalize_pair", "src/core/residual_mix.hpp", "r1", 1};
constexpr LauncherRef gate_up{"strix::moe_core::microscaled_r1::launch_gate_up", "src/moe_core/microscaled_r1.hpp", "r1.k8", 1};
constexpr LauncherRef down_combine{"strix::moe_core::microscaled_r1::launch_down_combine", "src/moe_core/microscaled_r1.hpp", "r1.k8", 1};
constexpr LauncherRef pick_part{"strix::cuda::core::head_argmax::pick_partial", "src/cuda/core/head_argmax.cuh", "plain", 1};
constexpr LauncherRef pick_final{"strix::cuda::core::head_argmax::pick_final", "src/cuda/core/head_argmax.cuh", "plain.index_or_none", 1};

template <class Args, OperandClass Act = OperandClass::f32_fma, Bf16DenseMode... Modes>
constexpr KernelEntry entry(graph::Op op, FormId form, weight::FormatId format,
                            std::string_view identity, const LauncherRef& launcher) {
    return {.key = {op, form, format, Act, RowsClass::r1},
            .arch = device::ArchMask::sm120, .identity = identity, .exact = lever::Exactness::bitwise,
            .cost = unknown_cost, .workspace = workspace, .bind = bind_site<Args, Modes...>, .launcher = &launcher};
}

template <class Args, MatrixFormat Format>
EntryLaunch bind_packed_site(const LaunchSite& site, EntryBinder& binder) {
    if (!site.backend_args || static_cast<const Args*>(site.backend_args)->format != Format) return {};
    return bind_site<Args>(site, binder);
}

template <class Args, MatrixFormat Format>
constexpr KernelEntry packed_entry(graph::Op op, FormId form, std::string_view identity, const LauncherRef& launcher) {
    constexpr weight::FormatId format = Format == MatrixFormat::nvfp4
        ? weight::FormatId{"nvidia.nvfp4", 1} : weight::FormatId{"block.mxfp8.e4m3-e8m0.g32", 1};
    auto instance = entry<Args>(op, form, format, identity, launcher);
    instance.bind = bind_packed_site<Args, Format>;
    return instance;
}

enum Leaf { embed, rms_bf16, kv_store, flash, finite_check, conv, head_clip, bank_norm, bank_mix,
            row_add, blend, round, gate, head_softcap, relative_d16, linear_fp32, linear_mma3, linear_mma3_round,
            controller_pair, gate_up_nvfp4, gate_up_mxfp8, down_nvfp4, down_mxfp8, greedy_part, greedy_final };
constexpr KernelEntry catalogue[] = {
    entry<Embedding>(graph::Op::gather, "embedding.bf16_to_f32", {"plain.bf16", 1}, "cuda-sm120-embedding-bf16-r1-v1", embedding),
    entry<RmsBf16>(graph::Op::norm, "rms.bf16_gain.f32_image", {"plain.bf16", 1}, "cuda-sm120-rms-bf16-image-r1-v1", rms),
    entry<AppendKvBf16>(graph::Op::store, "prepared.kv.bf16", {}, "cuda-sm120-prepared-bf16-kv-step-v1", append_kv),
    entry<PreparedFlashBf16>(graph::Op::attend, "prepared.flash.pad2.d128", {}, "cuda-sm120-prepared-flash-pad2-d128-v1", prepared_flash),
    entry<PreparedFinite>(graph::Op::elementwise, "prepared.finite_check", {}, "cuda-sm120-prepared-finite-check-v1", finite),
    entry<CausalConvolution>(graph::Op::elementwise, "causal_convolution.decode", {}, "cuda-sm120-causal-convolution-r1-v1", convolution),
    entry<HeadClipBf16>(graph::Op::norm, "head_clip.bf16", {}, "cuda-sm120-head-clip-bf16-r1-v1", clip),
    entry<residual::NormalizeArgs>(graph::Op::norm, "residual.bank", {}, "cuda-sm120-residual-bank-norm-r1-v1", normalize),
    entry<residual::MixArgs>(graph::Op::mix, "residual.bank", {}, "cuda-sm120-residual-bank-mix-r1-v1", mix),
    entry<residual::AddArgs>(graph::Op::combine, "residual.add.out_of_place", {}, "cuda-sm120-residual-add-r1-v1", add),
    entry<residual::ControllerArgs>(graph::Op::combine, "residual.controller", {}, "cuda-sm120-residual-controller-r1-v1", controller),
    entry<Bf16Round>(graph::Op::elementwise, "bf16.round", {}, "cuda-sm120-pointwise-bf16-round-v1", round_bf16),
    entry<Bf16SigmoidProduct>(graph::Op::elementwise, "bf16.sigmoid_product", {}, "cuda-sm120-pointwise-bf16-sigmoid-product-v1", sigmoid),
    entry<Bf16Softcap>(graph::Op::elementwise, "bf16.softcap", {}, "cuda-sm120-pointwise-bf16-softcap-v1", softcap),
    entry<Bf16RelativeProjection16>(graph::Op::linear, "relative.bf16.d16", {"plain.bf16", 1},
        "cuda-sm120-relative-bf16-d16-v1", relative),
    entry<Bf16DenseGroup, OperandClass::f32_fma, Bf16DenseMode::fp32>(graph::Op::linear, "r1.bf16_image.group", {"plain.bf16", 1},
        "cuda-sm120-bf16-r1-fp32-group-v1", dense_fp32),
    entry<Bf16DenseGroup, OperandClass::bf16_mma, Bf16DenseMode::mma3>(graph::Op::linear, "r1.bf16_image.group", {"plain.bf16", 1},
        "cuda-sm120-bf16-r1-mma3-group-v1", dense_mma3),
    entry<Bf16DenseGroup, OperandClass::bf16_mma, Bf16DenseMode::mma3_round>(graph::Op::linear, "r1.bf16_image.group_round", {"plain.bf16", 1},
        "cuda-sm120-bf16-r1-mma3-group-round-v1", dense_round),
    entry<residual::NormalizePairArgs>(graph::Op::norm, "residual.controller_pair", {},
        "cuda-sm120-controller-pair-norm-r1-v1", pair_norm),
    packed_entry<PackedGateUp, MatrixFormat::nvfp4>(graph::Op::linear, "microscaled.packed_gate_up.k8",
        "cuda-sm120-packed-nvfp4-gate-up-r1-k8-v1", gate_up),
    packed_entry<PackedGateUp, MatrixFormat::mxfp8>(graph::Op::linear, "microscaled.packed_gate_up.k8",
        "cuda-sm120-packed-mxfp8-gate-up-r1-k8-v1", gate_up),
    packed_entry<PackedDownCombine, MatrixFormat::nvfp4>(graph::Op::combine, "microscaled.packed_down_combine.k8",
        "cuda-sm120-packed-nvfp4-down-combine-r1-k8-v1", down_combine),
    packed_entry<PackedDownCombine, MatrixFormat::mxfp8>(graph::Op::combine, "microscaled.packed_down_combine.k8",
        "cuda-sm120-packed-mxfp8-down-combine-r1-k8-v1", down_combine),
    entry<GreedyPartial>(graph::Op::argmax, "greedy.partial", {}, "cuda-sm120-greedy-partial-r1-v1", pick_part),
    entry<GreedyFinal>(graph::Op::argmax, "greedy.final", {}, "cuda-sm120-greedy-final-r1-v1", pick_final),
};

template <class Args>
KernelBinding bind_leaf(Leaf leaf, Grid grid, const Args& args) {
    KernelBinding bound;
    bound.fn = thunk<Args>;
    bound.identity = catalogue[leaf].identity;
    bound.grid = grid;
    bound.predicted.launches = 1;
    const auto bytes = std::as_bytes(std::span(&args, 1));
    bound.args.assign(bytes.begin(), bytes.end());
    return bound;
}

Grid pointwise_grid(std::size_t elements) {
    if (elements == 0 || elements > std::numeric_limits<std::size_t>::max() / sizeof(float))
        throw std::invalid_argument("invalid CUDA decode pointwise size");
    const auto blocks = static_cast<unsigned>(std::min<std::size_t>((elements - 1U) / 256U + 1U, 65535U));
    return {{blocks, 1, 1}, {256, 1, 1}, 0};
}

}  // namespace

std::span<const KernelEntry> entries() noexcept { return catalogue; }

KernelBinding bind(const Embedding& a) {
    return bind_leaf(embed, {{1, 1, 1}, {256, 1, 1}, 0}, a);
}

KernelBinding bind(const RmsBf16& a) {
    return bind_leaf(rms_bf16, {{a.groups, 1, 1}, {256, 1, 1}, 0}, a);
}

KernelBinding bind(const AppendKvBf16& a) {
    if (!a.position) throw std::invalid_argument("missing device position for CUDA KV step");
    return bind_leaf(kv_store, {{a.kv_heads, 1, 1}, {std::min(a.width, 256U), 1, 1}, 0}, a);
}

KernelBinding bind(const PreparedFlashBf16& a) {
    if (!a.position) throw std::invalid_argument("missing device position for CUDA prepared flash step");
    // One position-driven CTA per head. PadLayoutShape<128> in attention_flash.cu
    // reserves two 32-key stages, with 32 and 16 padding bytes for K and V.
    constexpr auto shared = 2U * 32U * (4U * prepared_head_width + 32U + 16U);
    return bind_leaf(flash, {{a.heads, 1, 1}, {256, 1, 1}, shared}, a);
}

KernelBinding bind(const PreparedFinite& a) {
    const auto stride = std::max(a.width, a.relative_extent);
    if (a.rows.count != 1 || !a.failure || a.heads == 0 || stride == 0 ||
        a.heads > std::numeric_limits<std::size_t>::max() / stride)
        throw std::invalid_argument("invalid CUDA R1 prepared finite check");
    return bind_leaf(finite_check, pointwise_grid(std::size_t{a.heads} * stride), a);
}

KernelBinding bind(const CausalConvolution& a) {
    const auto args = convolution_args(a);
    if (!core::causal_conv::valid(a.plan, args)) throw std::invalid_argument("invalid CUDA R1 convolution");
    const auto grid = core::causal_conv::launch_plan(a.plan, args);
    return bind_leaf(conv, {{grid.channel_blocks, grid.row_blocks, 1}, {grid.threads, 1, 1}, 0}, a);
}

KernelBinding bind(const HeadClipBf16& a) {
    if (a.heads == 0 || a.width == 0 || a.width > 256) throw std::invalid_argument("invalid CUDA R1 head clip");
    const auto threads = a.width <= 32 ? 32U : a.width <= 128 ? 128U : 256U;
    return bind_leaf(head_clip, {{a.heads, 1, 1}, {threads, 1, 1}, 0}, a);
}

KernelBinding bind(const residual::NormalizeArgs& a) {
    if (a.rows != 1 || !residual::fits(a)) throw std::invalid_argument("invalid CUDA R1 residual norm");
    return bind_leaf(bank_norm, {{1, 1, 1}, {residual::detail::kThreads, 1, 1}, 0}, a);
}

KernelBinding bind(const residual::MixArgs& a) {
    if (a.rows != 1 || !residual::fits(a)) throw std::invalid_argument("invalid CUDA R1 residual mix");
    const auto threads = a.bank_count > 1 ? residual::detail::kPairedBankThreads : residual::detail::kThreads;
    return bind_leaf(bank_mix, {{1, 1, 1}, {threads, a.bank_count, 1}, 0}, a);
}

KernelBinding bind(const residual::AddArgs& a) {
    return bind_leaf(row_add, pointwise_grid(a.elements), a);
}

KernelBinding bind(const residual::ControllerArgs& a) {
    return bind_leaf(blend, pointwise_grid(a.elements), a);
}

KernelBinding bind(const Bf16Round& a) {
    return bind_leaf(round, pointwise_grid(a.elements), a);
}

KernelBinding bind(const Bf16SigmoidProduct& a) {
    return bind_leaf(gate, pointwise_grid(a.elements), a);
}

KernelBinding bind(const Bf16Softcap& a) {
    return bind_leaf(head_softcap, pointwise_grid(a.elements), a);
}

KernelBinding bind(const Bf16RelativeProjection16& a) {
    if (a.relative_extent == 0 || a.heads > std::numeric_limits<std::size_t>::max() / a.relative_extent)
        throw std::invalid_argument("invalid CUDA R1 relative projection");
    return bind_leaf(relative_d16, pointwise_grid(std::size_t{a.heads} * a.relative_extent), a);
}

KernelBinding bind(const Bf16DenseGroup& a) {
    if (a.count == 0 || a.count > a.projections.size()) throw std::invalid_argument("invalid CUDA R1 dense group size");
    const auto projections = std::span(a.projections).first(a.count);
    const bool mma = a.mode != Bf16DenseMode::fp32;
    const auto arithmetic = mma ? cuda::DenseArith::mma3 : cuda::DenseArith::fp32;
    if (!cuda::detail::valid_bf16_dense_row_group({a.input, a.input_width, projections}, 1, arithmetic))
        throw std::invalid_argument("invalid CUDA R1 dense group operands");
    const bool round_output = a.mode == Bf16DenseMode::mma3_round;
    std::uint32_t outputs = 0;
    for (const auto& projection : projections) {
        if (projection.round_bf16 != round_output || (mma && !cuda::detail::bf16_mma3_tile(a.input_width, projection.outputs)))
            throw std::invalid_argument("CUDA R1 dense group does not select one native launch");
        outputs += projection.outputs;
    }
    if (mma) {
        const auto leaf = round_output ? linear_mma3_round : linear_mma3;
        return bind_leaf(leaf, {{outputs / core::rows::kTileOut, 1, 1}, {256, 1, 1}, 0}, a);
    }
    // dense.cu by_rows<bf16, false> uses one output per warp at R1.
    if (outputs > std::numeric_limits<std::uint32_t>::max() - 7U)
        throw std::invalid_argument("CUDA R1 dense group exceeds the FP32 launch range");
    return bind_leaf(linear_fp32, {{(outputs + 7U) / 8U, 1, 1}, {256, 1, 1}, 0}, a);
}

KernelBinding bind(const residual::NormalizePairArgs& a) {
    if (a.first.rows != 1U || a.second.rows != 1U || !residual::fits(a.first) || !residual::fits(a.second))
        throw std::invalid_argument("invalid CUDA R1 controller pair norm");
    return bind_leaf(controller_pair, {{1, 2, 1}, {residual::detail::kThreads, 1, 1}, 0}, a);
}

namespace {

bool fits_packed(const packed::Args& args, MatrixFormat format) {
    switch (format) {
    case MatrixFormat::nvfp4: return packed::fits<MatrixFormat::nvfp4, kPackedExpertsPerToken>(args);
    case MatrixFormat::mxfp8: return packed::fits<MatrixFormat::mxfp8, kPackedExpertsPerToken>(args);
    }
    return false;
}

std::uint32_t greedy_chunks(const GreedyPartial& a) {
    const auto chunks = a.chunks != 0U ? a.chunks : head_argmax::default_chunks(a.count);
    if (!a.logits || !a.partial || a.count == 0U || chunks > head_argmax::kMaxChunks)
        throw std::invalid_argument("invalid CUDA R1 greedy reduction");
    return chunks;
}

}  // namespace

KernelBinding bind(const PackedGateUp& a) {
    if (!fits_packed(a.args, a.format)) throw std::invalid_argument("invalid CUDA R1 packed gate-up");
    const auto leaf = a.format == MatrixFormat::nvfp4 ? gate_up_nvfp4 : gate_up_mxfp8;
    const auto blocks = packed::gate_blocks(a.args.intermediate, packed::kCudaGeometry.gate_columns);
    return bind_leaf(leaf, {{blocks, kPackedExpertsPerToken, 1}, {packed::detail::kThreads, 1, 1}, 0}, a);
}

KernelBinding bind(const PackedDownCombine& a) {
    if (!fits_packed(a.args, a.format)) throw std::invalid_argument("invalid CUDA R1 packed down-combine");
    const auto leaf = a.format == MatrixFormat::nvfp4 ? down_nvfp4 : down_mxfp8;
    const auto blocks = packed::down_blocks(a.args.hidden, packed::kCudaGeometry.down_columns);
    return bind_leaf(leaf, {{blocks, 1, 1}, {packed::detail::kThreads, 1, 1}, 0}, a);
}

KernelBinding bind(const GreedyPartial& a) {
    return bind_leaf(greedy_part, {{greedy_chunks(a), 1, 1}, {argmax::detail::kThreads, 1, 1}, 0}, a);
}

KernelBinding bind(const GreedyFinal& a) {
    (void)greedy_chunks(a.reduction);
    if (!a.output) throw std::invalid_argument("missing CUDA R1 greedy output");
    return bind_leaf(greedy_final, {{1, 1, 1}, {argmax::detail::kThreads, 1, 1}, 0}, a);
}

}  // namespace strix::exec::cuda_decode
