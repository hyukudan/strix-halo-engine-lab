#include "catalog_cuda_decode.hpp"

#include "core/row_transforms.hpp"
#include "cuda/core/head_argmax.cuh"
#include "cuda/core/head_screened.cuh"
#include "cuda/core/dense/bf16_rows.hpp"
#include "cuda/core/dense/dense_mma.hpp"
#include "cuda/core/dense/rows_bf16_narrow.cuh"
#include "cuda/core/mix_cuda.cuh"
#include "cuda/core/mix_cuda_latency.cuh"
#include "cuda/core/moe_r1_pipe.cuh"
#include "cuda/core/mma_tile.cuh"
#include "cuda/core/dense/rows_bf16_epilogue.cuh"
#include "cuda/core/lookup_catalog.hpp"
#include "strix/cuda_core/attention.hpp"
#include "strix/cuda_core/dense.hpp"
#include "strix/cuda_core/norms.hpp"
#include "strix/device/runtime.hpp"

#include <algorithm>
#include <cmath>
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
namespace grouped_mma = ::strix::cuda::core::mma::microscaled;
namespace grouped_rows = moe_core::grouped_rows;
namespace lookup = ::strix::cuda::core::lookup;
using weight::microscaled::MatrixFormat;

cudaError_t enqueue(const Embedding& a, cudaStream_t stream) {
    lookup::View view{{a.table, a.vocabulary, a.width}, lookup::Form::bf16_to_f32, a.output, 1U};
    view.programmatic = true;
    return lookup::enqueue(view, lookup::TokenIds{a.token}, stream);
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
    return ::strix::cuda::mix::enqueue(a, stream);
}

cudaError_t enqueue(const residual::AddArgs& a, cudaStream_t stream) {
    return residual::launch_add(a, stream);
}

cudaError_t enqueue(const residual::ControllerArgs& a, cudaStream_t stream) {
    return residual::launch_controller(a, stream);
}

cudaError_t enqueue(const residual::BankBatchArgs& a, cudaStream_t stream) {
    return ::strix::cuda::mix::enqueue_batch(a, stream);
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

cudaError_t enqueue(const PackedDownCombineTail& a, cudaStream_t stream) {
    if (a.format == MatrixFormat::nvfp4)
        return packed::launch_down_combine_tail<MatrixFormat::nvfp4, kPackedExpertsPerToken>(a.args, a.tail, stream);
    return packed::launch_down_combine_tail<MatrixFormat::mxfp8, kPackedExpertsPerToken>(a.args, a.tail, stream);
}

cudaError_t enqueue(const BankReadoutTiled& a, cudaStream_t stream) {
    return ::strix::cuda::mix::enqueue_tiled(a.args, stream);
}

cudaError_t enqueue(const Bf16OutputSplitRound& a, cudaStream_t stream) {
    cuda::DenseWeight weight;
    weight.data = a.weight;
    weight.out = a.outputs;
    weight.in = a.input_width;
    return cuda::dense_bf16_mma3_r1_output_split(weight, a.output, a.input, stream);
}

cudaError_t enqueue(const PackedGateUpPipe& a, cudaStream_t stream) {
    if (a.format == MatrixFormat::nvfp4) return packed::pipe::launch_gate_up<MatrixFormat::nvfp4>(a.args, stream);
    return packed::pipe::launch_gate_up<MatrixFormat::mxfp8>(a.args, stream);
}

cudaError_t enqueue(const PackedDownCombinePipe& a, cudaStream_t stream) {
    if (a.format == MatrixFormat::nvfp4) return packed::pipe::launch_down_combine<MatrixFormat::nvfp4>(a.args, stream);
    return packed::pipe::launch_down_combine<MatrixFormat::mxfp8>(a.args, stream);
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

namespace screened = ::strix::cuda::core::head_screened;

cudaError_t enqueue(const ScreenedHeadPrep& a, cudaStream_t stream) {
    if (!core::screened_head_plan::launch_valid(a.launch)) return cudaErrorInvalidValue;
    return screened::launch_prep(a.launch, stream);
}
cudaError_t enqueue(const ScreenedHeadPass1& a, cudaStream_t stream) { return screened::launch_pass1(a.launch, stream); }
cudaError_t enqueue(const ScreenedHeadCompact& a, cudaStream_t stream) { return screened::launch_compact(a.launch, stream); }
cudaError_t enqueue(const ScreenedHeadPass2& a, cudaStream_t stream) { return screened::launch_pass2(a.launch, stream); }
cudaError_t enqueue(const ScreenedHeadPick& a, cudaStream_t stream) {
    if (!a.output) return cudaErrorInvalidValue;
    return screened::launch_pick(a.launch, argmax::EmitIndexOrNone{a.output}, stream);
}

struct EmitScreenedRows {
    RowOut* output;
    const StepBlock* descriptor;
    __device__ void operator()(std::uint32_t row, const head_argmax::Pick2& pick) const {
        if (descriptor && !step_row_valid(*descriptor, row)) {
            output[row].status = RowStatus::masked;
            return;
        }
        output[row] = {pick.first.nonfinite ? head_argmax::kNone : pick.first.index, pick.first.value, pick.value2,
                       RowStatus::picked};
    }
};

cudaError_t enqueue(const ScreenedHeadPickRows& a, cudaStream_t stream) {
    if (!a.output) return cudaErrorInvalidValue;
    return screened::launch_pick2(a.launch, EmitScreenedRows{a.output, a.descriptor}, stream);
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

cudaError_t enqueue(const cuda::MmaGroupTable& a, cudaStream_t stream) {
    return cuda::dense_bf16_mma3_r1(a, stream);
}

cudaError_t enqueue(const Bf16DenseGroup& a, cudaStream_t stream) {
    static_assert(std::tuple_size_v<decltype(a.projections)> == cuda::kMaxBf16RowProjections);
    if (a.count == 0 || a.count > a.projections.size()) return cudaErrorInvalidValue;
    cuda::DenseWeight weights[cuda::kMaxBf16RowProjections];
    float* output[cuda::kMaxBf16RowProjections];
    for (unsigned i = 0; i < a.count; ++i) {
        const auto& projection = a.projections[i];
        if (!projection.contiguous(a.input_width)) return cudaErrorInvalidValue;
        weights[i] = {projection.weights, cuda::WeightFormat::bf16, projection.outputs, a.input_width};
        output[i] = projection.output;
    }
    const auto n = static_cast<int>(a.count);
    if (a.mode == Bf16DenseMode::mma3_round)
        return cuda::dense_bf16_mma3_rows_round(weights, output, n, a.input, 1, stream);
    const auto arithmetic = a.mode == Bf16DenseMode::mma3 ? cuda::DenseArith::mma3 : cuda::DenseArith::fp32;
    return cuda::dense_rows_multi(weights, output, n, a.input, 1, nullptr, stream, arithmetic);
}

cudaError_t enqueue(const AppendKvHostBf16& a, cudaStream_t stream) { return enqueue(a.args, stream); }
cudaError_t enqueue(const PreparedFlashHostBf16& a, cudaStream_t stream) { return enqueue(a.args, stream); }

cudaError_t enqueue(const grouped_rows::GatherArgs& a, cudaStream_t stream) { return grouped_rows::gather(a, stream); }
cudaError_t enqueue(const grouped_rows::CombineArgs& a, cudaStream_t stream) { return grouped_rows::combine(a, stream); }

cudaError_t enqueue(const GroupedProjection& a, cudaStream_t stream) {
    if (a.format == MatrixFormat::nvfp4) return grouped_mma::launch<MatrixFormat::nvfp4>(a.args, stream);
    if (a.format == MatrixFormat::mxfp8) return grouped_mma::launch<MatrixFormat::mxfp8>(a.args, stream);
    return cudaErrorInvalidValue;
}

cudaError_t enqueue(const GroupedActivation& a, cudaStream_t stream) {
    return moe_core::expert_activation::launch(a.args, a.gate_cap, a.up_cap, stream);
}

bool valid(const Bf16Fp32Epilogue& a) {
    const auto& d = a.dense;
    return d.count == 1 && d.mode == Bf16DenseMode::fp32 &&
        d.projections[0].outputs <= std::numeric_limits<std::uint32_t>::max() - 7U &&
        cuda::detail::valid_bf16_dense_row_group({d.input, d.input_width, std::span(d.projections).first(1)},
                                                1, cuda::DenseArith::fp32) &&
        (a.form == graph::Form::store_bf16 ||
         (a.form == graph::Form::softcap && a.cap > 0.0F && std::isfinite(a.cap)));
}

cudaError_t enqueue(const Bf16Fp32Epilogue& a, cudaStream_t stream) {
    if (!valid(a)) return cudaErrorInvalidValue;
    const auto& d = a.dense;
    const auto& p = d.projections[0];
    const auto store = a.form == graph::Form::softcap ? cuda::Bf16Fp32Store::softcap : cuda::Bf16Fp32Store::round;
    return cuda::dense_bf16_fp32_r1_epilogue(p.weights, d.input, p.output, d.input_width, p.outputs, store, a.cap, stream);
}

cudaError_t enqueue(const RoundedCausalConvolution& a, cudaStream_t stream) {
    if (!a.convolution.round_bf16) return cudaErrorInvalidValue;
    return enqueue(a.convolution, stream);
}

bool valid(const RoundedCausalConvolution& a) {
    return a.convolution.round_bf16 &&
        core::causal_conv::valid(a.convolution.plan, convolution_args(a.convolution));
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
    if constexpr (std::is_same_v<Args, cuda::MmaGroupTable>) {
        if (!cuda::valid_bf16_mma3_r1(*static_cast<const Args*>(site.backend_args))) return {};
    }
    if constexpr (sizeof...(Modes) != 0) {
        const auto mode = static_cast<const Args*>(site.backend_args)->mode;
        if (!((mode == Modes) || ...)) return {};
    }
    return {catalogue_enqueue<Args>, const_cast<void*>(site.backend_args)};
}

EntryCost logical_cost(const LaunchSite& site, const device::Profile&) {
    return {std::numeric_limits<double>::infinity(), site.weight_bytes, site.state_bytes, site.macs, 1,
            site.bytes_written};
}

void workspace(const LaunchSite& site, WorkspaceNeed& need) { need = site.scratch; }

constexpr auto& embedding = lookup::detail::kEntries.front();
static_assert(embedding.key.form == "embedding.bf16_to_f32" && embedding.key.rows == RowsClass::r1);
constexpr LauncherRef rms{"strix::cuda_core::rmsnorm_bf16", "src/cuda/core/norms/norms.cu", "r1", 1};
constexpr LauncherRef append_kv{"strix::cuda_core::kv_store_prepared_bf16", "src/cuda/core/attn_production.cu", "step", 1};
constexpr LauncherRef prepared_flash{"strix::cuda_core::attention_prefill_prepared_bf16",
    "src/cuda/core/attn_flash.cu", "step.pad2", 1};
constexpr LauncherRef finite{"strix::attn_core::enqueue_validate_prepared_finite",
    "src/attn_core/prepared_gqa_attention.hpp", "r1", 1};
constexpr LauncherRef convolution{"strix::core::causal_conv::launch_with_store", "src/core/short_convolution.hpp", "r1", 1};
constexpr LauncherRef clip{"strix::ops::norm::l2_clip_bf16", "src/ops/norm/norm_device.hpp", "r1", 1};
constexpr LauncherRef normalize{"strix::core::residual_mix::launch_normalize", "src/core/residual_mix.hpp", "r1", 1};
constexpr LauncherRef mix{"strix::cuda::mix::enqueue", "src/cuda/core/mix_cuda.cuh", "r1", 1};
constexpr LauncherRef add{"strix::core::residual_mix::launch_add", "src/core/residual_mix.hpp", "r1", 1};
constexpr LauncherRef controller{"strix::core::residual_mix::launch_controller", "src/core/residual_mix.hpp", "r1", 1};
constexpr LauncherRef round_bf16{"strix::core::pointwise::elements", "src/core/pointwise.hpp", "bf16_round", 1};
constexpr LauncherRef sigmoid{"strix::core::pointwise::elements", "src/core/pointwise.hpp", "bf16_sigmoid_product", 1};
constexpr LauncherRef softcap{"strix::core::pointwise::elements", "src/core/pointwise.hpp", "bf16_softcap", 1};
constexpr LauncherRef relative{"strix::core::pointwise::elements", "src/core/row_transforms.hpp", "relative_bf16_d16", 1};
constexpr LauncherRef independent{"strix::cuda_core::dense_bf16_mma3_r1",
    "src/cuda/core/dense/dense_mma.cu", "mma3_independent", 1};
constexpr LauncherRef dense_fp32{"strix::cuda_core::dense_rows_multi", "src/cuda/core/dense/dense.cu", "fp32", 1};
constexpr LauncherRef dense_mma3{"strix::cuda_core::dense_rows_multi", "src/cuda/core/dense/dense.cu", "mma3", 1};
constexpr LauncherRef dense_round{"strix::cuda_core::dense_bf16_mma3_rows_round", "src/cuda/core/dense/dense_mma.cu", "mma3_round", 1};
constexpr LauncherRef bank_batch_launcher{"strix::cuda::mix::enqueue_batch", "src/cuda/core/mix_cuda_latency.cuh",
    "bank_softmax.f32.native.tiled.batch", 1};
constexpr LauncherRef pair_norm{"strix::core::residual_mix::launch_normalize_pair", "src/core/residual_mix.hpp", "r1", 1};
constexpr LauncherRef gate_up{"strix::moe_core::microscaled_r1::launch_gate_up", "src/moe_core/microscaled_r1.hpp", "r1.k8", 1};
constexpr LauncherRef down_combine{"strix::moe_core::microscaled_r1::launch_down_combine", "src/moe_core/microscaled_r1.hpp", "r1.k8", 1};
constexpr LauncherRef down_combine_tail{"strix::moe_core::microscaled_r1::launch_down_combine_tail",
    "src/moe_core/microscaled_r1.hpp", "r1.k8.conv4_residual", 1};
constexpr LauncherRef pick_part{"strix::cuda::core::head_argmax::pick_partial", "src/cuda/core/head_argmax.cuh", "plain", 1};
constexpr LauncherRef pick_final{"strix::cuda::core::head_argmax::pick_final", "src/cuda/core/head_argmax.cuh", "plain.index_or_none", 1};
constexpr LauncherRef append_kv_host{append_kv.symbol, append_kv.source, "host.r1", 1};
constexpr LauncherRef prepared_flash_host{prepared_flash.symbol, prepared_flash.source, "host.r1.pad2", 1};
constexpr LauncherRef grouped_gather{"strix::moe_core::grouped_rows::gather", "src/moe_core/grouped_rows.hpp", "r1", 1};
constexpr LauncherRef grouped_project{"strix::cuda::core::mma::microscaled::launch", "src/cuda/core/mma_tile.cuh", "r1", 1};
constexpr LauncherRef grouped_activate{"strix::moe_core::expert_activation::launch", "src/moe_core/expert_activation.hpp", "r1", 1};
constexpr LauncherRef grouped_combine{"strix::moe_core::grouped_rows::combine", "src/moe_core/grouped_rows.hpp", "r1", 1};
constexpr LauncherRef dense_fp32_round{"strix::cuda_core::dense_bf16_fp32_r1_epilogue",
    "src/cuda/core/dense/dense.cu", "fp32_round", 1};
constexpr LauncherRef dense_fp32_softcap{"strix::cuda_core::dense_bf16_fp32_r1_epilogue",
    "src/cuda/core/dense/dense.cu", "fp32_softcap", 1};
constexpr LauncherRef mix_tiled{"strix::cuda::mix::enqueue_tiled", "src/cuda/core/mix_cuda_latency.cuh",
    "bank_softmax.f32.native.tiled", 1};
constexpr LauncherRef dense_split{"strix::cuda_core::dense_bf16_mma3_r1_output_split", "src/cuda/core/dense/dense_mma.cu",
    "mma3_round.output_split", 1};
constexpr LauncherRef gate_up_pipe{"strix::moe_core::microscaled_r1::pipe::launch_gate_up",
    "src/cuda/core/moe_r1_pipe.cuh", "r1.k8.pipe", 1};
constexpr LauncherRef down_combine_pipe{"strix::moe_core::microscaled_r1::pipe::launch_down_combine",
    "src/cuda/core/moe_r1_pipe.cuh", "r1.k8.pipe", 1};
constexpr LauncherRef convolution_round{"strix::core::causal_conv::launch_with_store",
    "src/core/short_convolution.hpp", "r1.round_bf16", 1};
constexpr LauncherRef screen_prep{"strix::cuda::core::head_screened::launch_prep", "src/cuda/core/head_screened.cuh",
    "lanes", 1};
constexpr LauncherRef screen_pass1{"strix::cuda::core::head_screened::launch_pass1", "src/cuda/core/head_screened.cuh",
    "lanes", 1};
constexpr LauncherRef screen_compact{"strix::cuda::core::head_screened::launch_compact", "src/cuda/core/head_screened.cuh",
    "lanes", 1};
constexpr LauncherRef screen_pass2{"strix::cuda::core::head_screened::launch_pass2", "src/cuda/core/head_screened.cuh",
    "lanes.fp32_softcap_mapped", 1};
constexpr LauncherRef screen_pick{"strix::cuda::core::head_screened::launch_pick", "src/cuda/core/head_screened.cuh",
    "lanes.index_or_none", 1};
constexpr LauncherRef screen_pick_rows{"strix::cuda::core::head_screened::launch_pick2",
    "src/cuda/core/head_screened.cuh", "lanes.row_out", 1};

template <class Args, OperandClass Act = OperandClass::f32_fma, Bf16DenseMode... Modes>
constexpr KernelEntry entry(graph::Op op, FormId form, weight::FormatId format,
                            std::string_view identity, const LauncherRef& launcher) {
    return {.key = {op, form, format, Act, RowsClass::r1},
            .arch = device::ArchMask::sm120, .identity = identity, .exact = lever::Exactness::bitwise,
            .cost = logical_cost, .workspace = workspace, .bind = bind_site<Args, Modes...>, .launcher = &launcher};
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

template <graph::Form Form>
EntryLaunch bind_fp32_epilogue_site(const LaunchSite& site, EntryBinder& binder) {
    const auto* args = static_cast<const Bf16Fp32Epilogue*>(site.backend_args);
    if (!args || args->form != Form || !valid(*args)) return {};
    return bind_site<Bf16Fp32Epilogue>(site, binder);
}

template <graph::Form Form>
constexpr KernelEntry fp32_epilogue_entry(FormId key, std::string_view identity, const LauncherRef& launcher) {
    auto instance = entry<Bf16Fp32Epilogue>(graph::Op::linear, key, {"plain.bf16", 1}, identity, launcher);
    instance.bind = bind_fp32_epilogue_site<Form>;
    return instance;
}

EntryLaunch bind_rounded_conv_site(const LaunchSite& site, EntryBinder& binder) {
    const auto* args = static_cast<const RoundedCausalConvolution*>(site.backend_args);
    if (!args || !valid(*args)) return {};
    return bind_site<RoundedCausalConvolution>(site, binder);
}

constexpr KernelEntry rounded_conv_entry() {
    auto instance = entry<RoundedCausalConvolution>(graph::Op::elementwise, "causal_convolution.decode_round", {},
        "cuda-sm120-causal-convolution-round-r1-v1", convolution_round);
    instance.bind = bind_rounded_conv_site;
    return instance;
}

enum Leaf { embed, rms_bf16, kv_store, flash, finite_check, conv, head_clip, bank_norm, bank_mix,
            row_add, blend, round, gate, head_softcap, relative_d16, linear_fp32, linear_mma3, linear_mma3_round,
            controller_pair, gate_up_nvfp4, gate_up_mxfp8, down_nvfp4, down_mxfp8, greedy_part, greedy_final,
            kv_store_host, flash_host, group_gather, group_nvfp4, group_mxfp8, group_activation, group_combine,
            linear_fp32_round, linear_fp32_softcap, conv_round, linear_independent, bank_mix_tiled, linear_mma3_split,
            gate_up_pipe_nvfp4, gate_up_pipe_mxfp8, down_pipe_nvfp4, down_pipe_mxfp8, screened_prep, screened_pass1,
            screened_compact, screened_pass2, screened_pick, screened_pick_rows, down_tail_nvfp4,
            down_tail_mxfp8, bank_batch };
constexpr KernelEntry catalogue[] = {
    [] {
        auto instance = embedding;
        instance.bind = bind_site<Embedding>;
        return instance;
    }(),
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
    entry<AppendKvHostBf16>(graph::Op::store, "prepared.kv.bf16.host_position", {},
        "cuda-sm120-prepared-bf16-kv-host-r1-v1", append_kv_host),
    entry<PreparedFlashHostBf16>(graph::Op::attend, "prepared.flash.pad2.d128.host_position", {},
        "cuda-sm120-prepared-flash-pad2-d128-host-r1-v1", prepared_flash_host),
    entry<grouped_rows::GatherArgs>(graph::Op::gather, "microscaled.grouped.gather", {},
        "cuda-sm120-grouped-gather-r1-v1", grouped_gather),
    packed_entry<GroupedProjection, MatrixFormat::nvfp4>(graph::Op::linear, "microscaled.grouped.projection",
        "cuda-sm120-grouped-nvfp4-projection-r1-v1", grouped_project),
    packed_entry<GroupedProjection, MatrixFormat::mxfp8>(graph::Op::linear, "microscaled.grouped.projection",
        "cuda-sm120-grouped-mxfp8-projection-r1-v1", grouped_project),
    entry<GroupedActivation>(graph::Op::elementwise, "microscaled.grouped.swiglu_bf16", {},
        "cuda-sm120-grouped-swiglu-r1-v1", grouped_activate),
    entry<grouped_rows::CombineArgs>(graph::Op::combine, "microscaled.grouped.combine", {},
        "cuda-sm120-grouped-combine-r1-v1", grouped_combine),
    fp32_epilogue_entry<graph::Form::store_bf16>("r1.bf16_image.fp32_round", "cuda-sm120-bf16-r1-fp32-round-v1", dense_fp32_round),
    fp32_epilogue_entry<graph::Form::softcap>("r1.bf16_image.fp32_softcap", "cuda-sm120-bf16-r1-fp32-softcap-v1", dense_fp32_softcap),
    rounded_conv_entry(),
    entry<cuda::MmaGroupTable, OperandClass::bf16_mma>(graph::Op::linear, "r1.bf16_image.independent", {"plain.bf16", 1},
        "cuda-sm120-bf16-r1-mma3-independent-v1", independent),
    entry<BankReadoutTiled>(graph::Op::mix, "residual.bank.tiled", {}, "cuda-sm120-residual-bank-softmax-tiled-r1-v1",
        mix_tiled),
    entry<Bf16OutputSplitRound, OperandClass::bf16_mma>(graph::Op::linear, "r1.bf16_image.output_split_round",
        {"plain.bf16", 1}, "cuda-sm120-bf16-r1-mma3-output-split-round-v1", dense_split),
    packed_entry<PackedGateUpPipe, MatrixFormat::nvfp4>(graph::Op::linear, "microscaled.packed_gate_up.k8.pipe",
        "cuda-sm120-packed-nvfp4-gate-up-r1-k8-pipe-v1", gate_up_pipe),
    packed_entry<PackedGateUpPipe, MatrixFormat::mxfp8>(graph::Op::linear, "microscaled.packed_gate_up.k8.pipe",
        "cuda-sm120-packed-mxfp8-gate-up-r1-k8-pipe-v1", gate_up_pipe),
    packed_entry<PackedDownCombinePipe, MatrixFormat::nvfp4>(graph::Op::combine,
        "microscaled.packed_down_combine.k8.pipe", "cuda-sm120-packed-nvfp4-down-combine-r1-k8-pipe-v1",
        down_combine_pipe),
    packed_entry<PackedDownCombinePipe, MatrixFormat::mxfp8>(graph::Op::combine,
        "microscaled.packed_down_combine.k8.pipe", "cuda-sm120-packed-mxfp8-down-combine-r1-k8-pipe-v1",
        down_combine_pipe),
    entry<ScreenedHeadPrep>(graph::Op::norm, "screened_head.prep", {}, "cuda-sm120-screened-head-prep-v1", screen_prep),
    entry<ScreenedHeadPass1>(graph::Op::head, "screened_head.pass1", {}, "cuda-sm120-screened-head-pass1-v1",
        screen_pass1),
    entry<ScreenedHeadCompact>(graph::Op::select, "screened_head.compact", {}, "cuda-sm120-screened-head-compact-v1",
        screen_compact),
    entry<ScreenedHeadPass2>(graph::Op::head, "screened_head.pass2", {"plain.bf16", 1},
        "cuda-sm120-screened-head-pass2-fp32-softcap-v1", screen_pass2),
    entry<ScreenedHeadPick>(graph::Op::argmax, "screened_head.pick", {}, "cuda-sm120-screened-head-pick-v1", screen_pick),
    entry<ScreenedHeadPickRows>(graph::Op::argmax, "screened_head.pick_rows", {}, "cuda-sm120-screened-head-pick-rows-v1",
        screen_pick_rows),
    packed_entry<PackedDownCombineTail, MatrixFormat::nvfp4>(graph::Op::combine, "microscaled.packed_down_combine_tail.k8",
        "cuda-sm120-packed-nvfp4-down-combine-conv4-residual-r1-k8-v1", down_combine_tail),
    packed_entry<PackedDownCombineTail, MatrixFormat::mxfp8>(graph::Op::combine, "microscaled.packed_down_combine_tail.k8",
        "cuda-sm120-packed-mxfp8-down-combine-conv4-residual-r1-k8-v1", down_combine_tail),
    entry<residual::BankBatchArgs>(graph::Op::mix, "residual.bank.batch", {}, "cuda-sm120-residual-bank-batch-tiled-r1-v1",
        bank_batch_launcher),
};
static_assert(std::size(catalogue) == bank_batch + 1U, "decode catalogue order follows Leaf");

template <class Args>
KernelBinding bind_leaf(Leaf leaf, Grid grid, const Args& args, const LaunchSite& site = {}) {
    KernelBinding bound;
    bound.fn = thunk<Args>;
    bound.identity = catalogue[leaf].identity;
    bound.grid = grid;
    bound.predicted.launches = 1;
    bound.entry_cost = {&catalogue[leaf], catalogue[leaf].cost(site, {})};
    switch (leaf) {
    case embed:  // Token lookup explicitly selects the programmatic body.
    case rms_bf16:
    case linear_fp32:
    case linear_mma3:
    case linear_mma3_round:
    case linear_fp32_round:
    case linear_fp32_softcap:
    case linear_independent:
    case bank_batch:  // The tiled body waits before it reads the bank.
        bound.pdl.wait = PdlWait::before_step_io;
        break;
    case kv_store:
    case flash:
    case finite_check:
    case conv:
    case head_clip:
    case bank_norm:
    case bank_mix:
    case row_add:
    case blend:
    case round:
    case gate:
    case head_softcap:
    case relative_d16:
    case controller_pair:
    case gate_up_nvfp4:
    case gate_up_mxfp8:
    case down_nvfp4:
    case down_mxfp8:
    case down_tail_nvfp4:
    case down_tail_mxfp8:
    case greedy_part:
    case greedy_final:
    case kv_store_host:
    case flash_host:
    case group_gather:
    case group_nvfp4:
    case group_mxfp8:
    case group_activation:
    case group_combine:
    case conv_round:
    case screened_prep:
    case screened_pass1:
    case screened_compact:
    case screened_pick:
    case screened_pick_rows:
        bound.pdl = {PdlWait::absent, PdlOptOut::kernel_without_wait};
        break;
    case screened_pass2:  // the mapped rows body waits first, but the stage keeps completion edges (plain launch)
        bound.pdl = {PdlWait::before_step_io, PdlOptOut::legacy_schedule};
        break;
    case bank_mix_tiled:
    case linear_mma3_split:
    case gate_up_pipe_nvfp4:
    case gate_up_pipe_mxfp8:
    case down_pipe_nvfp4:
    case down_pipe_mxfp8:
        // Opt-in R1 kernels have no qualified native PDL waits. Preserve the undeclared policy.
        break;
    }
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

// Packed codes and scale planes are counted once per logical matrix operand.
// MACs cover projection math. Routing, conversion and epilogue math vary by leaf.
LaunchSite packed_projection_cost(MatrixFormat format, std::uint64_t inputs, std::uint64_t outputs,
                                  unsigned matrices, std::uint64_t read, std::uint64_t written) {
    const bool nvfp4 = format == MatrixFormat::nvfp4;
    const auto columns_per_scale = nvfp4 ? weight::microscaled::kNvfp4BlockColumns
                                        : weight::microscaled::kMxfp8BlockColumns;
    const auto row_bytes = (nvfp4 ? inputs / 2U : inputs) + inputs / columns_per_scale;
    const auto rows = matrices * outputs;
    if (rows > std::numeric_limits<std::uint64_t>::max() / 2U / inputs) return {};
    return {.weight_bytes = rows * row_bytes, .state_bytes = read, .macs = rows * inputs, .scratch = {}, .bytes_written = written};
}

}  // namespace

std::span<const KernelEntry> entries() noexcept { return catalogue; }

KernelBinding bind(const Embedding& a) {
    return bind_leaf(embed, {{1, 1, 1}, {256, 1, 1}, 0}, a,
        {.weight_bytes = std::uint64_t{a.width} * sizeof(std::uint16_t), .state_bytes = sizeof(*a.token), .scratch = {},
         .bytes_written = std::uint64_t{a.width} * sizeof(float)});
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
    // One position-driven CTA per head. PadLayoutShape<128> in attn_flash.cu
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
    namespace readout = ::strix::cuda::mix::detail;
    const auto threads = a.bank_count > 1 ? readout::kPairedLanes : readout::kLanes;
    return bind_leaf(bank_mix, {{1, 1, 1}, {threads, a.bank_count, 1}, 0}, a);
}

KernelBinding bind(const residual::BankBatchArgs& a) {
    const auto grid = ::strix::cuda::mix::latency_batch_grid(a);
    if (grid.tiles == 0U) throw std::invalid_argument("invalid CUDA R1 residual bank batch");
    return bind_leaf(bank_batch, {{grid.tiles, grid.queries, 1}, {::strix::cuda::mix::kLatencyBankThreads, grid.banks, 1}, 0}, a);
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
    const auto macs = std::uint64_t{outputs} * a.input_width;
    const LaunchSite cost{.weight_bytes = macs * sizeof(std::uint16_t),
                          .state_bytes = std::uint64_t{a.input_width} * sizeof(float),
                          .macs = macs, .scratch = {}, .bytes_written = std::uint64_t{outputs} * sizeof(float)};
    if (mma) {
        const auto leaf = round_output ? linear_mma3_round : linear_mma3;
        return bind_leaf(leaf, {{outputs / core::rows::kTileOut, 1, 1}, {256, 1, 1}, 0}, a, cost);
    }
    // dense.cu by_rows<bf16, false> uses one output per warp at R1.
    if (outputs > std::numeric_limits<std::uint32_t>::max() - 7U)
        throw std::invalid_argument("CUDA R1 dense group exceeds the FP32 launch range");
    return bind_leaf(linear_fp32, {{(outputs + 7U) / 8U, 1, 1}, {256, 1, 1}, 0}, a, cost);
}

std::vector<KernelBinding> bind_projection_batch(const core::Bf16ProjectionBatch& batch, const device::Profile& profile) {
    namespace rows = core::rows;
    if (!batch.valid_r1_operands() || !profile.units) throw std::invalid_argument("invalid independent projection operands");
    std::vector<KernelBinding> leaves;
    std::array<rows::Job, rows::kMaxJobs> jobs{};
    std::size_t count = 0;
    for (const auto& group : batch.groups()) {
        core::for_each_bf16_projection_group<cuda::kMaxBf16RowProjections>(batch.outputs(group), true, [&](auto stripe) {
            Bf16DenseGroup scalar{group.input, group.width, {}, 0, Bf16DenseMode::fp32};
            for (const auto& p : stripe) {
                if (cuda::detail::bf16_mma3_tile(group.width, p.outputs)) {
                    const auto chain = p.round_bf16 ? cuda::kRowsRoundBf16 : rows::ChainId{};
                    jobs[count++] = {{p.weights, weight::kBf16WeightStorage}, p.outputs, group.width,
                                     {group.input}, {p.output}, chain, {}};
                } else {
                    if (p.round_bf16) throw std::invalid_argument("untiled projection requires an explicit round");
                    scalar.projections[scalar.count++] = p;
                }
            }
            if (scalar.count) leaves.push_back(bind(scalar));
            return true;
        });
    }
    if (count) {
        const auto plan = rows::plan({std::span(jobs).first(count), 1, 0, rows::Packing::fewest_launches}, profile);
        for (const auto& launch : plan.launches()) {
            const auto args = cuda::mma_group_args(launch);
            auto leaf = bind_leaf(linear_independent, {{launch.tiles.total(), 1, 1}, {256, 1, 1}, 0}, args);
            // Encode which outputs are rounded in the chain identity used by the recipe/compiler.
            for (unsigned j = 0; j < args.tiles.count; ++j) leaf.chain_identity += args.jobs[j].round_bf16 ? 'r' : 'f';
            leaves.push_back(std::move(leaf));
        }
    }
    return leaves;
}

KernelBinding bind(const Bf16Fp32Epilogue& a) {
    if (!valid(a)) throw std::invalid_argument("invalid CUDA R1 FP32 dense epilogue");
    const auto outputs = a.dense.projections[0].outputs;
    const auto leaf = a.form == graph::Form::softcap ? linear_fp32_softcap : linear_fp32_round;
    return bind_leaf(leaf, {{(outputs - 1U) / 8U + 1U, 1, 1}, {256, 1, 1}, 0}, a);
}

KernelBinding bind(const RoundedCausalConvolution& a) {
    const auto& conv = a.convolution;
    if (!valid(a))
        throw std::invalid_argument("invalid CUDA R1 rounded convolution");
    const auto grid = core::causal_conv::launch_plan(conv.plan, convolution_args(conv));
    return bind_leaf(conv_round, {{grid.channel_blocks, grid.row_blocks, 1}, {grid.threads, 1, 1}, 0}, a);
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
    const auto cost = packed_projection_cost(a.format, a.args.hidden, 2ULL * a.args.intermediate, kPackedExpertsPerToken,
        std::uint64_t{a.args.hidden} * sizeof(float),
        std::uint64_t{kPackedExpertsPerToken} * a.args.intermediate * sizeof(*a.args.activation));
    return bind_leaf(leaf, {{blocks, kPackedExpertsPerToken, 1}, {packed::detail::kThreads, 1, 1}, 0}, a, cost);
}

KernelBinding bind(const PackedDownCombine& a) {
    if (!fits_packed(a.args, a.format)) throw std::invalid_argument("invalid CUDA R1 packed down-combine");
    const auto leaf = a.format == MatrixFormat::nvfp4 ? down_nvfp4 : down_mxfp8;
    const auto blocks = packed::down_blocks(a.args.hidden, packed::kCudaGeometry.down_columns);
    const auto cost = packed_projection_cost(a.format, a.args.intermediate, a.args.hidden, kPackedExpertsPerToken,
        std::uint64_t{kPackedExpertsPerToken} * a.args.intermediate * sizeof(*a.args.activation),
        std::uint64_t{a.args.hidden} * sizeof(float));
    return bind_leaf(leaf, {{blocks, 1, 1}, {packed::detail::kThreads, 1, 1}, 0}, a, cost);
}

// Same grid as PackedDownCombine; the tail adds its per-channel operands (weights, history, residual) and stores
// (history, optional convolution image, optional sum). Legality is the shared plan rule (one row, no aliasing).
KernelBinding bind(const PackedDownCombineTail& a) {
    if (!fits_packed(a.args, a.format) || !packed::admits_tail(a.args, a.tail, 1U, kPackedExpertsPerToken))
        throw std::invalid_argument("invalid CUDA R1 packed down-combine tail");
    const auto leaf = a.format == MatrixFormat::nvfp4 ? down_tail_nvfp4 : down_tail_mxfp8;
    const auto blocks = packed::down_blocks(a.args.hidden, packed::kCudaGeometry.down_columns);
    const std::uint64_t channel = std::uint64_t{a.args.hidden} * sizeof(float);
    constexpr std::uint64_t taps = kPackedConvTailTaps;
    const std::uint64_t tail_read = (2U * taps - 1U + (a.tail.residual ? 1U : 0U)) * channel;
    const std::uint64_t tail_written = (taps - 1U + (a.tail.conv_output ? 1U : 0U) + (a.tail.residual ? 1U : 0U)) * channel;
    const auto cost = packed_projection_cost(a.format, a.args.intermediate, a.args.hidden, kPackedExpertsPerToken,
        std::uint64_t{kPackedExpertsPerToken} * a.args.intermediate * sizeof(*a.args.activation) + tail_read,
        channel + tail_written);
    return bind_leaf(leaf, {{blocks, 1, 1}, {packed::detail::kThreads, 1, 1}, 0}, a, cost);
}

KernelBinding bind(const BankReadoutTiled& a) {
    if (!fits(a)) throw std::invalid_argument("invalid CUDA R1 tiled bank readout");
    const auto grid = ::strix::cuda::mix::latency_grid(a.args);
    return bind_leaf(bank_mix_tiled, {{grid.tiles, grid.queries, 1}, {::strix::cuda::mix::kLatencyBankThreads, grid.banks, 1}, 0},
        a, {.scratch = {}, .bytes_written = std::uint64_t{a.args.query_count} * a.args.width * sizeof(float)});
}

KernelBinding bind(const Bf16OutputSplitRound& a) {
    if (!fits(a)) throw std::invalid_argument("invalid CUDA R1 BF16 output split");
    const auto macs = std::uint64_t{a.outputs} * a.input_width;
    return bind_leaf(linear_mma3_split, {{cuda::bf16_narrow_blocks(a.input_width, a.outputs), 1, 1},
                                         {cuda::kBf16NarrowThreads, 1, 1}, 0}, a,
        {.weight_bytes = macs * sizeof(std::uint16_t), .state_bytes = std::uint64_t{a.input_width} * sizeof(float),
         .macs = macs, .scratch = {}, .bytes_written = std::uint64_t{a.outputs} * sizeof(float)});
}

static_assert(packed::pipe::kTop == kPackedExpertsPerToken);

KernelBinding bind(const PackedGateUpPipe& a) {
    if (!fits_packed(a.args, a.format)) throw std::invalid_argument("invalid CUDA R1 packed gate-up pipe");
    const auto leaf = a.format == MatrixFormat::nvfp4 ? gate_up_pipe_nvfp4 : gate_up_pipe_mxfp8;
    const auto cost = packed_projection_cost(a.format, a.args.hidden, 2ULL * a.args.intermediate, kPackedExpertsPerToken,
        std::uint64_t{a.args.hidden} * sizeof(float),
        std::uint64_t{kPackedExpertsPerToken} * a.args.intermediate * sizeof(*a.args.activation));
    return bind_leaf(leaf, {{packed::pipe::gate_blocks(a.args.intermediate), packed::pipe::kTop, 1},
                            {packed::pipe::kGateWarps * packed::pipe::kWarp, 1, 1}, 0}, a, cost);
}

KernelBinding bind(const PackedDownCombinePipe& a) {
    if (!fits_packed(a.args, a.format)) throw std::invalid_argument("invalid CUDA R1 packed down-combine pipe");
    const auto leaf = a.format == MatrixFormat::nvfp4 ? down_pipe_nvfp4 : down_pipe_mxfp8;
    const auto cost = packed_projection_cost(a.format, a.args.intermediate, a.args.hidden, kPackedExpertsPerToken,
        std::uint64_t{kPackedExpertsPerToken} * a.args.intermediate * sizeof(*a.args.activation),
        std::uint64_t{a.args.hidden} * sizeof(float));
    return bind_leaf(leaf, {{packed::pipe::down_blocks(a.args.hidden), 1, 1},
                            {packed::pipe::kTop * packed::pipe::kWarp, 1, 1}, 0}, a, cost);
}

KernelBinding bind(const GreedyPartial& a) {
    return bind_leaf(greedy_part, {{greedy_chunks(a), 1, 1}, {argmax::detail::kThreads, 1, 1}, 0}, a);
}

KernelBinding bind(const GreedyFinal& a) {
    (void)greedy_chunks(a.reduction);
    if (!a.output) throw std::invalid_argument("missing CUDA R1 greedy output");
    return bind_leaf(greedy_final, {{1, 1, 1}, {argmax::detail::kThreads, 1, 1}, 0}, a);
}

namespace {

namespace screen_plan = core::screened_head_plan;

const screen_plan::Launch& checked(const screen_plan::Launch& a) {
    if (!screen_plan::launch_valid(a)) throw std::invalid_argument("invalid CUDA screened head launch");
    return a;
}

}  // namespace

KernelBinding bind(const ScreenedHeadPrep& a) {
    const auto& l = checked(a.launch);
    return bind_leaf(screened_prep, {{l.lanes, 1, 1}, {256, 1, 1}, 0}, a,
        {.state_bytes = std::uint64_t{l.lanes} * l.head.width * sizeof(float),
         .scratch = {}, .bytes_written = std::uint64_t{l.lanes} * sizeof(screen_plan::State)});
}

KernelBinding bind(const ScreenedHeadPass1& a) {
    const auto& l = checked(a.launch);
    const auto plane = std::uint64_t{l.plane.rows} * l.plane.row_bytes;
    const auto groups = screen_plan::pass1_groups(l.lanes, l.pass1_lanes);
    return bind_leaf(screened_pass1, {{l.grids.pass1, groups, 1}, {screen_plan::kPass1Waves * 32U, 1, 1},
                                      static_cast<unsigned>(screened::pass1_shared_bytes(l))},
        a, {.weight_bytes = plane * groups, .state_bytes = std::uint64_t{l.lanes} * l.head.width * sizeof(float),
            .macs = std::uint64_t{l.plane.rows} * l.head.width * l.lanes,
            .scratch = {}, .bytes_written = std::uint64_t{l.lanes} * l.head.rows * sizeof(float)});
}

KernelBinding bind(const ScreenedHeadCompact& a) {
    const auto& l = checked(a.launch);
    return bind_leaf(screened_compact, {{l.grids.compact, l.lanes, 1}, {screen_plan::kCompactThreads, 1, 1}, 0}, a,
        {.state_bytes = std::uint64_t{l.lanes} * l.head.rows * sizeof(float), .scratch = {}});
}

KernelBinding bind(const ScreenedHeadPass2& a) {
    const auto& l = checked(a.launch);
    return bind_leaf(screened_pass2, {{l.grids.pass2, l.lanes, 1}, {256, 1, 1}, 0}, a);
}

KernelBinding bind(const ScreenedHeadPick& a) {
    const auto& l = checked(a.launch);
    if (!a.output) throw std::invalid_argument("missing CUDA screened head output");
    return bind_leaf(screened_pick, {{l.lanes, 1, 1}, {256, 1, 1}, 0}, a,
        {.scratch = {}, .bytes_written = std::uint64_t{l.lanes} * sizeof(std::uint32_t)});
}

KernelBinding bind(const ScreenedHeadPickRows& a) {
    const auto& l = checked(a.launch);
    if (!a.output || !l.runner_up) throw std::invalid_argument("CUDA screened head rows pick needs a runner-up launch");
    return bind_leaf(screened_pick_rows, {{l.lanes, 1, 1}, {256, 1, 1}, 0}, a,
        {.scratch = {}, .bytes_written = std::uint64_t{l.lanes} * sizeof(RowOut)});
}

KernelBinding bind(const AppendKvHostBf16& a) {
    if (a.args.position) throw std::invalid_argument("host-position KV append has a device position");
    return bind_leaf(kv_store_host, {{a.args.kv_heads, 1, 1}, {std::min(a.args.width, 256U), 1, 1}, 0}, a);
}

KernelBinding bind(const PreparedFlashHostBf16& a) {
    if (a.args.position) throw std::invalid_argument("host-position prepared flash has a device position");
    constexpr auto shared = 2U * 32U * (4U * prepared_head_width + 32U + 16U);
    return bind_leaf(flash_host, {{a.args.heads, 1, 1}, {256, 1, 1}, shared}, a);
}

KernelBinding bind(const grouped_rows::GatherArgs& a) {
    if (a.width == 0 || a.assignments == 0) throw std::invalid_argument("empty CUDA grouped gather");
    const auto grid = grouped_rows::detail::grid(a.width, a.assignments);
    return bind_leaf(group_gather, {{grid.x, grid.y, 1}, {grouped_rows::detail::kThreads, 1, 1}, 0}, a);
}

KernelBinding bind(const GroupedProjection& a) {
    if (!grouped_mma::fits(a.args) || a.args.max_tile_rows != 1U ||
        (a.format != MatrixFormat::nvfp4 && a.format != MatrixFormat::mxfp8))
        throw std::invalid_argument("invalid CUDA R1 grouped projection");
    using Shape = grouped_mma::TileShape< ::strix::cuda::core::native_mma::kDecodeMBlocks, grouped_mma::kDecodeOutputs>;
    const auto leaf = a.format == MatrixFormat::nvfp4 ? group_nvfp4 : group_mxfp8;
    // The R1 recipe allocates one tile per routed assignment. Count its planned logical work.
    const auto cost = packed_projection_cost(a.format, a.args.inputs, a.args.outputs, a.args.count,
        std::uint64_t{a.args.count} * a.args.inputs * sizeof(std::uint16_t),
        std::uint64_t{a.args.count} * a.args.outputs * sizeof(float));
    return bind_leaf(leaf, {{a.args.outputs / grouped_mma::kDecodeOutputs, 1, a.args.count},
                            {Shape::kThreads, 1, 1}, 0}, a, cost);
}

KernelBinding bind(const GroupedActivation& a) {
    if (a.args.rows == 0 || !moe_core::expert_activation::fits(a.args, a.gate_cap, a.up_cap))
        throw std::invalid_argument("invalid CUDA grouped activation");
    return bind_leaf(group_activation, {{static_cast<unsigned>((a.args.width - 1U) / 128U + 1U),
        static_cast<unsigned>(a.args.rows), 1}, {128, 1, 1}, 0}, a);
}

KernelBinding bind(const grouped_rows::CombineArgs& a) {
    if (a.width == 0 || a.rows != 1U) throw std::invalid_argument("invalid CUDA R1 grouped combine");
    const auto grid = grouped_rows::detail::grid(a.width, a.rows);
    return bind_leaf(group_combine, {{grid.x, grid.y, 1}, {grouped_rows::detail::kThreads, 1, 1}, 0}, a);
}

}  // namespace strix::exec::cuda_decode
