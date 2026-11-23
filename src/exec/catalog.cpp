#include "strix/exec/catalog.hpp"

#include "strix/device/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace strix::device {

ArchMask arch_mask(std::string_view arch) noexcept {
    if (arch == "sm_120" || arch == "sm_120a") return ArchMask::sm120;
    if (arch == "gfx1151") return ArchMask::gfx1151;
    return ArchMask::none;
}

} // namespace strix::device

namespace strix::exec {
namespace {

auto key_fields(const KernelKey& key) noexcept {
    return std::tie(key.op, key.form, key.weight.name, key.weight.revision, key.act, key.rows,
                    key.dim_class, key.prologue, key.epilogue, key.math);
}

template <OperandClass Act>
double peak_tops(const device::Peaks& peaks) {
    if constexpr (Act == OperandClass::f32_fma) return peaks.fp32;
    if constexpr (Act == OperandClass::bf16_mma) return peaks.bf16;
    if constexpr (Act == OperandClass::f16_mma) return peaks.fp16;
    if constexpr (Act == OperandClass::fp8_mma) return peaks.fp8;
    // The profile's tensor INT8 rate does not price scalar DP4A.
    return 0; // Populate when admitting an operand class with a matching device fact.
}

// An optimistic roofline estimate, in microseconds. Missing required profile
// facts yield infinity. Launch overhead and utilization need calibration.
double roof_time(double work, double rate, double units) {
    if (work == 0) return 0;
    return rate > 0 && std::isfinite(rate) ? work / rate / units : std::numeric_limits<double>::infinity();
}

template <OperandClass Act>
EntryCost estimate(const LaunchSite& site, const device::Profile& profile) {
    const auto bytes = static_cast<double>(site.weight_bytes) + static_cast<double>(site.state_bytes);
    const double memory = roof_time(bytes, profile.memory_gbytes_ps, 1e3);
    const double compute = roof_time(2.0 * static_cast<double>(site.macs), peak_tops<Act>(profile.peaks), 1e6);
    return {std::max(memory, compute), site.weight_bytes, site.state_bytes, site.macs, site.launches};
}

void workspace(const LaunchSite& site, WorkspaceNeed& need) { need = site.scratch; }

EntryCost unpriced_single_launch(const LaunchSite& site, const device::Profile&) {
    return {std::numeric_limits<double>::infinity(), site.weight_bytes, site.state_bytes, site.macs, 1U};
}

void no_workspace(const LaunchSite&, WorkspaceNeed& need) { need = {}; }

RowsClass rows_class(std::uint32_t rows) {
    if (rows == 1) return RowsClass::r1;
    if (rows <= 8) return RowsClass::r2_8;
    if (rows <= 32) return RowsClass::r9_32;
    return RowsClass::tall;
}

template <RowsClass Rows, const LauncherRef& Launcher>
EntryLaunch bind_existing(const LaunchSite& site, EntryBinder& arena) {
    if (!arena.resolve || site.rows == 0 || site.rows > Launcher.max_rows || rows_class(site.rows) != Rows) return {};
    return arena.resolve(arena.context, Launcher, site);
}

template <OperandClass Act, RowsClass Rows, const LauncherRef& Launcher,
          auto Cost = estimate<Act>, auto Workspace = workspace>
constexpr KernelEntry instance(graph::Op op, FormId form, weight::FormatId format, device::ArchMask arch,
                               std::string_view identity, bool capturable) {
    return {.key = {op, form, format, Act, Rows}, .arch = arch, .identity = identity,
            .exact = lever::Exactness::bitwise, .cost = Cost, .workspace = Workspace,
            .bind = bind_existing<Rows, Launcher>,
            .launcher = &Launcher, .capturable = capturable};
}

// BEGIN D4 INSTANCES
constexpr LauncherRef cuda_dense_rows{"strix::cuda_core::dense_rows",
    "src/cuda/core/dense/dense.cu", "fp32", 32U};
constexpr LauncherRef cuda_dense_rows_multi{"strix::cuda_core::dense_rows_multi",
    "src/cuda/core/dense/dense.cu", "fp32", 32U};
constexpr LauncherRef cuda_dense_mma{"strix::cuda_core::dense_mma_launch",
    "src/cuda/core/dense/dense_mma.cu", "mma3", 32U};
constexpr LauncherRef cuda_nvfp4_rows{"strix::cuda_core::nvfp4_dense_rows",
    "src/cuda/core/dense/nvfp4.cu", "w4a16.split_exact", 32U};
constexpr LauncherRef cuda_nvfp4_rows_multi{"strix::cuda_core::nvfp4_dense_rows_multi",
    "src/cuda/core/dense/nvfp4.cu", "w4a16.split_exact", 32U};
constexpr LauncherRef hip_dense_linear{"strix::core::dense_linear::enqueue_bf16_tight_rows",
    "src/core/dense_linear.hpp", "tight.fp32_store", 16U};
constexpr LauncherRef hip_fma_rows{"strix::core::fma_rows::launch_auto",
    "src/core/fma_rows.hpp", "pipe128.fp32_store", 32U};
constexpr LauncherRef hip_q8_gemv{"strix::q8_rows::launch",
    "src/hip_q8_rows_gemv.hpp", "linear_f32", 1U};
constexpr LauncherRef cuda_rms{"strix::cuda_core::rmsnorm",
    "src/cuda/core/norms/norms.cu", "fp32_gain", 65535U};
constexpr LauncherRef cuda_rms_bf16{"strix::cuda_core::rmsnorm_bf16",
    "src/cuda/core/norms/norms.cu", "bf16_gain.bf16_store", 65535U};
constexpr LauncherRef cuda_l2{"strix::cuda_core::l2norm",
    "src/cuda/core/norms/norms.cu", "fp32", 65535U};
constexpr LauncherRef cuda_gated_rms_swish{"strix::cuda_core::gated_rmsnorm",
    "src/cuda/core/norms/norms.cu", "swish", 65535U};
constexpr LauncherRef cuda_gated_rms_sigmoid{"strix::cuda_core::gated_rmsnorm",
    "src/cuda/core/norms/norms.cu", "sigmoid", 65535U};
constexpr LauncherRef hip_rms_bf16{"strix::ops::norm::enqueue_rms_bf16_rows",
    "src/ops/norm/rms_bf16_rows.hpp", "parallel", 2147483647U};
constexpr LauncherRef cuda_add{"strix::cuda_core::add_inplace",
    "src/cuda/core/norms/norms.cu", "inplace", 4294967040U};
constexpr LauncherRef hip_add{"strix::core::pointwise::elements",
    "src/core/pointwise.hpp", "add", 4294967295U};
constexpr LauncherRef hip_swiglu{"strix::ops::glu::launch_swiglu",
    "src/ops/glu/glu_device.hpp", "fp32", 4294967295U};
constexpr LauncherRef cuda_mix_bank_readout{"strix::cuda::mix::enqueue",
    "src/cuda/core/mix_cuda.cuh", "bank_softmax.f32.native", 4294967295U};
constexpr LauncherRef cuda_mix_bank_readout_tiled{"strix::cuda::mix::enqueue_tiled",
    "src/cuda/core/mix_cuda_latency.cuh", "bank_softmax.f32.native.tiled", 1U};
constexpr LauncherRef cuda_select_pick_ordered{"strix::cuda::core::select::enqueue_pick",
    "src/cuda/core/select_cuda.cuh", "ordered", 2147483647U};
constexpr LauncherRef cuda_select_pick_float{"strix::cuda::core::select::enqueue_pick",
    "src/cuda/core/select_cuda.cuh", "canonical_float", 2147483647U};
constexpr LauncherRef cuda_select_pick_raw{"strix::cuda::core::select::enqueue_pick",
    "src/cuda/core/select_cuda.cuh", "raw_float", 2147483647U};
constexpr LauncherRef cuda_select_pick_mask{"strix::cuda::core::select::enqueue_pick",
    "src/cuda/core/select_cuda.cuh", "eligible_float", 2147483647U};
constexpr LauncherRef cuda_select_score_relu_tree{"strix::cuda::core::select::enqueue_scores",
    "src/cuda/core/select_score.cuh", "relu_tree", 65535U};
constexpr LauncherRef cuda_select_score_weighted_fma{"strix::cuda::core::select::enqueue_scores",
    "src/cuda/core/select_score.cuh", "weighted_fma", 65535U};
constexpr LauncherRef cuda_select_score_weighted_bf16{"strix::cuda::core::select::enqueue_scores",
    "src/cuda/core/select_score.cuh", "weighted_bf16", 65535U};
constexpr LauncherRef cuda_select_score_weighted_bf16x3{"strix::cuda::core::select::enqueue_scores",
    "src/cuda/core/select_score.cuh", "weighted_bf16x3", 65535U};
constexpr LauncherRef cuda_select_score_weighted_fp4{"strix::cuda::core::select::enqueue_scores",
    "src/cuda/core/select_score.cuh", "weighted_fp4", 65535U};
constexpr LauncherRef cuda_select_score_block_max{"strix::cuda::core::select::enqueue_scores",
    "src/cuda/core/select_score.cuh", "block_max", 65535U};
constexpr LauncherRef cuda_select_score_block_max_tiled{"strix::cuda::core::select::enqueue_scores",
    "src/cuda/core/select_score.cuh", "block_max_tiled", 65535U};
constexpr LauncherRef cuda_select_score_weighted_fp8{"strix::cuda::core::select::enqueue_scores",
    "src/cuda/core/select_score.cuh", "weighted_fp8", 65535U};
constexpr LauncherRef cuda_decode_bf16_output_split{"strix::cuda_core::dense_bf16_mma3_r1_output_split",
    "src/cuda/core/dense/dense_mma.cu", "mma3_round.output_split", 1U};
constexpr LauncherRef cuda_decode_embedding{"strix::cuda_core::embedding_rows",
    "src/cuda/core/dense/dense.cu", "bf16", 1U};
constexpr LauncherRef cuda_decode_rms{"strix::cuda_core::rmsnorm_bf16",
    "src/cuda/core/norms/norms.cu", "r1", 1U};
constexpr LauncherRef cuda_decode_append_kv{"strix::cuda_core::kv_store_prepared_bf16",
    "src/cuda/core/attn_production.cu", "step", 1U};
constexpr LauncherRef cuda_decode_prepared_flash{"strix::cuda_core::attention_prefill_prepared_bf16",
    "src/cuda/core/attn_flash.cu", "step.pad2", 1U};
constexpr LauncherRef cuda_decode_prepared_flash_split{"strix::cuda_core::attention_decode_prepared_split_bf16",
    "src/cuda/core/attn_flash_decode_split.cu", "step.split", 1U};
constexpr LauncherRef cuda_decode_finite{"strix::attn_core::enqueue_validate_prepared_finite",
    "src/attn_core/prepared_gqa_attention.hpp", "r1", 1U};
constexpr LauncherRef cuda_decode_convolution{"strix::core::causal_conv::launch_with_store",
    "src/core/short_convolution.hpp", "r1", 1U};
constexpr LauncherRef cuda_decode_clip{"strix::ops::norm::l2_clip_bf16",
    "src/ops/norm/norm_device.hpp", "r1", 1U};
constexpr LauncherRef cuda_decode_normalize{"strix::core::residual_mix::launch_normalize",
    "src/core/residual_mix.hpp", "r1", 1U};
constexpr LauncherRef cuda_decode_add{"strix::core::residual_mix::launch_add",
    "src/core/residual_mix.hpp", "r1", 1U};
constexpr LauncherRef cuda_decode_controller{"strix::core::residual_mix::launch_controller",
    "src/core/residual_mix.hpp", "r1", 1U};
constexpr LauncherRef cuda_decode_round_bf16{"strix::core::pointwise::elements",
    "src/core/pointwise.hpp", "bf16_round", 1U};
constexpr LauncherRef cuda_decode_sigmoid{"strix::core::pointwise::elements",
    "src/core/pointwise.hpp", "bf16_sigmoid_product", 1U};
constexpr LauncherRef cuda_decode_softcap{"strix::core::pointwise::elements",
    "src/core/pointwise.hpp", "bf16_softcap", 1U};
constexpr LauncherRef cuda_decode_relative{"strix::core::pointwise::elements",
    "src/core/row_transforms.hpp", "relative_bf16_d16", 1U};
constexpr LauncherRef cuda_decode_dense_fp32{"strix::cuda_core::dense_rows_multi",
    "src/cuda/core/dense/dense.cu", "fp32", 1U};
constexpr LauncherRef cuda_decode_dense_mma3{"strix::cuda_core::dense_rows_multi",
    "src/cuda/core/dense/dense.cu", "mma3", 1U};
constexpr LauncherRef cuda_decode_independent{"strix::cuda_core::dense_bf16_mma3_r1",
    "src/cuda/core/dense/dense_mma.cu", "mma3_independent", 1U};
constexpr LauncherRef cuda_decode_dense_round{"strix::cuda_core::dense_bf16_mma3_rows_round",
    "src/cuda/core/dense/dense_mma.cu", "mma3_round", 1U};
constexpr LauncherRef cuda_decode_pair_norm{"strix::core::residual_mix::launch_normalize_pair",
    "src/core/residual_mix.hpp", "r1", 1U};
constexpr LauncherRef cuda_decode_bank_batch{"strix::cuda::mix::enqueue_batch",
    "src/cuda/core/mix_cuda_latency.cuh", "bank_softmax.f32.native.tiled.batch", 1U};
constexpr LauncherRef cuda_decode_gate_up_nvfp4{"strix::moe_core::microscaled_r1::launch_gate_up",
    "src/moe_core/microscaled_r1.hpp", "r1.k8", 1U};
constexpr LauncherRef cuda_decode_gate_up_mxfp8{"strix::moe_core::microscaled_r1::launch_gate_up",
    "src/moe_core/microscaled_r1.hpp", "r1.k8", 1U};
constexpr LauncherRef cuda_decode_down_combine_nvfp4{"strix::moe_core::microscaled_r1::launch_down_combine",
    "src/moe_core/microscaled_r1.hpp", "r1.k8", 1U};
constexpr LauncherRef cuda_decode_down_combine_mxfp8{"strix::moe_core::microscaled_r1::launch_down_combine",
    "src/moe_core/microscaled_r1.hpp", "r1.k8", 1U};
constexpr LauncherRef cuda_decode_pick_part{"strix::cuda::core::head_argmax::pick_partial",
    "src/cuda/core/head_argmax.cuh", "plain", 1U};
constexpr LauncherRef cuda_decode_pick_final{"strix::cuda::core::head_argmax::pick_final",
    "src/cuda/core/head_argmax.cuh", "plain.index_or_none", 1U};
constexpr LauncherRef cuda_decode_dense_fp32_round{"strix::cuda_core::dense_bf16_fp32_r1_epilogue",
    "src/cuda/core/dense/dense.cu", "fp32_round", 1U};
constexpr LauncherRef cuda_decode_dense_fp32_softcap{"strix::cuda_core::dense_bf16_fp32_r1_epilogue",
    "src/cuda/core/dense/dense.cu", "fp32_softcap", 1U};
constexpr LauncherRef cuda_decode_convolution_round{"strix::core::causal_conv::launch_with_store",
    "src/core/short_convolution.hpp", "r1.round_bf16", 1U};
constexpr LauncherRef cuda_decode_embedding_q8{"strix::cuda_core::embedding_rows",
    "src/cuda/core/dense/dense.cu", "q8_0", 1U};
constexpr LauncherRef cuda_decode_gate_up_mma3{"strix::cuda_core::dense_gate_up_silu_rows",
    "src/cuda/core/dense/dense.cu", "mma3", 1U};
constexpr LauncherRef cuda_decode_rope{"strix::cuda_core::rope_apply",
    "src/cuda/core/norms/rope.cu", "device_position", 1U};
constexpr LauncherRef cuda_decode_kv_store{"strix::cuda_core::kv_store",
    "src/cuda/core/attn_production.cu", "linear", 1U};
constexpr LauncherRef cuda_decode_pick_part_pdl{"strix::cuda::core::head_argmax::pick_partial",
    "src/cuda/core/head_argmax.cuh", "pdl", 1U};
constexpr LauncherRef cuda_decode_pick_final_pdl_index{"strix::cuda::core::head_argmax::pick_final",
    "src/cuda/core/head_argmax.cuh", "pdl.index", 1U};
constexpr LauncherRef cuda_decode_gdn_convolution{"strix::cuda_core::gdn_window_convolution",
    "src/cuda/core/recur_parity.cu", "linear.parity", 1U};
constexpr LauncherRef cuda_decode_gdn_recurrence{"strix::cuda_core::gdn_window_recurrence",
    "src/cuda/core/recur_parity.cu", "linear.parity", 1U};
constexpr LauncherRef cuda_decode_attention_scores{"strix::cuda_core::attention_decode_row_phase",
    "src/cuda/core/attn_production.cu", "bf16.linear.scores", 1U};
constexpr LauncherRef cuda_decode_attention_values{"strix::cuda_core::attention_decode_row_phase",
    "src/cuda/core/attn_production.cu", "bf16.linear.values", 1U};
constexpr LauncherRef cuda_decode_attention_combine{"strix::cuda_core::attention_decode_row_phase",
    "src/cuda/core/attn_production.cu", "bf16.linear.combine", 1U};
constexpr LauncherRef cuda_moer_pipe_gate_up_nvfp4{"strix::moe_core::microscaled_r1::pipe::launch_gate_up",
    "src/cuda/core/moe_r1_pipe.cuh", "r1.k8.pipe", 1U};
constexpr LauncherRef cuda_moer_pipe_gate_up_mxfp8{"strix::moe_core::microscaled_r1::pipe::launch_gate_up",
    "src/cuda/core/moe_r1_pipe.cuh", "r1.k8.pipe", 1U};
constexpr LauncherRef cuda_moer_pipe_down_combine_nvfp4{"strix::moe_core::microscaled_r1::pipe::launch_down_combine",
    "src/cuda/core/moe_r1_pipe.cuh", "r1.k8.pipe", 1U};
constexpr LauncherRef cuda_moer_pipe_down_combine_mxfp8{"strix::moe_core::microscaled_r1::pipe::launch_down_combine",
    "src/cuda/core/moe_r1_pipe.cuh", "r1.k8.pipe", 1U};

constexpr KernelEntry cuda_entries[] = {
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_dense_rows>(
        graph::Op::linear, "rows", {"plain.bf16", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-rows-fp32-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, cuda_dense_rows>(
        graph::Op::linear, "rows", {"plain.bf16", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-rows-fp32-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, cuda_dense_rows>(
        graph::Op::linear, "rows", {"plain.bf16", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-rows-fp32-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_dense_rows>(
        graph::Op::linear, "rows", {"plain.f32", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-rows-fp32-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, cuda_dense_rows>(
        graph::Op::linear, "rows", {"plain.f32", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-rows-fp32-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, cuda_dense_rows>(
        graph::Op::linear, "rows", {"plain.f32", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-rows-fp32-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_dense_rows>(
        graph::Op::linear, "rows", {"gguf.q8_0", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-rows-fp32-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, cuda_dense_rows>(
        graph::Op::linear, "rows", {"gguf.q8_0", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-rows-fp32-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, cuda_dense_rows>(
        graph::Op::linear, "rows", {"gguf.q8_0", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-rows-fp32-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_dense_rows_multi>(
        graph::Op::linear, "rows.multi", {"plain.bf16", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-rows-multi-fp32-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, cuda_dense_rows_multi>(
        graph::Op::linear, "rows.multi", {"plain.bf16", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-rows-multi-fp32-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, cuda_dense_rows_multi>(
        graph::Op::linear, "rows.multi", {"plain.bf16", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-rows-multi-fp32-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_dense_rows_multi>(
        graph::Op::linear, "rows.multi", {"plain.f32", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-rows-multi-fp32-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, cuda_dense_rows_multi>(
        graph::Op::linear, "rows.multi", {"plain.f32", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-rows-multi-fp32-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, cuda_dense_rows_multi>(
        graph::Op::linear, "rows.multi", {"plain.f32", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-rows-multi-fp32-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_dense_rows_multi>(
        graph::Op::linear, "rows.multi", {"gguf.q8_0", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-rows-multi-fp32-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, cuda_dense_rows_multi>(
        graph::Op::linear, "rows.multi", {"gguf.q8_0", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-rows-multi-fp32-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, cuda_dense_rows_multi>(
        graph::Op::linear, "rows.multi", {"gguf.q8_0", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-rows-multi-fp32-v1", true),
    instance<OperandClass::bf16_mma, RowsClass::r1, cuda_dense_mma>(
        graph::Op::linear, "rows.multi", {"plain.bf16", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-rows-mma3-v1", true),
    instance<OperandClass::bf16_mma, RowsClass::r2_8, cuda_dense_mma>(
        graph::Op::linear, "rows.multi", {"plain.bf16", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-rows-mma3-v1", true),
    instance<OperandClass::bf16_mma, RowsClass::r9_32, cuda_dense_mma>(
        graph::Op::linear, "rows.multi", {"plain.bf16", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-rows-mma3-v1", true),
    instance<OperandClass::bf16_mma, RowsClass::r1, cuda_dense_mma>(
        graph::Op::linear, "rows.multi", {"gguf.q8_0", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-rows-mma3-v1", true),
    instance<OperandClass::bf16_mma, RowsClass::r2_8, cuda_dense_mma>(
        graph::Op::linear, "rows.multi", {"gguf.q8_0", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-rows-mma3-v1", true),
    instance<OperandClass::bf16_mma, RowsClass::r9_32, cuda_dense_mma>(
        graph::Op::linear, "rows.multi", {"gguf.q8_0", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-rows-mma3-v1", true),
    instance<OperandClass::f16_mma, RowsClass::r1, cuda_nvfp4_rows>(
        graph::Op::linear, "rows", {"gguf.nvfp4_tile.strix", 1U}, device::ArchMask::sm120, "cuda-sm120-nvfp4-rows-w4a16-v1", false),
    instance<OperandClass::f16_mma, RowsClass::r2_8, cuda_nvfp4_rows>(
        graph::Op::linear, "rows", {"gguf.nvfp4_tile.strix", 1U}, device::ArchMask::sm120, "cuda-sm120-nvfp4-rows-w4a16-v1", false),
    instance<OperandClass::f16_mma, RowsClass::r9_32, cuda_nvfp4_rows>(
        graph::Op::linear, "rows", {"gguf.nvfp4_tile.strix", 1U}, device::ArchMask::sm120, "cuda-sm120-nvfp4-rows-w4a16-v1", false),
    instance<OperandClass::f16_mma, RowsClass::r1, cuda_nvfp4_rows_multi>(
        graph::Op::linear, "rows.multi", {"gguf.nvfp4_tile.strix", 1U}, device::ArchMask::sm120, "cuda-sm120-nvfp4-rows-multi-w4a16-v1", false),
    instance<OperandClass::f16_mma, RowsClass::r2_8, cuda_nvfp4_rows_multi>(
        graph::Op::linear, "rows.multi", {"gguf.nvfp4_tile.strix", 1U}, device::ArchMask::sm120, "cuda-sm120-nvfp4-rows-multi-w4a16-v1", false),
    instance<OperandClass::f16_mma, RowsClass::r9_32, cuda_nvfp4_rows_multi>(
        graph::Op::linear, "rows.multi", {"gguf.nvfp4_tile.strix", 1U}, device::ArchMask::sm120, "cuda-sm120-nvfp4-rows-multi-w4a16-v1", false),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_rms>(
        graph::Op::norm, "norm.rms", {"plain.f32", 1U}, device::ArchMask::sm120, "cuda-sm120-norm-rms-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, cuda_rms>(
        graph::Op::norm, "norm.rms", {"plain.f32", 1U}, device::ArchMask::sm120, "cuda-sm120-norm-rms-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, cuda_rms>(
        graph::Op::norm, "norm.rms", {"plain.f32", 1U}, device::ArchMask::sm120, "cuda-sm120-norm-rms-v1", true),
    instance<OperandClass::f32_fma, RowsClass::tall, cuda_rms>(
        graph::Op::norm, "norm.rms", {"plain.f32", 1U}, device::ArchMask::sm120, "cuda-sm120-norm-rms-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_rms_bf16>(
        graph::Op::norm, "norm.rms.bf16_store", {"plain.bf16", 1U}, device::ArchMask::sm120, "cuda-sm120-norm-rms-bf16-gain-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, cuda_rms_bf16>(
        graph::Op::norm, "norm.rms.bf16_store", {"plain.bf16", 1U}, device::ArchMask::sm120, "cuda-sm120-norm-rms-bf16-gain-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, cuda_rms_bf16>(
        graph::Op::norm, "norm.rms.bf16_store", {"plain.bf16", 1U}, device::ArchMask::sm120, "cuda-sm120-norm-rms-bf16-gain-v1", true),
    instance<OperandClass::f32_fma, RowsClass::tall, cuda_rms_bf16>(
        graph::Op::norm, "norm.rms.bf16_store", {"plain.bf16", 1U}, device::ArchMask::sm120, "cuda-sm120-norm-rms-bf16-gain-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_l2>(
        graph::Op::norm, "norm.l2", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-norm-l2-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, cuda_l2>(
        graph::Op::norm, "norm.l2", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-norm-l2-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, cuda_l2>(
        graph::Op::norm, "norm.l2", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-norm-l2-v1", true),
    instance<OperandClass::f32_fma, RowsClass::tall, cuda_l2>(
        graph::Op::norm, "norm.l2", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-norm-l2-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_gated_rms_swish>(
        graph::Op::norm, "norm.gated_rms.swish", {"plain.f32", 1U}, device::ArchMask::sm120, "cuda-sm120-norm-gated-rms-swish-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, cuda_gated_rms_swish>(
        graph::Op::norm, "norm.gated_rms.swish", {"plain.f32", 1U}, device::ArchMask::sm120, "cuda-sm120-norm-gated-rms-swish-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, cuda_gated_rms_swish>(
        graph::Op::norm, "norm.gated_rms.swish", {"plain.f32", 1U}, device::ArchMask::sm120, "cuda-sm120-norm-gated-rms-swish-v1", true),
    instance<OperandClass::f32_fma, RowsClass::tall, cuda_gated_rms_swish>(
        graph::Op::norm, "norm.gated_rms.swish", {"plain.f32", 1U}, device::ArchMask::sm120, "cuda-sm120-norm-gated-rms-swish-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_gated_rms_sigmoid>(
        graph::Op::norm, "norm.gated_rms.sigmoid", {"plain.f32", 1U}, device::ArchMask::sm120, "cuda-sm120-norm-gated-rms-sigmoid-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, cuda_gated_rms_sigmoid>(
        graph::Op::norm, "norm.gated_rms.sigmoid", {"plain.f32", 1U}, device::ArchMask::sm120, "cuda-sm120-norm-gated-rms-sigmoid-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, cuda_gated_rms_sigmoid>(
        graph::Op::norm, "norm.gated_rms.sigmoid", {"plain.f32", 1U}, device::ArchMask::sm120, "cuda-sm120-norm-gated-rms-sigmoid-v1", true),
    instance<OperandClass::f32_fma, RowsClass::tall, cuda_gated_rms_sigmoid>(
        graph::Op::norm, "norm.gated_rms.sigmoid", {"plain.f32", 1U}, device::ArchMask::sm120, "cuda-sm120-norm-gated-rms-sigmoid-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_add>(
        graph::Op::combine, "residual.add", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-pointwise-add-inplace-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, cuda_add>(
        graph::Op::combine, "residual.add", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-pointwise-add-inplace-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, cuda_add>(
        graph::Op::combine, "residual.add", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-pointwise-add-inplace-v1", true),
    instance<OperandClass::f32_fma, RowsClass::tall, cuda_add>(
        graph::Op::combine, "residual.add", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-pointwise-add-inplace-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_mix_bank_readout, unpriced_single_launch, no_workspace>(
        graph::Op::mix, "residual.bank", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-residual-bank-softmax-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, cuda_mix_bank_readout, unpriced_single_launch, no_workspace>(
        graph::Op::mix, "residual.bank", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-residual-bank-softmax-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, cuda_mix_bank_readout, unpriced_single_launch, no_workspace>(
        graph::Op::mix, "residual.bank", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-residual-bank-softmax-v1", true),
    instance<OperandClass::f32_fma, RowsClass::tall, cuda_mix_bank_readout, unpriced_single_launch, no_workspace>(
        graph::Op::mix, "residual.bank", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-residual-bank-softmax-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_mix_bank_readout_tiled, unpriced_single_launch, no_workspace>(
        graph::Op::mix, "residual.bank.tiled", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-residual-bank-softmax-tiled-r1-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_select_pick_ordered>(
        graph::Op::select, "pick.ordered", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-pick-ordered-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, cuda_select_pick_ordered>(
        graph::Op::select, "pick.ordered", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-pick-ordered-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, cuda_select_pick_ordered>(
        graph::Op::select, "pick.ordered", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-pick-ordered-v1", true),
    instance<OperandClass::f32_fma, RowsClass::tall, cuda_select_pick_ordered>(
        graph::Op::select, "pick.ordered", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-pick-ordered-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_select_pick_float>(
        graph::Op::select, "pick.float", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-pick-float-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, cuda_select_pick_float>(
        graph::Op::select, "pick.float", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-pick-float-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, cuda_select_pick_float>(
        graph::Op::select, "pick.float", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-pick-float-v1", true),
    instance<OperandClass::f32_fma, RowsClass::tall, cuda_select_pick_float>(
        graph::Op::select, "pick.float", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-pick-float-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_select_pick_raw>(
        graph::Op::select, "pick.raw_float", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-pick-raw-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, cuda_select_pick_raw>(
        graph::Op::select, "pick.raw_float", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-pick-raw-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, cuda_select_pick_raw>(
        graph::Op::select, "pick.raw_float", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-pick-raw-v1", true),
    instance<OperandClass::f32_fma, RowsClass::tall, cuda_select_pick_raw>(
        graph::Op::select, "pick.raw_float", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-pick-raw-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_select_pick_mask>(
        graph::Op::select, "pick.eligible_float", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-pick-mask-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, cuda_select_pick_mask>(
        graph::Op::select, "pick.eligible_float", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-pick-mask-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, cuda_select_pick_mask>(
        graph::Op::select, "pick.eligible_float", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-pick-mask-v1", true),
    instance<OperandClass::f32_fma, RowsClass::tall, cuda_select_pick_mask>(
        graph::Op::select, "pick.eligible_float", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-pick-mask-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_select_score_relu_tree>(
        graph::Op::select, "score.relu_tree", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-relu-tree-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, cuda_select_score_relu_tree>(
        graph::Op::select, "score.relu_tree", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-relu-tree-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, cuda_select_score_relu_tree>(
        graph::Op::select, "score.relu_tree", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-relu-tree-v1", true),
    instance<OperandClass::f32_fma, RowsClass::tall, cuda_select_score_relu_tree>(
        graph::Op::select, "score.relu_tree", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-relu-tree-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_select_score_weighted_fma>(
        graph::Op::select, "score.weighted_fma", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-weighted-fma-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, cuda_select_score_weighted_fma>(
        graph::Op::select, "score.weighted_fma", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-weighted-fma-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, cuda_select_score_weighted_fma>(
        graph::Op::select, "score.weighted_fma", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-weighted-fma-v1", true),
    instance<OperandClass::f32_fma, RowsClass::tall, cuda_select_score_weighted_fma>(
        graph::Op::select, "score.weighted_fma", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-weighted-fma-v1", true),
    instance<OperandClass::bf16_mma, RowsClass::r1, cuda_select_score_weighted_bf16>(
        graph::Op::select, "score.weighted_bf16", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-weighted-bf16-v1", true),
    instance<OperandClass::bf16_mma, RowsClass::r2_8, cuda_select_score_weighted_bf16>(
        graph::Op::select, "score.weighted_bf16", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-weighted-bf16-v1", true),
    instance<OperandClass::bf16_mma, RowsClass::r9_32, cuda_select_score_weighted_bf16>(
        graph::Op::select, "score.weighted_bf16", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-weighted-bf16-v1", true),
    instance<OperandClass::bf16_mma, RowsClass::tall, cuda_select_score_weighted_bf16>(
        graph::Op::select, "score.weighted_bf16", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-weighted-bf16-v1", true),
    instance<OperandClass::bf16_mma, RowsClass::r1, cuda_select_score_weighted_bf16x3>(
        graph::Op::select, "score.weighted_bf16x3", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-weighted-bf16x3-v1", true),
    instance<OperandClass::bf16_mma, RowsClass::r2_8, cuda_select_score_weighted_bf16x3>(
        graph::Op::select, "score.weighted_bf16x3", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-weighted-bf16x3-v1", true),
    instance<OperandClass::bf16_mma, RowsClass::r9_32, cuda_select_score_weighted_bf16x3>(
        graph::Op::select, "score.weighted_bf16x3", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-weighted-bf16x3-v1", true),
    instance<OperandClass::bf16_mma, RowsClass::tall, cuda_select_score_weighted_bf16x3>(
        graph::Op::select, "score.weighted_bf16x3", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-weighted-bf16x3-v1", true),
    instance<OperandClass::i8_dp4a, RowsClass::r1, cuda_select_score_weighted_fp4>(
        graph::Op::select, "score.weighted_fp4", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-weighted-fp4-v1", true),
    instance<OperandClass::i8_dp4a, RowsClass::r2_8, cuda_select_score_weighted_fp4>(
        graph::Op::select, "score.weighted_fp4", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-weighted-fp4-v1", true),
    instance<OperandClass::i8_dp4a, RowsClass::r9_32, cuda_select_score_weighted_fp4>(
        graph::Op::select, "score.weighted_fp4", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-weighted-fp4-v1", true),
    instance<OperandClass::i8_dp4a, RowsClass::tall, cuda_select_score_weighted_fp4>(
        graph::Op::select, "score.weighted_fp4", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-weighted-fp4-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_select_score_block_max>(
        graph::Op::select, "score.block_max", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-block-max-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, cuda_select_score_block_max>(
        graph::Op::select, "score.block_max", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-block-max-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, cuda_select_score_block_max>(
        graph::Op::select, "score.block_max", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-block-max-v1", true),
    instance<OperandClass::f32_fma, RowsClass::tall, cuda_select_score_block_max>(
        graph::Op::select, "score.block_max", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-block-max-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_select_score_block_max_tiled>(
        graph::Op::select, "score.block_max_tiled", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-block-max-tiled-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, cuda_select_score_block_max_tiled>(
        graph::Op::select, "score.block_max_tiled", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-block-max-tiled-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, cuda_select_score_block_max_tiled>(
        graph::Op::select, "score.block_max_tiled", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-block-max-tiled-v1", true),
    instance<OperandClass::f32_fma, RowsClass::tall, cuda_select_score_block_max_tiled>(
        graph::Op::select, "score.block_max_tiled", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-block-max-tiled-v1", true),
    instance<OperandClass::fp8_mma, RowsClass::r1, cuda_select_score_weighted_fp8>(
        graph::Op::select, "score.weighted_fp8", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-weighted-fp8-v1", true),
    instance<OperandClass::fp8_mma, RowsClass::r2_8, cuda_select_score_weighted_fp8>(
        graph::Op::select, "score.weighted_fp8", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-weighted-fp8-v1", true),
    instance<OperandClass::fp8_mma, RowsClass::r9_32, cuda_select_score_weighted_fp8>(
        graph::Op::select, "score.weighted_fp8", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-weighted-fp8-v1", true),
    instance<OperandClass::fp8_mma, RowsClass::tall, cuda_select_score_weighted_fp8>(
        graph::Op::select, "score.weighted_fp8", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-select-score-weighted-fp8-v1", true),
    instance<OperandClass::bf16_mma, RowsClass::r1, cuda_decode_bf16_output_split, unpriced_single_launch, no_workspace>(
        graph::Op::linear, "r1.bf16_image.output_split_round", {"plain.bf16", 1U}, device::ArchMask::sm120, "cuda-sm120-bf16-r1-mma3-output-split-round-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_embedding, unpriced_single_launch, workspace>(
        graph::Op::gather, "embedding.bf16_to_f32", {"plain.bf16", 1U}, device::ArchMask::sm120, "cuda-sm120-embedding-bf16-r1-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_rms, unpriced_single_launch, workspace>(
        graph::Op::norm, "rms.bf16_gain.f32_image", {"plain.bf16", 1U}, device::ArchMask::sm120, "cuda-sm120-rms-bf16-image-r1-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_append_kv, unpriced_single_launch, workspace>(
        graph::Op::store, "prepared.kv.bf16", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-prepared-bf16-kv-step-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_prepared_flash, unpriced_single_launch, workspace>(
        graph::Op::attend, "prepared.flash.pad2.d128", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-prepared-flash-pad2-d128-v1", true),
    [] {
        auto e = instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_prepared_flash_split, unpriced_single_launch, workspace>(
        graph::Op::attend, "prepared.flash.split.d128", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-prepared-flash-decode-split-d128-v1", true);
        e.exact = lever::Exactness::bit_changing;
        return e;
    }(),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_finite, unpriced_single_launch, workspace>(
        graph::Op::elementwise, "prepared.finite_check", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-prepared-finite-check-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_convolution, unpriced_single_launch, workspace>(
        graph::Op::elementwise, "causal_convolution.decode", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-causal-convolution-r1-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_clip, unpriced_single_launch, workspace>(
        graph::Op::norm, "head_clip.bf16", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-head-clip-bf16-r1-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_normalize, unpriced_single_launch, workspace>(
        graph::Op::norm, "residual.bank", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-residual-bank-norm-r1-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_add, unpriced_single_launch, workspace>(
        graph::Op::combine, "residual.add.out_of_place", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-residual-add-r1-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_controller, unpriced_single_launch, workspace>(
        graph::Op::combine, "residual.controller", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-residual-controller-r1-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_round_bf16, unpriced_single_launch, workspace>(
        graph::Op::elementwise, "bf16.round", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-pointwise-bf16-round-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_sigmoid, unpriced_single_launch, workspace>(
        graph::Op::elementwise, "bf16.sigmoid_product", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-pointwise-bf16-sigmoid-product-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_softcap, unpriced_single_launch, workspace>(
        graph::Op::elementwise, "bf16.softcap", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-pointwise-bf16-softcap-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_relative, unpriced_single_launch, workspace>(
        graph::Op::linear, "relative.bf16.d16", {"plain.bf16", 1U}, device::ArchMask::sm120, "cuda-sm120-relative-bf16-d16-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_dense_fp32, unpriced_single_launch, workspace>(
        graph::Op::linear, "r1.bf16_image.group", {"plain.bf16", 1U}, device::ArchMask::sm120, "cuda-sm120-bf16-r1-fp32-group-v1", true),
    instance<OperandClass::bf16_mma, RowsClass::r1, cuda_decode_dense_mma3, unpriced_single_launch, workspace>(
        graph::Op::linear, "r1.bf16_image.group", {"plain.bf16", 1U}, device::ArchMask::sm120, "cuda-sm120-bf16-r1-mma3-group-v1", true),
    instance<OperandClass::bf16_mma, RowsClass::r1, cuda_decode_independent, unpriced_single_launch, workspace>(
        graph::Op::linear, "r1.bf16_image.independent", {"plain.bf16", 1U}, device::ArchMask::sm120, "cuda-sm120-bf16-r1-mma3-independent-v1", true),
    instance<OperandClass::bf16_mma, RowsClass::r1, cuda_decode_dense_round, unpriced_single_launch, workspace>(
        graph::Op::linear, "r1.bf16_image.group_round", {"plain.bf16", 1U}, device::ArchMask::sm120, "cuda-sm120-bf16-r1-mma3-group-round-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_pair_norm, unpriced_single_launch, workspace>(
        graph::Op::norm, "residual.controller_pair", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-controller-pair-norm-r1-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_bank_batch, unpriced_single_launch, no_workspace>(
        graph::Op::mix, "residual.bank.batch", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-residual-bank-batch-tiled-r1-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_gate_up_nvfp4, unpriced_single_launch, workspace>(
        graph::Op::linear, "microscaled.packed_gate_up.k8", {"nvidia.nvfp4", 1U}, device::ArchMask::sm120, "cuda-sm120-packed-nvfp4-gate-up-r1-k8-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_gate_up_mxfp8, unpriced_single_launch, workspace>(
        graph::Op::linear, "microscaled.packed_gate_up.k8", {"block.mxfp8.e4m3-e8m0.g32", 1U}, device::ArchMask::sm120, "cuda-sm120-packed-mxfp8-gate-up-r1-k8-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_down_combine_nvfp4, unpriced_single_launch, workspace>(
        graph::Op::combine, "microscaled.packed_down_combine.k8", {"nvidia.nvfp4", 1U}, device::ArchMask::sm120, "cuda-sm120-packed-nvfp4-down-combine-r1-k8-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_down_combine_mxfp8, unpriced_single_launch, workspace>(
        graph::Op::combine, "microscaled.packed_down_combine.k8", {"block.mxfp8.e4m3-e8m0.g32", 1U}, device::ArchMask::sm120, "cuda-sm120-packed-mxfp8-down-combine-r1-k8-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_pick_part, unpriced_single_launch, workspace>(
        graph::Op::argmax, "greedy.partial", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-greedy-partial-r1-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_pick_final, unpriced_single_launch, workspace>(
        graph::Op::argmax, "greedy.final", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-greedy-final-r1-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_dense_fp32_round, unpriced_single_launch, workspace>(
        graph::Op::linear, "r1.bf16_image.fp32_round", {"plain.bf16", 1U}, device::ArchMask::sm120, "cuda-sm120-bf16-r1-fp32-round-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_dense_fp32_softcap, unpriced_single_launch, workspace>(
        graph::Op::linear, "r1.bf16_image.fp32_softcap", {"plain.bf16", 1U}, device::ArchMask::sm120, "cuda-sm120-bf16-r1-fp32-softcap-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_convolution_round, unpriced_single_launch, workspace>(
        graph::Op::elementwise, "causal_convolution.decode_round", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-causal-convolution-round-r1-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_embedding_q8, unpriced_single_launch, workspace>(
        graph::Op::gather, "embedding.q8_to_f32", {"gguf.q8_0", 1U}, device::ArchMask::sm120, "cuda-sm120-embedding-q8-r1-v1", true),
    instance<OperandClass::bf16_mma, RowsClass::r1, cuda_decode_gate_up_mma3, unpriced_single_launch, workspace>(
        graph::Op::linear, "rows.gate_up_silu", {"gguf.q8_0", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-gate-up-mma3-r1-v1", true),
    instance<OperandClass::bf16_mma, RowsClass::r1, cuda_decode_gate_up_mma3, unpriced_single_launch, workspace>(
        graph::Op::linear, "rows.gate_up_silu", {"plain.bf16", 1U}, device::ArchMask::sm120, "cuda-sm120-dense-gate-up-mma3-r1-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_rope, unpriced_single_launch, workspace>(
        graph::Op::elementwise, "rope.device_position", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-rope-device-position-r1-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_kv_store, unpriced_single_launch, workspace>(
        graph::Op::store, "kv.bf16.linear", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-kv-store-bf16-linear-r1-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_pick_part_pdl, unpriced_single_launch, workspace>(
        graph::Op::argmax, "greedy.partial.pdl", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-greedy-partial-pdl-r1-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_pick_final_pdl_index, unpriced_single_launch, workspace>(
        graph::Op::argmax, "greedy.final.pdl_index", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-greedy-final-pdl-index-r1-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_gdn_convolution, unpriced_single_launch, workspace>(
        graph::Op::elementwise, "gdn.convolution.parity", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-gdn-convolution-parity-r1-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_gdn_recurrence, unpriced_single_launch, workspace>(
        graph::Op::elementwise, "gdn.recurrence.parity", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-gdn-recurrence-parity-r1-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_attention_scores, unpriced_single_launch, workspace>(
        graph::Op::attend, "attention.decode.bf16.scores", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-attention-decode-bf16-scores-r1-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_attention_values, unpriced_single_launch, workspace>(
        graph::Op::attend, "attention.decode.bf16.values", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-attention-decode-bf16-values-r1-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_decode_attention_combine, unpriced_single_launch, workspace>(
        graph::Op::combine, "attention.decode.bf16.combine", {"", 0U}, device::ArchMask::sm120, "cuda-sm120-attention-decode-bf16-combine-r1-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_moer_pipe_gate_up_nvfp4, unpriced_single_launch, no_workspace>(
        graph::Op::linear, "microscaled.packed_gate_up.k8.pipe", {"nvidia.nvfp4", 1U}, device::ArchMask::sm120, "cuda-sm120-packed-nvfp4-gate-up-r1-k8-pipe-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_moer_pipe_gate_up_mxfp8, unpriced_single_launch, no_workspace>(
        graph::Op::linear, "microscaled.packed_gate_up.k8.pipe", {"block.mxfp8.e4m3-e8m0.g32", 1U}, device::ArchMask::sm120, "cuda-sm120-packed-mxfp8-gate-up-r1-k8-pipe-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_moer_pipe_down_combine_nvfp4, unpriced_single_launch, no_workspace>(
        graph::Op::combine, "microscaled.packed_down_combine.k8.pipe", {"nvidia.nvfp4", 1U}, device::ArchMask::sm120, "cuda-sm120-packed-nvfp4-down-combine-r1-k8-pipe-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, cuda_moer_pipe_down_combine_mxfp8, unpriced_single_launch, no_workspace>(
        graph::Op::combine, "microscaled.packed_down_combine.k8.pipe", {"block.mxfp8.e4m3-e8m0.g32", 1U}, device::ArchMask::sm120, "cuda-sm120-packed-mxfp8-down-combine-r1-k8-pipe-v1", true),
};

constexpr KernelEntry hip_entries[] = {
    instance<OperandClass::f32_fma, RowsClass::r1, hip_dense_linear>(
        graph::Op::linear, "rows", {"plain.bf16", 1U}, device::ArchMask::gfx1151, "hip-gfx1151-dense-rows-direct-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, hip_dense_linear>(
        graph::Op::linear, "rows", {"plain.bf16", 1U}, device::ArchMask::gfx1151, "hip-gfx1151-dense-rows-direct-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, hip_dense_linear>(
        graph::Op::linear, "rows", {"plain.bf16", 1U}, device::ArchMask::gfx1151, "hip-gfx1151-dense-rows-direct-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, hip_fma_rows>(
        graph::Op::linear, "rows.column", {"plain.f16", 1U}, device::ArchMask::gfx1151, "hip-gfx1151-dense-rows-column-pipe128-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, hip_fma_rows>(
        graph::Op::linear, "rows.column", {"plain.f16", 1U}, device::ArchMask::gfx1151, "hip-gfx1151-dense-rows-column-pipe128-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, hip_fma_rows>(
        graph::Op::linear, "rows.column", {"plain.f16", 1U}, device::ArchMask::gfx1151, "hip-gfx1151-dense-rows-column-pipe128-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, hip_fma_rows>(
        graph::Op::linear, "rows.column", {"plain.bf16", 1U}, device::ArchMask::gfx1151, "hip-gfx1151-dense-rows-column-pipe128-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, hip_fma_rows>(
        graph::Op::linear, "rows.column", {"plain.bf16", 1U}, device::ArchMask::gfx1151, "hip-gfx1151-dense-rows-column-pipe128-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, hip_fma_rows>(
        graph::Op::linear, "rows.column", {"plain.bf16", 1U}, device::ArchMask::gfx1151, "hip-gfx1151-dense-rows-column-pipe128-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, hip_fma_rows>(
        graph::Op::linear, "rows.column", {"plain.f32", 1U}, device::ArchMask::gfx1151, "hip-gfx1151-dense-rows-column-pipe128-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, hip_fma_rows>(
        graph::Op::linear, "rows.column", {"plain.f32", 1U}, device::ArchMask::gfx1151, "hip-gfx1151-dense-rows-column-pipe128-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, hip_fma_rows>(
        graph::Op::linear, "rows.column", {"plain.f32", 1U}, device::ArchMask::gfx1151, "hip-gfx1151-dense-rows-column-pipe128-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, hip_q8_gemv>(
        graph::Op::linear, "rows", {"gguf.q8_0", 1U}, device::ArchMask::gfx1151, "hip-gfx1151-dense-rows-q8-fp32-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, hip_rms_bf16>(
        graph::Op::norm, "norm.rms.bf16_store", {"plain.bf16", 1U}, device::ArchMask::gfx1151, "hip-gfx1151-norm-rms-bf16-store-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, hip_rms_bf16>(
        graph::Op::norm, "norm.rms.bf16_store", {"plain.bf16", 1U}, device::ArchMask::gfx1151, "hip-gfx1151-norm-rms-bf16-store-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, hip_rms_bf16>(
        graph::Op::norm, "norm.rms.bf16_store", {"plain.bf16", 1U}, device::ArchMask::gfx1151, "hip-gfx1151-norm-rms-bf16-store-v1", true),
    instance<OperandClass::f32_fma, RowsClass::tall, hip_rms_bf16>(
        graph::Op::norm, "norm.rms.bf16_store", {"plain.bf16", 1U}, device::ArchMask::gfx1151, "hip-gfx1151-norm-rms-bf16-store-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, hip_add>(
        graph::Op::combine, "residual.add", {"", 0U}, device::ArchMask::gfx1151, "hip-gfx1151-pointwise-add-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, hip_add>(
        graph::Op::combine, "residual.add", {"", 0U}, device::ArchMask::gfx1151, "hip-gfx1151-pointwise-add-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, hip_add>(
        graph::Op::combine, "residual.add", {"", 0U}, device::ArchMask::gfx1151, "hip-gfx1151-pointwise-add-v1", true),
    instance<OperandClass::f32_fma, RowsClass::tall, hip_add>(
        graph::Op::combine, "residual.add", {"", 0U}, device::ArchMask::gfx1151, "hip-gfx1151-pointwise-add-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r1, hip_swiglu>(
        graph::Op::elementwise, "act.swiglu", {"", 0U}, device::ArchMask::gfx1151, "hip-gfx1151-pointwise-swiglu-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r2_8, hip_swiglu>(
        graph::Op::elementwise, "act.swiglu", {"", 0U}, device::ArchMask::gfx1151, "hip-gfx1151-pointwise-swiglu-v1", true),
    instance<OperandClass::f32_fma, RowsClass::r9_32, hip_swiglu>(
        graph::Op::elementwise, "act.swiglu", {"", 0U}, device::ArchMask::gfx1151, "hip-gfx1151-pointwise-swiglu-v1", true),
    instance<OperandClass::f32_fma, RowsClass::tall, hip_swiglu>(
        graph::Op::elementwise, "act.swiglu", {"", 0U}, device::ArchMask::gfx1151, "hip-gfx1151-pointwise-swiglu-v1", true),
};
// END D4 INSTANCES

Catalog make_catalog(Backend backend) {
    Catalog result;
    result.add(Catalog::builtin_entries(backend));
    return result;
}

} // namespace

bool KernelKey::operator==(const KernelKey& other) const noexcept { return key_fields(*this) == key_fields(other); }

bool Catalog::Less::operator()(const KernelKey& lhs, const KernelKey& rhs) const noexcept {
    return key_fields(lhs) < key_fields(rhs);
}

void Catalog::add(std::span<const KernelEntry> entries) {
    // Stage additions so a failed allocation or validation leaves existing
    // lookups valid. Node merge also preserves successful lookup spans.
    decltype(entries_) added;
    for (const KernelEntry& entry : entries) {
        if (entry.key.form.empty() || entry.arch == device::ArchMask::none || entry.identity.empty() ||
            !entry.cost || !entry.workspace || !entry.bind || !entry.launcher || entry.launcher->symbol.empty() ||
            entry.launcher->source.empty() || entry.launcher->max_rows == 0) {
            throw std::invalid_argument("incomplete kernel entry");
        }
        if (entries_.contains(entry.key) || !added.emplace(entry.key, &entry).second) {
            throw std::invalid_argument("duplicate kernel key");
        }
    }
    entries_.merge(added);
}

std::span<const KernelEntry* const> Catalog::find(const KernelKey& key, device::ArchMask arches) const {
    const auto it = entries_.find(key);
    if (it == entries_.end() || (it->second->arch & arches) == device::ArchMask::none) return {};
    return {&it->second, 1};
}

std::span<const KernelEntry* const> Catalog::find(const KernelKey& key, const device::Capabilities& caps) const {
    return find(key, caps.major == 12 && caps.minor == 0 ? device::ArchMask::sm120 : device::ArchMask::none);
}

std::span<const KernelEntry> Catalog::builtin_entries(Backend backend) noexcept {
    switch (backend) {
    case Backend::cuda: return cuda_entries;
    case Backend::hip: return hip_entries;
    case Backend::cpu:
    case Backend::sycl: return {};
    }
    return {};
}

const Catalog& Catalog::builtin(Backend backend) {
    switch (backend) {
    case Backend::cuda: {
        static const Catalog cuda = make_catalog(Backend::cuda);
        return cuda;
    }
    case Backend::hip: {
        static const Catalog hip = make_catalog(Backend::hip);
        return hip;
    }
    case Backend::cpu:
    case Backend::sycl: {
        static const Catalog empty;
        return empty;
    }
    }
    throw std::invalid_argument("unknown catalogue backend");
}

} // namespace strix::exec
