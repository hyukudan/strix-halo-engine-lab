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
    return 0; // Populate when admitting an operand class with a matching device fact.
}

// An optimistic roofline estimate, in microseconds. Missing required profile
// facts yield infinity. Launch overhead and utilization need calibration.
double roof_time(double work, double rate, double units) {
    if (work == 0) return 0;
    return rate > 0 && std::isfinite(rate) ? work / rate / units : std::numeric_limits<double>::infinity();
}

template <OperandClass Act>
Cost estimate(const LaunchSite& site, const device::Profile& profile) {
    const auto bytes = static_cast<double>(site.weight_bytes) + static_cast<double>(site.state_bytes);
    const double memory = roof_time(bytes, profile.memory_gbytes_ps, 1e3);
    const double compute = roof_time(2.0 * static_cast<double>(site.macs), peak_tops<Act>(profile.peaks), 1e6);
    return {std::max(memory, compute), site.weight_bytes, site.state_bytes, site.macs, site.launches};
}

void workspace(const LaunchSite& site, WorkspaceNeed& need) { need = site.scratch; }

RowsClass rows_class(std::uint32_t rows) {
    if (rows == 1) return RowsClass::r1;
    if (rows <= 8) return RowsClass::r2_8;
    if (rows <= 32) return RowsClass::r9_32;
    return RowsClass::tall;
}

template <RowsClass Rows, const LauncherRef& Launcher>
BoundLaunch bind_existing(const LaunchSite& site, Arena& arena) {
    if (!arena.resolve || site.rows == 0 || site.rows > Launcher.max_rows || rows_class(site.rows) != Rows) return {};
    return arena.resolve(arena.context, Launcher, site);
}

template <OperandClass Act, RowsClass Rows, const LauncherRef& Launcher>
constexpr KernelEntry instance(graph::Op op, FormId form, weight::FormatId format, device::ArchMask arch,
                               std::string_view identity, bool capturable) {
    return {.key = {op, form, format, Act, Rows}, .arch = arch, .identity = identity,
            .exact = lever::Exactness::bitwise, .cost = estimate<Act>, .workspace = workspace,
            .bind = bind_existing<Rows, Launcher>,
            .launcher = &Launcher, .capturable = capturable};
}

// BEGIN D4 INSTANCES
constexpr LauncherRef cuda_dense_rows{"strix::cuda_core::dense_rows",
    "src/cuda_core/dense/dense.cu", "fp32", 32U};
constexpr LauncherRef cuda_dense_rows_multi{"strix::cuda_core::dense_rows_multi",
    "src/cuda_core/dense/dense.cu", "fp32", 32U};
constexpr LauncherRef cuda_dense_mma{"strix::cuda_core::dense_mma_launch",
    "src/cuda_core/dense/dense_mma.cu", "mma3", 32U};
constexpr LauncherRef cuda_nvfp4_rows{"strix::cuda_core::nvfp4_dense_rows",
    "src/cuda_core/dense/nvfp4.cu", "w4a16.split_exact", 32U};
constexpr LauncherRef cuda_nvfp4_rows_multi{"strix::cuda_core::nvfp4_dense_rows_multi",
    "src/cuda_core/dense/nvfp4.cu", "w4a16.split_exact", 32U};
constexpr LauncherRef hip_dense_linear{"strix::core::dense_linear::enqueue_bf16_tight_rows",
    "src/core/dense_linear.hpp", "tight.fp32_store", 16U};
constexpr LauncherRef hip_fma_rows{"strix::core::fma_rows::launch_auto",
    "src/core/fma_rows.hpp", "pipe128.fp32_store", 32U};
constexpr LauncherRef hip_q8_gemv{"strix::q8_rows::launch",
    "src/hip_q8_rows_gemv.hpp", "linear_f32", 1U};
constexpr LauncherRef cuda_rms{"strix::cuda_core::rmsnorm",
    "src/cuda_core/norms/norms.cu", "fp32_gain", 65535U};
constexpr LauncherRef cuda_rms_bf16{"strix::cuda_core::rmsnorm_bf16",
    "src/cuda_core/norms/norms.cu", "bf16_gain.bf16_store", 65535U};
constexpr LauncherRef cuda_l2{"strix::cuda_core::l2norm",
    "src/cuda_core/norms/norms.cu", "fp32", 65535U};
constexpr LauncherRef cuda_gated_rms_swish{"strix::cuda_core::gated_rmsnorm",
    "src/cuda_core/norms/norms.cu", "swish", 65535U};
constexpr LauncherRef cuda_gated_rms_sigmoid{"strix::cuda_core::gated_rmsnorm",
    "src/cuda_core/norms/norms.cu", "sigmoid", 65535U};
constexpr LauncherRef hip_rms_bf16{"strix::ops::norm::enqueue_rms_bf16_rows",
    "src/ops/norm/rms_bf16_rows.hpp", "parallel", 2147483647U};
constexpr LauncherRef cuda_add{"strix::cuda_core::add_inplace",
    "src/cuda_core/norms/norms.cu", "inplace", 4294967040U};
constexpr LauncherRef hip_add{"strix::core::pointwise::elements",
    "src/core/pointwise.hpp", "add", 4294967295U};
constexpr LauncherRef hip_swiglu{"strix::ops::glu::launch_swiglu",
    "src/ops/glu/glu_device.hpp", "fp32", 4294967295U};

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
