// deepseek41 K2 routed experts on the shared trellis MoE family (see hip_ds41_k2_experts.hpp). Compile with
// -ffp-contract=off (engine flags): the finishes keep the library's operation order.
#include "strix/hip_ds41_k2_experts.hpp"

#include "moe_core/split_trellis_kernels.hpp"

#include <hip/hip_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace strix::ds41::k2 {
namespace {

namespace kernels = strix::moe_core::split_trellis;

constexpr int kBits = 2;
constexpr int kWaves = 4;   // waves per gate/up and down CTA (one 16-column tile column each)
constexpr int kDepth = 8;   // k tiles in flight per wave (R1 single depth of the strix-trellis S4 configuration)
constexpr unsigned kStageHalves = 1024;  // LDS halves per CTA: >= the 128-blocks covering a gate split, >= a down split

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("ds41 k2 experts: ") + what + ": " + hipGetErrorString(e));
}

}  // namespace

DeviceLayer upload_layer(const LayerImage& image) {
    DeviceLayer L;
    L.layer = image.layer;
    L.expert_first = image.expert_first;
    L.expert_count = image.expert_count;
    L.hidden = image.geometry.hidden;
    L.inter = image.geometry.inter;
    L.local = image.slice.count;
    L.plan = split_plan(image.geometry, image.slice);
    L.tile_bytes = image.gate_tile_bytes;
    if (image.geometry.bits != static_cast<std::uint32_t>(kBits)) throw std::runtime_error("ds41 k2 experts: the device form is K2 only");
    if (L.hidden % (16U * kWaves) != 0U || L.local % (16U * kWaves) != 0U || L.local % 128U != 0U)
        throw std::runtime_error("ds41 k2 experts: H / I range not a multiple of the CTA tile columns");
    const unsigned gate_ch = L.hidden / L.plan.gate_splits;
    const unsigned down_ch = L.inter / L.plan.down_splits;
    if ((gate_ch + 254U) / 128U * 128U > kStageHalves || down_ch > kStageHalves || gate_ch % 16U != 0U || down_ch % 16U != 0U)
        throw std::runtime_error("ds41 k2 experts: split widths do not fit the LDS stage");
    const std::size_t tiles = image.gate.size() + image.up.size() + image.down.size();
    const std::size_t halves = image.gate_suh.size() + image.up_suh.size() + image.down_suh.size() + image.gate_svh.size() +
                               image.up_svh.size() + image.down_svh.size();
    std::vector<std::uint8_t> host(tiles + 2U * halves);
    std::size_t at = 0;
    auto put = [&](const void* src, std::size_t bytes) {
        std::memcpy(host.data() + at, src, bytes);
        const std::size_t off = at;
        at += bytes;
        return off;
    };
    const std::size_t o_gate = put(image.gate.data(), image.gate.size()), o_up = put(image.up.data(), image.up.size()),
                      o_down = put(image.down.data(), image.down.size());
    const std::size_t o_gsuh = put(image.gate_suh.data(), 2U * image.gate_suh.size()), o_usuh = put(image.up_suh.data(), 2U * image.up_suh.size()),
                      o_dsuh = put(image.down_suh.data(), 2U * image.down_suh.size());
    const std::size_t o_gsvh = put(image.gate_svh.data(), 2U * image.gate_svh.size()), o_usvh = put(image.up_svh.data(), 2U * image.up_svh.size()),
                      o_dsvh = put(image.down_svh.data(), 2U * image.down_svh.size());
    check(hipMalloc(&L.memory, host.size()), "hipMalloc");
    check(hipMemcpy(L.memory, host.data(), host.size(), hipMemcpyHostToDevice), "upload");
    auto* base = static_cast<std::uint8_t*>(L.memory);
    L.gate = reinterpret_cast<const std::uint32_t*>(base + o_gate);
    L.up = reinterpret_cast<const std::uint32_t*>(base + o_up);
    L.down = reinterpret_cast<const std::uint32_t*>(base + o_down);
    L.gate_suh = reinterpret_cast<const std::uint16_t*>(base + o_gsuh);
    L.up_suh = reinterpret_cast<const std::uint16_t*>(base + o_usuh);
    L.down_suh = reinterpret_cast<const std::uint16_t*>(base + o_dsuh);
    L.gate_svh = reinterpret_cast<const std::uint16_t*>(base + o_gsvh);
    L.up_svh = reinterpret_cast<const std::uint16_t*>(base + o_usvh);
    L.down_svh = reinterpret_cast<const std::uint16_t*>(base + o_dsvh);
    return L;
}

void free_layer(DeviceLayer& layer) noexcept {
    if (layer.memory != nullptr) (void)hipFree(layer.memory);
    layer = DeviceLayer{};
}

std::size_t gu_part_floats(const DeviceLayer& layer, std::uint32_t entries) noexcept {
    return std::size_t{entries} * 2U * layer.plan.gate_splits * layer.local;
}
std::size_t down_part_floats(const DeviceLayer& layer, std::uint32_t entries) noexcept {
    return std::size_t{entries} * layer.plan.down_splits * layer.hidden;
}

void enqueue_k2_experts(const DecodeView& v, void* stream) {
    const DeviceLayer& L = *v.layer;
    const auto s = static_cast<hipStream_t>(stream);
    const unsigned entries = v.rows * v.top_k;
    if (entries == 0U) return;
    const dim3 gate_grid(L.local / 16U / kWaves, L.plan.gate_splits, entries * 2U);
    kernels::gate_up_kernel<DeviceLayer, kBits, kWaves, kDepth, kStageHalves>
        <<<gate_grid, 32 * kWaves, 0, s>>>(L, v.x, v.ids, v.top_k, v.gu_part);
    check(hipGetLastError(), "gate/up launch");
    kernels::finish_gate_up_kernel<DeviceLayer><<<dim3(L.local / 128U, entries), 32, 0, s>>>(
        L, v.ids, v.swiglu_limit, v.gu_part, v.h);
    check(hipGetLastError(), "finish launch");
    const dim3 down_grid(L.hidden / 16U / kWaves, L.plan.down_end - L.plan.down_first, entries);
    kernels::down_kernel<DeviceLayer, kBits, kWaves, kDepth, kStageHalves>
        <<<down_grid, 32 * kWaves, 0, s>>>(L, v.ids, v.h, v.down_part);
    check(hipGetLastError(), "down launch");
}

void enqueue_k2_down_finish(const DecodeView& v, void* stream) {
    const DeviceLayer& L = *v.layer;
    if (v.rows == 0U) return;
    const auto s = static_cast<hipStream_t>(stream);
    kernels::finish_down_kernel<DeviceLayer><<<dim3(L.hidden / 128U, v.rows), 32, 0, s>>>(
        L, v.ids, v.weights, v.top_k, v.down_part, v.y);
    check(hipGetLastError(), "down finish launch");
}

}  // namespace strix::ds41::k2
