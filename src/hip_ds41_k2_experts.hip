// deepseek41 K2 routed experts on the shared trellis MoE family (see hip_ds41_k2_experts.hpp). Compile with
// -ffp-contract=off (engine flags): the finishes keep the library's operation order.
#include "strix/hip_ds41_k2_experts.hpp"

#include "strix/ds41_moe.hpp"

#include "moe_core/exl3_trellis.hpp"

#include <hip/hip_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace strix::ds41::k2 {
namespace {

namespace mc = strix::moe_core;

constexpr int kBits = 2;
constexpr int kWaves = 4;   // waves per gate/up and down CTA (one 16-column tile column each)
constexpr int kDepth = 8;   // k tiles in flight per wave (R1 single depth of the strix-trellis S4 configuration)
constexpr unsigned kStageHalves = 1024;  // LDS halves per CTA: >= the 128-blocks covering a gate split, >= a down split

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("ds41 k2 experts: ") + what + ": " + hipGetErrorString(e));
}

__device__ __forceinline__ const std::uint32_t* words_at(const std::uint32_t* base, std::uint64_t byte_offset, std::uint64_t back_bytes) {
    // back_bytes: a rank's down matrix holds k tiles [k0, ...); the reader addresses k tiles from 0.
    return reinterpret_cast<const std::uint32_t*>(reinterpret_cast<std::uintptr_t>(base) + byte_offset - back_bytes);
}

// grid (local / 16 / kWaves, gate_splits, entries * 2), block 32 * kWaves.
__global__ __launch_bounds__(32 * kWaves) void k2_gate_up_kernel(DeviceLayer L, const float* x, const std::int32_t* ids,
                                                                 std::uint32_t top_k, float* gu_part) {
    __shared__ _Float16 xs[kStageHalves];
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned entry = blockIdx.z >> 1U, m = blockIdx.z & 1U, split = blockIdx.y;
    const unsigned splits = L.plan.gate_splits, split_ch = L.hidden / splits;
    const unsigned e = static_cast<unsigned>(ids[entry]) - L.expert_first;
    const std::uint16_t* suh = (m == 0 ? L.gate_suh : L.up_suh) + static_cast<std::size_t>(e) * L.hidden;
    const float* row = x + static_cast<std::size_t>(entry / top_k) * L.hidden;
    const unsigned c0 = split * split_ch, b0 = c0 / 128U, b1 = (c0 + split_ch + 127U) / 128U;
    for (unsigned b = b0 + wave; b < b1; b += kWaves) mc::transform_block_at(row, suh, xs, b * 128U, (b - b0) * 128U, lane);
    __syncthreads();
    const unsigned col = blockIdx.x * kWaves + wave;
    const std::uint32_t* w = words_at(m == 0 ? L.gate : L.up, static_cast<std::uint64_t>(e) * L.tile_bytes, 0);
    float* part = gu_part + ((static_cast<std::size_t>(entry) * 2U + m) * splits + split) * L.local;
    mc::expert_split_rows<kBits, 1, kDepth>(w, L.local, L.hidden / 16U, splits, col, split, split_ch, xs + (c0 - b0 * 128U), 1, lane,
                                            part, 0);
}

// grid (local / 128, entries), block 32.
__global__ __launch_bounds__(32) void k2_finish_kernel(DeviceLayer L, const std::int32_t* ids, float limit, const float* gu_part,
                                                       _Float16* h) {
    const unsigned lane = threadIdx.x, block = blockIdx.x, entry = blockIdx.y;
    const unsigned splits = L.plan.gate_splits;
    const unsigned e = static_cast<unsigned>(ids[entry]) - L.expert_first;
    const float* pg = gu_part + (static_cast<std::size_t>(entry) * 2U) * splits * L.local + block * 128U + lane;
    const float* pu = pg + static_cast<std::size_t>(splits) * L.local;
    float g[4], u[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) g[j] = u[j] = 0.0F;
    for (unsigned s = 0; s < splits; ++s)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            g[j] += pg[static_cast<std::size_t>(s) * L.local + 32U * static_cast<unsigned>(j)];
            u[j] += pu[static_cast<std::size_t>(s) * L.local + 32U * static_cast<unsigned>(j)];
        }
    mc::hadamard128(g, lane);
    mc::hadamard128(u, lane);
    const std::size_t eo = static_cast<std::size_t>(e) * L.local;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned i = block * 128U + lane + 32U * static_cast<unsigned>(j);
        const float act = swiglu_clamped(g[j] * mc::half_bits(L.gate_svh[eo + i]), u[j] * mc::half_bits(L.up_svh[eo + i]), limit);
        const float scaled = mc::materialize(act * mc::half_bits(L.down_suh[eo + i]));
        v[j] = static_cast<float>(static_cast<_Float16>(scaled));
    }
    mc::hadamard128(v, lane);
    _Float16* out = h + static_cast<std::size_t>(entry) * L.local;
#pragma unroll
    for (int j = 0; j < 4; ++j) out[block * 128U + lane + 32U * static_cast<unsigned>(j)] = static_cast<_Float16>(v[j]);
}

// grid (H / 16 / kWaves, down_end - down_first, entries), block 32 * kWaves.
__global__ __launch_bounds__(32 * kWaves) void k2_down_kernel(DeviceLayer L, const std::int32_t* ids, const _Float16* h, float* down_part) {
    __shared__ _Float16 xs[kStageHalves];
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned entry = blockIdx.z, split = L.plan.down_first + blockIdx.y;
    const unsigned splits = L.plan.down_splits, inter = L.inter;
    const unsigned split_ch = inter / splits, local0 = split * split_ch - L.plan.down_k_tile0 * 16U;
    const unsigned e = static_cast<unsigned>(ids[entry]) - L.expert_first;
    const _Float16* src = h + static_cast<std::size_t>(entry) * L.local + local0;
    for (unsigned i = threadIdx.x; i < split_ch; i += 32U * kWaves) xs[i] = src[i];
    __syncthreads();
    const unsigned col = blockIdx.x * kWaves + wave;
    const std::uint64_t back = static_cast<std::uint64_t>(L.plan.down_k_tile0) * (L.hidden / 16U) * kBits * 32U;
    const std::uint32_t* w = words_at(L.down, static_cast<std::uint64_t>(e) * L.tile_bytes, back);
    float* part = down_part + (static_cast<std::size_t>(entry) * splits + split) * L.hidden;
    mc::expert_split_rows<kBits, 1, kDepth>(w, L.hidden, inter / 16U, splits, col, split, split_ch, xs, 1, lane, part, 0);
}

// grid (H / 128, rows), block 32.
__global__ __launch_bounds__(32) void k2_down_finish_kernel(DeviceLayer L, const std::int32_t* ids, const float* weights,
                                                            std::uint32_t top_k, const float* down_part, float* y) {
    const unsigned lane = threadIdx.x, block = blockIdx.x, row = blockIdx.y;
    const unsigned splits = L.plan.down_splits;
    float out[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    for (unsigned slot = 0; slot < top_k; ++slot) {
        const unsigned entry = row * top_k + slot;
        const unsigned e = static_cast<unsigned>(ids[entry]) - L.expert_first;
        const float* p = down_part + static_cast<std::size_t>(entry) * splits * L.hidden + block * 128U + lane;
        float acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        for (unsigned s = 0; s < splits; ++s)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[j] += p[static_cast<std::size_t>(s) * L.hidden + 32U * static_cast<unsigned>(j)];
        mc::hadamard128(acc, lane);
        const float wgt = weights[entry];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned i = block * 128U + lane + 32U * static_cast<unsigned>(j);
            const float ye = mc::materialize(acc[j] * mc::half_bits(L.down_svh[static_cast<std::size_t>(e) * L.hidden + i]));
            out[j] = out[j] + wgt * ye;
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) y[static_cast<std::size_t>(row) * L.hidden + block * 128U + lane + 32U * static_cast<unsigned>(j)] = out[j];
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
    k2_gate_up_kernel<<<dim3(L.local / 16U / kWaves, L.plan.gate_splits, entries * 2U), 32 * kWaves, 0, s>>>(L, v.x, v.ids, v.top_k,
                                                                                                         v.gu_part);
    check(hipGetLastError(), "gate/up launch");
    k2_finish_kernel<<<dim3(L.local / 128U, entries), 32, 0, s>>>(L, v.ids, v.swiglu_limit, v.gu_part, v.h);
    check(hipGetLastError(), "finish launch");
    k2_down_kernel<<<dim3(L.hidden / 16U / kWaves, L.plan.down_end - L.plan.down_first, entries), 32 * kWaves, 0, s>>>(L, v.ids, v.h,
                                                                                                                   v.down_part);
    check(hipGetLastError(), "down launch");
}

void enqueue_k2_down_finish(const DecodeView& v, void* stream) {
    const DeviceLayer& L = *v.layer;
    if (v.rows == 0U) return;
    k2_down_finish_kernel<<<dim3(L.hidden / 128U, v.rows), 32, 0, static_cast<hipStream_t>(stream)>>>(L, v.ids, v.weights, v.top_k,
                                                                                                      v.down_part, v.y);
    check(hipGetLastError(), "down finish launch");
}

}  // namespace strix::ds41::k2
