#pragma once
// Row addressing and window split of the shared GDN prefill recurrence core (qpf2k_gdn::gdn_qpf2k_cols_kernel,
// src/hip_gdn_qpf2k_prefill.inc). Row r of a call reads q / k / v at qkv + r * qkv_stride (layout [q | k | v], key
// heads x 128, key heads x 128, value heads x 128), beta / gate at r * scalar_stride + value head, and writes its
// output at core + r * core_stride + value head x 128 + column. The Qwen MoE staging (GdnPrefillRows) keeps the four in
// one row with one stride; dense engines on hip_gdn_decode_step_dims keep separate row-major buffers (qkv: channels,
// beta / gate: value heads, core: value heads x 128). The strides only move addresses: every value and every
// operation of the core is the same for any of them.
//
// The core stages Chunk rows per step, so a call covers a multiple of Chunk rows; the remaining rows of a window run the
// incumbent per-row kernel from the state the core stored (an FP32 store / load round trip is exact), which keeps
// every row's bits equal to one kernel over the whole window.
#include <cstdint>

namespace strix::gdn_recurrence {

struct Rows {
    const float* qkv;
    const float* beta;
    const float* gate;
    float* core;
    std::uint64_t qkv_stride, scalar_stride, core_stride;
};

// One stride for the four tensors (the GdnPrefillRows staging layout).
constexpr Rows uniform(const float* qkv, const float* beta, const float* gate, float* core, std::uint64_t stride) {
    return Rows{qkv, beta, gate, core, stride, stride, stride};
}

// Rows `first`.. of the same tensors.
constexpr Rows advance(const Rows& r, std::uint64_t first) {
    return Rows{r.qkv + first * r.qkv_stride, r.beta + first * r.scalar_stride, r.gate + first * r.scalar_stride,
                r.core + first * r.core_stride, r.qkv_stride, r.scalar_stride, r.core_stride};
}

struct Split {
    std::uint32_t bulk;  // rows run by the chunked core (a multiple of chunk)
    std::uint32_t tail;  // rows run by the per-row kernel after it
};
constexpr Split split(std::uint32_t rows, std::uint32_t chunk) {
    return Split{rows / chunk * chunk, rows % chunk};
}

} // namespace strix::gdn_recurrence
