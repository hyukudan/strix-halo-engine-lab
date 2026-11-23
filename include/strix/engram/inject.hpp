#pragma once
// Engram gated injection: the shared contract (backend-neutral; the HIP kernel is src/engram/hip/inject_kernel.hpp).
//
// One Engram layer adds, to every hyper-connection copy c of a row's residual stream h_c [hidden], a gated value
// computed from the row's projected Engram embedding kv = [key_0 .. key_{copies-1}, value] (each [hidden]). The
// skeleton is common: per copy, chunked reductions over hidden (core::chunked_dot_of: 640-element chunks, lane-strided
// lane dots of 32, butterfly, chunks summed in ascending order), a scalar gate, then a per-element update. Two forms:
//
// folded (DeepSeek V4.1): w0 = q_norm [copies][hidden], w1 = k_norm [copies][hidden]; the norms are folded into scalars:
//   rh = rsqrt(sum h^2 / hidden + eps), rk likewise for key, dot = (sum (h * (q * k)) * key) * (rh * rk) * inv_sqrt_hidden,
//   gate = numeric::engram_gate_of(dot), h[i] = h[i] + gate * value[i]; gate_off: gate 0 (the update still runs).
// normalized_bf16: w0 = gate weight [copies][hidden] (the folded norm_key * norm_query weight),
//   w1 = [copy_scale [copies][hidden] | value_gamma [hidden]] (one contiguous block); BF16 rounding points of the model:
//   query = bf(h * rq), key = bf(k * rk), term = bf(bf(key * w0) * query), d = bf(sum(term) * inv_sqrt_hidden),
//   g = bf(sigmoid(sign(d) * sqrt(max(|d|, 1e-6)))) (sign(0) = 0), m = bf(gamma * bf(value * rv)),
//   h[i] = bf(h[i] + bf(bf(g * m) + bf(scale_c[i] * m))); gate_off: the row is skipped (Engram output zero).
// Rows are independent; inv_sqrt_hidden = float(1 / sqrt(double(hidden))).
#include "strix/core/chunked_dot.hpp"
#include "strix/numeric/declared.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace strix::engram {

enum class InjectForm : std::uint8_t { folded, normalized_bf16 };

// Device launch view (pointers are device memory; the same fields drive the host reference below).
struct InjectView {
    InjectForm form = InjectForm::folded;
    float* stream = nullptr;               // [rows][copies][hidden]
    const float* kv = nullptr;             // [rows][copies + 1][hidden]
    const float* w0 = nullptr;             // folded: q_norm; normalized_bf16: gate weight ([copies][hidden])
    const float* w1 = nullptr;             // folded: k_norm; normalized_bf16: copy_scale [copies][hidden] then gamma [hidden]
    const std::uint8_t* gate_off = nullptr;  // [rows], optional
    std::uint32_t rows = 0, hidden = 0, copies = 4;
    float eps = 0.0F;
};

inline constexpr std::uint32_t inject_chunk = 640;  // = core::dot_chunk (the declared reduction rule)
inline constexpr std::uint32_t inject_wave = 32;

// Admission shared by every backend: one block per row, one wave per (copy, chunk), at most 32 such waves.
[[nodiscard]] inline bool inject_supported(const InjectView& v) {
    const std::uint32_t chunks = (v.hidden + inject_chunk - 1U) / inject_chunk;
    return v.stream && v.kv && v.w0 && v.w1 && v.rows && v.hidden && v.copies == 4U && chunks * v.copies <= 32U;
}
[[nodiscard]] inline std::uint32_t inject_threads(const InjectView& v) {
    return inject_wave * v.copies * ((v.hidden + inject_chunk - 1U) / inject_chunk);
}

STRIX_DS41_HD inline float inv_sqrt_hidden_of(std::uint32_t hidden) {
    return static_cast<float>(1.0 / std::sqrt(static_cast<double>(hidden)));
}
// The normalized_bf16 gate of the BF16-rounded dot d (both forms share engram_gate_of's sqrt floor; this one keeps
// sign(0) = 0, so d = +-0 gives sigmoid(0)).
STRIX_DS41_HD inline float normalized_gate_of(float d) {
    const float sign = d > 0.0F ? 1.0F : (d < 0.0F ? -1.0F : 0.0F);
    const float a = d < 0.0F ? -d : d;
    return numeric::round_to_bf16(numeric::declared_sigmoid(sign * numeric::declared_sqrt(a < 1e-6F ? 1e-6F : a)));
}

// ---- host reference (declared order; the device kernel reproduces it bitwise) -------------------------------------
// One row: stream [copies * hidden] updated in place; kv [(copies + 1) * hidden]; w0/w1 as in InjectView.
inline void inject_row_reference(InjectForm form, std::uint32_t copies, std::uint32_t hidden, float eps, float* stream,
                                 const float* kv, const float* w0, const float* w1, bool gate_off, float* gates_out = nullptr) {
    using core::chunked_dot_of;
    const float inv = inv_sqrt_hidden_of(hidden);
    const float* value = kv + std::size_t{copies} * hidden;
    const auto sq = [hidden](const float* x) {
        return chunked_dot_of([x](std::size_t i) { return x[i]; }, [x](std::size_t i) { return x[i]; }, hidden);
    };
    std::vector<float> gates(copies);
    if (form == InjectForm::folded) {
        for (std::uint32_t c = 0; c < copies; ++c) {
            const float* h = stream + std::size_t{c} * hidden;
            const float* key = kv + std::size_t{c} * hidden;
            const float* q = w0 + std::size_t{c} * hidden;
            const float* k = w1 + std::size_t{c} * hidden;
            const float rh = numeric::declared_rsqrt(sq(h) / static_cast<float>(hidden) + eps);
            const float rk = numeric::declared_rsqrt(sq(key) / static_cast<float>(hidden) + eps);
            const float dk = chunked_dot_of([&](std::size_t i) { return h[i] * (q[i] * k[i]); },
                                            [&](std::size_t i) { return key[i]; }, hidden);
            gates[c] = gate_off ? 0.0F : numeric::engram_gate_of((dk * (rh * rk)) * inv);
        }
        for (std::uint32_t c = 0; c < copies; ++c)
            for (std::size_t i = 0; i < hidden; ++i) {
                float& x = stream[std::size_t{c} * hidden + i];
                x = x + gates[c] * value[i];
            }
    } else {
        using numeric::round_to_bf16;
        const float* scale = w1;
        const float* gamma = w1 + std::size_t{copies} * hidden;
        const float rv = numeric::declared_rsqrt(sq(value) / static_cast<float>(hidden) + eps);
        for (std::uint32_t c = 0; c < copies; ++c) {
            const float* h = stream + std::size_t{c} * hidden;
            const float* key = kv + std::size_t{c} * hidden;
            const float* w = w0 + std::size_t{c} * hidden;
            const float rq = numeric::declared_rsqrt(sq(h) / static_cast<float>(hidden) + eps);
            const float rk = numeric::declared_rsqrt(sq(key) / static_cast<float>(hidden) + eps);
            const float sum = chunked_dot_of(
                [&](std::size_t i) { return round_to_bf16(round_to_bf16(round_to_bf16(key[i] * rk) * w[i]) * round_to_bf16(h[i] * rq)); },
                [](std::size_t) { return 1.0F; }, hidden);
            gates[c] = normalized_gate_of(round_to_bf16(sum * inv));
        }
        if (!gate_off)
            for (std::uint32_t c = 0; c < copies; ++c)
                for (std::size_t i = 0; i < hidden; ++i) {
                    const float m = round_to_bf16(gamma[i] * round_to_bf16(value[i] * rv));
                    const float out = round_to_bf16(round_to_bf16(gates[c] * m) + round_to_bf16(scale[std::size_t{c} * hidden + i] * m));
                    float& x = stream[std::size_t{c} * hidden + i];
                    x = round_to_bf16(x + out);
                }
    }
    if (gates_out != nullptr)
        for (std::uint32_t c = 0; c < copies; ++c) gates_out[c] = gates[c];
}

// Every row of a view whose pointers are host memory (tests, CPU engines).
inline void inject_reference(const InjectView& v) {
    if (!inject_supported(v)) throw std::invalid_argument("engram inject: view");
    const std::size_t s = std::size_t{v.copies} * v.hidden, k = std::size_t{v.copies + 1U} * v.hidden;
    for (std::uint32_t r = 0; r < v.rows; ++r)
        inject_row_reference(v.form, v.copies, v.hidden, v.eps, v.stream + r * s, v.kv + r * k, v.w0, v.w1,
                             v.gate_off != nullptr && v.gate_off[r] != 0U);
}

}  // namespace strix::engram
