// Host reader of one EXL3 trellis linear as the dense weight the device computes: the source side of the transcoder from
// third-party trellis packs (GLM, MiniMax, DeepSeek dense roles, Nemotron) to the Strix pack (SP4 / SP3 / IU4 through
// weight::host_lowbit). docs/engine/trellis-transcode.md states the conversion and what remains.
//
// The device forward of an EXL3 linear (tools/trellis_reference.py: input_stage, core_gemv, output_stage) is
//   y = svh (.) H_out( Q^T H_in( suh (.) x ) ),
// with Q the decoded trellis coefficients (inputs x outputs, src/weight/formats/exl3.cpp decode_exl3: transformed domain)
// and H the normalised Sylvester Hadamard over 128-element blocks (involutory: H H = I). The dense weight is therefore
//   W[n][k] = svh[n] * sum_n' H[n][n'] * sum_q Q[q][n'] * H[q][k] * suh[k],
// computed here in FP32 (output-major rows, W[n * inputs + k]). The device rounds its stages to FP16; the dense form is
// the operator those stages approximate, not their bits (a transcoded pack is a lossy lever with its own gate).
// Pure C++, host only.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/trellis_scalar.hpp"
#include "strix/weight/formats/exl3_transport.hpp"
#include "strix/weight/formats/ieee_half.hpp"

namespace strix::weight::exl3_dense {

inline constexpr unsigned kBlock = exl3_transport::kTransformBlock;  // 128

// In-place normalised Sylvester Hadamard over consecutive 128-element blocks of `values` (stride `step` between
// elements of one vector): the transform of tools/trellis_reference.py hadamard(), scaled by 1 / sqrt(128).
inline void hadamard128(float* values, std::size_t count, std::size_t step = 1) {
    if (count % kBlock != 0) throw std::invalid_argument("EXL3 Hadamard needs whole 128-element blocks");
    const float scale = 1.0F / std::sqrt(static_cast<float>(kBlock));
    for (std::size_t base = 0; base < count; base += kBlock) {
        for (std::size_t stride = 1; stride < kBlock; stride *= 2)
            for (std::size_t start = 0; start < kBlock; start += 2 * stride)
                for (std::size_t i = start; i < start + stride; ++i) {
                    float& lo = values[(base + i) * step];
                    float& hi = values[(base + i + stride) * step];
                    const float a = lo, b = hi;
                    lo = a + b;
                    hi = a - b;
                }
        for (std::size_t i = 0; i < kBlock; ++i) values[(base + i) * step] *= scale;
    }
}

// One EXL3 linear: the input-tile-major trellis rings (Format::tile order, 32 * bits bytes per 16 x 16 tile) and the
// binary16 input / output scales (suh: one per input, svh: one per output).
struct Linear {
    std::span<const std::uint8_t> rings;
    std::span<const std::uint16_t> suh, svh;
    unsigned inputs = 0, outputs = 0, bits = 0;
};

inline void validate(const Linear& linear) {
    if (linear.bits < 1 || linear.bits > 8) throw std::invalid_argument("EXL3 bits must be in 1..8");
    if (!exl3_transport::aligned_shape(linear.inputs, linear.outputs))
        throw std::invalid_argument("EXL3 dense form needs inputs and outputs in whole 128-element blocks");
    if (linear.suh.size() != linear.inputs || linear.svh.size() != linear.outputs)
        throw std::invalid_argument("EXL3 scales do not match the matrix shape");
    const std::size_t tiles = std::size_t{linear.inputs / 16U} * (linear.outputs / 16U);
    if (linear.rings.size() != tiles * static_cast<std::size_t>(core::trellis::Format::tile_bytes(int(linear.bits))))
        throw std::invalid_argument("EXL3 ring bytes do not match the matrix shape");
    for (const auto scale : linear.suh)
        if (!exl3_transport::finite_scale(static_cast<std::uint8_t>(scale >> 8U)))
            throw std::runtime_error("nonfinite EXL3 input scale");
    for (const auto scale : linear.svh)
        if (!exl3_transport::finite_scale(static_cast<std::uint8_t>(scale >> 8U)))
            throw std::runtime_error("nonfinite EXL3 output scale");
}

// Transformed-domain coefficient Q[k][n] (the codebook value of its ring state).
inline float coefficient(const Linear& linear, unsigned k, unsigned n) {
    using Format = core::trellis::Format;
    const auto* tile = linear.rings.data() + Format::tile(k / 16U, n / 16U, linear.outputs / 16U, int(linear.bits)) * 4U;
    return core::trellis::mul1_value(Format::state_bytes(tile, int(linear.bits), Format::position(k % 16U, n % 16U)));
}

// The dense weight W (outputs x inputs, output-major) of one EXL3 linear.
inline std::vector<float> dense_rows(const Linear& linear) {
    validate(linear);
    const std::size_t inputs = linear.inputs, outputs = linear.outputs;
    std::vector<float> w(inputs * outputs);
    for (unsigned n = 0; n < linear.outputs; ++n)
        for (unsigned k = 0; k < linear.inputs; ++k) w[n * inputs + k] = coefficient(linear, k, n);  // Q^T
    for (std::size_t n = 0; n < outputs; ++n) hadamard128(w.data() + n * inputs, inputs);         // Q^T H_in
    for (std::size_t k = 0; k < inputs; ++k) hadamard128(w.data() + k, outputs, inputs);          // H_out Q^T H_in
    for (std::size_t n = 0; n < outputs; ++n) {
        const float out_scale = ieee_half::decode(linear.svh[n]);
        for (std::size_t k = 0; k < inputs; ++k) w[n * inputs + k] *= out_scale * ieee_half::decode(linear.suh[k]);
    }
    return w;
}

}  // namespace strix::weight::exl3_dense
