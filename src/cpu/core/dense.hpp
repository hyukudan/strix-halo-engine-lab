#pragma once

#include "cpu/core/isa.hpp"

#include <cstddef>
#include <vector>

namespace strix::cpu {

struct Chunk16Dot {
    Isa isa;
    float (*run)(const float* weights, const float* input, std::uint32_t width) noexcept;
};

// The kernel may be capped below the host ISA, including for reference comparisons.
Chunk16Dot chunk16_kernel(Isa maximum = Isa::avx512) noexcept;

struct OrderedFma {
    Isa isa;
    void (*matvec)(const float* weights, const float* input, std::size_t width,
                   float* out, std::size_t outputs) noexcept;
    // Eight independent inputs are interleaved as [width][8].
    void (*dot8)(const float* weights, const float* inputs, std::size_t width, float* out) noexcept;
};

// Each result has its own ascending-column FMA chain, including on SIMD.
OrderedFma ordered_fma_kernel(bool allow_simd) noexcept;

// Writes an independent output range for all rows. Decode maps one stored coefficient
// to FP32. The caller owns validation, output scheduling and format-specific epilogues.
template<class Decode>
void chunk16_linear(Chunk16Dot dot, std::uint32_t width, std::uint32_t outputs, const float* input,
                    std::size_t rows, float* out, std::size_t begin, std::size_t end, Decode decode) {
    std::vector<float> weights(width);
    for (std::size_t o = begin; o < end; ++o) {
        for (std::uint32_t i = 0; i < width; ++i) weights[i] = decode(static_cast<std::uint32_t>(o), i);
        for (std::size_t r = 0; r < rows; ++r)
            out[r * outputs + o] = dot.run(weights.data(), input + r * width, width);
    }
}

}  // namespace strix::cpu
