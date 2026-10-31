#pragma once

// Host reference recipe of g4-a8-v1 (see a8.hpp), installed into the g4-v1 host graph through
// HostTarget's LinearRecipe hook. BF16/F32 views return false and keep the g4-v1 recipe.
#include "a8.hpp"
#include "gemma4/ref/target.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace strix::gemma4::a8 {

// Quantizes rows x [rows, width] into codes [rows, width] and meta [rows, width / 32].
inline void quantize_rows(const float* x, std::size_t rows, std::uint32_t width, std::vector<std::int8_t>& codes,
                          std::vector<Meta>& meta) {
    if (!width || width % 256U)
        throw std::runtime_error("g4-a8-v1 quantized linear width must be a multiple of 256");
    codes.resize(rows * width);
    meta.resize(rows * (width / group_width));
    for (std::size_t i = 0; i < rows * width; ++i)
        if (!std::isfinite(x[i]))
            throw std::runtime_error("nonfinite g4-a8-v1 activation");
    for (std::size_t g = 0; g < rows * (width / group_width); ++g)
        meta[g] = quantize_group(x + g * group_width, codes.data() + g * group_width);
}
inline LinearRecipe host_recipe() {
    return [](LinearView v, const float* x, std::size_t rows, float* out, const ParallelOutputs& parallel) {
        if (!quantized(v.format))
            return false;
        std::vector<std::int8_t> codes;
        std::vector<Meta> meta;
        quantize_rows(x, rows, v.inputs, codes, meta);
        const auto groups = v.inputs / group_width;
        parallel(v.outputs, [&](std::size_t begin, std::size_t end) {
            for (auto o = begin; o < end; ++o)
                for (std::size_t r = 0; r < rows; ++r)
                    out[r * v.outputs + o] =
                        dot(v, std::uint32_t(o), codes.data() + r * v.inputs, meta.data() + r * groups);
        });
        return true;
    };
}

} // namespace strix::gemma4::a8
