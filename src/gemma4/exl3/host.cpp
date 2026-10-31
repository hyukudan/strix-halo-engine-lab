#include "format.hpp"
#include <cfenv>
#include <stdexcept>

namespace strix::gemma4::exl3 {
namespace {
void finite(float x) {
    if (!std::isfinite(x)) throw std::runtime_error("nonfinite EXL3 arithmetic");
}
float half(float x) {
    std::uint16_t h = 0;
    if (!prefill::half_bits(x, h)) throw std::runtime_error("EXL3 FP16 overflow");
    return fp16_value(h);
}
}
void validate(LinearView w) {
    if (!geometry(w)) throw std::invalid_argument("EXL3 tensor geometry");
    for (std::size_t i = 0; i < std::size_t(w.inputs) + w.outputs; ++i)
        finite(fp16_value(load16(w.data + 2 * i)));
}
void hadamard(float* block) {
    for (unsigned stride = 1; stride < 128; stride *= 2)
        for (unsigned base = 0; base < 128; base += 2 * stride)
            for (unsigned j = 0; j < stride; ++j) {
                const float a = block[base + j], b = block[base + stride + j];
                block[base + j] = a + b; block[base + stride + j] = a - b;
                finite(block[base + j]); finite(block[base + stride + j]);
            }
    for (unsigned i = 0; i < 128; ++i) { block[i] = block[i] * hscale; finite(block[i]); }
}
void input_transform(LinearView w, const float* x, unsigned rows, float* out) {
    for (unsigned r = 0; r < rows; ++r) {
        auto* row = out + std::size_t(r) * w.inputs;
        for (unsigned i = 0; i < w.inputs; ++i) {
            const float product = x[std::size_t(r) * w.inputs + i] * fp16_value(load16(w.data + 2ULL * i));
            row[i] = half(product);
        }
        for (unsigned i = 0; i < w.inputs; i += 128) hadamard(row + i);
        for (unsigned i = 0; i < w.inputs; ++i) row[i] = half(row[i]);
    }
}
void output_transform(LinearView w, float* core, unsigned rows) {
    for (unsigned r = 0; r < rows; ++r) {
        auto* row = core + std::size_t(r) * w.outputs;
        for (unsigned i = 0; i < w.outputs; i += 128) hadamard(row + i);
        for (unsigned i = 0; i < w.outputs; ++i) {
            row[i] = row[i] * fp16_value(load16(w.data + 2ULL * (w.inputs + i)));
            finite(row[i]);
        }
    }
}
namespace {
float core_v2(LinearView w, const float* x, unsigned o) {
    const unsigned chunks = w.inputs / 16U, splits = v2_splits(w.inputs, w.outputs), span = chunks / splits;
    float value = 0.0F;
    for (unsigned s = 0; s < splits; ++s) {
        float acc[4] = {};
        for (unsigned a = 0; a < 4; ++a)
            for (unsigned k = s * span; k < (s + 1U) * span; ++k)
                for (unsigned e : {2U * a, 2U * a + 1U, 2U * a + 8U, 2U * a + 9U}) {
                    const auto i = k * 16U + e;
                    acc[a] = acc[a] + rotated_at(w, i, o) * x[i];
                }
        const float partial = (acc[0] + acc[2]) + (acc[1] + acc[3]);
        value = s ? value + partial : partial;
    }
    return value;
}
}
void core_linear(LinearView w, const float* transformed, unsigned rows, float* out, unsigned begin, unsigned end) {
    if (v2_active()) {
        for (unsigned o = begin; o < end; ++o)
            for (unsigned r = 0; r < rows; ++r) {
                const float value = core_v2(w, transformed + std::size_t(r) * w.inputs, o);
                finite(value); out[std::size_t(r) * w.outputs + o] = value;
            }
        return;
    }
    for (unsigned o = begin; o < end; ++o)
        for (unsigned r = 0; r < rows; ++r) {
            const float value = chunk16_dot([&](unsigned i) { return rotated_at(w, i, o); },
                                            transformed + std::size_t(r) * w.inputs, w.inputs);
            finite(value); out[std::size_t(r) * w.outputs + o] = value;
        }
}
void linear(LinearView w, const float* x, unsigned rows, float* out) {
    validate(w);
    if (!rows || !x || !out || std::fegetround() != FE_TONEAREST) throw std::invalid_argument("EXL3 linear arguments/RNE");
    std::vector<float> transformed(std::size_t(rows) * w.inputs);
    input_transform(w, x, rows, transformed.data());
    core_linear(w, transformed.data(), rows, out, 0, w.outputs);
    output_transform(w, out, rows);
}
} // namespace strix::gemma4::exl3
