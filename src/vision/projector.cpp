// Shared projector ops of the vision runtime (host side): the FP64 activation and the host graph of
// DenseOps::merge_project. Built without contraction into FMA (non-MSVC), as the family projectors were: the rows
// are cache content, and the device kernel evaluates the same formula in the same order.
#include "strix/vision/encoder.hpp"

#include "encoder_layout.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace strix::vision {

float activate(Activation a, float v) {
    const double x = static_cast<double>(v);
    switch (a) {
        case Activation::gelu: return static_cast<float>(0.5 * x * (1.0 + std::erf(x * 0.70710678118654752440)));
        case Activation::gelu_tanh: return static_cast<float>(0.5 * x * (1.0 + std::tanh(0.79788456080286535588 * (x + 0.044715 * x * x * x))));
        case Activation::silu: return static_cast<float>(x / (1.0 + std::exp(-x)));
    }
    return v;
}

unsigned MergeProjector::tokens(unsigned grid_h, unsigned grid_w) const {
    if (!ratio) throw std::invalid_argument("vision encoder: merge projector ratio");
    return ((grid_h + ratio - 1U) / ratio) * ((grid_w + ratio - 1U) / ratio);
}

void DenseOps::merge_project(const EncoderOutput& hidden, unsigned grid_h, unsigned grid_w, const MergeProjector& m, float* rows) {
    const unsigned R = m.ratio, W = hidden.width;
    if (!R || !W || !rows || !m.w1 || !m.w2 || hidden.rows != grid_h * grid_w || hidden.hidden.size() != std::size_t{hidden.rows} * W ||
        m.w1.cols != std::size_t{W} * R * R || m.w2.cols != m.w1.rows)
        throw std::invalid_argument("vision encoder: merge projector operands");
    const unsigned mh = (grid_h + R - 1U) / R, mw = (grid_w + R - 1U) / R, tokens = mh * mw;
    const std::size_t in = std::size_t{W} * R * R, P = m.w1.rows;
    // Unshuffle; the vector starts as zeros, so cells past the grid edge stay zero.
    std::vector<float> merged(std::size_t{tokens} * in);
    for (unsigned my = 0; my < mh; ++my)
        for (unsigned mx = 0; mx < mw; ++mx) {
            float* dst = &merged[(std::size_t{my} * mw + mx) * in];
            for (unsigned dy = 0; dy < R; ++dy)
                for (unsigned dx = 0; dx < R; ++dx) {
                    if (my * R + dy >= grid_h || mx * R + dx >= grid_w) continue;
                    const float* src = &hidden.hidden[(std::size_t{my * R + dy} * grid_w + mx * R + dx) * W];
                    const unsigned slot = m.row_major ? dy * R + dx : dx * R + dy;
                    for (unsigned c = 0; c < W; ++c) dst[std::size_t{c} * R * R + slot] = src[c];
                }
        }
    std::vector<float> mid(std::size_t{tokens} * P);
    linear(m.w1, m.b1, merged.data(), tokens, mid.data());
    merged = {};
    layout::parallel_ranges(mid.size(), 16U, [&](std::size_t i0, std::size_t i1) {  // elementwise: the split cannot change a value
        for (std::size_t i = i0; i < i1; ++i) mid[i] = activate(m.act, mid[i]);
    });
    linear(m.w2, m.b2, mid.data(), tokens, rows);
}

}  // namespace strix::vision
