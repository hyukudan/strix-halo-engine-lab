// Host oracle of the compressed group indexer contract (include/strix/attn/group_indexer.hpp).
#include "strix/attn/group_indexer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace strix::attn::group_indexer {

float head_weight_scale(const Shape& s) {
    // The reference multiplies an fp32 tensor by the Python double heads**-0.5 * head_dim**-0.5.
    const double scale = (1.0 / std::sqrt(static_cast<double>(s.heads))) * (1.0 / std::sqrt(static_cast<double>(s.head_dim)));
    return static_cast<float>(scale);
}

void pool_group_gated_ape(const Shape& s, const float* keys, const GatedApeWeights& w, float (*exp_fn)(float), float* out) {
    if (s.pooling != Pooling::GatedSoftmaxApe || s.ratio != 4U)
        throw std::invalid_argument("group_indexer: pool_group_gated_ape needs Pooling::GatedSoftmaxApe and ratio 4");
    constexpr unsigned R = 4U;
    const unsigned d = s.head_dim;
    std::vector<float> pooled(d);
    for (unsigned o = 0; o < d; ++o) {
        float logits[R], values[R];
        const float* wv = w.wv + static_cast<std::size_t>(o) * d;
        const float* wg = w.wg + static_cast<std::size_t>(o) * d;
        for (unsigned j = 0; j < R; ++j) {
            const float* k = keys + static_cast<std::size_t>(j) * d;
            float v = 0.0F, g = 0.0F;
            for (unsigned i = 0; i < d; ++i) {
                v = fma3(wv[i], k[i], v);
                g = fma3(wg[i], k[i], g);
            }
            values[j] = v;
            logits[j] = g + w.ape[static_cast<std::size_t>(j) * d + o];
        }
        pooled[o] = gated_pool_channel<R>(logits, values, exp_fn);
    }
    float ss = 0.0F;
    for (unsigned i = 0; i < d; ++i) ss = fma3(pooled[i], pooled[i], ss);
    const float r = 1.0F / std::sqrt(ss / static_cast<float>(d) + w.eps);
    for (unsigned i = 0; i < d; ++i) out[i] = bf16_round(w.norm[i] * (pooled[i] * r));
}

void score_row(const Shape& s, const float* q, const float* w, float c, const float* group_keys, unsigned visible,
               float* scores) {
    for (unsigned g = 0; g < visible; ++g) {
        const float v = score_group(s, q, group_keys + static_cast<std::size_t>(g) * s.head_dim, w, c);
        if (std::isnan(v)) throw std::domain_error("group_indexer: NaN score");
        scores[g] = v;
    }
}

std::vector<std::uint32_t> select_groups(const Shape& s, const float* scores, unsigned visible) {
    std::vector<std::uint32_t> idx(visible);
    std::iota(idx.begin(), idx.end(), 0U);
    const unsigned k = std::min(pick_count(s), visible);
    if (k < visible) {
        std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(), [scores](std::uint32_t a, std::uint32_t b) {
            const std::uint32_t ka = score_key(scores[a]), kb = score_key(scores[b]);
            return ka != kb ? ka > kb : a < b;
        });
        idx.resize(k);
        std::sort(idx.begin(), idx.end());
    }
    return idx;
}

std::vector<std::uint32_t> expand_tokens(const Shape& s, const std::vector<std::uint32_t>& groups, std::uint32_t t) {
    const std::uint32_t tail = tail_begin(s, t);
    std::vector<std::uint32_t> tokens;
    tokens.reserve(groups.size() * s.ratio + (t + 1U - tail));
    for (const std::uint32_t g : groups) {
        if (g >= visible_groups(s, t)) throw std::invalid_argument("group_indexer: selected group not visible");
        for (unsigned j = 0; j < s.ratio; ++j) tokens.push_back(g * s.ratio + j);
    }
    for (std::uint32_t p = tail; p <= t; ++p) tokens.push_back(p);
    return tokens;
}

}  // namespace strix::attn::group_indexer
