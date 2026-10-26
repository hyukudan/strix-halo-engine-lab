#include "strix/minimax_msa.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace strix::minimax::msa {

RopeRow rope_row(std::uint64_t position, double theta) {
    RopeRow row;
    for (unsigned i = 0; i < kRotaryHalf; ++i) {
        const double inv = 1.0 / std::pow(theta, static_cast<double>(2 * i) / static_cast<double>(kRotaryDim));
        const double angle = static_cast<double>(position) * inv;
        row.cos[i] = static_cast<float>(std::cos(angle));
        row.sin[i] = static_cast<float>(std::sin(angle));
    }
    return row;
}

void head_norm_rope(float* x, const float* gemma_w, const RopeRow& rope) {
    float lanes[kLanes];
    for (unsigned l = 0; l < kLanes; ++l) {
        float acc = 0.0F;
        for (unsigned i = 0; i < kDimsPerLane; ++i) acc = std::fmaf(x[l * kDimsPerLane + i], x[l * kDimsPerLane + i], acc);
        lanes[l] = acc;
    }
    butterfly(lanes);
    const float r = 1.0F / std::sqrt(lanes[0] / static_cast<float>(kHeadDim) + kNormEps);
    for (unsigned i = 0; i < kHeadDim; ++i) x[i] = (x[i] * r) * (1.0F + gemma_w[i]);
    for (unsigned i = 0; i < kRotaryHalf; ++i) {
        const float a = x[i], b = x[i + kRotaryHalf];
        x[i] = std::fmaf(-b, rope.sin[i], a * rope.cos[i]);
        x[i + kRotaryHalf] = std::fmaf(a, rope.sin[i], b * rope.cos[i]);
    }
}

void index_block_scores(const float* q, const float* keys, std::uint64_t count, std::span<float> out) {
    const std::uint64_t blocks = block_count(count);
    if (out.size() < blocks) throw std::invalid_argument("index_block_scores: output too small");
    for (std::uint64_t b = 0; b < blocks; ++b) {
        float best = -std::numeric_limits<float>::infinity();
        const std::uint64_t end = std::min<std::uint64_t>(count, (b + 1) * kBlock);
        for (std::uint64_t t = b * kBlock; t < end; ++t) best = std::max(best, dot128(q, keys + t * kIdxDim) * kScale);
        out[b] = best;
    }
}

std::vector<std::uint32_t> select_blocks(std::span<const float> block_scores) {
    const auto blocks = static_cast<std::uint32_t>(block_scores.size());
    std::vector<std::uint32_t> chosen;
    if (blocks <= kTopK) {
        chosen.resize(blocks);
        std::iota(chosen.begin(), chosen.end(), 0U);
        return chosen;
    }
    const std::uint32_t newest = blocks - 1;
    std::vector<std::uint32_t> rest(newest);
    std::iota(rest.begin(), rest.end(), 0U);
    std::partial_sort(rest.begin(), rest.begin() + (kTopK - 1), rest.end(), [&](std::uint32_t a, std::uint32_t b) {
        if (block_scores[a] != block_scores[b]) return block_scores[a] > block_scores[b];
        return a < b;
    });
    chosen.assign(rest.begin(), rest.begin() + (kTopK - 1));
    chosen.push_back(newest);
    std::sort(chosen.begin(), chosen.end());
    return chosen;
}

void sparse_attention(const float* q, unsigned heads, const float* keys, const float* values, std::uint64_t count,
                      std::span<const std::uint32_t> blocks, float* out) {
    if (blocks.size() > kTopK) throw std::invalid_argument("sparse_attention: too many blocks");
    const unsigned chunks = static_cast<unsigned>((blocks.size() + kChunkBlocks - 1) / kChunkBlocks);
    for (unsigned h = 0; h < heads; ++h) {
        const float* qh = q + static_cast<std::size_t>(h) * kHeadDim;
        float cm[kMaxChunks], cl[kMaxChunks], cacc[kMaxChunks][kHeadDim];
        for (unsigned c = 0; c < chunks; ++c) {
            float m = -std::numeric_limits<float>::infinity(), l = 0.0F;
            float* acc = cacc[c];
            std::fill(acc, acc + kHeadDim, 0.0F);
            bool first = true;
            // The chunk's visible keys, ascending.
            std::vector<std::uint64_t> pos;
            for (std::size_t bi = c * kChunkBlocks; bi < std::min<std::size_t>(blocks.size(), (c + 1) * kChunkBlocks); ++bi) {
                const std::uint64_t lo = static_cast<std::uint64_t>(blocks[bi]) * kBlock;
                for (std::uint64_t t = lo; t < std::min<std::uint64_t>(count, lo + kBlock); ++t) pos.push_back(t);
            }
            for (std::size_t t0 = 0; t0 < pos.size(); t0 += kTile) {
                const std::size_t t1 = std::min(pos.size(), t0 + kTile);
                float s[kTile];
                float tm = -std::numeric_limits<float>::infinity();
                for (std::size_t j = t0; j < t1; ++j) {
                    s[j - t0] = dot128(qh, keys + pos[j] * kHeadDim) * kScale;
                    tm = std::max(tm, s[j - t0]);
                }
                const float mn = std::max(m, tm);
                const float corr = first ? 0.0F : exp_f(m - mn);
                l = l * corr;
                for (unsigned d = 0; d < kHeadDim; ++d) acc[d] = acc[d] * corr;
                for (std::size_t j = t0; j < t1; ++j) {
                    const float p = exp_f(s[j - t0] - mn);
                    l = l + p;
                    const float* v = values + pos[j] * kHeadDim;
                    for (unsigned d = 0; d < kHeadDim; ++d) acc[d] = std::fmaf(p, v[d], acc[d]);
                }
                m = mn;
                first = false;
            }
            cm[c] = m;
            cl[c] = l;
        }
        float big = -std::numeric_limits<float>::infinity();
        for (unsigned c = 0; c < chunks; ++c) big = std::max(big, cm[c]);
        float L = 0.0F;
        float e[kMaxChunks];
        for (unsigned c = 0; c < chunks; ++c) {
            e[c] = exp_f(cm[c] - big);
            L = std::fmaf(cl[c], e[c], L);
        }
        float* o = out + static_cast<std::size_t>(h) * kHeadDim;
        for (unsigned d = 0; d < kHeadDim; ++d) {
            float acc = 0.0F;
            for (unsigned c = 0; c < chunks; ++c) acc = std::fmaf(cacc[c][d], e[c], acc);
            o[d] = acc / L;
        }
    }
}

}  // namespace strix::minimax::msa
