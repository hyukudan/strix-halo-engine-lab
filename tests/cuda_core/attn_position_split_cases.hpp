#pragma once

#include "attn_core/position_split.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace b6_fixture {

inline void require(bool ok, const char* check) {
    if (ok) return;
    std::fprintf(stderr, "b6 FAIL %s\n", check);
    std::exit(EXIT_FAILURE);
}

inline bool same_bits(float a, float b) {
    return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
}

inline void compare(const char* check, const std::vector<float>& expected, const std::vector<float>& actual) {
    require(expected.size() == actual.size(), check);
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::isfinite(expected[i]) && same_bits(expected[i], actual[i])) continue;
        std::fprintf(stderr, "b6 FAIL %s at %zu: %08x vs %08x\n", check, i,
                     std::bit_cast<std::uint32_t>(expected[i]), std::bit_cast<std::uint32_t>(actual[i]));
        std::exit(EXIT_FAILURE);
    }
}

enum class Layout { head, slot, ring };

struct Row {
    unsigned pos, plane, self, query;
};

struct Case {
    const char* name;
    unsigned dim, heads, kv_heads, context, window, slots;
    Layout layout;
    std::array<Row, 4> rows;

    [[nodiscard]] std::size_t plane_size() const {
        return std::size_t{slots} * kv_heads * dim;
    }
    [[nodiscard]] std::size_t row_size() const { return std::size_t{heads} * dim; }
    [[nodiscard]] std::size_t offset(unsigned plane, unsigned key, unsigned head) const {
        const unsigned slot = layout == Layout::ring ? key % slots : key;
        const std::size_t row = layout == Layout::head ?
            std::size_t{head} * slots + slot : std::size_t{slot} * kv_heads + head;
        return plane * plane_size() + row * dim;
    }
};

inline constexpr Case cases[] = {
    {"verify", 64, 8, 1, 900, 0, 900, Layout::slot,
     {{{510, 0, 510, 0}, {511, 0, 511, 1}, {512, 0, 512, 2}, {513, 0, 513, 3}}}},
    {"tree", 256, 4, 2, 1024, 0, 1024, Layout::head,
     {{{511, 0, 511, 0}, {512, 0, 905, 1}, {512, 0, 906, 2}, {513, 0, 513, 3}}}},
    {"cohort", 37, 10, 2, 1200, 270, 512, Layout::ring,
     {{{1026, 1, 1026, 0}, {255, 0, 255, 1}, {812, 1, 812, 2}, {511, 0, 511, 3}}}},
};

struct Data {
    std::vector<float> q, k, v; // K and V are exactly representable in BF16.
};

inline Data make_data(const Case& c, bool general_scores) {
    Data data{std::vector<float>(c.rows.size() * c.row_size()),
              std::vector<float>(2 * c.plane_size()), std::vector<float>(2 * c.plane_size())};
    for (const auto& row : c.rows) {
        for (unsigned h = 0; h < c.heads; ++h) {
            float* q = data.q.data() + row.query * c.row_size() + h * c.dim;
            q[0] = 1.0F;
            if (general_scores)
                for (unsigned d = 1; d < c.dim; ++d)
                    q[d] = static_cast<float>(static_cast<int>((d + 3U * h + row.query) % 7U) - 3) / 32.0F;
        }
    }
    constexpr float values[] = {16777216.0F, 1.0F, -16777216.0F, 2.0F, 0.5F, -3.0F, 4.0F};
    for (unsigned plane = 0; plane < 2; ++plane) {
        for (unsigned slot = 0; slot < c.slots; ++slot) {
            for (unsigned h = 0; h < c.kv_heads; ++h) {
                const auto base = c.offset(plane, slot, h);
                data.k[base] = (slot + h + plane) % 5U == 0 ? -128.0F : 0.0F;
                for (unsigned d = 0; d < c.dim; ++d) {
                    if (general_scores)
                        data.k[base + d] = static_cast<float>(static_cast<int>((slot + 5U * d + 7U * h) % 23U) - 11) / 16.0F;
                    data.v[base + d] = values[(slot + 3U * d + 2U * plane + h) % 7U];
                }
            }
        }
    }
    return data;
}

// Scalar execution of the incumbent per-row order. The serial arm buckets a
// single increasing key walk by absolute position. The plan arm uses the
// same walk in explicitly planned chunks.
inline std::vector<float> serial_row(const Case& c, const Data& data, const Row& row,
                                     const strix::attn_core::PositionSplitPlan* plan = nullptr) {
    constexpr unsigned chunk = 256, lanes = 32;
    const unsigned end = row.pos + 1U;
    const unsigned lo = c.window != 0 && end > c.window ? end - c.window : 0U;
    const auto visible = plan ? plan->row_keys(row.pos) : strix::attn_core::mask::Range{lo, end};
    require(visible.end != 0, "admitted oracle row");
    const unsigned first = plan ? strix::attn_core::position_split(visible.begin) : lo / chunk;
    const unsigned last = plan ? strix::attn_core::position_split(visible.end - 1U) : row.pos / chunk;
    std::vector<float> out(c.row_size());
    for (unsigned h = 0; h < c.heads; ++h) {
        const unsigned kv_head = h / (c.heads / c.kv_heads);
        const float* q = data.q.data() + row.query * c.row_size() + h * c.dim;
        const auto physical = [&](unsigned key) { return c.offset(row.plane, key == row.pos ? row.self : key, kv_head); };
        std::vector<float> scores(end);
        float max_score = -std::numeric_limits<float>::infinity();
        for (unsigned key = lo; key < end; ++key) {
            const float* k = data.k.data() + physical(key);
            std::array<float, lanes> sums{};
            const unsigned lane_dims = (c.dim + lanes - 1U) / lanes;
            for (unsigned lane = 0; lane < lanes; ++lane)
                for (unsigned d = lane * lane_dims; d < std::min(c.dim, (lane + 1U) * lane_dims); ++d)
                    sums[lane] += q[d] * k[d];
            for (unsigned step = lanes / 2U; step != 0; step /= 2U)
                for (unsigned lane = 0; lane < step; ++lane) sums[lane] += sums[lane + step];
            scores[key] = sums[0];
            max_score = std::max(max_score, scores[key]);
        }

        struct Partial {
            std::array<float, chunk> weights{};
            std::vector<float> values;
        };
        std::vector<Partial> partials(last - first + 1U);
        for (auto& partial : partials) partial.values.resize(c.dim);
        const auto consume = [&](unsigned key, unsigned split, unsigned begin) {
            const float weight = std::exp(scores[key] - max_score);
            auto& partial = partials[split - first];
            partial.weights[key - begin] = weight;
            const float* v = data.v.data() + physical(key);
            for (unsigned d = 0; d < c.dim; ++d) partial.values[d] += weight * v[d];
        };
        if (plan) {
            for (unsigned split = first; split <= last; ++split) {
                const unsigned begin = strix::attn_core::position_split_begin(split);
                for (unsigned key = std::max(begin, visible.begin); key < std::min(begin + chunk, visible.end); ++key)
                    consume(key, split, begin);
            }
        } else {
            for (unsigned key = lo; key < end; ++key) consume(key, key / chunk, key - key % chunk);
        }

        float denominator = 0.0F;
        float* acc = out.data() + h * c.dim;
        for (auto& partial : partials) {
            float split_denom = 0.0F;
            for (unsigned base = 0; base < chunk; base += lanes) {
                for (unsigned step = lanes / 2U; step != 0; step /= 2U)
                    for (unsigned lane = 0; lane < step; ++lane)
                        partial.weights[base + lane] += partial.weights[base + lane + step];
                split_denom += partial.weights[base];
            }
            denominator += split_denom;
            for (unsigned d = 0; d < c.dim; ++d) acc[d] += partial.values[d];
        }
        for (unsigned d = 0; d < c.dim; ++d) acc[d] /= denominator;
    }
    return out;
}

}  // namespace b6_fixture
