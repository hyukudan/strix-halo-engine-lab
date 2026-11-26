#pragma once

#include "../core/linear_reference.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace strix::attn_core {

// Largest key first, breaking ties by the lowest position. Return the selected
// positions in causal order so the following reduction has a fixed order.
template<class Key>
std::vector<std::uint32_t> ordered_top_k(std::span<const Key> keys, unsigned k) {
    std::vector<std::uint32_t> order(keys.size());
    for (std::uint32_t i = 0; i < order.size(); ++i) order[i] = i;
    const std::size_t take = std::min<std::size_t>(k, order.size());
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(take), order.end(),
        [&](std::uint32_t x, std::uint32_t y) { return keys[x] != keys[y] ? keys[x] > keys[y] : x < y; });
    order.resize(take);
    std::sort(order.begin(), order.end());
    return order;
}

// Quantize independent blocks of 32 E2M1 values with a power-of-two scale.
// Both index-query arithmetic and the packed key writer use this contract.
inline void quantize_index_fp4_reference(const float* x, unsigned n, std::uint8_t* codes, int* exponents) {
    namespace codec = weight::scalar_float;
    for (unsigned group = 0; group < n / 32; ++group) {
        float amax = 0.0F;
        for (unsigned i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[32 * group + i]));
        amax = std::fmax(amax, 7.05296657e-38F); // 6 * 2^-126
        const int exponent = codec::log2_ceil_positive_normal(amax * (1.0F / 6.0F));
        const float scale = codec::pow2f(exponent);
        for (unsigned i = 0; i < 32; ++i)
            codes[32 * group + i] = codec::e2m1_encode(std::clamp(x[32 * group + i] / scale, -6.0F, 6.0F));
        exponents[group] = exponent;
    }
}

inline std::vector<std::uint8_t> pack_index_fp4_reference(std::span<const float> key) {
    if (key.empty() || key.size() % 32U != 0U)
        throw std::invalid_argument("index key: expected nonempty 32-channel blocks");
    std::vector<std::uint8_t> codes(key.size()), entry(key.size() / 2U + key.size() / 32U);
    std::vector<int> exponents(key.size() / 32U);
    quantize_index_fp4_reference(key.data(), static_cast<unsigned>(key.size()), codes.data(), exponents.data());
    for (std::size_t i = 0; i < key.size() / 2U; ++i)
        entry[i] = static_cast<std::uint8_t>(codes[2U * i] | (codes[2U * i + 1U] << 4U));
    for (std::size_t i = 0; i < exponents.size(); ++i)
        entry[key.size() / 2U + i] = static_cast<std::uint8_t>(exponents[i] + 127);
    return entry;
}

inline void unpack_index_fp4_reference(std::span<const std::uint8_t> entry, std::span<float> key) {
    namespace codec = weight::scalar_float;
    if (key.empty() || key.size() % 32U != 0U || entry.size() != key.size() / 2U + key.size() / 32U)
        throw std::invalid_argument("index key: packed shape differs");
    for (std::size_t i = 0; i < key.size(); ++i) {
        const auto code = static_cast<std::uint8_t>((entry[i / 2U] >> (4U * (i & 1U))) & 15U);
        key[i] = codec::e2m1(code) * codec::e8m0(entry[key.size() / 2U + i / 32U]);
    }
}

// Head gates select the strictly positive logits. An all-negative row retains
// the strongest head. The caller supplies only the committed prefix and the
// current entry. The top-k is dense when the causal prefix fits.
template<class KeyAt>
std::vector<std::uint32_t> gated_index_top_k_reference(std::span<const float> queries, std::span<const float> weights,
    std::span<const float> gates, unsigned dim, std::size_t visible, unsigned top_k, KeyAt key_at) {
    if (dim == 0 || weights.empty() || gates.size() != weights.size() || queries.size() != weights.size() * dim)
        throw std::invalid_argument("sparse index: invalid query or head gate shape");
    const auto best_head = static_cast<std::size_t>(std::ranges::max_element(gates) - gates.begin());
    std::vector<std::size_t> selected_heads;
    for (std::size_t h = 0; h < gates.size(); ++h)
        if (gates[h] > 0.0F || h == best_head) selected_heads.push_back(h);
    std::vector<float> scores(visible), key(dim);
    if (visible > top_k) {
        for (std::size_t token = 0; token < visible; ++token) {
            unpack_index_fp4_reference(key_at(token), key);
            for (const auto h : selected_heads) {
                const float dot = core::dot_f32(queries.subspan(h * dim, dim), key);
                scores[token] = std::fma(weights[h], std::max(0.0F, dot), scores[token]);
            }
        }
    }
    return ordered_top_k<float>(scores, top_k);
}

} // namespace strix::attn_core
