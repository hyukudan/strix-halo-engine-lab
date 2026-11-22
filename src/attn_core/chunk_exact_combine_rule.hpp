#pragma once
// Host statement of the shared EXACT chunk-partial combine (chunk_exact_combine.hpp): the chains every form evaluates.
// Plain C++ (no SDK): partial layout pt[cc * (2 + dim) + {0: m, 1: l, 2 + d: acc[d]}], chunks cc < cnt.
#include <cmath>
#include <cstddef>

namespace strix::attn_core::chunk_exact {

template <class Exp> void combine_rule(const float* pt, unsigned cnt, unsigned dim, Exp exp_fn, float* o) {
    const std::size_t part = 2U + std::size_t{dim};
    float big = -INFINITY;
    for (unsigned cc = 0; cc < cnt; ++cc) {
        const float m = pt[cc * part];
        big = big < m ? m : big;
    }
    float L = 0.0F;
    for (unsigned cc = 0; cc < cnt; ++cc) L = std::fma(pt[cc * part + 1], exp_fn(pt[cc * part] - big), L);
    for (unsigned d = 0; d < dim; ++d) {
        float a = 0.0F;
        for (unsigned cc = 0; cc < cnt; ++cc) a = std::fma(pt[cc * part + 2 + d], exp_fn(pt[cc * part] - big), a);
        o[d] = a / L;
    }
}

} // namespace strix::attn_core::chunk_exact
