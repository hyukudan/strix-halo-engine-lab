#pragma once

#include <array>
#include <cstddef>

namespace strix::core {

// Lane dot where a is produced on the fly: a(i) for i in [0, n).
template <class A, class B>
[[nodiscard]] float lane_dot_of(A a, B b, std::size_t n) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__clang__)
#pragma clang fp contract(off)
#endif
    std::array<float, 32> v{};
    for (std::size_t l = 0; l < 32; ++l)
        for (std::size_t i = l; i < n; i += 32) v[l] = v[l] + a(i) * b(i);
    for (std::size_t d = 16; d; d >>= 1U) {
        std::array<float, 32> next{};
        for (std::size_t l = 0; l < 32; ++l) next[l] = v[l] + v[l ^ d];
        v = next;
    }
    return v[0];
}

// Chunked dot (the long reductions: RMSNorm, hyper-connection mixes, Engram gate): the n elements are
// cut into chunks of dot_chunk; each chunk is a lane dot of its own elements (lane l takes chunk
// elements l, l + 32, ...), and total = total + chunk_sum in ascending chunk order from 0. Every chunk
// can run in its own wave.
inline constexpr std::size_t dot_chunk = 640;
template <class A, class B>
[[nodiscard]] float chunked_dot_of(A a, B b, std::size_t n, std::size_t chunk = dot_chunk) {
    float total = 0.0F;
    for (std::size_t c0 = 0; c0 < n; c0 += chunk) {
        const std::size_t len = n - c0 < chunk ? n - c0 : chunk;
        total = total + lane_dot_of([&](std::size_t i) { return a(c0 + i); }, [&](std::size_t i) { return b(c0 + i); }, len);
    }
    return total;
}
} // namespace strix::core
