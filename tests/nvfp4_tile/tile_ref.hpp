#pragma once
// Test-side reference for the NVFP4 tile GEMM (strix/cuda/core/nvfp4_tile.hpp): operands built from an INDEPENDENT transcription of
// the decode record layout (strix/cuda_core/nvfp4.hpp), the host quantiser of nvfp4-mma-f32-v1 applied per row, and a plain
// double-precision reference. Shared by the host oracle (tile_host.cpp) and the device fixture (tile_device.cu).
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "strix/cuda/core/nvfp4.hpp"
#include "strix/cuda/core/nvfp4_tile.hpp"

namespace tile_ref {
namespace T = strix::cuda::core::nvfp4::tile;
namespace N = strix::cuda::core::nvfp4;

// decode record layout, written from its documentation (not from rec_slot): byte of input k of row o, and the scale byte
inline std::size_t rec_code(std::uint32_t o, std::uint32_t k, std::uint32_t in) {
    const std::uint32_t m = o / 16, h = (o % 16) / 8, g = o % 8, c = k / 128, kc = k % 128, j = kc / 16, e = kc % 16;
    const std::uint32_t p = e / 8, t = (e % 8) / 2;
    return (static_cast<std::size_t>(m) * (in / 128) + c) * 1152 + h * 512 + (4 * g + t) * 16 + (j / 2) * 4 + 2 * (j % 2) + p;
}
inline std::size_t rec_scale(std::uint32_t o, std::uint32_t k, std::uint32_t in) {
    const std::uint32_t m = o / 16, h = (o % 16) / 8, g = o % 8, c = k / 128, j = (k % 128) / 16;
    return (static_cast<std::size_t>(m) * (in / 128) + c) * 1152 + 1024 + 16 * g + 8 * h + j;
}

struct Weight {
    std::uint32_t out = 0, in = 0;
    float gs = 1.f;
    std::vector<std::uint8_t> rec;  // decode record layout
};
inline Weight random_weight(std::uint32_t out, std::uint32_t in, std::uint32_t seed) {
    Weight w;
    w.out = out;
    w.in = in;
    w.gs = 0.0123f * static_cast<float>(1 + seed % 3);
    w.rec.resize(T::w_tile_bytes(out, in));
    std::mt19937 r(seed);
    for (std::uint32_t o = 0; o < out; ++o)
        for (std::uint32_t k = 0; k < in; k += 2) w.rec[rec_code(o, k, in)] = static_cast<std::uint8_t>(r());
    for (std::uint32_t o = 0; o < out; ++o)
        for (std::uint32_t k = 0; k < in; k += 16) w.rec[rec_scale(o, k, in)] = static_cast<std::uint8_t>(0x28 + r() % 0x20);
    return w;
}
// e2m1 * e4m3 of weight element (o, k) (gscale excluded)
inline double w_value(const Weight& w, std::uint32_t o, std::uint32_t k) {
    const std::uint8_t b = w.rec[rec_code(o, k, w.in)];
    return static_cast<double>(N::e2m1(static_cast<std::uint8_t>((k & 1) ? b >> 4 : b & 15))) * N::e4m3(w.rec[rec_scale(o, k, w.in)]);
}

struct Act {
    std::uint32_t rows = 0, rows_pad = 0, in = 0;
    std::vector<std::uint8_t> codes, scales, tile;  // canonical [rows][in/2], [rows][in/16]; X tile layout
    std::vector<float> rs;                          // [rows_pad]
};
// per-row quantisation of the tile contract: rs = amax / 2688, nvfp4::quantize with tensor_scale = rs; zero rows stay zero
inline Act quantize(const std::vector<float>& x, std::uint32_t rows, std::uint32_t in) {
    Act a;
    a.rows = rows;
    a.rows_pad = T::rows_padded(rows);
    a.in = in;
    a.codes.assign(static_cast<std::size_t>(rows) * in / 2, 0);
    a.scales.assign(static_cast<std::size_t>(rows) * in / 16, 0);
    a.tile.assign(T::x_tile_bytes(a.rows_pad, in), 0);
    a.rs.assign(a.rows_pad, 0.f);
    for (std::uint32_t r = 0; r < rows; ++r) {
        float amax = 0.f;
        for (std::uint32_t k = 0; k < in; ++k) amax = std::fmax(amax, std::fabs(x[static_cast<std::size_t>(r) * in + k]));
        const float rs = amax / T::kRowScaleDiv;
        a.rs[r] = rs;
        if (rs > 0.f)
            N::quantize(&x[static_cast<std::size_t>(r) * in], in, 1, in, &a.codes[static_cast<std::size_t>(r) * in / 2], in / 2,
                        &a.scales[static_cast<std::size_t>(r) * in / 16], in / 16, rs);
        for (std::uint32_t k = 0; k < in; k += 2) a.tile[T::x_code_byte(r, k, in)] = a.codes[(static_cast<std::size_t>(r) * in + k) / 2];
        for (std::uint32_t q = 0; q < in / 16; ++q) a.tile[T::x_scale_byte(r, q, in)] = a.scales[static_cast<std::size_t>(r) * in / 16 + q];
    }
    return a;
}
inline double x_value(const Act& a, std::uint32_t r, std::uint32_t k) {
    const std::uint8_t b = a.codes[(static_cast<std::size_t>(r) * a.in + k) / 2];
    return static_cast<double>(N::e2m1(static_cast<std::uint8_t>((k & 1) ? b >> 4 : b & 15))) *
           N::e4m3(a.scales[(static_cast<std::size_t>(r) * a.in + k) / 16]);
}
inline std::vector<float> random_x(std::uint32_t rows, std::uint32_t in, std::uint32_t seed) {
    std::mt19937 r(seed);
    std::normal_distribution<float> d(0.f, 1.f);
    std::vector<float> x(static_cast<std::size_t>(rows) * in);
    for (auto& v : x) v = d(r);
    for (std::uint32_t k = 0; k < in; k += 97) x[k] = 0.f;  // a few exact zeros
    if (rows > 3)
        for (std::uint32_t k = 0; k < in; ++k) x[3 * static_cast<std::size_t>(in) + k] = 0.f;  // one all-zero row
    if (rows > 5) x[5 * static_cast<std::size_t>(in) + 7] = 3.0e4f;                       // one outlier
    return x;
}
// reference dot in double: sum_k W(o, k) X(r, k) (scales excluded)
inline double dot(const Weight& w, const Act& a, std::uint32_t o, std::uint32_t r) {
    double s = 0;
    for (std::uint32_t k = 0; k < w.in; ++k) s += w_value(w, o, k) * x_value(a, r, k);
    return s;
}
inline double silu(double v) { return v / (1.0 + std::exp(-v)); }
}  // namespace tile_ref
