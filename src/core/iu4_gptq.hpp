#pragma once
// Error-compensated (GPTQ / OBQ-style) quantisation into the shared IU4 g64 layout (core/iu4_onfly.hpp: packed nibbles
// via iu4_weight_byte + one FP16 scale per (output, K64 group)), host only, any model. Instead of rounding every weight
// to its nearest code (RTN), the input dimensions are visited in order and the rounding error of each weight is pushed
// onto the not-yet-quantised weights of the same output through the inverse input Hessian, so the OUTPUT error
// E ||(W - Q) x||^2 over the calibration rows is minimised. Each group's FP16 scale is chosen by the same rule as the
// load-time converter (onfly::quantise_group) from the error-updated weights when the group is reached; codes are
// iu4_requant_full with that scale, so the device reads the result with the unchanged IU4 kernels.
// Invariant (tests/iu4_gptq_tests.cpp): with a diagonal Hessian no error moves, and the result equals
// onfly::convert_q8_0_host byte for byte.
#include "core/iu4_onfly.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace strix::core::gptq {

// H += sum over rows of x x^T (K x K, row-major, double); x = rows x K floats.
inline void accumulate_hessian(const float* x, std::size_t rows, std::size_t k, std::vector<double>& h) {
    if (h.size() != k * k) h.assign(k * k, 0.0);
    for (std::size_t r = 0; r < rows; ++r) {
        const float* v = x + r * k;
        for (std::size_t i = 0; i < k; ++i) {
            const double a = v[i];
            if (a == 0.0) continue;
            double* hi = h.data() + i * k;
            for (std::size_t j = 0; j <= i; ++j) hi[j] += a * static_cast<double>(v[j]);
        }
    }
}

// Lower Cholesky in place (row-major K x K, lower triangle used); throws if not positive definite.
inline void cholesky_lower(std::vector<double>& a, std::size_t k) {
    for (std::size_t j = 0; j < k; ++j) {
        double d = a[j * k + j];
        for (std::size_t p = 0; p < j; ++p) d -= a[j * k + p] * a[j * k + p];
        if (!(d > 0.0)) throw std::runtime_error("gptq: Hessian not positive definite (raise damp)");
        const double l = std::sqrt(d);
        a[j * k + j] = l;
        for (std::size_t i = j + 1; i < k; ++i) {
            double s = a[i * k + j];
            const double* ai = a.data() + i * k;
            const double* aj = a.data() + j * k;
            for (std::size_t p = 0; p < j; ++p) s -= ai[p] * aj[p];
            a[i * k + j] = s / l;
        }
        for (std::size_t p = j + 1; p < k; ++p) a[j * k + p] = 0.0;
    }
}

// From the symmetric H (lower triangle filled by accumulate_hessian): U upper with H^-1 = U^T U (row-major, float),
// after the usual damping (damp * mean diagonal) and dead inputs (zero diagonal) set to 1.
inline std::vector<float> inverse_factor(std::vector<double> h, std::size_t k, double damp, std::vector<std::uint8_t>& dead,
                                         unsigned threads = 1U) {
    dead.assign(k, 0);
    double mean = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = 0; j < i; ++j) h[j * k + i] = h[i * k + j];
        if (h[i * k + i] == 0.0) { h[i * k + i] = 1.0; dead[i] = 1; }
        mean += h[i * k + i];
    }
    mean /= static_cast<double>(k);
    for (std::size_t i = 0; i < k; ++i) h[i * k + i] += damp * mean;
    cholesky_lower(h, k);  // H = L L^T
    // Columns of Linv (lower) by forward substitution, stored column-major (col c = L^-1 e_c, rows >= c); then
    // Hinv[i][j] = sum_p Linv[p][i] Linv[p][j] (p >= max(i, j)) as dots of contiguous columns. Threads split columns /
    // rows; every value is one fixed-order sum, so the thread count never changes a bit.
    std::vector<double> li(k * k, 0.0);
    std::vector<double> hinv(k * k, 0.0);
    const unsigned nt = std::max(1U, std::min(threads, static_cast<unsigned>(k)));
    const auto par = [nt](std::size_t count, const auto& body) {
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < nt; ++t)
            pool.emplace_back([&, t] {
                for (std::size_t c = t; c < count; c += nt) body(c);
            });
        for (auto& th : pool) th.join();
    };
    par(k, [&](std::size_t c) {
        double* y = li.data() + c * k;
        y[c] = 1.0 / h[c * k + c];
        for (std::size_t i = c + 1; i < k; ++i) {
            double acc = 0.0;
            const double* li_row = h.data() + i * k;
            for (std::size_t p = c; p < i; ++p) acc += li_row[p] * y[p];
            y[i] = -acc / h[i * k + i];
        }
    });
    par(k, [&](std::size_t i) {
        const double* ci = li.data() + i * k;
        for (std::size_t j = 0; j <= i; ++j) {
            const double* cj = li.data() + j * k;
            double acc = 0.0;
            for (std::size_t p = i; p < k; ++p) acc += ci[p] * cj[p];
            hinv[i * k + j] = acc;
            hinv[j * k + i] = acc;
        }
    });
    cholesky_lower(hinv, k);  // Hinv = M M^T, U = M^T
    std::vector<float> u(k * k, 0.0F);
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = i; j < k; ++j) u[i * k + j] = static_cast<float>(hinv[j * k + i]);
    return u;
}

// Block-diagonal form for wide inputs (K x K too large to factor): H restricted to diagonal blocks of `block` inputs
// (a multiple of 64 dividing K), each block damped and factored alone; U is K x K with zeros off the blocks.
inline std::vector<float> block_inverse_factor(const float* x, std::size_t rows, std::size_t k, std::size_t block, double damp,
                                               std::vector<std::uint8_t>& dead, unsigned threads = 1U) {
    if (!block || block % 64U || k % block) throw std::invalid_argument("gptq: block % 64 == 0 and divides K");
    std::vector<float> u(k * k, 0.0F), sub(rows * block);
    dead.assign(k, 0);
    for (std::size_t b0 = 0; b0 < k; b0 += block) {
        for (std::size_t r = 0; r < rows; ++r) std::copy(x + r * k + b0, x + r * k + b0 + block, sub.begin() + static_cast<std::ptrdiff_t>(r * block));
        std::vector<double> h;
        accumulate_hessian(sub.data(), rows, block, h);
        std::vector<std::uint8_t> bd;
        const auto ub = inverse_factor(h, block, damp, bd, threads);
        for (std::size_t i = 0; i < block; ++i) {
            dead[b0 + i] = bd[i];
            std::copy(ub.begin() + static_cast<std::ptrdiff_t>(i * block), ub.begin() + static_cast<std::ptrdiff_t>((i + 1U) * block),
                      u.begin() + static_cast<std::ptrdiff_t>((b0 + i) * k + b0));
        }
    }
    return u;
}

// W: n x k row-major dequantised source weights. u = inverse_factor(...) or empty for RTN (no error feedback).
// Returns the IU4 g64 copy; `dequant` (if non-null) receives the n x k quantised values (code * FP16 scale).
inline onfly::HostCopy quantise(const float* w_src, std::size_t n, std::size_t k, const std::vector<float>& u,
                                const std::vector<std::uint8_t>& dead, int rule, std::vector<float>* dequant = nullptr,
                                unsigned threads = 1U, std::size_t block = 0) {
    if (!w_src || !n || !k || k % 64U || !iu4_scale_rule_valid(rule)) throw std::invalid_argument("gptq: K % 64 == 0, valid rule");
    if (block && (block % 64U || k % block)) throw std::invalid_argument("gptq: block % 64 == 0 and divides K");
    if (!u.empty() && u.size() != k * k) throw std::invalid_argument("gptq: factor size");
    const std::size_t padded = onfly::padded_outputs(n), groups = k / 64U;
    onfly::HostCopy out;
    out.packed.assign(onfly::packed_bytes(padded, k), 0);
    out.scales.assign(padded * groups, 0);
    if (dequant) dequant->assign(n * k, 0.0F);
    const auto rows = [&](std::size_t c0, std::size_t c1) {
        std::vector<float> row(k);
        int qg[64];
        for (std::size_t c = c0; c < c1; ++c) {
            std::copy(w_src + c * k, w_src + (c + 1U) * k, row.begin());
            for (std::size_t i = 0; i < k; ++i)
                if (!dead.empty() && dead[i]) row[i] = 0.0F;
            float inv = 1.0F, scale = 1.0F;
            for (std::size_t i = 0; i < k; ++i) {
                if (i % 64U == 0U) {
                    const std::uint16_t bits = onfly::quantise_group(row.data() + i, rule, qg);
                    out.scales[c * groups + i / 64U] = bits;
                    scale = onfly::f16_bits_to_f32(bits);
                    inv = 1.0F / scale;
                }
                const int q = iu4_requant_full(row[i], inv);
                const float dq = scale * static_cast<float>(q);
                if (dequant) (*dequant)[c * k + i] = dq;
                auto& byte = out.packed[iu4_weight_byte(static_cast<int>(i & ~std::size_t{1}), static_cast<int>(c), static_cast<int>(padded))];
                byte = static_cast<std::uint8_t>(byte | ((static_cast<unsigned>(q) & 15U) << ((i & 1U) * 4U)));
                if (u.empty()) continue;
                const float* ui = u.data() + i * k;
                const float err = (row[i] - dq) / ui[i];
                const std::size_t end = block ? (i / block + 1U) * block : k;  // block-diagonal Hessian: feedback stays in the block
                for (std::size_t j = i + 1U; j < end; ++j) row[j] -= err * ui[j];
            }
        }
    };
    threads = std::max(1U, std::min<unsigned>(threads, static_cast<unsigned>(n)));
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; ++t) pool.emplace_back(rows, n * t / threads, n * (t + 1U) / threads);
    for (auto& t : pool) t.join();
    return out;
}

// Relative output error sum ||(Wq - W) x||^2 / sum ||W x||^2 over rows x (n x k weights, rows x k inputs).
inline double relative_output_error(const float* w, const float* wq, std::size_t n, std::size_t k, const float* x, std::size_t rows,
                                    unsigned threads = 1U) {
    const unsigned nt = std::max(1U, std::min(threads, static_cast<unsigned>(n)));
    std::vector<double> num(nt, 0.0), den(nt, 0.0);
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < nt; ++t)
        pool.emplace_back([&, t] {
            for (std::size_t c = t; c < n; c += nt)
                for (std::size_t r = 0; r < rows; ++r) {
                    double a = 0.0, b = 0.0;
                    const float* xr = x + r * k;
                    for (std::size_t i = 0; i < k; ++i) {
                        a += static_cast<double>(w[c * k + i]) * xr[i];
                        b += static_cast<double>(wq[c * k + i]) * xr[i];
                    }
                    num[t] += (b - a) * (b - a);
                    den[t] += a * a;
                }
        });
    for (auto& th : pool) th.join();
    double sn = 0.0, sd = 0.0;
    for (unsigned t = 0; t < nt; ++t) { sn += num[t]; sd += den[t]; }
    return sd > 0.0 ? sn / sd : 0.0;
}

}  // namespace strix::core::gptq
