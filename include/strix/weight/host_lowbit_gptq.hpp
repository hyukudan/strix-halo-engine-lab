#pragma once
// Calibrated packing of the low-bit resident rows (strix/weight/host_lowbit.hpp q4_g64 / q6_g64), host only, any model.
// Same run-time bytes layout and the same device decode; only the choice of scales and codes changes.
//
// Inputs: the n x k source weights W (row-major, output-major) and the second moments H = sum x x^T of the matrix's
// INPUT rows x over a calibration set (k x k, see accumulate_second_moments). Two techniques, separately selectable:
//   importance  the clip search of every 64-element group weights each element's squared error by H[j][j]
//               (the input energy of that column) instead of 1.
//   feedback    error-compensated rounding (OBS / GPTQ order): the columns of a row are coded in ascending K and the
//               rounding error of column i is moved onto the columns not coded yet through the inverse-Hessian factor
//               U (H^-1 = U^T U, U upper), which minimises the OUTPUT error sum ||(W - Q) x||^2 over the calibration rows.
//               A group's FP16 scale is chosen when its first column is reached, from the error-updated weights.
// Invariants (tests/weight_lowbit_gptq_tests.cpp): with neither technique the bytes equal lowbit::encode_block; with a
// diagonal H the feedback moves nothing; the result never depends on the thread count.
#include "strix/weight/host_lowbit.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace strix::weight::lowbit::gptq {

namespace detail {
// Slice t of nt runs c = t, t + nt, ... (each c is independent: the results do not depend on the thread count). The
// calling thread runs slice 0 and every slice whose thread could not be created; every started thread is joined on
// every path, and the first exception of any slice is rethrown on the calling thread after the joins.
template <class Body>
inline void parallel_for(std::size_t count, unsigned threads, const Body& body) {
    const auto nt = static_cast<unsigned>(std::max<std::size_t>(1U, std::min<std::size_t>(threads, count)));
    if (nt <= 1U) {
        for (std::size_t c = 0; c < count; ++c) body(c);
        return;
    }
    std::mutex mu;
    std::exception_ptr first;
    const auto slice = [&](unsigned t) noexcept {
        try {
            for (std::size_t c = t; c < count; c += nt) body(c);
        } catch (...) {
            const std::lock_guard<std::mutex> lock(mu);
            if (!first) first = std::current_exception();
        }
    };
    std::vector<std::thread> pool;
    struct Joiner {
        std::vector<std::thread>& threads;
        ~Joiner() {
            for (auto& th : threads)
                if (th.joinable()) th.join();
        }
    } joiner{pool};
    unsigned started = 1U;
    try {
        pool.reserve(nt - 1U);
        for (; started < nt; ++started) pool.emplace_back(slice, started);
    } catch (...) {  // thread creation failed: the calling thread runs the slices that have no thread
    }
    slice(0U);
    for (unsigned t = started; t < nt; ++t) slice(t);
    for (auto& th : pool) th.join();
    if (first) std::rethrow_exception(first);
}
}  // namespace detail

// h (k x k row-major double, lower triangle) += sum over the rows of x x^T; x = rows x k floats. Every element is one
// fixed-order sum over the rows (ascending), whatever the thread count.
inline void accumulate_second_moments(const float* x, std::size_t rows, std::size_t k, std::vector<double>& h,
                                      unsigned threads = 1U) {
    if (h.size() != k * k) h.assign(k * k, 0.0);
    constexpr std::size_t chunk = 64;  // rows per pass: the chunk stays in cache while every H row visits it
    for (std::size_t r0 = 0; r0 < rows; r0 += chunk) {
        const std::size_t r1 = std::min(rows, r0 + chunk);
        detail::parallel_for(k, threads, [&](std::size_t i) {
            double* hi = h.data() + i * k;
            for (std::size_t r = r0; r < r1; ++r) {
                const float* v = x + r * k;
                const double a = v[i];
                if (a == 0.0) continue;
                for (std::size_t j = 0; j <= i; ++j) hi[j] += a * static_cast<double>(v[j]);
            }
        });
    }
}

// Lower Cholesky in place (row-major k x k, lower triangle read, upper zeroed); throws if not positive definite.
inline void cholesky_lower(std::vector<double>& a, std::size_t k, unsigned threads = 1U) {
    for (std::size_t j = 0; j < k; ++j) {
        double d = a[j * k + j];
        for (std::size_t p = 0; p < j; ++p) d -= a[j * k + p] * a[j * k + p];
        if (!(d > 0.0)) throw std::runtime_error("lowbit gptq: second moments not positive definite (raise damp)");
        const double l = std::sqrt(d);
        a[j * k + j] = l;
        const std::size_t below = k - j - 1U;
        detail::parallel_for(below, below >= 256U ? threads : 1U, [&](std::size_t t) {
            const std::size_t i = j + 1U + t;
            double s = a[i * k + j];
            const double* ai = a.data() + i * k;
            const double* aj = a.data() + j * k;
            for (std::size_t p = 0; p < j; ++p) s -= ai[p] * aj[p];
            a[i * k + j] = s / l;
        });
        for (std::size_t p = j + 1U; p < k; ++p) a[j * k + p] = 0.0;
    }
}

// From H (lower triangle as accumulate_second_moments leaves it): U upper (row-major float) with H^-1 = U^T U after
// damping by damp * mean(diag). Inputs with zero energy (dead[i] = 1) get diagonal 1; the encoder zeroes their weights
// (they never reach the output on the calibration rows).
inline std::vector<float> inverse_factor(std::vector<double> h, std::size_t k, double damp, std::vector<std::uint8_t>& dead,
                                         unsigned threads = 1U) {
    if (h.size() != k * k) throw std::invalid_argument("lowbit gptq: second moment size");
    dead.assign(k, 0);
    double mean = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = 0; j < i; ++j) h[j * k + i] = h[i * k + j];
        if (h[i * k + i] == 0.0) {
            h[i * k + i] = 1.0;
            dead[i] = 1;
        }
        mean += h[i * k + i];
    }
    mean /= static_cast<double>(k);
    for (std::size_t i = 0; i < k; ++i) h[i * k + i] += damp * mean;
    cholesky_lower(h, k, threads);  // H = L L^T
    // Columns of L^-1 by forward substitution (column-major: column c holds rows >= c), then
    // H^-1[i][j] = sum_p L^-1[p][i] L^-1[p][j] over p >= max(i, j); each value one fixed-order sum.
    std::vector<double> li(k * k, 0.0), hinv(k * k, 0.0);
    detail::parallel_for(k, threads, [&](std::size_t c) {
        double* y = li.data() + c * k;
        y[c] = 1.0 / h[c * k + c];
        for (std::size_t i = c + 1U; i < k; ++i) {
            double acc = 0.0;
            const double* row = h.data() + i * k;
            for (std::size_t p = c; p < i; ++p) acc += row[p] * y[p];
            y[i] = -acc / h[i * k + i];
        }
    });
    detail::parallel_for(k, threads, [&](std::size_t i) {
        const double* ci = li.data() + i * k;
        for (std::size_t j = 0; j <= i; ++j) {
            const double* cj = li.data() + j * k;
            double acc = 0.0;
            for (std::size_t p = i; p < k; ++p) acc += ci[p] * cj[p];
            hinv[i * k + j] = acc;
        }
    });
    cholesky_lower(hinv, k, threads);  // H^-1 = M M^T, U = M^T
    std::vector<float> u(k * k, 0.0F);
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = i; j < k; ++j) u[i * k + j] = static_cast<float>(hinv[j * k + i]);
    return u;
}

struct Options {
    const float* importance = nullptr;         // k weights for the clip search (H diagonal), or null
    const float* factor = nullptr;             // k x k U from inverse_factor, or null (no error feedback)
    const std::uint8_t* dead = nullptr;        // k flags from inverse_factor (with factor), or null
    unsigned threads = 1U;
    float* dequant = nullptr;                  // optional n x k output: the decoded weights
};

// Codec policy of the calibrated row coder: the superblock geometry, the per-group scale choice (from the
// error-updated weights, when the group's first column is reached), the code of one weight for that scale, its decoded
// value and the superblock packing. FixedGridCodec is the q4_g64 / q6_g64 grid (FP16 scale per 64, signed linear
// codes); StrixPackCodec is the Strix pack grid (scale byte per 32, sign + table index codes, sp::Header).
template <class Storage>
struct FixedGridCodec {
    static constexpr std::size_t elements = Storage::elements, group = Storage::group, groups = Storage::groups,
                                 bytes = Storage::bytes;
    struct State {
        std::uint16_t bits = 0;
        double sd = 0.0, inv = 0.0;
    };
    EncodeStatus::Code begin(const float* w, const float* importance, State& st) const {
        int scratch[Storage::group];
        if (const auto c = lowbit::detail::encode_group<Storage>(w, scratch, st.bits, importance); c != EncodeStatus::ok)
            return c;
        st.sd = static_cast<double>(f16_bits_to_f32(st.bits));
        st.inv = st.sd > 0.0 ? 1.0 / st.sd : 0.0;
        return EncodeStatus::ok;
    }
    int code(float w, const State& st) const { return st.sd > 0.0 ? lowbit::detail::quantise<Storage>(w, st.sd, st.inv) : 0; }
    float value(int c, const State& st) const { return static_cast<float>(st.sd) * static_cast<float>(c); }
    void pack(const int* q, const State* st, std::uint8_t* out) const {
        std::uint16_t scales[Storage::groups];
        for (std::size_t g = 0; g < Storage::groups; ++g) scales[g] = st[g].bits;
        pack_block<Storage>(q, scales, out);
    }
};

template <class S>
struct StrixPackCodec {
    static constexpr std::size_t elements = S::elements, group = S::group, groups = S::groups, bytes = S::bytes;
    const sp::Header* header = nullptr;
    struct State {
        std::uint8_t byte = 0;
        bool zero = true;
        double s = 0.0;
    };
    EncodeStatus::Code begin(const float* w, const float* importance, State& st) const {
        unsigned scratch[S::group];
        sp::encode_group<S>(w, *header, scratch, st.byte, importance);
        st.zero = true;
        for (std::size_t j = 0; j < S::group; ++j) st.zero = st.zero && w[j] == 0.0F;
        st.s = static_cast<double>(sp::scale_value(st.byte, header->base));
        return EncodeStatus::ok;
    }
    int code(float w, const State& st) const {
        return st.zero ? 0 : static_cast<int>(sp::quantise<S>(w, st.s, header->table.data()));
    }
    float value(int c, const State& st) const {
        const auto u = static_cast<unsigned>(c);
        const float m = static_cast<float>(header->table[u & (S::levels - 1U)]) * static_cast<float>(st.s);
        return (u >> (S::bits - 1U)) != 0U ? -m : m;
    }
    void pack(const int* q, const State* st, std::uint8_t* out) const {
        unsigned codes[S::elements];
        std::uint8_t scales[S::groups];
        for (std::size_t j = 0; j < S::elements; ++j) codes[j] = static_cast<unsigned>(q[j]);
        for (std::size_t g = 0; g < S::groups; ++g) scales[g] = st[g].byte;
        sp::pack_block<S>(codes, scales, out);
    }
};

// n x k source weights -> n rows of k / Codec::elements superblocks (out: n * (k / elements) * bytes bytes).
template <class Codec>
inline EncodeStatus encode_rows_with(const Codec& codec, const float* w, std::size_t n, std::size_t k, std::uint8_t* out,
                                     const Options& opt = {}) {
    if (!w || !out || !n || !k || k % Codec::elements != 0U) throw std::invalid_argument("lowbit gptq: K % 128 == 0");
    for (std::size_t e = 0; e < n * k; ++e)
        if (!std::isfinite(w[e])) return {EncodeStatus::nonfinite};
    const std::size_t blocks = k / Codec::elements, row_bytes = blocks * Codec::bytes;
    constexpr std::size_t kRowBlock = 16;  // rows coded together so each U row is read once per block
    const std::size_t row_blocks = (n + kRowBlock - 1U) / kRowBlock;
    std::vector<std::uint8_t> failed(row_blocks, 0);
    detail::parallel_for(row_blocks, opt.threads, [&](std::size_t b) {
        const NearestRounding nearest;
        const std::size_t r0 = b * kRowBlock, rn = std::min(n - r0, kRowBlock);
        std::vector<float> rows(rn * k), err(rn);
        std::vector<int> q(rn * Codec::elements);
        std::vector<typename Codec::State> states(rn * Codec::groups);
        for (std::size_t r = 0; r < rn; ++r) {
            std::copy(w + (r0 + r) * k, w + (r0 + r + 1U) * k, rows.begin() + static_cast<std::ptrdiff_t>(r * k));
            if (opt.factor && opt.dead)
                for (std::size_t i = 0; i < k; ++i)
                    if (opt.dead[i]) rows[r * k + i] = 0.0F;
        }
        for (std::size_t i = 0; i < k; ++i) {
            const std::size_t in_block = i % Codec::elements, group = in_block / Codec::group;
            for (std::size_t r = 0; r < rn; ++r) {
                float* row = rows.data() + r * k;
                auto& st = states[r * Codec::groups + group];
                if (in_block % Codec::group == 0U &&
                    codec.begin(row + i, opt.importance ? opt.importance + i : nullptr, st) != EncodeStatus::ok) {
                    failed[b] = 1;
                    return;
                }
                const int code = codec.code(row[i], st);
                q[r * Codec::elements + in_block] = code;
                const float dq = codec.value(code, st);
                if (opt.dequant) opt.dequant[(r0 + r) * k + i] = dq;
                err[r] = opt.factor ? (row[i] - dq) / opt.factor[i * k + i] : 0.0F;
                if (in_block + 1U == Codec::elements)
                    codec.pack(q.data() + r * Codec::elements, states.data() + r * Codec::groups,
                               out + (r0 + r) * row_bytes + (i / Codec::elements) * Codec::bytes);
            }
            if (!opt.factor) continue;
            const float* ui = opt.factor + i * k;
            for (std::size_t r = 0; r < rn; ++r) {
                const float e = err[r];
                float* row = rows.data() + r * k;
                for (std::size_t j = i + 1U; j < k; ++j) row[j] -= e * ui[j];
            }
        }
    });
    for (const auto f : failed)
        if (f) return {EncodeStatus::scale_overflow};
    return {EncodeStatus::ok};
}

// q4_g64 / q6_g64: n x k source weights -> n rows of k / Storage::elements superblocks (n * Storage::row_bytes(k) bytes).
template <class Storage>
inline EncodeStatus encode_rows(const float* w, std::size_t n, std::size_t k, std::uint8_t* out, const Options& opt = {}) {
    return encode_rows_with(FixedGridCodec<Storage>{}, w, n, k, out, opt);
}

// Strix pack: the same coder on the SP grid of header h (its table and base; see sp::make_header / sp::fit_table).
template <class S>
inline EncodeStatus encode_rows_sp(const sp::Header& h, const float* w, std::size_t n, std::size_t k, std::uint8_t* out,
                                   const Options& opt = {}) {
    StrixPackCodec<S> codec;
    codec.header = &h;
    return encode_rows_with(codec, w, n, k, out, opt);
}

// Output error of a coded matrix against its source under input second moments h (lower triangle, k x k):
// sum_c d_c^T H d_c with d_c = Wq_c - W_c, and the same form for W itself (the signal). The ratio is the relative
// output error sum ||(Wq - W) x||^2 / sum ||W x||^2 over the rows that produced h.
struct OutputError {
    double error = 0.0, signal = 0.0;
    // The signal must be positive and both sums finite; otherwise the ratio does not exist (NaN, never a zero error).
    [[nodiscard]] bool available() const { return std::isfinite(signal) && std::isfinite(error) && signal > 0.0; }
    [[nodiscard]] double relative() const {
        return available() ? error / signal : std::numeric_limits<double>::quiet_NaN();
    }
};

inline OutputError output_error(const float* w, const float* wq, std::size_t n, std::size_t k, const std::vector<double>& h,
                                unsigned threads = 1U) {
    if (h.size() != k * k) throw std::invalid_argument("lowbit gptq: second moment size");
    std::vector<double> err(n, 0.0), sig(n, 0.0);
    detail::parallel_for(n, threads, [&](std::size_t c) {
        const float* a = w + c * k;
        const float* b = wq + c * k;
        double e = 0.0, s = 0.0;
        for (std::size_t i = 0; i < k; ++i) {
            const double di = static_cast<double>(b[i]) - a[i], wi = a[i];
            const double* hi = h.data() + i * k;
            double ed = 0.0, sd = 0.0;
            for (std::size_t j = 0; j < i; ++j) {
                ed += hi[j] * (static_cast<double>(b[j]) - a[j]);
                sd += hi[j] * a[j];
            }
            e += di * (2.0 * ed + hi[i] * di);
            s += wi * (2.0 * sd + hi[i] * wi);
        }
        err[c] = e;
        sig[c] = s;
    });
    OutputError total;
    for (std::size_t c = 0; c < n; ++c) {
        total.error += err[c];
        total.signal += sig[c];
    }
    return total;
}

}  // namespace strix::weight::lowbit::gptq
