// No-model fixture of the FAST small-M form of the grouped microscaled tile (grouped_microscaled::Form::small_f16) at the
// Inkling 2 Mini expert shapes (gate_up 2304 x 2304 interleaved halves, down 2304 x 1152; 128 experts, top 8), NVFP4 and
// MXFP8. Finite-data checks seed output buffers with NaN bytes and require finite outputs. Checks: (1) every output
// of a routed 8-row visit against a double reference of the decoded weights and the BF16 inputs (normalised by
// sum |x w|), for the small form and the incumbent BF16 tile; (2) tile launches of 1 / 16 / 17 / 32 / 33 / 48 / 64 rows:
// the small form against the BF16 tile, and EVERY row of every tile run alone (a one-row tile) gives the same bits as in
// its tile, whatever its position; (3) the full route (grouped_dispatch::enqueue: token rows, the in-place gate-up
// launch_indirect, activation, down, combine) of 1 / 8 / 16 / 17 / 33 / 64 token rows: the small form against the
// BF16 tile, and every token row enqueued alone gives the same bits as in its batch (batch invariance of the form).
// Then ms per visit (gate_up + down projections) of R1 rows (8-row chunks, activation and combine included), the
// incumbent tile and the small form, over rows 1..40, uniform and skewed routes, and GB/s over the distinct expert
// bytes; and ms per full route (enqueue) beside R1 at rows 1 / 8 / 16. Exit 1 on a failed check. --check-only skips the
// timings (ctest), --bench-only the checks.
#include <hip/hip_runtime.h>

#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "moe_core/grouped_dispatch.hpp"
#include "moe_core/grouped_microscaled.hpp"
#include "moe_core/microscaled_r1.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

namespace packed = strix::weight::microscaled;
namespace sf = strix::weight::scalar_float;
namespace grouped = strix::moe_core::grouped_microscaled;
namespace r1 = strix::moe_core::microscaled_r1;
namespace dispatch = strix::moe_core::grouped_dispatch;
using strix::moe_core::device_grouped::ExpertWeights;
using Format = packed::MatrixFormat;

constexpr unsigned kExperts = 128U, kTopK = 8U, kHidden = 2304U, kInter = 1152U, kGate = 2U * kInter;
constexpr unsigned kMaxRows = 64U;  // token rows of the largest checked launch
constexpr unsigned kVariants = 6U;  // route draws cycled per timed rep (distinct experts per rep, cold weights)

void require_hip(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        std::cerr << "HIP failure: " << what << ": " << hipGetErrorString(status) << '\n';
        std::exit(2);
    }
}

std::uint16_t bf16(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7FFFU + ((u >> 16U) & 1U);
    return static_cast<std::uint16_t>(u >> 16U);
}
float widen(std::uint16_t h) {
    const std::uint32_t u = std::uint32_t{h} << 16U;
    float x;
    std::memcpy(&x, &u, 4);
    return x;
}

// NaN-propagating maximum (std::max keeps the old value against a NaN) and an all-finite check of host outputs.
void worse(double& max, double value) {
    if (!(value <= max)) max = value;
}
bool finite(const std::vector<float>& v) {
    return std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
}

template <class T>
T* device_copy(const std::vector<T>& host) {
    T* p = nullptr;
    require_hip(hipMalloc(&p, host.size() * sizeof(T)), "malloc");
    require_hip(hipMemcpy(p, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice), "copy");
    return p;
}

struct Matrix {  // host bytes of one expert matrix
    std::vector<std::uint8_t> codes, scales;
};

// Random valid codes; scales in a range that keeps every product finite.
Matrix make_matrix(Format f, unsigned rows, unsigned cols, std::mt19937& rng) {
    Matrix m;
    const bool nv = f == Format::nvfp4;
    m.codes.resize(std::size_t{rows} * (nv ? cols / 2U : cols));
    m.scales.resize(std::size_t{rows} * cols / (nv ? 16U : 32U));
    for (std::size_t i = 0; i < m.codes.size(); i += 4U) {
        const auto word = static_cast<std::uint32_t>(rng());
        for (std::size_t j = 0; j < 4U && i + j < m.codes.size(); ++j) {
            auto b = static_cast<std::uint8_t>(word >> (8U * j));
            if (!nv && (b & 0x7FU) == 0x7FU) b = static_cast<std::uint8_t>(b ^ 1U);  // no E4M3 NaN
            m.codes[i + j] = b;
        }
    }
    std::uniform_int_distribution<int> s4(0x20, 0x40), s8(118, 124);
    for (auto& s : m.scales) s = static_cast<std::uint8_t>(nv ? s4(rng) : s8(rng));
    return m;
}

double weight(Format f, const Matrix& m, float outer, unsigned cols, std::size_t prow, unsigned k) {
    if (f == Format::nvfp4) {
        const auto byte = m.codes[prow * (cols / 2U) + k / 2U];
        const auto code = static_cast<std::uint8_t>(k % 2U ? byte >> 4U : byte & 0x0FU);
        return double{sf::e2m1(code)} * double{sf::e4m3(m.scales[prow * (cols / 16U) + k / 16U])} * double{outer};
    }
    return double{sf::e4m3(m.codes[prow * cols + k])} * double{sf::e8m0(m.scales[prow * (cols / 32U) + k / 32U])};
}

struct Routes {
    std::vector<unsigned> ids;  // [rows][8] token-major
    unsigned distinct;
};

Routes draw(unsigned rows, bool skewed, std::mt19937& rng) {
    std::vector<double> p(kExperts, 1.0);
    if (skewed) {
        std::vector<unsigned> perm(kExperts);
        std::iota(perm.begin(), perm.end(), 0U);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (unsigned e = 0; e < kExperts; ++e) p[perm[e]] = 1.0 / std::pow(double(e + 1U), 0.9);
    }
    Routes r{{}, 0U};
    std::vector<int> seen(kExperts, 0);
    for (unsigned row = 0; row < rows; ++row) {
        auto w = p;
        for (unsigned k = 0; k < kTopK; ++k) {
            std::discrete_distribution<unsigned> d(w.begin(), w.end());
            const unsigned e = d(rng);
            w[e] = 0.0;
            r.ids.push_back(e);
            seen[e] = 1;
        }
    }
    r.distinct = static_cast<unsigned>(std::count(seen.begin(), seen.end(), 1));
    return r;
}

struct Format_ {
    Format f;
    std::vector<Matrix> gu, dn;
    std::vector<float> outer;
    std::vector<ExpertWeights> host;  // device addresses
    ExpertWeights* dev{};
    double expert_bytes{};
};

Format_ make_format(Format f, std::mt19937& rng) {
    Format_ F{f, {}, {}, {}, {}, nullptr, 0.0};
    const bool nv = f == Format::nvfp4;
    for (unsigned e = 0; e < kExperts; ++e) {
        F.gu.push_back(make_matrix(f, kGate, kHidden, rng));
        F.dn.push_back(make_matrix(f, kHidden, kInter, rng));
        F.outer.push_back(nv ? 0.75F + 0.001F * float(e) : 1.0F);
        const auto upload = [&](const Matrix& m, unsigned rows, unsigned cols, packed::RowOrder order) {
            packed::PackedMatrix pm{device_copy(m.codes), device_copy(m.scales), rows, cols, F.outer.back(), order};
            return pm;
        };
        F.host.push_back({upload(F.gu.back(), kGate, kHidden, packed::RowOrder::interleaved_halves),
                          upload(F.dn.back(), kHidden, kInter, packed::RowOrder::contiguous)});
    }
    F.dev = device_copy(F.host);
    const auto& m0 = F.gu[0];
    const auto& d0 = F.dn[0];
    F.expert_bytes = double(m0.codes.size() + m0.scales.size() + d0.codes.size() + d0.scales.size());
    return F;
}

// Expert-major tiles of one route draw over gathered rows (assignment a = position in expert-major order).
struct Plan {
    std::vector<grouped::TileRows> gu, dn;
    std::vector<unsigned> token;  // assignment -> token row
    unsigned max_rows{};
};

Plan plan_of(const Format_& F, const Routes& r, unsigned rows, std::uint16_t* gu_in, float* gu_out, std::uint16_t* dn_in,
             float* dn_out) {
    Plan p;
    for (unsigned e = 0; e < kExperts; ++e) {
        const unsigned first = static_cast<unsigned>(p.token.size());
        for (unsigned row = 0; row < rows; ++row)
            for (unsigned k = 0; k < kTopK; ++k)
                if (r.ids[row * kTopK + k] == e) p.token.push_back(row);
        const unsigned n = static_cast<unsigned>(p.token.size()) - first;
        if (n == 0U) continue;
        p.max_rows = std::max(p.max_rows, n);
        p.gu.push_back({F.host[e].gate_up, gu_in + std::size_t{first} * kHidden, gu_out + std::size_t{first} * kGate,
                        kHidden, kGate, n});
        p.dn.push_back({F.host[e].down, dn_in + std::size_t{first} * kInter, dn_out + std::size_t{first} * kHidden,
                        kInter, kHidden, n});
    }
    return p;
}

struct Device {
    std::uint16_t *gu_in{}, *dn_in{};
    float *gu_out{}, *dn_out{};
    grouped::TileRows *gu_tiles{}, *dn_tiles{};
    // R1
    unsigned* ids{};
    float *weights{}, *x{}, *y{};
    std::uint16_t* act{};
    unsigned* failure{};
};

constexpr unsigned kMaxAssign = kMaxRows * kTopK;

Device make_device() {
    Device d;
    require_hip(hipMalloc(&d.gu_in, std::size_t{kMaxAssign} * kHidden * 2U), "m");
    require_hip(hipMalloc(&d.dn_in, std::size_t{kMaxAssign} * kInter * 2U), "m");
    require_hip(hipMalloc(&d.gu_out, std::size_t{kMaxAssign} * kGate * 4U), "m");
    require_hip(hipMalloc(&d.dn_out, std::size_t{kMaxAssign} * kHidden * 4U), "m");
    require_hip(hipMalloc(&d.gu_tiles, kMaxAssign * sizeof(grouped::TileRows)), "m");  // one-row tiles: one per assignment
    require_hip(hipMalloc(&d.dn_tiles, kMaxAssign * sizeof(grouped::TileRows)), "m");
    require_hip(hipMalloc(&d.ids, kMaxAssign * 4U), "m");
    require_hip(hipMalloc(&d.weights, kMaxAssign * 4U), "m");
    require_hip(hipMalloc(&d.x, std::size_t{kMaxRows} * kHidden * 4U), "m");
    require_hip(hipMalloc(&d.y, std::size_t{kMaxRows} * kHidden * 4U), "m");
    require_hip(hipMalloc(&d.act, std::size_t{kMaxAssign} * kInter * 2U), "m");
    require_hip(hipMalloc(&d.failure, kMaxRows * 4U), "m");
    require_hip(hipMemset(d.failure, 0, kMaxRows * 4U), "m");
    return d;
}

grouped::Args args_of(const grouped::TileRows* tiles, unsigned count, unsigned inputs, unsigned outputs, unsigned max_rows,
                      grouped::Form form) {
    grouped::Args a{};
    a.tiles = tiles;
    a.count = count;
    a.inputs = inputs;
    a.outputs = outputs;
    a.max_tile_rows = max_rows;
    a.form = form;
    return a;
}

template <Format F>
std::vector<float> edge_projection(const Device& d, const grouped::TileRows& tile, grouped::Form form,
                                   const std::uint32_t* map = nullptr) {
    std::vector<float> out(tile.rows * tile.output_row, std::numeric_limits<float>::infinity());
    require_hip(hipMemcpy(d.gu_out, out.data(), out.size() * sizeof(float), hipMemcpyHostToDevice), "edge output sentinel");
    require_hip(hipMemcpy(d.gu_tiles, &tile, sizeof(tile), hipMemcpyHostToDevice), "edge tile");
    const auto args = args_of(d.gu_tiles, 1U, static_cast<unsigned>(tile.input_row),
                              static_cast<unsigned>(tile.output_row), tile.rows, form);
    const auto status = map ? grouped::launch_indirect<F>({args, map, tile.input}, nullptr)
                            : grouped::launch<F>(args, nullptr);
    if (status != hipSuccess) {
        std::cerr << "edge " << (F == Format::nvfp4 ? "NVFP4" : "MXFP8") << " K=" << args.inputs
                  << " N=" << args.outputs << ' ' << (map ? "indirect" : "direct") << ": "
                  << hipGetErrorString(status) << '\n';
        return {};
    }
    require_hip(hipMemcpy(out.data(), d.gu_out, out.size() * sizeof(float), hipMemcpyDeviceToHost), "edge read");
    return out;
}

bool mxfp8_edge_check(const Device& d) {
    constexpr unsigned k = 128U, n = 128U;
    const std::vector<unsigned> bytes{246U, 247U, 254U, 255U};
    constexpr unsigned positive_nan_col = 4U, negative_nan_col = 5U;
    std::vector<std::uint8_t> codes(k * n), scales(k * n / 32U, 127U);
    for (unsigned c = 0; c < bytes.size(); ++c) {
        codes[c * k] = 0x38U;  // E4M3 1
        scales[c * (k / 32U)] = static_cast<std::uint8_t>(bytes[c]);
    }
    // Unit scales isolate the two reserved codes, in opposite halves of an FP16 pair.
    codes[positive_nan_col * k] = 0x7FU;
    codes[negative_nan_col * k + 1U] = 0xFFU;
    std::vector<std::uint16_t> input(2U * k, 0U);
    input[0] = input[1] = bf16(1.0F);  // one unit contribution per column; second row is all zero
    require_hip(hipMemcpy(d.gu_in, input.data(), input.size() * sizeof(input[0]), hipMemcpyHostToDevice), "scale input");
    auto* device_codes = device_copy(codes);
    auto* device_scales = device_copy(scales);
    const packed::PackedMatrix w{device_codes, device_scales, n, k, 1.0F};
    const grouped::TileRows tile{w, d.gu_in, d.gu_out, k, n, 2U};
    const auto out = edge_projection<Format::mxfp8>(d, tile, grouped::Form::small_f16);
    const auto ref = edge_projection<Format::mxfp8>(d, tile, grouped::Form::bf16_tile);
    require_hip(hipFree(device_codes), "free edge codes");
    require_hip(hipFree(device_scales), "free edge scales");
    if (out.empty() || ref.empty()) return false;
    bool ok = true;
    for (unsigned c = 0; c < bytes.size(); ++c) {
        const bool matches = bytes[c] == 255U
            ? std::isnan(out[c]) && std::isnan(out[n + c]) && std::isnan(ref[c]) && std::isnan(ref[n + c])
            : out[c] == std::ldexp(1.0F, static_cast<int>(bytes[c]) - 127) && out[n + c] == 0.0F &&
              out[c] == ref[c] && out[n + c] == ref[n + c];
        if (!matches) {
            std::cerr << "MXFP8 scale " << bytes[c] << ": small " << out[c] << ", " << out[n + c]
                      << " incumbent " << ref[c] << ", " << ref[n + c] << '\n';
            ok = false;
        }
    }
    for (unsigned c : {positive_nan_col, negative_nan_col}) {
        if (!std::isnan(out[c]) || !std::isnan(out[n + c]) || !std::isnan(ref[c]) || !std::isnan(ref[n + c])) {
            std::cerr << "MXFP8 reserved weight code " << (c == positive_nan_col ? "0x7f" : "0xff")
                      << ": small " << out[c] << ", " << out[n + c] << ", incumbent " << ref[c] << ", " << ref[n + c] << '\n';
            ok = false;
        }
    }
    return ok;
}

template <Format F, unsigned k = 256U, unsigned n = 256U>
bool activation_range_check(const Device& d) {
    constexpr unsigned rows = 33U, ref_n = std::max(n, grouped::gemm::kBN);
    constexpr bool nv = F == Format::nvfp4;
    constexpr unsigned code_stride = nv ? k / 2U : k, scale_stride = k / (nv ? 16U : 32U);
    constexpr std::uint8_t one = nv ? 0x02U : 0x38U;
    std::vector<std::uint8_t> codes(n * code_stride), scales(n * scale_stride, nv ? 0x38U : 127U);
    for (unsigned c = 0; c < n; ++c) {
        codes[c * code_stride] = one;
        codes[(c + 1U) * code_stride - 1U] = nv ? 0x20U : one;
    }
    // The final column reads only a sub-FP16 value, distinguishing small from BF16 for ordinary rows.
    codes[(n - 1U) * code_stride] = nv ? 0x20U : 0U;
    if constexpr (!nv) codes[(n - 1U) * code_stride + 1U] = one;
    codes[n * code_stride - 1U] = 0U;

    std::vector<std::uint16_t> input(rows * k, 0U);
    std::vector<std::uint32_t> reverse(rows);
    for (unsigned r = 0; r < rows; ++r) {
        input[r * k] = bf16((r & 1U) ? -65280.0F : 65280.0F);
        input[r * k + 1U] = bf16(0x1p-30F);
        reverse[r] = rows - 1U - r;
    }
    input[k] = bf16(65536.0F);
    input[16U * k] = bf16(-65536.0F);
    input[31U * k + k - 1U] = bf16(-65536.0F);
    input[32U * k + k - 1U] = bf16(65536.0F);
    require_hip(hipMemcpy(d.gu_in, input.data(), input.size() * sizeof(input[0]), hipMemcpyHostToDevice), "range input");
    auto* device_codes = device_copy(codes);
    auto* device_scales = device_copy(scales);
    auto* map = device_copy(reverse);
    const packed::PackedMatrix w{device_codes, device_scales, n, k, 1.0F};
    const grouped::TileRows tile{w, d.gu_in, d.gu_out, k, n, rows};
    auto* ref_codes = device_codes;
    auto* ref_scales = device_scales;
    if constexpr (ref_n != n) {
        // The BF16 entry requires a full 128-column tile. Pad only its weight planes, leaving the small form's
        // matrices at their actual width so its exceptional-row fallback must stay within the final 64 columns.
        codes.resize(ref_n * code_stride);
        scales.resize(ref_n * scale_stride, nv ? 0x38U : 127U);
        ref_codes = device_copy(codes);
        ref_scales = device_copy(scales);
    }
    const grouped::TileRows ref_tile{{ref_codes, ref_scales, ref_n, k, 1.0F}, d.gu_in, d.gu_out, k, ref_n, rows};
    bool ok = true;
    for (const auto* indices : {static_cast<std::uint32_t*>(nullptr), map}) {
        const auto out = edge_projection<F>(d, tile, grouped::Form::small_f16, indices);
        const auto ref = edge_projection<F>(d, ref_tile, grouped::Form::bf16_tile, indices);
        if (out.empty() || ref.empty()) {
            ok = false;
            continue;
        }
        for (unsigned r = 0; r < rows; ++r) {
            const unsigned source = indices ? reverse[r] : r;
            const bool outside = source == 1U || source == 16U || source == 31U || source == 32U;
            const float expected = widen(input[source * k]) + widen(input[(source + 1U) * k - 1U]);
            if (out[r * n] != expected ||
                (outside ? std::memcmp(out.data() + r * n, ref.data() + r * ref_n, n * sizeof(float)) != 0
                         : out[(r + 1U) * n - 1U] != 0.0F || ref[r * ref_n + n - 1U] != 0x1p-30F)) {
                std::cerr << "BF16 activation range (" << (nv ? "NVFP4" : "MXFP8") << ", "
                          << (indices ? "indirect" : "direct") << "), K=" << k << " N=" << n
                          << ", input row " << source << '\n';
                ok = false;
            }
        }
    }
    if constexpr (ref_n != n) {
        require_hip(hipFree(ref_codes), "free padded edge codes");
        require_hip(hipFree(ref_scales), "free padded edge scales");
    }
    require_hip(hipFree(map), "free edge map");
    require_hip(hipFree(device_codes), "free edge codes");
    require_hip(hipFree(device_scales), "free edge scales");
    return ok;
}

template <Format F>
void run_tiles(const Plan& p, const Device& d, grouped::Form form) {
    require_hip(grouped::launch<F>(args_of(d.gu_tiles, unsigned(p.gu.size()), kHidden, kGate, p.max_rows, form), nullptr),
                "gate_up");
    require_hip(grouped::launch<F>(args_of(d.dn_tiles, unsigned(p.dn.size()), kInter, kHidden, p.max_rows, form), nullptr),
                "down");
}

template <Format F>
void run_r1(const Format_& Fm, const Device& d, unsigned rows) {
    for (unsigned r0 = 0; r0 < rows; r0 += r1::kMaxRows) {
        const unsigned n = std::min(r1::kMaxRows, rows - r0);
        const r1::Args a{Fm.dev, d.ids + r0 * kTopK, d.weights + r0 * kTopK, d.failure + r0, d.x + std::size_t{r0} * kHidden,
                         d.act, d.y + std::size_t{r0} * kHidden, kExperts, kHidden, kInter, 4.0F, 25.0F};
        require_hip(r1::launch<F, kTopK>(a, nullptr, n), "r1");
    }
}

void upload_plan(const Plan& p, const Device& d) {
    require_hip(hipMemcpy(d.gu_tiles, p.gu.data(), p.gu.size() * sizeof(grouped::TileRows), hipMemcpyHostToDevice), "t");
    require_hip(hipMemcpy(d.dn_tiles, p.dn.data(), p.dn.size() * sizeof(grouped::TileRows), hipMemcpyHostToDevice), "t");
}

template <Format F>
bool check(const Format_& Fm, const Device& d, std::mt19937& rng) {
    constexpr unsigned rows = 8U;
    const auto routes = draw(rows, false, rng);
    const auto p = plan_of(Fm, routes, rows, d.gu_in, d.gu_out, d.dn_in, d.dn_out);
    const unsigned A = unsigned(p.token.size());
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<std::uint16_t> xg(std::size_t{A} * kHidden), xd(std::size_t{A} * kInter);
    for (auto& v : xg) v = bf16(nd(rng));
    for (auto& v : xd) v = bf16(nd(rng));
    require_hip(hipMemcpy(d.gu_in, xg.data(), xg.size() * 2U, hipMemcpyHostToDevice), "x");
    require_hip(hipMemcpy(d.dn_in, xd.data(), xd.size() * 2U, hipMemcpyHostToDevice), "x");
    upload_plan(p, d);
    // Reference (double) per tile row.
    std::vector<double> ref_gu(std::size_t{A} * kGate), ref_dn(std::size_t{A} * kHidden), mag_gu(ref_gu.size()),
        mag_dn(ref_dn.size());
    unsigned a = 0;
    for (std::size_t t = 0; t < p.gu.size(); ++t) {
        unsigned e = 0;
        while (Fm.host[e].gate_up.codes != p.gu[t].weights.codes) ++e;
        for (unsigned r = 0; r < p.gu[t].rows; ++r, ++a) {
            for (unsigned n = 0; n < kGate; ++n) {
                const auto prow = packed::physical_row(packed::RowOrder::interleaved_halves, kGate, n);
                double s = 0.0, m = 0.0;
                for (unsigned k = 0; k < kHidden; ++k) {
                    const double v = double(widen(xg[std::size_t{a} * kHidden + k])) * weight(F, Fm.gu[e], Fm.outer[e], kHidden, prow, k);
                    s += v;
                    m += std::fabs(v);
                }
                ref_gu[std::size_t{a} * kGate + n] = s;
                mag_gu[std::size_t{a} * kGate + n] = m;
            }
            for (unsigned n = 0; n < kHidden; ++n) {
                double s = 0.0, m = 0.0;
                for (unsigned k = 0; k < kInter; ++k) {
                    const double v = double(widen(xd[std::size_t{a} * kInter + k])) * weight(F, Fm.dn[e], Fm.outer[e], kInter, n, k);
                    s += v;
                    m += std::fabs(v);
                }
                ref_dn[std::size_t{a} * kHidden + n] = s;
                mag_dn[std::size_t{a} * kHidden + n] = m;
            }
        }
    }
    bool ok = true;
    std::vector<float> gu(ref_gu.size()), dn(ref_dn.size());
    double small_err = 0.0;
    for (auto form : {grouped::Form::bf16_tile, grouped::Form::small_f16}) {
        require_hip(hipMemset(d.gu_out, 0xFF, gu.size() * 4U), "m");
        require_hip(hipMemset(d.dn_out, 0xFF, dn.size() * 4U), "m");
        run_tiles<F>(p, d, form);
        require_hip(hipDeviceSynchronize(), "sync");
        require_hip(hipMemcpy(gu.data(), d.gu_out, gu.size() * 4U, hipMemcpyDeviceToHost), "y");
        require_hip(hipMemcpy(dn.data(), d.dn_out, dn.size() * 4U, hipMemcpyDeviceToHost), "y");
        double eg = 0.0, ed = 0.0;
        for (std::size_t i = 0; i < gu.size(); ++i) worse(eg, std::fabs(double(gu[i]) - ref_gu[i]) / (mag_gu[i] + 1e-30));
        for (std::size_t i = 0; i < dn.size(); ++i) worse(ed, std::fabs(double(dn[i]) - ref_dn[i]) / (mag_dn[i] + 1e-30));
        const bool small = form == grouped::Form::small_f16;
        if (!finite(gu) || !finite(dn)) ok = false;
        if (small) small_err = std::max(eg, ed);
        std::cout << (F == Format::nvfp4 ? "nvfp4" : "mxfp8") << (small ? " small_f16" : " bf16_tile")
                  << " vs double: max |err| / sum|xw| gate_up " << eg << " down " << ed << '\n';
    }
    if (!(small_err < 1e-5)) ok = false;
    // Row invariance: the small form on one-row tiles (each tile's first row) gives row 0 of each tile bitwise.
    auto one = p;
    for (auto& t : one.gu) t.rows = 1U;
    for (auto& t : one.dn) t.rows = 1U;
    one.max_rows = 1U;
    upload_plan(one, d);
    std::vector<float> gu1(gu.size()), dn1(dn.size());
    require_hip(hipMemset(d.gu_out, 0xFF, gu.size() * 4U), "m");
    require_hip(hipMemset(d.dn_out, 0xFF, dn.size() * 4U), "m");
    run_tiles<F>(one, d, grouped::Form::small_f16);
    require_hip(hipDeviceSynchronize(), "sync");
    require_hip(hipMemcpy(gu1.data(), d.gu_out, gu.size() * 4U, hipMemcpyDeviceToHost), "y");
    require_hip(hipMemcpy(dn1.data(), d.dn_out, dn.size() * 4U, hipMemcpyDeviceToHost), "y");
    unsigned diff = 0U, first = 0U;
    for (std::size_t t = 0; t < p.gu.size(); ++t) {
        diff += std::memcmp(&gu[std::size_t{first} * kGate], &gu1[std::size_t{first} * kGate], kGate * 4U) != 0;
        diff += std::memcmp(&dn[std::size_t{first} * kHidden], &dn1[std::size_t{first} * kHidden], kHidden * 4U) != 0;
        first += p.gu[t].rows;
    }
    std::cout << (F == Format::nvfp4 ? "nvfp4" : "mxfp8") << " small_f16 row invariance (tile row 0 alone vs in its tile): "
              << (diff == 0U ? "SAME" : "DIFF") << " (" << p.gu.size() << " tiles)\n";
    if (diff != 0U) ok = false;
    std::cout << (F == Format::nvfp4 ? "nvfp4" : "mxfp8") << " check " << (ok ? "PASS" : "FAIL") << '\n';
    return ok;
}

template <Format F>
void bench(const Format_& Fm, const Device& d, std::mt19937& rng) {
    const char* name = F == Format::nvfp4 ? "nvfp4" : "mxfp8";
    std::normal_distribution<float> nd(0.0F, 1.0F);
    {
        std::vector<std::uint16_t> xg(std::size_t{kMaxAssign} * kHidden), xd(std::size_t{kMaxAssign} * kInter);
        for (auto& v : xg) v = bf16(nd(rng));
        for (auto& v : xd) v = bf16(nd(rng));
        require_hip(hipMemcpy(d.gu_in, xg.data(), xg.size() * 2U, hipMemcpyHostToDevice), "x");
        require_hip(hipMemcpy(d.dn_in, xd.data(), xd.size() * 2U, hipMemcpyHostToDevice), "x");
        std::vector<float> x(std::size_t{kMaxRows} * kHidden);
        for (auto& v : x) v = nd(rng);
        require_hip(hipMemcpy(d.x, x.data(), x.size() * 4U, hipMemcpyHostToDevice), "x");
    }
    hipEvent_t e0, e1;
    require_hip(hipEventCreate(&e0), "e");
    require_hip(hipEventCreate(&e1), "e");
    for (unsigned rows : {1U, 2U, 4U, 8U, 16U, 40U}) {
        for (bool skewed : {false, true}) {
            std::vector<Routes> rs;
            std::vector<Plan> ps;
            double distinct = 0.0;
            for (unsigned v = 0; v < kVariants; ++v) {
                rs.push_back(draw(rows, skewed, rng));
                ps.push_back(plan_of(Fm, rs.back(), rows, d.gu_in, d.gu_out, d.dn_in, d.dn_out));
                distinct += rs.back().distinct;
            }
            distinct /= kVariants;
            const auto time = [&](auto&& body) {
                std::vector<float> ms;
                for (unsigned rep = 0; rep < 4U * kVariants; ++rep) {
                    const unsigned v = rep % kVariants;
                    std::vector<float> wts(rs[v].ids.size(), 0.125F);
                    require_hip(hipMemcpy(d.ids, rs[v].ids.data(), rs[v].ids.size() * 4U, hipMemcpyHostToDevice), "ids");
                    require_hip(hipMemcpy(d.weights, wts.data(), wts.size() * 4U, hipMemcpyHostToDevice), "w");
                    upload_plan(ps[v], d);
                    require_hip(hipDeviceSynchronize(), "sync");
                    require_hip(hipEventRecord(e0, nullptr), "rec");
                    body(ps[v]);
                    require_hip(hipEventRecord(e1, nullptr), "rec");
                    require_hip(hipEventSynchronize(e1), "sync");
                    float t = 0.0F;
                    require_hip(hipEventElapsedTime(&t, e0, e1), "el");
                    if (rep >= kVariants) ms.push_back(t);  // first round warms up
                }
                std::sort(ms.begin(), ms.end());
                return double(ms[ms.size() / 2U]);
            };
            const double t_r1 = time([&](const Plan&) { run_r1<F>(Fm, d, rows); });
            const double t_tile = time([&](const Plan& p) { run_tiles<F>(p, d, grouped::Form::bf16_tile); });
            const double t_small = time([&](const Plan& p) { run_tiles<F>(p, d, grouped::Form::small_f16); });
            const double bytes = distinct * Fm.expert_bytes;
            std::cout << name << " rows " << rows << (skewed ? " skewed " : " uniform") << " U " << distinct
                      << " ms r1 " << t_r1 << " tile " << t_tile << " small " << t_small << " | GB/s r1 "
                      << bytes / t_r1 * 1e-6 << " tile " << bytes / t_tile * 1e-6 << " small " << bytes / t_small * 1e-6
                      << " | small/r1 " << t_small / t_r1 << '\n';
        }
    }
}

// Largest |a - b| over the row's root mean square of b (b: the BF16 tile), across all rows of width n.
double max_rel(const std::vector<float>& a, const std::vector<float>& b, std::size_t n) {
    double worst = 0.0;
    for (std::size_t row = 0; row < b.size() / n; ++row) {
        double ss = 0.0;
        for (std::size_t i = 0; i < n; ++i) ss += double(b[row * n + i]) * double(b[row * n + i]);
        const double rms = std::sqrt(ss / double(n)) + 1e-30;
        for (std::size_t i = 0; i < n; ++i) worse(worst, std::fabs(double(a[row * n + i]) - double(b[row * n + i])) / rms);
    }
    return worst;
}

// Rows of width n whose bytes differ between a and b.
unsigned rows_differ(const std::vector<float>& a, const std::vector<float>& b, std::size_t n) {
    unsigned diff = 0U;
    for (std::size_t row = 0; row < a.size() / n; ++row) diff += std::memcmp(&a[row * n], &b[row * n], n * 4U) != 0;
    return diff;
}

// (2) Tile launches over `rows` token rows: the small form against the BF16 tile, and every row of every tile run alone
// as a one-row tile at its own input / output rows: the same bits as in its tile, whatever its position.
template <Format F>
bool tile_rows_check(const Format_& Fm, const Device& d, std::mt19937& rng, unsigned rows) {
    const char* name = F == Format::nvfp4 ? "nvfp4" : "mxfp8";
    const auto routes = draw(rows, false, rng);
    const auto p = plan_of(Fm, routes, rows, d.gu_in, d.gu_out, d.dn_in, d.dn_out);
    const unsigned A = unsigned(p.token.size());
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<std::uint16_t> xg(std::size_t{A} * kHidden), xd(std::size_t{A} * kInter);
    for (auto& v : xg) v = bf16(nd(rng));
    for (auto& v : xd) v = bf16(nd(rng));
    require_hip(hipMemcpy(d.gu_in, xg.data(), xg.size() * 2U, hipMemcpyHostToDevice), "x");
    require_hip(hipMemcpy(d.dn_in, xd.data(), xd.size() * 2U, hipMemcpyHostToDevice), "x");
    Plan one;
    one.max_rows = 1U;
    for (std::size_t t = 0; t < p.gu.size(); ++t) {
        for (unsigned r = 0; r < p.gu[t].rows; ++r) {
            auto g = p.gu[t];
            auto n = p.dn[t];
            g.input += std::size_t{r} * g.input_row;
            g.output += std::size_t{r} * g.output_row;
            n.input += std::size_t{r} * n.input_row;
            n.output += std::size_t{r} * n.output_row;
            g.rows = n.rows = 1U;
            one.gu.push_back(g);
            one.dn.push_back(n);
        }
    }
    const auto run = [&](const Plan& plan, grouped::Form form, std::vector<float>& gu, std::vector<float>& dn) {
        gu.assign(std::size_t{A} * kGate, 0.0F);
        dn.assign(std::size_t{A} * kHidden, 0.0F);
        require_hip(hipMemset(d.gu_out, 0xFF, gu.size() * 4U), "m");  // NaN sentinel: an unwritten output fails
        require_hip(hipMemset(d.dn_out, 0xFF, dn.size() * 4U), "m");
        upload_plan(plan, d);
        run_tiles<F>(plan, d, form);
        require_hip(hipDeviceSynchronize(), "sync");
        require_hip(hipMemcpy(gu.data(), d.gu_out, gu.size() * 4U, hipMemcpyDeviceToHost), "y");
        require_hip(hipMemcpy(dn.data(), d.dn_out, dn.size() * 4U, hipMemcpyDeviceToHost), "y");
    };
    std::vector<float> gs, ds, gt, dt, g1, d1;
    run(p, grouped::Form::small_f16, gs, ds);
    run(p, grouped::Form::bf16_tile, gt, dt);
    run(one, grouped::Form::small_f16, g1, d1);
    const bool all_finite = finite(gs) && finite(ds) && finite(gt) && finite(dt) && finite(g1) && finite(d1);
    const double eg = max_rel(gs, gt, kGate), ed = max_rel(ds, dt, kHidden);
    const unsigned diff = rows_differ(gs, g1, kGate) + rows_differ(ds, d1, kHidden);
    // Against the BF16 tile, not a reference: for NVFP4 the tile rounds decoded weights to BF16 (about 1 % of a row's rms
    // here); the accuracy of the small form is judged against the double reference in check().
    const bool ok = all_finite && eg < 5e-2 && ed < 5e-2 && diff == 0U;
    std::cout << name << " tile rows " << rows << " (" << A << " assignments, " << p.gu.size() << " tiles, max tile "
              << p.max_rows << "): finite " << (all_finite ? "yes" : "NO") << "; small vs bf16_tile max |d|/rms gate_up "
              << eg << " down " << ed << "; every row alone vs in its tile: " << (diff == 0U ? "SAME" : "DIFF") << " ("
              << diff << " of " << 2U * A << " rows differ) " << (ok ? "PASS" : "FAIL") << '\n';
    return ok;
}

// (3) The full route through the shared dispatcher (grouped_dispatch::enqueue) at the fixture's expert shapes.
enum class Proj : std::uint8_t { gate_up, down };
template <Format F>
struct RouteConfig {
    using Projection = Proj;
    static constexpr unsigned hidden = kHidden;
    static constexpr unsigned intermediate = kInter;
    static constexpr unsigned experts = kExperts;
    static constexpr unsigned top_k = kTopK;
    static constexpr unsigned layers = 1U;
    static constexpr unsigned max_rows = kMaxRows;
    static constexpr Format format(unsigned) { return F; }
    static constexpr packed::RowOrder row_order(Proj projection) {
        return projection == Proj::gate_up ? packed::RowOrder::interleaved_halves : packed::RowOrder::contiguous;
    }
    struct Caps {
        static constexpr float gate = 4.0F;
        static constexpr float up = 25.0F;
    };
    using Activation = dispatch::SoftcappedSwigluBf16<Caps>;
    using Combine = dispatch::Bf16Combine;
};

struct RouteDevice {
    float *input{}, *output{}, *alone{}, *fp32{}, *assignment_weights{};
    std::uint16_t *bf16{}, *activation{};
    std::uint32_t *assignment_rows{}, *combine_rows{};
    grouped::TileRows *gate_up{}, *down{};
    unsigned* failure{};
};

RouteDevice make_route_device() {
    const auto need = dispatch::workspace_requirement<RouteConfig<Format::nvfp4>>(kMaxRows);  // shape only
    RouteDevice d;
    const auto rows_bytes = std::size_t{kMaxRows} * kHidden * 4U;
    require_hip(hipMalloc(&d.input, rows_bytes), "m");
    require_hip(hipMalloc(&d.output, rows_bytes), "m");
    require_hip(hipMalloc(&d.alone, rows_bytes), "m");
    require_hip(hipMalloc(&d.fp32, need.fp32 * 4U), "m");
    require_hip(hipMalloc(&d.bf16, need.bf16 * 2U), "m");
    require_hip(hipMalloc(&d.activation, need.activation * 2U), "m");
    require_hip(hipMalloc(&d.assignment_rows, kMaxAssign * 4U), "m");
    require_hip(hipMalloc(&d.combine_rows, kMaxAssign * 4U), "m");
    require_hip(hipMalloc(&d.assignment_weights, kMaxAssign * 4U), "m");
    require_hip(hipMalloc(&d.gate_up, kMaxAssign * sizeof(grouped::TileRows)), "m");
    require_hip(hipMalloc(&d.down, kMaxAssign * sizeof(grouped::TileRows)), "m");
    require_hip(hipMalloc(&d.failure, 4U), "m");
    require_hip(hipMemset(d.failure, 0, 4U), "m");
    return d;
}

template <Format F>
dispatch::Workspace route_workspace(const RouteDevice& d, unsigned rows) {
    const auto need = dispatch::workspace_requirement<RouteConfig<F>>(rows);
    return {{d.bf16, need.bf16}, {d.fp32, need.fp32}, {d.activation, need.activation}};
}

// Host plan of `rows` token rows (the shared expert-major planner), uploaded as the device plan of one enqueue.
template <Format F>
dispatch::DevicePlan upload_route(const Format_& Fm, const RouteDevice& d, unsigned rows,
                                  std::span<const std::uint32_t> ids, std::span<const float> weights, grouped::Form form) {
    const auto plan = dispatch::prepare_form<RouteConfig<F>, dispatch::HipBackend>(
        0U, rows, ids, weights, std::span<const ExpertWeights>(Fm.host), route_workspace<F>(d, rows), form);
    const std::size_t A = std::size_t{rows} * kTopK;
    require_hip(hipMemcpy(d.assignment_rows, plan.routes.grouping.assignment_rows.data(), A * 4U, hipMemcpyHostToDevice), "p");
    require_hip(hipMemcpy(d.combine_rows, plan.routes.combine_rows.data(), A * 4U, hipMemcpyHostToDevice), "p");
    require_hip(hipMemcpy(d.assignment_weights, plan.routes.assignment_weights.data(), A * 4U, hipMemcpyHostToDevice), "p");
    require_hip(hipMemcpy(d.gate_up, plan.gate_up.data(), plan.gate_up.size() * sizeof(grouped::TileRows),
                          hipMemcpyHostToDevice), "p");
    require_hip(hipMemcpy(d.down, plan.down.data(), plan.down.size() * sizeof(grouped::TileRows), hipMemcpyHostToDevice),
                "p");
    dispatch::DevicePlan device{0U, rows, {d.assignment_rows, A}, {d.combine_rows, A}, {d.assignment_weights, A},
                                {d.gate_up, plan.gate_up.size()}, {d.down, plan.down.size()}, nullptr, d.failure};
    device.form = form;
    return device;
}

template <Format F>
void enqueue_route(const dispatch::DevicePlan& plan, const RouteDevice& d, unsigned rows, const float* input,
                   float* output) {
    const std::size_t n = std::size_t{rows} * kHidden;
    require_hip(dispatch::enqueue<RouteConfig<F>, dispatch::HipBackend>(plan, {input, n}, {output, n},
                                                                        route_workspace<F>(d, rows), nullptr),
                "enqueue");
}

template <Format F>
bool route_check(const Format_& Fm, const RouteDevice& d, std::mt19937& rng) {
    const char* name = F == Format::nvfp4 ? "nvfp4" : "mxfp8";
    bool ok = true;
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::uniform_real_distribution<float> ud(0.05F, 0.3F);
    for (unsigned rows : {1U, 8U, 16U, 17U, 33U, 64U}) {
        const auto routes = draw(rows, false, rng);
        const std::vector<std::uint32_t> ids(routes.ids.begin(), routes.ids.end());
        std::vector<float> weights(ids.size()), x(std::size_t{rows} * kHidden);
        for (auto& w : weights) w = ud(rng);
        for (auto& v : x) v = nd(rng);
        require_hip(hipMemcpy(d.input, x.data(), x.size() * 4U, hipMemcpyHostToDevice), "x");
        const auto run = [&](grouped::Form form, std::vector<float>& batch, std::vector<float>& alone) {
            batch.assign(x.size(), 0.0F);
            alone.assign(x.size(), 0.0F);
            require_hip(hipMemset(d.output, 0xFF, x.size() * 4U), "m");  // NaN sentinel
            require_hip(hipMemset(d.alone, 0xFF, x.size() * 4U), "m");
            enqueue_route<F>(upload_route<F>(Fm, d, rows, ids, weights, form), d, rows, d.input, d.output);
            require_hip(hipDeviceSynchronize(), "sync");
            for (unsigned r = 0; r < rows; ++r) {
                const auto one = upload_route<F>(Fm, d, 1U, std::span(ids).subspan(r * kTopK, kTopK),
                                                 std::span(weights).subspan(r * kTopK, kTopK), form);
                enqueue_route<F>(one, d, 1U, d.input + std::size_t{r} * kHidden, d.alone + std::size_t{r} * kHidden);
                require_hip(hipDeviceSynchronize(), "sync");
            }
            require_hip(hipMemcpy(batch.data(), d.output, x.size() * 4U, hipMemcpyDeviceToHost), "y");
            require_hip(hipMemcpy(alone.data(), d.alone, x.size() * 4U, hipMemcpyDeviceToHost), "y");
        };
        std::vector<float> sb, sa, tb, ta;
        run(grouped::Form::small_f16, sb, sa);
        run(grouped::Form::bf16_tile, tb, ta);
        unsigned failure = 1U;
        require_hip(hipMemcpy(&failure, d.failure, 4U, hipMemcpyDeviceToHost), "f");
        const bool all_finite = finite(sb) && finite(sa) && finite(tb) && finite(ta);
        const double err = max_rel(sb, tb, kHidden);
        const unsigned small_diff = rows_differ(sb, sa, kHidden), tile_diff = rows_differ(tb, ta, kHidden);
        const bool pass = all_finite && failure == 0U && err < 5e-2 && small_diff == 0U;
        ok = ok && pass;
        std::cout << name << " route rows " << rows << ": finite " << (all_finite ? "yes" : "NO") << ", failure word "
                  << failure << "; small vs bf16_tile max |d|/rms " << err << "; each row alone vs in its batch: small "
                  << (small_diff == 0U ? "SAME" : "DIFF") << " (" << small_diff << " rows differ), bf16_tile "
                  << (tile_diff == 0U ? "SAME" : "DIFF") << " (" << tile_diff << ") " << (pass ? "PASS" : "FAIL") << '\n';
    }
    return ok;
}

// ms of one full route (enqueue from an uploaded plan: token rows, gate-up, activation, down, combine; the device-side
// routing of a served step is not included) for the BF16 tile and the small form, beside R1 (activation and combine
// included), uniform routes.
template <Format F>
void route_bench(const Format_& Fm, const Device& d, const RouteDevice& rd, std::mt19937& rng) {
    const char* name = F == Format::nvfp4 ? "nvfp4" : "mxfp8";
    std::normal_distribution<float> nd(0.0F, 1.0F);
    {
        std::vector<float> x(std::size_t{kMaxRows} * kHidden);
        for (auto& v : x) v = nd(rng);
        require_hip(hipMemcpy(rd.input, x.data(), x.size() * 4U, hipMemcpyHostToDevice), "x");
    }
    hipEvent_t e0, e1;
    require_hip(hipEventCreate(&e0), "e");
    require_hip(hipEventCreate(&e1), "e");
    for (unsigned rows : {1U, 8U, 16U}) {
        std::vector<std::vector<std::uint32_t>> ids;
        double distinct = 0.0;
        for (unsigned v = 0; v < kVariants; ++v) {
            const auto r = draw(rows, false, rng);
            ids.emplace_back(r.ids.begin(), r.ids.end());
            distinct += r.distinct;
        }
        distinct /= kVariants;
        const std::vector<float> wts(std::size_t{rows} * kTopK, 0.125F);
        const auto time = [&](auto&& body) {
            std::vector<float> ms;
            for (unsigned rep = 0; rep < 4U * kVariants; ++rep) {
                const unsigned v = rep % kVariants;
                require_hip(hipMemcpy(d.ids, ids[v].data(), ids[v].size() * 4U, hipMemcpyHostToDevice), "ids");
                require_hip(hipMemcpy(d.weights, wts.data(), wts.size() * 4U, hipMemcpyHostToDevice), "w");
                const auto small = upload_route<F>(Fm, rd, rows, ids[v], wts, grouped::Form::small_f16);
                auto tile = small;
                tile.form = grouped::Form::bf16_tile;
                require_hip(hipDeviceSynchronize(), "sync");
                require_hip(hipEventRecord(e0, nullptr), "rec");
                body(small, tile);
                require_hip(hipEventRecord(e1, nullptr), "rec");
                require_hip(hipEventSynchronize(e1), "sync");
                float t = 0.0F;
                require_hip(hipEventElapsedTime(&t, e0, e1), "el");
                if (rep >= kVariants) ms.push_back(t);  // first round warms up
            }
            std::sort(ms.begin(), ms.end());
            return double(ms[ms.size() / 2U]);
        };
        using DP = dispatch::DevicePlan;
        const double t_r1 = time([&](const DP&, const DP&) { run_r1<F>(Fm, d, rows); });
        const double t_tile = time([&](const DP&, const DP& t) { enqueue_route<F>(t, rd, rows, rd.input, rd.output); });
        const double t_small = time([&](const DP& s, const DP&) { enqueue_route<F>(s, rd, rows, rd.input, rd.output); });
        std::cout << name << " route rows " << rows << " uniform U " << distinct << " ms r1 " << t_r1 << " bf16_tile "
                  << t_tile << " small " << t_small << " | small/r1 " << t_small / t_r1 << " small/bf16_tile "
                  << t_small / t_tile << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? std::string(argv[1]) : std::string();
    const bool bench_only = mode == "--bench-only", check_only = mode == "--check-only";
    std::mt19937 rng(20261123U);
    const auto dev = make_device();
    const auto route = make_route_device();
    bool ok = true;
    if (!bench_only) {
        const bool mxfp8_edges = mxfp8_edge_check(dev);
        const bool nvfp4_activations = activation_range_check<Format::nvfp4>(dev);
        const bool mxfp8_activations = activation_range_check<Format::mxfp8>(dev);
        const bool nvfp4_last_panel = activation_range_check<Format::nvfp4, 128U, 64U>(dev);
        const bool mxfp8_last_panel = activation_range_check<Format::mxfp8, 128U, 64U>(dev);
        ok = mxfp8_edges && nvfp4_activations && mxfp8_activations && nvfp4_last_panel && mxfp8_last_panel;
        std::cout << "small-form weight code, scale and activation edges: " << (ok ? "PASS" : "FAIL") << '\n';
    }
    {
        auto nv = make_format(Format::nvfp4, rng);
        if (!bench_only) {
            ok = check<Format::nvfp4>(nv, dev, rng) && ok;
            for (unsigned rows : {1U, 16U, 17U, 32U, 33U, 48U, 64U})
                ok = tile_rows_check<Format::nvfp4>(nv, dev, rng, rows) && ok;
            ok = route_check<Format::nvfp4>(nv, route, rng) && ok;
        }
        if (!check_only) {
            bench<Format::nvfp4>(nv, dev, rng);
            route_bench<Format::nvfp4>(nv, dev, route, rng);
        }
    }
    {
        auto mx = make_format(Format::mxfp8, rng);
        if (!bench_only) {
            ok = check<Format::mxfp8>(mx, dev, rng) && ok;
            for (unsigned rows : {1U, 16U, 17U, 32U, 33U, 48U, 64U})
                ok = tile_rows_check<Format::mxfp8>(mx, dev, rng, rows) && ok;
            ok = route_check<Format::mxfp8>(mx, route, rng) && ok;
        }
        if (!check_only) {
            bench<Format::mxfp8>(mx, dev, rng);
            route_bench<Format::mxfp8>(mx, dev, route, rng);
        }
    }
    std::cout << (ok ? "ALL PASS" : "FAILED") << '\n';
    return ok ? 0 : 1;
}
