// EXL3 dense reader (strix/weight/formats/exl3_dense.hpp), the source side of the trellis -> Strix pack transcoder:
//  1. the Hadamard over 128-blocks is involutory and orthonormal (H H x = x, |H x| = |x|);
//  2. the dense weight W reproduces the device's staged operator y = svh (.) H( Q^T H( suh (.) x ) ) on synthetic
//     packs (2..8 bits, several shapes), staged here in FP32 without the device's FP16 roundings;
//  3. the transformed-domain coefficients equal the shared registry decoder's (src/weight/formats/exl3.cpp layout:
//     output-major rows, Format::tile(input tile, output tile));
//  4. malformed shapes, ring sizes and nonfinite scales are refused.
#include "strix/weight/formats/exl3_dense.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

namespace ed = strix::weight::exl3_dense;

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}

template <class F> bool refuses(F&& f) {
    try {
        f();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

std::uint16_t half_of(float x) {
    std::uint16_t h = 0;
    (void)strix::weight::ieee_half::encode_finite(x, h);
    return h;
}

struct Pack {
    std::vector<std::uint8_t> rings;
    std::vector<std::uint16_t> suh, svh;
    ed::Linear linear;
};

Pack synthetic(unsigned inputs, unsigned outputs, unsigned bits, std::mt19937& rng) {
    Pack p;
    const std::size_t tiles = std::size_t{inputs / 16U} * (outputs / 16U);
    p.rings.resize(tiles * 32U * bits);
    for (auto& byte : p.rings) byte = static_cast<std::uint8_t>(rng() & 255U);
    std::uniform_real_distribution<float> scale(-0.25F, 0.25F);
    for (unsigned i = 0; i < inputs; ++i) p.suh.push_back(half_of(scale(rng)));
    for (unsigned o = 0; o < outputs; ++o) p.svh.push_back(half_of(scale(rng)));
    p.linear = {p.rings, p.suh, p.svh, inputs, outputs, bits};
    return p;
}

void hadamard_properties(std::mt19937& rng) {
    std::normal_distribution<float> g;
    std::vector<float> x(256), y;
    for (auto& v : x) v = g(rng);
    y = x;
    ed::hadamard128(y.data(), y.size());
    double nx = 0, ny = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        nx += double(x[i]) * x[i];
        ny += double(y[i]) * y[i];
    }
    check(std::fabs(nx - ny) <= 1e-4 * nx, "the 128-block Hadamard keeps the norm");
    ed::hadamard128(y.data(), y.size());
    float worst = 0;
    for (std::size_t i = 0; i < x.size(); ++i) worst = std::fmax(worst, std::fabs(y[i] - x[i]));
    check(worst <= 1e-5F, "the 128-block Hadamard is its own inverse");
    std::vector<float> unit(128, 0.0F);
    unit[0] = 1.0F;
    ed::hadamard128(unit.data(), unit.size());
    bool flat = true;
    for (const auto v : unit) flat = flat && std::fabs(v - 1.0F / std::sqrt(128.0F)) <= 1e-7F;
    check(flat, "H e0 is the flat 1 / sqrt(128) vector (Sylvester, natural order)");
}

void staged_equivalence(unsigned inputs, unsigned outputs, unsigned bits, std::mt19937& rng) {
    const auto pack = synthetic(inputs, outputs, bits, rng);
    const auto w = ed::dense_rows(pack.linear);
    std::normal_distribution<float> g;
    std::vector<float> x(inputs);
    for (auto& v : x) v = g(rng);
    // Staged operator: a = H(suh x); p = Q^T a; y = svh H(p).
    std::vector<float> a(inputs), p(outputs, 0.0F);
    for (unsigned k = 0; k < inputs; ++k) a[k] = x[k] * strix::weight::ieee_half::decode(pack.suh[k]);
    ed::hadamard128(a.data(), a.size());
    for (unsigned n = 0; n < outputs; ++n) {
        double acc = 0;
        for (unsigned k = 0; k < inputs; ++k) acc += double(ed::coefficient(pack.linear, k, n)) * a[k];
        p[n] = static_cast<float>(acc);
    }
    ed::hadamard128(p.data(), p.size());
    double err = 0, ref = 0;
    for (unsigned n = 0; n < outputs; ++n) {
        double y = 0;
        for (unsigned k = 0; k < inputs; ++k) y += double(w[std::size_t{n} * inputs + k]) * x[k];
        const double staged = double(p[n]) * strix::weight::ieee_half::decode(pack.svh[n]);
        err = std::fmax(err, std::fabs(y - staged));
        ref = std::fmax(ref, std::fabs(staged));
    }
    char what[128];
    std::snprintf(what, sizeof what, "dense W x equals the staged operator (%u x %u, %u bits): err %.3g of %.3g",
                  inputs, outputs, bits, err, ref);
    check(ref > 0 && err <= 2e-5 * ref, what);
}

void registry_layout(std::mt19937& rng) {
    // decode_exl3 (shared registry) reads out[row * width + col] with col = input, row = output and tile
    // Format::tile(col / 16, row / 16, rows / 16): the reader's coefficient(k, n) must address the same ring state.
    const auto pack = synthetic(128, 256, 4, rng);
    using Format = strix::core::trellis::Format;
    bool same = true;
    for (unsigned n = 0; n < 256; n += 7)
        for (unsigned k = 0; k < 128; k += 5) {
            const auto* tile = pack.rings.data() + Format::tile(k / 16U, n / 16U, 256U / 16U, 4) * 4U;
            const float registry = strix::core::trellis::mul1_value(
                Format::state_bytes(tile, 4, Format::position(k % 16U, n % 16U)));
            same = same && registry == ed::coefficient(pack.linear, k, n);
        }
    check(same, "the reader's coefficients are the registry decoder's");
}

void refusals(std::mt19937& rng) {
    auto pack = synthetic(128, 128, 4, rng);
    auto bad = pack.linear;
    bad.inputs = 64;
    check(refuses([&] { (void)ed::dense_rows(bad); }), "a shape outside whole 128-blocks is refused");
    bad = pack.linear;
    bad.rings = bad.rings.first(bad.rings.size() - 1);
    check(refuses([&] { (void)ed::dense_rows(bad); }), "a short ring stream is refused");
    bad = pack.linear;
    bad.bits = 9;
    check(refuses([&] { (void)ed::dense_rows(bad); }), "a rate above 8 bits is refused");
    pack.suh[3] = 0x7c00U;  // +inf
    check(refuses([&] { (void)ed::dense_rows(pack.linear); }), "a nonfinite input scale is refused");
}

}  // namespace

int main() {
    std::mt19937 rng(20261127U);
    hadamard_properties(rng);
    for (const unsigned bits : {2U, 3U, 4U, 5U, 6U, 8U}) staged_equivalence(128, 256, bits, rng);
    staged_equivalence(384, 128, 6, rng);
    registry_layout(rng);
    refusals(rng);
    if (failures) return 1;
    std::printf("exl3 dense reader: all checks passed\n");
    return 0;
}
