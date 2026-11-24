// Host test of the NVFP4 encoder and the MXFP8 -> NVFP4 transcode (strix/weight/formats/microscaled_encode.hpp):
// (1) NVFP4-representable rows round-trip bitwise in value; (2) the mse@2 block scale is never worse than the amax rule
// alone; (3) Gaussian rows stay within the 4-bit error band; (4) a random MXFP8 matrix transcodes with a bounded relative
// error and the reported error matches a recomputation; (5) malformed inputs throw.
#include "strix/weight/formats/microscaled_encode.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <new>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

namespace ms = strix::weight::microscaled;
namespace sf = strix::weight::scalar_float;

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}

float decoded(const std::vector<std::uint8_t>& codes, const std::vector<std::uint8_t>& scales, float outer, std::size_t c) {
    const auto byte = codes[c / 2U];
    const auto code = static_cast<std::uint8_t>(c % 2U == 0U ? byte & 0x0FU : byte >> 4U);
    return ms::nvfp4_value(code, sf::e4m3(scales[c / 16U]), outer);
}

void representable(std::mt19937& rng) {
    constexpr std::size_t n = 256;
    std::uniform_int_distribution<int> code(0, 15), scale(0x10, 0x60);
    std::vector<float> x(n);
    for (std::size_t b = 0; b < n / 16U; ++b) {
        const auto s = static_cast<std::uint8_t>(scale(rng));
        for (std::size_t i = 0; i < 16U; ++i) {
            // One element per block at |code| 6 makes amax / 6 the block scale exactly.
            const auto c = i == 3U ? std::uint8_t{7} : static_cast<std::uint8_t>(code(rng));
            x[b * 16U + i] = sf::e2m1(c) * sf::e4m3(s);
        }
    }
    std::vector<std::uint8_t> codes(n / 2U), scales(n / 16U);
    ms::encode_nvfp4_row(x, 1.0F, codes, scales);
    bool same = true;
    for (std::size_t c = 0; c < n; ++c) same = same && decoded(codes, scales, 1.0F, c) == x[c];
    expect(same, "representable row round-trips");
}

void mse_rule(std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0F, 1.0F);
    double mse_total = 0.0, amax_total = 0.0;
    const float outer = 0.01F;
    for (int trial = 0; trial < 2000; ++trial) {
        float block[16];
        float amax = 0.0F;
        for (float& v : block) {
            v = nd(rng) * (trial % 7 == 0 ? 0.01F : 1.0F);
            amax = std::fmax(amax, std::fabs(v));
        }
        std::uint8_t codes[8], scale[1];
        ms::encode_nvfp4_row(std::span<const float>(block, 16), outer, codes, scale);
        const double mse = ms::nvfp4_block_error(block, outer, scale[0], nullptr);
        const auto rule = static_cast<std::uint8_t>(sf::e4m3_encode(amax / (6.0F * outer)) & 0x7F);
        const double amax_error = ms::nvfp4_block_error(block, outer, rule, nullptr);
        expect(mse <= amax_error, "mse@2 never worse than the amax rule");
        mse_total += mse;
        amax_total += amax_error;
    }
    std::printf("block SSE mse@2 %.4f amax rule %.4f (ratio %.3f)\n", mse_total, amax_total, mse_total / amax_total);
}

void transcode(std::mt19937& rng) {
    constexpr std::size_t rows = 64, cols = 1152;
    std::vector<std::uint8_t> codes(rows * cols), scales(rows * cols / 32U);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::uniform_int_distribution<int> e(118, 124);
    for (auto& s : scales) s = static_cast<std::uint8_t>(e(rng));
    for (auto& c : codes) c = sf::e4m3_encode(nd(rng) * 64.0F);
    const auto out = ms::transcode_mxfp8_to_nvfp4(codes, scales, rows, cols);
    // Recompute the error from the packed bytes.
    double sse = 0.0, energy = 0.0;
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c) {
            const double src = double{sf::e4m3(codes[r * cols + c])} * double{sf::e8m0(scales[r * (cols / 32U) + c / 32U])};
            const auto byte = out.codes[(r * cols + c) / 2U];
            const auto code = static_cast<std::uint8_t>(c % 2U == 0U ? byte & 0x0FU : byte >> 4U);
            const double d = double{ms::nvfp4_value(code, sf::e4m3(out.scales[r * (cols / 16U) + c / 16U]), out.outer)} - src;
            sse += d * d;
            energy += src * src;
        }
    const double rel = std::sqrt(sse / energy);
    std::printf("MXFP8 -> NVFP4 relative RMS error %.4f (reported %.4f), outer %g\n", rel, out.error.relative(),
                double{out.outer});
    expect(std::fabs(rel - out.error.relative()) < 1e-9, "reported error matches");
    expect(rel > 0.0 && rel < 0.15, "4-bit error band");
}

void malformed() {
    bool threw = false;
    try {
        std::vector<float> x(15, 1.0F);
        std::vector<std::uint8_t> c(7), s(1);
        ms::encode_nvfp4_row(x, 1.0F, c, s);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "short row throws");
    threw = false;
    try {
        std::vector<float> x(16, 1.0F);
        x[5] = INFINITY;
        std::vector<std::uint8_t> c(8), s(1);
        ms::encode_nvfp4_row(x, 1.0F, c, s);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "non-finite value throws");
    threw = false;
    try {
        std::vector<std::uint8_t> c(64), s(2);
        c[0] = 0x7F;  // E4M3 NaN
        (void)ms::transcode_mxfp8_to_nvfp4(c, s, 1, 64);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "inconsistent or NaN MXFP8 throws");
}

// (6) A MXFP8 expert slab read from its source: every expert equals the one-matrix transcode, the bytes do not depend on
// the thread count, the device-style binding matches a checkpoint NVFP4 slab (strides, outer per expert, row order), a
// NVFP4 slab and a NaN source are refused.
struct ThrowingSource final : strix::weight::Source {
    std::uint64_t bytes{};
    explicit ThrowingSource(std::uint64_t size) : bytes(size) {}
    strix::weight::Status read(strix::weight::ByteRange, std::span<strix::weight::Byte>) const override {
        throw std::bad_alloc{};
    }
    std::uint64_t size() const noexcept override { return bytes; }
    strix::weight::Digest256 identity() const override { return {}; }
};

void slab(std::mt19937& rng) {
    namespace w = strix::weight;
    constexpr std::size_t experts = 5, rows = 6, cols = 128;
    std::vector<std::uint8_t> bytes(experts * rows * cols + experts * rows * cols / 32U);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::uniform_int_distribution<int> e(116, 126);
    const std::size_t code_bytes = experts * rows * cols;
    for (std::size_t i = 0; i < code_bytes; ++i) bytes[i] = sf::e4m3_encode(nd(rng) * 32.0F);
    for (std::size_t i = code_bytes; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(e(rng));
    const auto source = w::memory_source(std::as_bytes(std::span(bytes)));
    const ms::ExpertSlabView view{ms::MatrixFormat::mxfp8, ms::RowOrder::interleaved_halves, experts, rows, cols,
                                  {source, {0, code_bytes}}, {source, {code_bytes, bytes.size() - code_bytes}}, {}, {}};
    ms::Nvfp4Slab one, many;
    expect(static_cast<bool>(ms::transcode_slab_to_nvfp4(view, one, 1U)), "slab transcode, one thread");
    expect(static_cast<bool>(ms::transcode_slab_to_nvfp4(view, many, 3U)), "slab transcode, three threads");
    expect(one.codes == many.codes && one.scales == many.scales && one.outer == many.outer, "thread-count invariant");
    double sse = 0.0, energy = 0.0;
    for (std::size_t x = 0; x < experts; ++x) {
        const std::span<const std::uint8_t> codes(bytes.data() + x * rows * cols, rows * cols);
        const std::span<const std::uint8_t> scales(bytes.data() + code_bytes + x * rows * cols / 32U, rows * cols / 32U);
        const auto m = ms::transcode_mxfp8_to_nvfp4(codes, scales, rows, cols);
        expect(std::equal(m.codes.begin(), m.codes.end(), one.codes.begin() + static_cast<std::ptrdiff_t>(x * one.code_stride())),
               "expert codes equal the matrix transcode");
        expect(std::equal(m.scales.begin(), m.scales.end(),
                          one.scales.begin() + static_cast<std::ptrdiff_t>(x * one.scale_stride())),
               "expert scales equal the matrix transcode");
        expect(m.outer == one.outer[x], "outer per expert");
        sse += m.error.sse;
        energy += m.error.energy;
    }
    expect(one.error.sse == sse && one.error.energy == energy, "slab error is the expert sum in order");
    std::printf("slab relative RMS error %.4f\n", one.error.relative());

    // Binding: strides and metadata of a checkpoint NVFP4 slab of the same geometry.
    std::vector<ms::PackedMatrix> matrices(experts);
    expect(static_cast<bool>(one.bind_storage(one.codes, one.scales, matrices)), "bind");
    for (std::size_t x = 0; x < experts; ++x) {
        const auto& m = matrices[x];
        expect(m.codes == one.codes.data() + x * rows * cols / 2U && m.block_scales == one.scales.data() + x * rows * cols / 16U &&
                   m.rows == rows && m.columns == cols && m.outer_scale == one.outer[x] &&
                   m.row_order == ms::RowOrder::interleaved_halves,
               "binding layout");
    }
    expect(!one.bind_storage(std::span(one.codes).first(one.codes.size() - 1U), one.scales, matrices), "short storage refused");

    ms::Nvfp4Slab refused;
    ms::ExpertSlabView fp4 = view;
    fp4.format = ms::MatrixFormat::nvfp4;
    expect(!ms::transcode_slab_to_nvfp4(fp4, refused), "NVFP4 slab refused");
    bytes[3 * rows * cols + 7] = 0x7F;  // E4M3 NaN in expert 3 (memory_source keeps its own copy: a new source)
    ms::ExpertSlabView nan_view = view;
    nan_view.codes.source = nan_view.block_scales.source = w::memory_source(std::as_bytes(std::span(bytes)));
    const auto status = ms::transcode_slab_to_nvfp4(nan_view, refused, 2U);
    expect(!status && refused.codes.empty(), "NaN source refused, output untouched");

    // Source::read may throw (e.g. an allocation failure). The worker must
    // propagate to the caller after joining, not terminate the process.
    auto throwing_view = view;
    throwing_view.codes.source = throwing_view.block_scales.source =
        std::make_shared<ThrowingSource>(bytes.size());
    bool caught = false;
    try {
        (void)ms::transcode_slab_to_nvfp4(throwing_view, refused, 3U);
    } catch (const std::bad_alloc&) {
        caught = true;
    }
    expect(caught && refused.codes.empty(), "worker exception propagated, output untouched");
}

}  // namespace

int main() {
    std::mt19937 rng(1123U);
    representable(rng);
    mse_rule(rng);
    transcode(rng);
    malformed();
    slab(rng);
    std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
